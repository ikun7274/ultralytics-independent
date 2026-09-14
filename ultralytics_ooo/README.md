# ultralytics_ooo

从深度侵入式 Ultralytics 改造中提取的**即插即用**增强包：一行 `install()` 把「在线数据增强 + 混合样本池 + 修补续训 + 切片验证 + 双口径 mAP」装到任意干净 Ultralytics 上，**不修改上游任何源码、不改上游 `default.yaml`**。

## 这是什么

原本深度侵入在 `ultralytics/data/{base,augment}.py`、`engine/trainer.py`、`models/yolo/detect/val.py` 里的功能，被抽成独立包：

| 功能 | 效果 | 开关 |
|---|---|---|
| **在线数据增强** | 训练时在线切片(SAHI)、宽高比填充、运动模糊、合成、气象退化(雨/雾/噪声)、遮挡，标签同步重映射 | `slice_prob`, `blur_keep`, `ratio_pad_keep`, `compose_keep`, `weather_keep`, `occlusion_keep` |
| **混合样本池** | 每张原图每 epoch 动态扩展成多个训练样本（7 区段） | `slice_prob>0` 即启用 |
| **修补续训** | resume 已训完的 ckpt 时自动修补元数据续训到更长总轮数 | `resume_extend_epochs` |
| **切片验证** (SAHI eval) | val 时把大图切子块分别推理，预测 remap 回原图 + 类别级 NMS 融合 | `val_slice_enable` |
| **双口径 mAP** | 每轮跑切片(主)+整图(副)两遍；主 fitness 选 `best.pt`，副 fitness 另存 `best_whole.pt` | `val_slice_dual_metric` |
| **分组采样** | 让同一原图的所有子样本连续进出，raw LRU 命中、避免重复解码 | `slice_grouped_sampler`(默认开) |

## 三步上手

```python
# 1. 在任何干净 Ultralytics 环境里
from ultralytics_ooo import install
install()                      # 改的是运行时对象，不落盘改源码

# 2. 像平常一样训练，多传几个开关
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(
    data="coco8.yaml",
    epochs=100, imgsz=640, batch=16, workers=8,
    # 在线增强 + 混合样本池（不传这些就完全等于上游行为）
    slice_prob=1.0, slice_keep_origin=True,
    blur_keep=True, ratio_pad_keep=True, compose_keep=True,
    weather_keep=True, occlusion_keep=True,
    # 切片验证 + 双口径
    val_slice_enable=True, val_slice_overlap_ratio=0.2,
    val_slice_dual_metric=True,
)

# 3. 续训延长：一个已跑完的 last.pt，直接续到 200 轮
model.train(resume="path/to/last.pt", resume_extend_epochs=200)
```

不传任何 `slice_* / *_keep / resume_extend_epochs / val_slice_*` 时，行为与上游逐字节一致（全部默认关闭）。

## 工作原理

`install()` 在运行时做的事，全部 monkey-patch / 多继承，零侵入：

1. `build.YOLODataset` → `InstalledYOLODataset(OnlinePoolDataset, YOLODataset)`（协作多继承，MRO 融合；类在顶层模块，Windows spawn 多 worker 可 pickle）
2. `dataset.v8_transforms` → 在线增强感知版（装配 `OnlineSlice`、镜像各 `*_keep` 属性）
3. `BaseTrainer`：`on_train_epoch_start` 回调发布 `set_epoch`；`resume_training`/`check_resume` 实现 `resume_extend_epochs`；`validate`/`save_model` 实现双口径
4. `DetectionValidator`：`val_slice_enable` 时用 `SliceValDataset` 包装 val dataloader 并融合 tile 预测
5. `build_dataloader`：单机训练时改用 `GroupedImageSampler`（解码局部性）
6. `get_img_files`：fraction 舍入为 0 时回退保留源图，避免小数据集空集崩溃

在线超参注册到运行时 `DEFAULT_CFG`（不改上游 `default.yaml`）；上游不认识的 fork 调试参数由 `_compat` 按签名自动过滤。

## 包结构

```
ultralytics_ooo/
  installer.py          # install() 唯一入口
  dataset_class.py      # InstalledYOLODataset（顶层，spawn 多 worker 可 pickle）
  core/                 # 纯函数：退化算子、切片几何、unicode-safe 保存
  pool/
    dataset.py          # OnlinePoolDataset（混合样本池 7 区段）+ fraction 兜底
    augment_setup.py    # OnlineSlice + 在线感知 v8_transforms
    sampler.py          # GroupedImageSampler + patch_build_dataloader
    resume.py           # resume_extend_epochs
    valslice.py         # SliceValDataset + DetectionValidator patch
    dual.py             # val_slice_dual_metric 双口径 + best_whole.pt
    constants.py        # 在线超参默认表
```

## 测试与端到端验证

纯单元测试（不下载数据/权重）：

```bash
python -m pytest tests/test_ooo_core.py tests/test_ooo_augment.py \
                 tests/test_ooo_resume.py tests/test_ooo_valslice.py -q
```

在干净原版 **Ultralytics 8.4.126** 上端到端跑通的真实训练：

- 基础池：coco8 4 图 → 17 训练样本，1-epoch 真实训练 mAP50≈0.91
- 全开区段（slice/ratio/blur/compose/weather/occlusion）：池扩到 25 样本，1-epoch 干净
- **多 worker**：`workers=2` 训练干净（Windows spawn，`_mp_epoch` 跨进程掩码同步）
- **修补续训**：1 轮 ckpt 用 `resume_extend_epochs=3` 自动续训成功
- **双口径**：`val_slice_dual_metric=True` 一轮产出 `best.pt` + `best_whole.pt` + `last.pt`
- **fraction 兜底**：`fraction=0.01`（4 图舍入为 0）触发 guard，保留 4 图继续
- **高级切片**：`center_bias/center_constraint/full_box_only/background_ratio` 训练干净

## 兼容性说明

- 已在 **Ultralytics 8.4.126**（干净原版）上端到端验证。
- 所有与上游版本相关的构造差异（如 `Mosaic` 的 fork 调试参数）由 `_compat` 运行时按签名过滤，小版本间有一定容错。
- 多 worker 走 Windows spawn：`InstalledYOLODataset` 必须保持在顶层可 import（已如此设计）；训练脚本需有 `if __name__ == "__main__"` 保护。
- 仅支持 `task=detect` 的切片验证分支；`segment/pose/obb` 的 val 切片未接入。
- 与原改造版的一处有意差异：fork 的 `_hyp_get`「缺键即 raise」改为包内默认表静默兜底——这是为**不改上游 default.yaml** 所做的妥协。
