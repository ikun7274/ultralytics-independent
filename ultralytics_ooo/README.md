# ultralytics_ooo

从深度侵入式 Ultralytics 改造中提取的**即插即用**增强包：一行 `install()` 把「在线数据增强 + 混合样本池 + 修补续训 + 切片验证」装到任意干净 Ultralytics 上，**不修改上游任何源码、不改上游 `default.yaml`**。

## 这是什么

四个原本分散深度侵入在 `ultralytics/data/{base,augment}.py`、`engine/trainer.py`、`models/yolo/detect/val.py` 里的功能，被抽成独立包：

| 功能 | 效果 | 开关 |
|---|---|---|
| **在线数据增强** | 训练时在线切片(SAHI)、宽高比填充、运动模糊、合成，标签同步重映射 | `slice_prob`, `blur_keep`, `ratio_pad_keep`, `compose_keep` ... |
| **混合样本池** | 每张原图在每个 epoch 动态扩展成多个训练样本（切片/退化/合成子样本） | `slice_prob>0` 即启用 |
| **修补续训** | resume 一个已训完的 ckpt 时，自动修补元数据续训到更长总轮数，无需离线 fix 脚本 | `resume_extend_epochs` |
| **切片验证** (SAHI eval) | val 时把大图切成子块分别推理，预测 remap 回原图 + 类别级 NMS 融合 | `val_slice_enable` |

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
    epochs=100, imgsz=640, batch=16,
    # 在线增强 + 混合样本池（不传这些就完全等于上游行为）
    slice_prob=1.0, slice_keep_origin=True,
    blur_keep=True, ratio_pad_keep=True, compose_keep=True,
    # 切片验证
    val_slice_enable=True, val_slice_overlap_ratio=0.2,
)

# 3. 续训延长：一个已跑完的 last.pt，直接续到 200 轮
model.train(resume="path/to/last.pt", resume_extend_epochs=200)
```

不传任何 `slice_* / *_keep / resume_extend_epochs / val_slice_*` 时，行为与上游逐字节一致（全部默认关闭）。

## 工作原理

`install()` 在运行时做四件事，全部 monkey-patch / 多继承，零侵入：

1. `build.YOLODataset` → `InstalledYOLODataset(OnlinePoolDataset, YOLODataset)`（协作多继承，MRO 自动融合）
2. `dataset.v8_transforms` → 在线增强感知版（装配 `OnlineSlice`、镜像各 `*_keep` 属性）
3. `BaseTrainer`：`on_train_epoch_start` 回调发布 `set_epoch`；`resume_training`/`check_resume` 实现 `resume_extend_epochs`
4. `DetectionValidator`：`val_slice_enable` 时用 `SliceValDataset` 包装 val dataloader 并融合 tile 预测

在线超参注册到运行时 `DEFAULT_CFG`（不改上游 `default.yaml`）；上游不认识的 fork 调试参数由 `_compat` 自动过滤。

## 包结构

```
ultralytics_ooo/
  installer.py          # install() 唯一入口
  core/                 # 纯函数：退化算子、切片几何、unicode-safe 保存
  pool/
    dataset.py          # OnlinePoolDataset（混合样本池）
    augment_setup.py    # OnlineSlice + 在线感知 v8_transforms
    resume.py           # resume_extend_epochs
    valslice.py         # SliceValDataset + DetectionValidator patch
    constants.py        # 在线超参默认表
```

## 测试

纯单元测试，不下载数据/权重：

```bash
python -m pytest tests/test_ooo_core.py tests/test_ooo_augment.py \
                 tests/test_ooo_resume.py tests/test_ooo_valslice.py -q
```

已在干净原版上做过端到端验证：coco8 上 4 图 → 17 训练样本跑通 1-epoch 真实训练（mAP50≈0.91），`resume_extend_epochs` 把 1 轮 ckpt 延长续训成功。

## 兼容性说明

- 已在 **Ultralytics 8.4.126**（干净原版）上端到端验证。
- 所有与上游版本相关的构造差异（如 `Mosaic` 的 fork 调试参数）由 `_compat` 运行时按签名过滤，小版本间有一定容错。
- 未在其他 8.4.x 版本逐一枚举验证；若升级上游后开关失效，先跑上面的测试套件定位。
- 仅支持 `task=detect` 的切片验证分支；`segment/pose/obb` 的 val 切片未接入。
