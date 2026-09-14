# ultralytics_ooo：Ultralytics 即插即用增强包

> **一行 `install()`，不改上游任何源码，就把「在线数据增强 + 混合样本池 + 修补续训 + 切片验证 + 双口径 mAP」装进你现有的 Ultralytics 训练流程。**

本仓库 = **干净 Ultralytics 8.4.126 原版** + **零侵入增强包 `ultralytics_ooo`**。
这些能力原本是深度改进 Ultralytics 源码的（动 `data/base.py`、`data/augment.py`、`engine/trainer.py`、`detect/val.py`、`cfg/default.yaml`）；本项目把它们提取成独立包，用「子类化 + 运行时 monkey-patch」挂到干净原版上。

**即插即用意味着：**
- ✅ **不修改上游源码**：所有改动发生在运行时对象上，不落盘；
- ✅ **不改 `default.yaml`**：新超参在 `install()` 时运行时注册；
- ✅ **可整体关闭**：不传任何开关，训练/验证行为与上游逐字节一致；
- ✅ **克隆即用**：仓库自包含，原版源码 + 增强包同在，可直接跑测试与训练。

```python
from ultralytics_ooo import install
install()                      # 就这一行

from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(
    data="coco8.yaml", epochs=100, imgsz=640, batch=16,
    slice_prob=1.0, blur_keep=True, ratio_pad_keep=True,
    compose_keep=True, weather_keep=True, occlusion_keep=True,
    val_slice_enable=True, val_slice_dual_metric=True,
)
```

---

## 功能清单（均在干净 8.4.126 上端到端验证）

| 功能 | 效果 | 关键开关 |
|---|---|---|
| **在线切片增强** | 训练时在线 SAHI 切片，标签同步重映射；目标感知切缝 / 背景占比约束 | `slice_prob` `slice_overlap_ratio` `slice_center_bias` |
| **混合样本池** | 每张原图每 epoch 动态扩展成多训练样本（7 区段：切片/原图/比例/模糊/合成/气象/遮挡） | 各 `*_keep` |
| **宽高比填充** | 在线 pad 到目标宽高比，独立池区段 | `ratio_pad_keep` `ratio_pad_target` |
| **运动模糊** | 轴对齐/旋转运动模糊，模拟无人机运动失焦 | `blur_keep` `blur_long_len_max` |
| **在线合成** | 每 4 图合成 1 张 2×2 大图，提供多目标上下文 | `compose_keep` `compose_max_side` |
| **气象退化** | 雨 / 雾 / 噪声，提升恶劣天气鲁棒性 | `weather_keep` `weather_types` |
| **遮挡退化** | 矩形/条带语义遮挡（树冠/电线/阴影） | `occlusion_keep` `occlusion_types` |
| **修补续训** | resume 已训完的 ckpt 时自动修补元数据，续到更长总轮数 | `resume_extend_epochs` |
| **切片验证 (SAHI eval)** | val 大图切子块推理，预测还原回原图坐标 + NMS 融合 | `val_slice_enable` `val_slice_nms_iou` |
| **双口径 mAP** | 每轮切片(主)+整图(副)跑两遍；主→`best.pt`，副→`best_whole.pt` | `val_slice_dual_metric` |
| **分组采样** | 同一原图子样本连续进出，raw LRU 命中、免重复解码 | `slice_grouped_sampler`（默认开） |
| **fraction 兜底** | 小数据集 fraction 舍入为 0 时自动保留源图 | 自动生效 |

## 三步上手

```python
from ultralytics_ooo import install
install()                                  # 1. 安装（运行时 patch，不改源码）

from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(data="coco8.yaml", epochs=100, # 2. 像平常一样训练，多传几个开关
            slice_prob=1.0, blur_keep=True, weather_keep=True)

model.train(resume="last.pt",              # 3. 修补续训：已跑完的 ckpt 直接续到 200 轮
            resume_extend_epochs=200)
```

> Windows 多 worker 训练脚本需有 `if __name__ == "__main__":` 保护（spawn 要求）。

## 仓库结构

```
ultralytics-main/
├── ultralytics/          # 干净原版 Ultralytics 8.4.126 源码（未修改）
├── ultralytics_ooo/      # ★ 即插即用包
│   ├── installer.py      # install() 唯一入口
│   ├── core/             # 纯函数：退化算子、切片几何、unicode-safe 保存（零依赖）
│   └── pool/             # 混合样本池 / 在线切片 / 分组采样 / 修补续训 / 切片验证 / 双口径
├── tests/                # ooo 自测试
├── weights/              # 本地权重（gitignored）
├── train.py              # 全开关验证脚本
└── 项目说明.md            # 完整架构与设计文档
```

## 验证状态

- 单元测试：`pytest tests/test_ooo_*.py -q` → 16 项通过
- 真实全开训练：8 图 → 74 样本池，2 epoch，`best.pt` + `best_whole.pt` + `last.pt` 全部产出
- 多 worker：Windows spawn `workers=2` 干净
- 修补续训 / 双口径 / 高级切片 / fraction 兜底均实跑通过

详细原理、开关表、与原深度侵入版的差异见 [`项目说明.md`](项目说明.md)。
