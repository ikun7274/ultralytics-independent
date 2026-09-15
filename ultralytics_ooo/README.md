# ultralytics_ooo

从深度侵入式 Ultralytics 改造中提取的**即插即用**增强包：一行 `install()` 把「在线数据增强 + 混合样本池 + 修补续训 + 切片验证 + 双口径 mAP」装到任意干净 Ultralytics 上，**不修改上游任何源码、不改上游 `default.yaml`**。

## 这是什么

原本深度侵入在 `ultralytics/data/{base,augment}.py`、`engine/trainer.py`、`models/yolo/detect/val.py` 里的功能，被抽成独立包：

| 功能 | 效果 | 开关 |
|---|---|---|
| **在线数据增强** | 训练时在线切片(SAHI)、宽高比填充、运动模糊、合成、气象退化(雨/雾/噪声)、遮挡，标签同步重映射 | `slice_prob`, `blur_keep`, `ratio_pad_keep`, `compose_keep`, `weather_keep`, `occlusion_keep` |
| **混合样本池** | 每张原图每 epoch 动态扩展成多个训练样本（7 区段，**段长由 ratio 决定**，见下节） | `slice_prob=True` 即启用 |
| **每轮分支选中数日志** | 主进程每 epoch 打印各分支实际选中的原图数，配置与行为不一致时一眼可见 | 自动（`LOGGER.info`） |
| **修补续训** | resume 已训完的 ckpt 时自动修补元数据续训到更长总轮数 | `resume_extend_epochs` |
| **切片验证** (SAHI eval) | val 时把大图切子块分别推理，预测 remap 回原图 + 类别级 NMS 融合 | `val_slice_enable` |
| **双口径 mAP** | 每轮跑切片(主)+整图(副)两遍；主 fitness 选 `best.pt`，副 fitness 另存 `best_whole.pt` | `val_slice_dual_metric` |
| **分组采样** | 让同一原图的所有子样本连续进出，raw LRU 命中、避免重复解码 | `slice_grouped_sampler`(默认开) |

## 样本池布局：`*_keep` 开关占不占位，`*_ratio` 定段长

| 段 | 长度 | 说明 |
|---|---|---|
| base | `n_per*K_slice` | 只有被 `slice_ratio` 选中的图占位，各占 `n_per` 个（`slice_all_tiles=True` 时 4，否则 1）；**未被选中的图在 base 段不占任何位** |
| origin | `N`（`img_origin=True`）/ `0`（关闭） | `img_origin`：为**每张原图**恒定补 1 个整图槽。这是**全局统一覆盖旋钮**（对所有增强分支生效），取代旧的 `slice_keep_origin`（后者只覆盖被切片选中的图） |
| ratio / weather / occlusion | `K_x` | 每张**被选中**的原图 1 个位 |
| blur | `2*K_blur` | 每张被选中的原图 2 个位（短+长，同命运） |
| compose | `K_compose` | 每个**被选中的组** 1 个位（组 = 连续 4 张原图） |

其中 `K_x = round(x_ratio × count_x)`，`count_x` = 原图数 N（compose 为 `ceil(N/4)` 组）。

三条要点：

- **降低 ratio 会真的缩小池子**：未被选中的图不再占这个区段的位，所以不存在"位保留、内容退回原图"造成的重复膨胀。想彻底不要某条分支 → 关它的 `*_keep`。
- **`ratio >= 1` 时布局逐位退化为旧的满宽形状**（`4N / N / N / 2N / ceil(N/4) / N / N`，即 base/origin/ratio/blur/compose/weather/occlusion），可用于复现旧实验与 A/B 对照；`tests/test_ooo_branches.py::test_ratio_1_layout_is_bit_identical_to_the_legacy_shape` 锁死这条不变量。
- **池长全程恒定**：`K_x` 只由配置的 ratio 决定，与"选中了哪几张图"无关；每 epoch 只换选中的集合，不动段宽 —— mosaic buffer 的索引、`nb`/`nw`、sampler 的 units 全部按开始时的池长预计算，不能中途变化。

实测（N=40，六条 ratio 全 0.1）：池 `410 → 77`（**5.32× 更小**），"原图回退"占比 `91% → ~0%`，24/40 张图只占 1 个位。`python tools/ooo_pool_measure.py --n 40 --ratio 0.1` 可复现。

## 切片侧 5 个旋钮 + 1 个全局统一覆盖旋钮

切片相关的旋钮**不是并列的强度旋钮**，改错层会得到完全不同的池子。诊断工具：`tools/ooo_slice_knobs.py`。

| 层 | 旋钮 | 决定什么 | 取值 |
|---|---|---|---|
| ① 总开关 | `slice_prob` | 0 = 关闭切片；>0 才创建切片管线。**是开关，所以可以直接写 `True` / `False`**（等价 `1.0` / `0.0`；数字写法与老 args.yaml 里的 `0.5` 仍兼容） | `True` |
| ② 结构 | `slice_all_tiles` | 每图占 4 个槽位（4 片）还是 1 个槽位（1 片）→ base 段宽 `4K_slice` / `K_slice` | 小目标密集 → True |
| ③ 每轮选谁 | `slice_ratio` | 抽 `K=round(x·N)` 张图走切片；**未被抽中的图在 base 段不占位**，是否入池由全局 `img_origin` 决定 | 0.25~1.0 |
| ④ 多出的槽位放什么 | `slice_background_ratio` | 空片保留为负样本，还是换成整图（配额 = `x × 正片数`） | 稀疏 0~0.1；密集 -1 |
| ⑤ ②=False 时取哪一片 | `slice_target_tiles` | 槽位装**含目标的那一片**（不是盲抽的一片）：每轮取队列下一段，整个队列走完才回绕重排 | `all_tiles=False` 且要正片密度 → True |
| ⑥ 全局统一覆盖 | `img_origin` | **跨分支**：为每张原图恒定补 1 个整图槽（`origin` 段宽 = `N`）。关闭 = 未被任何分支选中的图**真正从池中丢弃** | 需要整图视角/不丢图时 True（默认 True） |

三条实测出来的要点：

- **`slice_prob=True/False` 是正式支持的写法**：它由 `augment_setup._resolve_slice_prob` 归一化（bool → 1.0/0.0）、校验范围（超出 `[0,1]` 直接报错，`-1` 不再有"隐式关闭"的含义）、并对 `0<p<1` 告警一次。bool 能写进 `args.yaml`（`slice_prob: true`）并被 resume 读回。
- **`slice_prob` 不是强度旋钮**：取 0~1 之间是**每个槽位各自抛硬币**，没中的槽位退回整图。实测 `p=0.5` + `all_tiles=True`：32 个槽位只有 14 个是真切片、16 个变成整图。调强度用 `slice_ratio`。
- **`slice_background_ratio` 是配额，不是固定比例**：条件为"空片数 ≤ x × 正片数"，按进程累计、每 epoch 重置，所以早期（正片还少）空片更容易被替换。实测稀疏场景（1600×1200 单 60×40 框）：`-1` → 24 个空片；`0.2` → 2 空片 + 22 整图；`0` → 24 整图。
- **稀疏数据慎用 `-1`**：池子会被纯负样本占满。但限空片的代价是槽位变成整图，而且那些整图是**纯重复**：实测 400 图稀疏场景下，`bg=0.25` 的 275 个回填槽只覆盖 100 张不同的图，而这 100 张**已经在 origin 段各有一个整图槽** —— 等于每张被选中的图每轮以整图形式出现 **3.75 次**。所以"回填整图"不是"整图稀释切片"，而是**同一张整图把切片挤掉了**。
- **硬规则：`img_origin=True` 时不要配 `slice_background_ratio > 0`**。origin 已经保证每张图每轮恰好一次整图，此时 bg 配额换来的整图**只可能**是已在池中的图的副本。实测 `0.25 → -1`：池大小不变（800），切片占比 **15.6% → 50%**，重复整图 **275 → 0**，正片一颗不丢。低密度下 `bg` 几乎是二值的（配额 = `空片 ≤ bg×正片`，正片只占切片的 1/4，要留住 4 片得 `bg ≥ 3`），**别指望微调 `bg` 能把池子变成"以切片为主"**。
- **正片数只由 `slice_ratio` 决定**：`.25/.5/1.0 → 100/200/400`，与 `bg`、`img_origin` 完全无关。冲小目标召回就调它，其余两个旋钮只决定"剩下的槽位装什么"。
- **`slice_all_tiles=False` 默认是省成本旋钮，不是召回旋钮**：它走 `k = random.randrange(4)` **盲抽** 1 片（`augment_setup.py`），命中框概率 ≈ 1/4，池子小 4 倍但**正片塌掉**（`r=.25`：100 → 19）。
- **⑤ `slice_target_tiles=True` 把"盲抽"换成"按目标挑"**：base 槽位绑定到**含目标的 (图, 片)**，每轮取队列下一段，**整个队列走完才回绕**（回绕时重新洗牌，即"重新开始选取"）。实测 400 图稀疏集（每图 1 个小框，队列 400 单位）：`r=.25` 时 base 100 槽 **正片 24 → 100**，池长 100 不变；每轮 100 个、4 轮一轮回，**轮内零重复**，一轮恰好覆盖全部 400 个含目标片。代价：池里几乎没有空片（全是正片会推高背景误检），要负样本就调小 `slice_ratio` 或开 `img_origin`。只在 `slice_prob>0` 且 `slice_all_tiles=False` 时生效，其它组合告警一次。调度按 `(数据集文件表, epoch)` 推导，worker 可独立重算，主进程与各 worker 必然一致。

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
    slice_prob=True, img_origin=True,
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
2. `dataset.v8_transforms` → 在线增强感知版（装配 `OnlineSlice`、镜像各 `*_keep` / `*_ratio` 属性）
3. `BaseTrainer`：`on_train_epoch_start` 回调发布 `set_epoch`（读 `trainer.train_loader`）；`resume_training`/`check_resume` 实现 `resume_extend_epochs`；`validate`/`save_model` 实现双口径
4. `DetectionValidator`：`val_slice_enable` 时用 `SliceValDataset` 包装 val dataloader 并融合 tile 预测
5. `build_dataloader`：单机训练时改用 `GroupedImageSampler`（解码局部性）
6. `get_img_files`：fraction 舍入为 0 时回退保留源图，避免小数据集空集崩溃

在线超参注册到运行时 `DEFAULT_CFG`（不改上游 `default.yaml`）；上游不认识的 fork 调试参数由 `_compat` 按签名自动过滤。

## 包结构

```
ultralytics_ooo/
  installer.py          # install() 唯一入口
  dataset_class.py      # InstalledYOLODataset（顶层，spawn 多 worker 可 pickle）
  ruff.toml             # 本包专属 lint 配置（不动上游 pyproject.toml）
  core/                 # 纯函数：退化算子、切片几何、unicode-safe 保存（零 ultralytics 依赖）
  pool/                 # 所有触碰 ultralytics 的代码与运行时 patch
    dataset.py          # OnlinePoolDataset（混合样本池 7 区段）
    augment_setup.py    # OnlineSlice + 在线感知 v8_transforms
    sampler.py          # GroupedImageSampler + patch_build_dataloader
    resume.py           # resume_extend_epochs
    valslice.py         # SliceValDataset + DetectionValidator patch
    dual.py             # val_slice_dual_metric 双口径 + best_whole.pt
    fraction.py         # fraction 舍入为 0 的兜底
    constants.py        # 在线超参默认表（仅项目自有键）
```

## 测试与端到端验证

全部为**离线**测试：不下载数据、不下载权重（切片验证/混合池测试自建 4 张图的临时数据集）。

```bash
python -m pytest tests/ -k ooo -q
```

| 测试文件 | 覆盖 |
|---|---|
| `tests/test_ooo_core.py` | 纯函数自校；`core/` 不依赖 ultralytics |
| `tests/test_ooo_augment.py` | `OnlineSlice` / `_hyp_get` / `_compat` |
| `tests/test_ooo_guards.py` | 切片几何不退化为 0 面积、构造期参数校验、`only=k` 等价性、配置表不覆盖上游键、**epoch 回调确实发布了 `set_epoch`** |
| `tests/test_ooo_sampler.py` | 分组采样种子对齐上游、loader 属性与上游公式一致 |
| `tests/test_ooo_branches.py` | 真实数据集端到端：7 区段总长、各分支标签结构、**ratio=1.0 布局逐位兼容旧形状**、ratio 定长切分、池长跨 epoch 恒定、`ratio=0` 边界、遮挡剔除、`close_aug_epoch` |
| `tests/test_ooo_resume.py` | 修补续训 |
| `tests/test_ooo_valslice.py` | 切片验证 |
| `tests/test_ooo_dual.py` | 双口径跨 epoch 不退化为整图口径、`best_whole.pt` 判定 |

配套工具：

| 工具 | 用途 |
|---|---|
| `tools/ooo_slice_knobs.py` | **切片 5 个旋钮的实际产出诊断**：槽位里有多少真切片 / 空片 / 整图，`--grid` 扫 `slice_background_ratio`。调切片参数前先跑它 |
| `tools/ooo_pool_measure.py` | 打印各段真实槽位、每个原图占几个位、与旧布局的倍率对比（`--n --ratio`） |
| `tools/ooo_ab_ratios.py` | `*_ratio` 行为对照模板：干跑打印两 arm 的实际选中数与池长，`--train` 跑两次训练并汇总 `results.csv` |
| `tools/ooo_layout_snapshot.py` | 存/比对"索引 → 样本"逐位快照，用于验证布局改动；`--compare` 有差异时退出码 1 |

## 兼容性说明

- 已在 **Ultralytics 8.4.126**（干净原版）上端到端验证。
- 所有与上游版本相关的构造差异（如 `Mosaic` 的 fork 调试参数）由 `_compat` 运行时按签名过滤，小版本间有一定容错。
- **与上游实现对齐的三处**（改动前会静默偏离）：
  1. `GroupedImageSampler` 所在的 loader 尾部与上游 `build_dataloader` 逐项对齐（含 `+ seed` 与 npu/xpu 的 `pin_memory_device`），由 `tests/test_ooo_sampler.py` 锁死；
  2. `pool/constants.py` 的默认表**只保留上游没有的键**，`install()` 会自检是否与上游键撞车；
  3. epoch 回调通过 **`trainer.train_loader`** 取训练数据集（上游没有 `train_dataloader` 这个属性）。这一处曾长期静默失效：回调是空操作 → `set_epoch` 从未执行 → 每轮 ratio 抽样与 `close_aug_epoch` 全程冻结在第 0 轮，且没有任何报错。现在由 `tests/test_ooo_guards.py` 的两条测试（上游属性名 + 回调真的调到 `set_epoch`）守住，且回调注册按标记幂等，重复 `install()` 不会重复注册。
- 多 worker 走 Windows spawn：`InstalledYOLODataset` 必须保持在顶层可 import（已如此设计）；训练脚本需有 `if __name__ == "__main__"` 保护。
- 仅支持 `task=detect` 的切片验证分支；`segment/pose/obb` 的 val 切片未接入。
- 已移除 `prefetch_factor` 配置键：它此前被读入 `dataset.prefetch_factor` 但**从未被任何 loader 使用**（上游硬编码 4），属于"传了没反应"的哑开关。现在传入会直接报未知参数，而不是静默忽略。
