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
| **双口径 mAP** | 每轮跑切片(主)+整图(副)两遍；主 fitness 选 `best.pt`/`last.pt`，副 fitness 另存 `best_whole.pt`/`last_whole.pt`（两个整图文件在收尾时同样被 strip 成 fp16 推理快照） | `val_slice_dual_metric` |
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

## 退化侧旋钮：一个换算关系 + 两个「默认别动」的键

六个在线退化/变换模块（blur / weather / occlusion / ratio / compose / origin）共用的工作分辨率与
算子选择旋钮。完整实测见 `六模块性能与JPEG解码分析.md`，这里只放"调之前必须知道"的部分。

| 旋钮 | 默认 | 含义与代价 |
|---|---|---|
| `degrade_max_side` | `0` = auto = `2*imgsz` | 退化算子的**工作分辨率上限**（长边像素）。`<0` = 关闭上限、在原分辨率上跑，会**告警一次**：实测 weather-noise 峰值 `18.4 MB → 180 MB`、稠密 PSF 模糊 `52 → 451 ms`。**这是增强强度参数，不是性能缺陷参数** |
| `compose_max_side` | `0` = auto = `2*imgsz` | 同上，作用于 compose 的每个源（画布因此上界为它） |
| `blur_axis_aligned` | `True` | **不要为了"更通用"关掉**：轴对齐 PSF 精确等于均匀箱式，输出与"任意角恰好抽到 0°/90°"**逐位相同**；关掉它长档模糊 `8.3 → 49.4 ms`（**5.9×**），等于让全包最贵的算子再贵 6 倍 |
| `degrade_resample` | `"linear"` | `"area"` 只在确实需要抗混叠、且愿意付 `5.09 → 20.62 ms`（**4.05×**）时用 |
| `*_keep` / `*_ratio` | 见上节 | `*_keep` 决定这一分支**占不占位**，`*_ratio` 决定段**多长**；两者都是配方旋钮，不是加速旋钮 |

**`degrade_max_side` 的换算关系**：上限与六个算子的成本都按**像素面积**缩放，所以
`2*imgsz`(1280) → ≈`1.2*imgsz`(768) 在退化侧是 **2.8×**（按 4K 全分支开的成本模型约 **整图 15%**）。
它成立的三个前提：① `768 > imgsz`，收尾 resize 仍是**下采样**，工作分辨率契约不破；
② `_degrade_frame` 已把 PSF 长度 / 雨线长度按 `scale` 同步缩放，缩回 `imgsz` 后相对外观基本不变
（既有实测 mean\|diff\| = 2.11 / corr = 0.9982）；③ **代价是像素真的会变**，所以它是一个
**需要质量 A/B 的改动，不能当默认值直接改**（改了会让历史实验与新实验不可比）。

> 一句话：`blur_axis_aligned` / `degrade_resample` / `degrade_max_side` 的默认值**都已经在便宜的那条路上**。
> 想省退化侧的时间，先动 `*_ratio`（少取样）而不是这三个键（提高单次取样成本）。

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

不传任何 `slice_* / *_keep / resume_extend_epochs / val_slice_*` 时，**训练行为**与上游一致：池恰为 `N` 张原图、段布局退化为 `[0, N, 0, 0, 0, 0, 0]`，样本内容逐字节相同（`_verify_probe/p1b_buffer_cause.py` 的 `upstream` / `installed` 两种模式给出 8/8 一致的摘要）。

两处**内部状态**不同，都已在代码与文档里写明：

- `self.ims` 是空的 —— 池内样本（含 origin 段）走原始分辨率的 raw LRU，这是第九轮定下的性能取舍；
- 由此 `cache='ram'` / `cache='disk'` 对池内样本无效：`install()` 后 `cache='ram'` 会降级为 `cache=False` 并告警，`cache='disk'` 给一条 `INFO`（见 `更新说明.md` 第二十节 §20.4）。

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
    dual.py             # val_slice_dual_metric 双口径 + best_whole.pt / last_whole.pt + 收尾 strip
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
| `tests/test_ooo_branches.py` | 真实数据集端到端：7 区段总长、各分支标签结构、**ratio=1.0 布局逐位兼容旧形状**、ratio 定长切分、池长跨 epoch 恒定、`ratio=0` 边界、遮挡剔除、`close_aug_epoch`、**无开关时池就是原数据集且 mosaic buffer 不会被缓存命中喂脏**、**buffer 只记录解码不记录访问** |
| `tests/test_ooo_resume.py` | 修补续训 |
| `tests/test_ooo_valslice.py` | 切片验证 |
| `tests/test_ooo_dual.py` | 双口径跨 epoch 不退化为整图口径、`best_whole.pt` 判定、`last_whole.pt` 逐轮镜像、双开关都关时不产出整图文件、`final_eval` 确实 strip 了两个整图文件（用真 `strip_optimizer` 验证 fp16/optimizer=None/epoch=-1）、**剥掉 `whole_*` 后 `results.png` 仍落在 run 目录** |

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
- **与上游实现对齐的四类**（改动前会静默偏离）：
  1. `GroupedImageSampler` 所在的 loader 尾部与上游 `build_dataloader` 逐项对齐（含 `+ seed` 与 npu/xpu 的 `pin_memory_device`），由 `tests/test_ooo_sampler.py` 锁死；
  2. `pool/constants.py` 的默认表**只保留上游没有的键**，`install()` 会自检是否与上游键撞车；
  3. epoch 回调通过 **`trainer.train_loader`** 取训练数据集（上游没有 `train_dataloader` 这个属性）。这一处曾长期静默失效：回调是空操作 → `set_epoch` 从未执行 → 每轮 ratio 抽样与 `close_aug_epoch` 全程冻结在第 0 轮，且没有任何报错。现在由 `tests/test_ooo_guards.py` 的两条测试（上游属性名 + 回调真的调到 `set_epoch`）守住，且回调注册按标记幂等，重复 `install()` 不会重复注册。
  4. **mosaic buffer 的语义与上游一致：只在"真的解码了"时记账，不在"访问了"时记账**。上游的 append 在 `load_image` 的 `if im is None:` 里（命中 `self.ims` 时提前返回），而池内样本不走 `load_image`，于是曾经每次访问都 append —— `Mosaic.get_indexes()` 从这里抽混样伙伴，重复项会把不同的图挤出混样池。实测（8 张图）：上游 `[0]` / `[0,1,2,3]`，修复前 `[0,0,0,0]` / `[1,0,0,3,2,0,1]`，8 个样本里 7 个与上游内容不同。现由 `_touch_buffer_for_decode` 统一把关，`tests/test_ooo_branches.py` 两条测试 + `_perf_review/inject_check.py` 的一条注入守住。
- 多 worker 走 Windows spawn：`InstalledYOLODataset` 必须保持在顶层可 import（已如此设计）；训练脚本需有 `if __name__ == "__main__"` 保护。
- 仅支持 `task=detect` 的切片验证分支；`segment/pose/obb` 的 val 切片未接入。
- `prefetch_factor` 是**真实生效**的配置键（默认 `4` = 上游硬编码值，即默认行为零变更）。它此前被读入 `dataset.prefetch_factor` 却从未被任何 loader 使用（上游硬编码 4），属于"传了没反应"的哑开关；第九轮把它接到了分组 loader 上。**作用域有限**：只在分组采样生效的训练路径上应用；走上游 `build_dataloader` 时（验证、多卡、`slice_grouped_sampler=False`）会**告警一次**说明它无法生效，而不是静默忽略。

## 迁移到另一个 Ultralytics（新版本 / 新环境）

本包靠**重绑上游已存在的名字**工作（不编辑上游源码），代价是：上游一旦改名或挪动符号，补丁**不会报错，只会静默失效**。本项目已经因此栽过一次——epoch 回调读的是 `trainer.train_dataloader`，而 `BaseTrainer` 从来没有这个属性，于是 `set_epoch`、连同每轮 `*_ratio` 抽签与 `close_aug_epoch`，在整个项目生命周期内冻结在第 0 轮且不打印任何东西。

所以迁移的正确顺序是：**先自检，再训练。**

### 第 0 步：跑迁移自检

```bash
python tools/ooo_compat_check.py          # 人类可读报告，任一项 FAIL 则退出码 1
python tools/ooo_compat_check.py --out res.json   # 同时落一份机器可读结果
```

它检查四层，共约 63 项，离线、不下载权重：

| 层 | 内容 | 失败意味着 |
|---|---|---|
| **A** 上游符号存在性 | `install()` 重绑的 29 个 `模块.属性` 是否仍可解析 | 上游改名/挪模块 → 对应补丁静默失效 |
| **B** 签名 / 调用契约 | 被包装函数的**前导参数名与顺序**、`BaseDataset.__init__` 的 `hyp`/`fraction` 关键字、`BaseTrainer.train_loader` 属性名、`final_eval` 是否还在 | 转发参数会被错位解释，且多半不报错 |
| **C** `install()` 生效性 | 打完补丁后逐个核对标记位（`_ooo_sampler_patched` 等）与身份（`is` 判等）、整图 strip 包装是否真的落上（`_ooo_final_eval_wrapper`）、以及项目自有键是否真的注册到了 `DEFAULT_CFG` | 补丁没落上、或在线超参根本没进配置 |
| **D** 端到端冒烟 | 真实 4 图数据集走真实工厂：7 段池几何、池层标签契约、变换后样本、分组采样器类别、epoch 回调真的调到 `set_epoch` | 装得上但跑起来不对 |

**不要靠"没报错"判断成功**：A/B 只能告诉你"名字还在"，C/D 才是"补丁真的做了事"的证据。这个检查器自身也做过注入验证（4 类失败模式全部被捕获，脚本在 `_perf_review/compat_inject.py`）。

### 第 1 步：搬哪些文件

| 路径 | 必要性 | 说明 |
|---|---|---|
| `ultralytics_ooo/` | **必需** | 15 个 `.py` + `ruff.toml`。含 `core/`（纯 numpy/cv2，零 ultralytics 依赖）与 `pool/`（所有触碰上游的代码与补丁） |
| `tools/ooo_compat_check.py` | **强烈建议** | 上一步的自检器；换环境后第一件事 |
| `tools/ooo_*.py`（其余） | 可选 | 池布局诊断 / 旋钮体检 / 性能测量 |
| `tests/test_ooo_*.py` | 可选 | 11 个文件，**需放进新仓库的 `tests/` 目录**（它们用上游的 `tests/conftest.py`） |
| `train.py` / `_mini_val_set/` | 参考 | 使用示例与最小数据集，不是包的一部分 |
| 上游 `ultralytics/` | **不要动** | 零侵入是本包的设计前提；改上游会让后续升级无法合并 |

本包**没有独立的构建配置**（`ultralytics_ooo/` 下只有 `ruff.toml`，没有 `pyproject.toml` / `setup.py`），所以它是"随源码走"而不是 `pip install` 的包。三种搬法：

```bash
# (a) 整仓沿用（最省事）：新环境里直接用这份仓库，只把 ultralytics 升级/替换
#     注意：若 pip 里另有一个 ultralytics，而仓库根在 sys.path 前面，用的是仓库这份。

# (b) 只带扩展包：把 ultralytics_ooo/ 复制到新仓库根目录，保证仓库根在 sys.path
#     目录名必须保持 ultralytics_ooo（worker 靠重新 import 这个名字反序列化数据集）

# (c) 想 pip 安装：需要自己补一个 build 配置（本包未提供），
#     且构建产物里的包名必须仍是 ultralytics_ooo
```

### 第 2 步：三条时序契约（违反会静默失效或直接报错）

1. **`install()` 必须在任何 `YOLO(...).train(...)` 之前**。在线超参是在 `install()` 时注册到运行时 `DEFAULT_CFG` 的；先 `train()` 会撞上 `check_dict_alignment` 并报 `not a valid YOLO argument`。

2. **`install()` 必须早于"按名字导入" `build_dataloader`**。这是本项目踩过的坑，模板如下：

   ```python
   from ultralytics.data.build import build_dataloader   # ← 绑定了补丁前的函数，永久失效
   from ultralytics_ooo import install
   install()
   ```

   正确写法是**在 `install()` 之后从模块属性取**：

   ```python
   from ultralytics_ooo import install
   install()
   from ultralytics.data import build as B   # 此时 B.build_dataloader 才是补丁版
   dl = B.build_dataloader(ds, batch, workers, True, -1, False, True, "cpu")
   ```

   自带工具 `tools/ooo_perf_run.py` 曾长期因此报 `RandomSampler`，让人误判"分组采样没效果"。

3. **不要依赖"多调一次没关系"**。`install()` 本身按标记幂等，但若发生 `sys.modules` 清理后重新 import，或环境里存在两份包副本，epoch 回调会被注册两次——**每轮的 `*_ratio` 抽签会跑两遍**。自检 C 层会断言回调恰好一个。

### 第 3 步：训练脚本的两个形态要求

- **Windows 走 spawn 多进程**：训练脚本必须有 `if __name__ == "__main__":` 保护，否则 worker 重新 import 主模块会递归启动。`InstalledYOLODataset` 已刻意放在顶层模块，就是为了让 worker 能反序列化到它。
- **`workers` 受内存约束**：每个 spawn worker 固定约 340 MB，外加预取批次与原始图 LRU。在 7.9 GB 的机器上 `workers=4` 会直接 `MemoryError`；`workers<=2` 才稳。别照抄大内存机器上的 `workers=8`。

### 第 4 步：按自检失败的那一层定位

| 层 | 症状 | 通常要改 |
|---|---|---|
| A FAIL | 符号找不到 | 该行 `->` 指出的文件；上游把函数挪了模块或改了名 |
| B FAIL | 参数名/顺序变了 | 同上；重点看 `pool/sampler.py`（分组 loader 尾部是**故意重复**上游写法的）与 `pool/valslice.py` |
| B WARN | `Mosaic` 现在带 `**kwargs` | `_compat` 的过滤失效，检查 fork-only 调试参数是否被原样转发 |
| C WARN | 键表与上游撞车 | `pool/constants.py`：把上游也已定义的键从 `_ONLINE_DEFAULTS` 删掉，让上游保持唯一真源 |
| D FAIL | 装上了但跑不对 | 池几何看 `pool/dataset.py`；采样器看 `pool/sampler.py`；变换看 `pool/augment_setup.py` |

**已知验证版本：`8.4.126`（干净原版）**。小版本间有一定容错（`_compat` 按签名过滤差异），但**每换一次上游版本都要重跑自检**，把上面那张表当作准入清单。

