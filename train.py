# -*- coding: utf-8 -*-

import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

from ultralytics_ooo import install
install()

from ultralytics import YOLO

'''
样本池 = 区段式布局, 每个区段只受自己的独立开关控制 (互不影响), 且【段长由 ratio 决定】:
    base      n_per*K_slice  被选中的图 → n_per 片; 未选中的图【不占 base 位】(由 img_origin 统一覆盖)
    origin    N              img_origin (统一覆盖: 每张原图 1 个整图槽, 取代旧 slice_keep_origin)
    ratio     K_ratio        ratio_pad_keep    
    blur      2*K_blur       blur_keep         (短+长)
    compose   K_compose      compose_keep      (2×2 大图, 组级)
    weather   K_weather      weather_keep     (雨/雾/噪声)
    occlusion K_occlusion    occlusion_keep    (rect/stripe)
其中 K_x = round(x_ratio × count_x) —— 只由配置的 ratio 决定, 与"选中了哪几张图"无关。

N=8 全开 ratio=1.0: 32 + 8 + 8 + 16 + 2 + 8 + 8 = 82 张混合样本池 (与改造前逐位一致)
N=8 全开 ratio=0.1:  4 + 8 + 1 + 2 + 0 + 1 + 1 = 17 张 (更小, 无"原图回退"重复位)

★ *_keep 与 *_ratio 的分工 (容易误解, 务必看清):
  · *_keep  = 这条分支【参不参与】→ 关掉它, 该区段长度归零, 池子里完全没有这类样本。
  · *_ratio = 这条分支的【强度】→ 决定这个区段里有多少张原图被选中做增强, 其余原图【不占这个区段的位】。
              未被任何增强分支选中的原图, 若 img_origin 开启则由 origin 段统一保留 1 个整图位(统一覆盖),
              否则从池中直接丢弃。所以降低 ratio 会【真的缩小池子】, 不会留下重复原图。
              选中数 = round(ratio × count); compose 是组级 count=ceil(N/4), 其余是原图级 count=N。
              注意 Python 的 round 是"四舍六入五取偶": round(0.5)=0, 所以 compose_ratio=0.5 在
              N=4 上会选中 0 组 (= 该分支不产生任何合成样本)。这种情况日志会告警并标注 "<-- NONE"。
              想让池子变小 → 降 ratio; 想彻底不要这条分支 → 关它的 *_keep。
  · 每条分支每 epoch 的【实际选中数】会打印在日志里 (主进程, 每轮一行, 由 trainer 的
    on_train_epoch_start 回调触发 set_epoch):
        augment masks @ epoch 3: slice 4/8 (ratio 0.5) | ratio 1/8 (ratio 0.1) | ...
    这是核对"配置是否真的生效"的第一手依据 —— 早期版本里 ratio/blur/compose 三个 ratio 键从未到达
    dataset, 而且发布 epoch 的回调读错了属性名(train_dataloader), 整行日志根本没出现过。
  · 想【逐位复现】ratio 三个键生效之前的旧行为(含旧的 4N/2N/N 段宽): 把它们显式设为 1.0
    (= 全量增强, 段宽退化回旧形状), 用于做修复前后的 A/B 对照; 见 tools/ooo_ab_ratios.py。
  · 改动样本池布局前先跑 tools/ooo_layout_snapshot.py 存基线: ratio=1.0 的"索引→样本"映射必须逐位不变。
'''

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/26/yolo26n.yaml')
    model.load('weights/yolo26n.pt')
    model.train(
        #---------训练参数---------------
        data='_mini_val_set/_mini_data.yaml',
        cache=False,                
        imgsz=320,
        epochs=2,
        batch=4,
        workers=0,                   
        optimizer='MuSGD',
        device='cpu',
        # resume=r'C:\Users\Administrator\Desktop\ultralytics-improved\runs\exp-3\weights\last.pt',  # 断点续训: 改成你本机 last.pt 路径
        # resume_extend_epochs=15,  # (int, 0=关闭) 续训自动延长: 自动修补ckpt元数据(epochs/patience), 从旧停点续训到该轮数; 需>ckpt已完成轮数
        patience=0,
        amp=True,
        fraction=1.0,
        project="runs",  # 相对工作目录, 便于仓库搬迁
        name='exp',


        # ---------Mosaic在线增强---------------
        mosaic=1.0,
        close_mosaic=1,

        # ---------SAHI在线切片 (slice_*)-----------
        # 切片侧共 4 个旋钮(①②选图, ③④结构/内容), 另有 1 个【全局统一覆盖】旋钮 img_origin(见下方独立说明, 不属于切片)。
        # 切片旋钮之间不要互相替代, 调参前先量密度(含 --grid 扫 bg):
        #     python tools/ooo_slice_knobs.py --data _mini_val_set/_mini_data.yaml --grid
        # ① 总开关: slice_prob
        # ② 每轮选谁: slice_ratio —— 抽 K=round(x*N) 张图走切片, 未抽中的图不占 base 位(由下方 img_origin 统一覆盖)。

        # ③ 结构: slice_all_tiles —— 每图占 4 个槽位(4 片) 还是 1 个槽位(随机 1 片) → base 段宽 4K / N。
        #    它的价值是"保证 4 片都出现"(小目标不会因为随机取片而漏掉), 代价是每图多 3 个槽位。
        # ④ 被 ③ 多出来的槽位放什么: slice_background_ratio(背景保留 vs 换整图)。

        # 【全局统一覆盖】img_origin —— 取代旧 slice_keep_origin, 但作用范围扩大到【所有增强分支】(不只是切片):
        #   开启时为【每一张原图】各保留 1 个整图槽(origin 段, 共 N 个), 无论它是否被切片/合成/模糊/天气/遮挡任一分支选中 → "未选中"的原图统一保底出现。
        #   关闭时, 未被【任何】增强分支选中的原图将真正从池中丢弃(=你说的"直接丢弃"): 仅被某分支选中的图才进池。
        #   注意: 它不属于切片旋钮; 这里紧邻切片参数仅出于位置便利, 其语义对 compose/blur/weather/occlusion 同样生效。
        slice_prob=True,              # 独立开关, 开启在线sahi切片增强 (区段 4K)
        slice_ratio=0.5,              # 每个epoch随机选 round(ratio*N) 张原图走切片, 其余不占 base 位;
                                      # 1.0=纯切片, 0.5=一半原图切片, 0=全整图(等效关闭切片); 每epoch重新随机。
        slice_all_tiles=True,         # True：每张被选中的原图 4 片全进池(区段 4K); False：随机 1 片(base=K), 可能漏掉目标。
                                      # 图片数量少建议设置True; 图片数量多时,可以设置False。
        slice_background_ratio=-1,    # 切片背景保留比例: 背景数 <= x*正片数; -1=全留; 0=一个不留
                                      # ⚠ 目标稀疏时不要用 -1: 实测 1600x1200 单小框场景 32 个槽位里 24 个是背景。
                                      #   稀疏建议 0~0.1; 密集可用 -1 或 0.2~0.5。
                                      #   注意"限背景"的代价是那些槽位变成整图 —— 同一张图会重复出现几次
        img_origin=True,              # 全局统一覆盖: 每张原图各 1 张整图进池(origin 段 N 个槽, 对所有分支生效); 关闭=未选中且未被任何增强选中的原图直接丢弃。

        slice_overlap_ratio=0.2,      # 相邻切片重叠比例 [0,1); 例: 原图4000x3000+重叠0.2 -> 切片2400x1800
        slice_min_tile_area_ratio=0.005,  # 切片块面积下界: 切片面积 < 原图x该值 的切片丢弃
        slice_min_box_retain_ratio=0.4,   # 目标框保留下界: 目标在切片内可见面积占原框比例 < 该值则丢弃
        slice_center_constraint=True,     # 目标唯一归属: 每个目标只分配给"中心所在"切片, 防同一目标被切两半重复出现
        slice_min_center_retain_ratio=0.6,# 中心不在本片时, 若本片内可见面积占比 >= 该值仍保留; 1.0=严格只留中心片
        slice_full_box_only=False,        # 目标必须完整落在切片内才保留, 被边界切开即过滤; 开启时优先于 slice_center_constraint
        slice_center_bias=False,           # 目标感知切缝: 切缝按本图目标中心分布微移, 落在最稀疏区间, 减少目标被劈碎; 与 center_constraint 互补建议同开
        slice_bias_margin=0.25,           # 切缝偏移窗口: 切缝只在 [margin, 1-margin] 区间内微移, 保证 tile 不过小
        slice_bias_jitter=0.05,           # 每图每调用随机扰动切缝, 防同一图每 epoch 切缝相同而过拟合; 0=关闭


        # ---------在线合成 (compose_*): 每 4 张原图拼 1 张 2x2 大图, 提供更大范围多目标上下文--------- 
        compose_keep=True,  # 独立开关; 每 4 张原图额外合成 1 张 2×2 大图进样本池 (区段 +ceil(N/4))
        compose_max_side=0, # 合成2x2大图拼后降采样最长边上限(像素): 0=自动=2×imgsz(默认开启, 降内存), >0=手动指定(如2560); 不想要此优化可设 compose_max_side 为一个很大的值关闭。
        compose_ratio=0.5,  # 每epoch随机选 round(x*ceil(N/4)) 组做合成(组级); 未选中的组【不占位】, 段长=K_compose; 1.0=全量合成
                            # 注意组级基数最小: N=8 → ceil(N/4)=2 组, 0.1 → round(0.2)=0 组 (该分支等于关闭, 日志会告警)


        # ---------在线比例调整 (ratio_pad_*): 在线加边框统一宽高比---------
        ratio_pad_keep=True,          # 独立开关, 开启在线比例调整 (区段 +K_ratio)
        ratio_pad_ratio=0.5,          # 每epoch随机选 round(x*N) 张原图做比例调整(原图级); 未选中的图不占本段位; 1.0=全量
        ratio_pad_target="auto",      # auto: 4:3↔16:9 双向 + 其他比例转最近（默认）
        ratio_pad_color="gray",       # 边框颜色 black/gray/white


        # ---------在线运动模糊 (blur_*): 模拟无人机运动失焦, 每张原图生成 2 张模糊副本(短+长), 标签不变---------
        blur_keep=True,              # 独立开关, 开启在线运动模糊 (区段 +2*K_blur, 短+长)
        blur_ratio=0.5,               # 每epoch随机选 round(x*N) 张原图做运动模糊(原图级, 短+长同命运, 所以段长=2*K); 1.0=全量
        blur_short_len_min=5,         # 短模糊(轻度, 无失焦) 长度下限(像素)
        blur_short_len_max=12,        # 短模糊(轻度, 无失焦) 长度上限(像素)
        blur_long_len_min=20,         # 长模糊(重度) 长度下限(像素)
        blur_long_len_max=35,         # 长模糊(重度) 长度上限(像素)
        blur_long_defocus_sigma=1.0,  # 长模糊失焦高斯 σ 上限; 0=不加失焦


        # ---------在线气象退化 (weather_*): 每张原图生成 1 张雨/雾/噪声退化图, 提升恶劣天气鲁棒性, 标签不变---------
        weather_keep=True,           # 独立开关, 开启在线气象退化 (区段 +K_weather)
        weather_ratio=0.5,            # 每epoch随机选 round(x*N) 张原图做气象退化(原图级); 未选中的图不占本段位; 0.3~0.6 推荐, 1.0=全量
        weather_types="rain,haze,noise", # (str) 退化类型池, 逗号分隔; 每张图随机抽 1 种; 可子集如 "rain,haze"
        weather_rain_density=0.15,    # 雨线密度 = 雨线数量 / max(h,w), 越大雨越密
        weather_rain_length=15.0,     # 雨线长度上限(像素), 每根随机取 0.5~1.0 倍
        weather_haze_beta=0.4,        # 雾浓度 [0,1), 越大雾越浓
        weather_noise_std=15.0,       # 高斯噪声标准差(每通道独立), 模拟传感器/弱光噪点


        # ---------在线遮挡模拟 (occlusion_*): 语义遮挡块 (树冠/电线/阴影), 提升被遮挡目标鲁棒性, 标签不变---------
        occlusion_keep=True,          # 独立开关, 开启在线遮挡模拟 (区段 +K_occlusion); 与 blur/weather 同构
        occlusion_ratio=0.5,          # 每epoch随机选 round(x*N) 张原图做遮挡(原图级); 未选中的图不占本段位; 0.3~0.6 推荐
        occlusion_types="stripe",# 遮挡类型, 逗号分隔; rect=随机矩形(树冠/阴影), stripe=细长条带(电线/枝干/云影)
        occlusion_blocks=1,           # 每图遮挡块数 (1~3)
        occlusion_size_ratio=0.2,     # 单块面积上限(相对原图面积), 防目标被完全盖住
        occlusion_color="auto",       # auto=采样图像深色分位均值(融入场景); 或 black/gray 固定色
        occlusion_max_cover=0.95,     # 目标被遮挡面积占比 >= 该值则从标签剔除 (完全被盖住的目标=纯噪声); 1.0=标签永不变


        # ---------训练后期关闭在线增强 (close_aug_epoch): 与 close_mosaic 同构的时间维衰减---------
        # 训练最后 N 个 epoch 关闭切片/比例/模糊/气象退化/遮挡/合成, 让模型在真实分布上收敛。
        # 实现: 段长保持不变(段长由 ratio 决定, 不能在训练中途变化 —— mosaic buffer 索引/nb/sampler 都按它预计算),
        # 但每条分支的构建器直接输出原图。所以收尾 epoch 会多出若干重复原图(约等于 K_* 之和), 池子整体仍是缩小后的规模。
        close_aug_epoch=1, # 0=关闭该调度(默认, 完全向后兼容)


        # ---------验证侧在线切片评估 (val_slice_*): 验证集切片推理 + 坐标还原 + NMS 融合 (SAHI评估)---------
        # 训练侧已在线切片, 验证侧整图直推会因小目标被降采样而低估切片训练收益.
        #
        # 口径与产物 (修复后才有意义): best.pt 按【切片口径】选, best_whole.pt 按【整图口径】选。
        # 每轮验证跑两遍 = 双口径的代价, 所以按场景收敛 ——
        #   · 调参 / 想看清两个口径的差距 : val_slice_enable=True , val_slice_dual_metric=True  (当前配置)
        #   · 线上是整图推理            : val_slice_enable=False                         (验证最省时, best.pt 即产物)
        #   · 线上是切片(SAHI)推理      : val_slice_enable=True , val_slice_dual_metric=False (只跑切片主口径)
        # 与历史实验对比时注意口径对齐: 新的 best_whole.pt 才能对上旧运行的 best.pt (两者都是整图口径;
        # 旧 best.pt 本身口径是混合的 —— epoch1 按切片选、epoch2 起按整图选)。
        val_slice_enable=True,       # 总开关: 验证时把验证图切成 2x2 重叠子图独立推理, 子图框还原到原图坐标,
                                      #         跨切片重复框 NMS 融合后与原图 GT 算 mAP; False=回归原生整图验证
        val_slice_all_tiles=True,    # True=每张验证图全部 2x2 子图都推理(与训练侧对齐); False=每图随机1片(快速验证)
        val_slice_ratio=0.5,          # 每轮验证随机选 round(x*N_val) 张验证图走切片, 其余整图直通; 1.0=全部切片
        val_slice_overlap_ratio=0.2,  # 验证侧切片重叠比例 [0,1), 建议与训练侧 slice_overlap_ratio 一致
        val_slice_nms_iou=0.5,        # 跨切片重复框 NMS 融合 IoU 阈值
        val_slice_dual_metric=True,   # 双口径: 先跑切片验证(主, 驱动 fitness/早停/best.pt), 再跑整图验证(参考),
                                      #         输出 whole_* 指标并额外保存 best_whole.pt/last_whole.pt (耗时为两遍验证)

        
        # ---- 在线增强保存 (人工检查切片是否正确) ----
        # 默认全部关闭: 保存是"同步 JPEG 编码 + 落盘", 直接跑在 DataLoader 取样路径上。
        # 实测单次带标注写盘 1280x960 = ~24 ms, 4000x3000 = ~183 ms; 切片分支保存用的是
        # 原图分辨率 tile(不受 degrade_max_side 限幅), 单张可达数十毫秒。6 条分支全开 =
        # 每样本一次同步写盘, 首轮训练会被 IO 主导, 并产生数万张 JPEG。
        # 需要人工抽查时: 只开一条分支, 用 slice_save_max 限到 50~200 张。
        # slice_save_annotated=True,   # 保存时画标注框+类别 (仅在下面任一 *_save_dir 非空时生效)
        # slice_save_max=100,          # 最多保存张数; 0=不限 (不限量会把整轮训练拖成 IO 瓶颈)
        # compose_save=False,          # 保存合成图 (默认关闭)
        # slice_save_dir=r"ultralytics-main\img\sliced_save_dir",
        # compose_save_dir=r"ultralytics-main\img\composed_save_dir",
        # ratio_pad_save_dir=r"ultralytics-main\img\change_proportion_save_dir",
        # blur_save_dir=r"ultralytics-main\img\motion_blur_save_dir",
        # weather_save_dir=r"ultralytics-main\img\weather_save_dir",
        # occlusion_save_dir=r"ultralytics-main\img\occlusion_save_dir",
        # 【注意】mosaic_save_* 在本仓库(干净上游)上是**无效键**: 上游 Mosaic 的签名是
        # Mosaic(dataset, imgsz, p, n), 不接受任何 save 参数, 装配期会被 _compat 静默剔除。
        # 设成非默认值只会触发一次告警, 不会写盘。需要抽查 mosaic 请另接回调或改上游。
        # mosaic_save_dir=r"ultralytics-main\img\mosaic_save_dir",
    )