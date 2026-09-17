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
假设：8张原图
	切片：slice_ratio=0.5, slice_all_tiles=False, slice_target_tiles=True, --->  4张有目标切片
	合成：compose_ratio=0.5, --->  1张合成图
	比例合成：ratio_pad_ratio=0.5, --->  4张比例图
	运动模糊：blur_ratio=0.5, --->  8张模糊图(4长4短)
	气象退化: weather_ratio=0.5, --->  4张退化图
	遮挡: occlusion_ratio=0.5, --->  4张遮挡图
若img_origin=True, 样本池中共有25+8=33张
若img_origin=False, 样本池中共有25张
'''

'''
关于图片切片产生的背景如何处理的问题(3个相关参数)：
  slice_all_tiles: True：每张被选中的原图 4 片全进池(区段 4K); False：随机 1 片(base=K), 可能漏掉目标。
  slice_target_tiles: =True, 搭配 slice_all_tiles=False 使用, 只选取有目标的切片。
  slice_background_ratio：背景数量=ratio * 正片数; -1背景全部保留, 0背景全部丢弃, 该位置被替换成原图

两种使用方式：
  1. 误检多
  slice_all_tiles=True, slice_target_tiles=False, slice_background_ratio=-1
     → 切片全部进入样本池, 包括背景切片。
  2. 漏检多
  slice_all_tiles=False, slice_target_tiles=True
     → 从4张切片中选取1张有目标的切片进入样本池, 背景切片不进池。

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
        workers=2,
        optimizer='MuSGD',
        device="cpu",
        # resume=r'C:\Users\Administrator\Desktop\ultralytics-improved\runs\exp-3\weights\last.pt',  # 断点续训: 改成你本机 last.pt 路径
        # resume_extend_epochs=15,  # (int, 0=关闭) 续训自动延长: 自动修补ckpt元数据(epochs/patience), 从旧停点续训到该轮数; 需>ckpt已完成轮数
        patience=0,
        amp=True,
        fraction=1.0,
        project=r"C:\Users\Administrator\Desktop\ultralytics-main\runs", 
        name='exp',


        # ---------原图---------------
        img_origin=False, # 是否在样本池里多增加1份原图参与训练            

        # ---------Mosaic在线增强---------------
        mosaic=1.0,
        close_mosaic=1,

        # ---------SAHI在线切片 (slice_*)-----------
        slice_prob=True,              # 独立开关, 开启在线sahi切片增强 (区段 4K)
        slice_ratio=0.5,              # 每个epoch随机选 round(ratio*N) 张原图走切片, 其余不占 base 位;
                                      # 1.0=纯切片, 0.5=一半原图切片, 0=全整图(等效关闭切片); 每epoch重新随机。
        slice_all_tiles=False, 
        slice_target_tiles=True,

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
        compose_ratio=0.5,  # 每epoch随机选 round(x*ceil(N/4)) 组做合成(组级);  1.0=全量合成
        compose_max_side=0, # 合成2x2大图拼后降采样最长边上限(像素): 0=自动=2×imgsz(默认开启, 降内存), >0=手动指定(如2560); 不想要此优化可设 compose_max_side 为一个很大的值关闭。


        # ---------在线比例调整 (ratio_pad_*): 在线加边框统一宽高比---------
        ratio_pad_keep=True,          # 独立开关, 开启在线比例调整 (区段 +K_ratio)
        ratio_pad_ratio=0.5,          # 每epoch随机选 round(x*N) 张原图做比例调整(原图级); 1.0=全量
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
        weather_ratio=0.5,            # 每epoch随机选 round(x*N) 张原图做气象退化(原图级); 0.3~0.6 推荐, 1.0=全量
        weather_types="rain,haze,noise", # (str) 退化类型池, 逗号分隔; 每张图随机抽 1 种; 可子集如 "rain,haze"
        weather_rain_density=0.15,    # 雨线密度 = 雨线数量 / max(h,w), 越大雨越密
        weather_rain_length=15.0,     # 雨线长度上限(像素), 每根随机取 0.5~1.0 倍
        weather_haze_beta=0.4,        # 雾浓度 [0,1), 越大雾越浓
        weather_noise_std=15.0,       # 高斯噪声标准差(每通道独立), 模拟传感器/弱光噪点


        # ---------在线遮挡模拟 (occlusion_*): 语义遮挡块 (树冠/电线/阴影), 提升被遮挡目标鲁棒性, 标签不变---------
        occlusion_keep=True,          # 独立开关, 开启在线遮挡模拟 (区段 +K_occlusion); 与 blur/weather 同构
        occlusion_ratio=0.5,          # 每epoch随机选 round(x*N) 张原图做遮挡(原图级); 0.3~0.6 推荐, 1.0=全量
        occlusion_types="stripe",     # 遮挡类型, 逗号分隔; rect=随机矩形(树冠/阴影), stripe=细长条带(电线/枝干/云影)
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
        val_slice_enable=True,       # 总开关: 验证时把验证图切成 2x2 重叠子图独立推理, 子图框还原到原图坐标, 跨切片重复框 NMS 融合后与原图 GT 算 mAP; False=回归原生整图验证
        val_slice_all_tiles=True,    # True=每张验证图全部 2x2 子图都推理(与训练侧对齐); False=每图随机1片(快速验证)
        val_slice_ratio=0.5,          # 每轮验证随机选 round(x*N_val) 张验证图走切片, 其余整图直通; 1.0=全部切片
        val_slice_overlap_ratio=0.2,  # 验证侧切片重叠比例 [0,1), 建议与训练侧 slice_overlap_ratio 一致
        val_slice_nms_iou=0.5,        # 跨切片重复框 NMS 融合 IoU 阈值
        val_slice_dual_metric=True,   # 双口径: 先跑切片验证(主, 驱动 fitness/早停/best.pt), 再跑整图验证(参考),
                                      #         输出 whole_* 指标并额外保存 best_whole.pt + last_whole.pt (耗时为两遍验证)
                                      #         这两个整图文件在收尾时与 best/last 一样被 strip 成 fp16 推理快照
                                      #         注: last_whole.pt 与 last.pt 内容等价(权重相同)但非逐字节相同
                                      #             (zip 条目前缀带文件名, torch.save 本就不保证字节可复现)

        
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
    )