import cv2
import numpy as np
from logger import info, debug, warn
from colors.builder import build_all

DEBUG_EVERY = 300  # ✅ 每 300 帧打一条（约 10 秒一次）


def extract_color_features(
    video,
    sample_rate=4,
    center_ratio=0.6,
    active_colors=None
):
    info("COLOR", f"开始提取色彩特征: {video}")

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30

    COLOR_FNS = build_all(
        active_names=active_colors,
        center_ratio=center_ratio
    )

    if not COLOR_FNS:
        warn("COLOR", "没有启用任何色彩特征")
        return {}

    info("COLOR", f"启用色彩: {list(COLOR_FNS.keys())}")

    frame_idx = 0
    color_lists = {name: [] for name in COLOR_FNS}

    while True:
        ret, f = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % sample_rate != 0:
            continue

        # ✅ 只打“心跳”，不打数值
        if frame_idx % DEBUG_EVERY == 0:
            debug("COLOR", f"processing frame={frame_idx}")

        for name, fn in COLOR_FNS.items():
            feat = fn(f)
            color_lists[name].append(feat[name])

    cap.release()
    info("COLOR", f"总处理帧数={frame_idx}")

    if frame_idx == 0:
        warn("COLOR", "未提取到任何帧")
        return {}

    step = max(1, int(fps / sample_rate))
    secs = max(1, len(next(iter(color_lists.values()))) // step)

    def agg(vals):
        return np.array([
            np.mean(vals[i * step:(i + 1) * step])
            for i in range(secs)
        ])

    result = {}
    for name, vals in color_lists.items():
        arr = agg(vals)
        result[name] = arr

        # ✅ 只打统计信息（非常重要）
        debug(
            "COLOR",
            f"{name}: len={len(arr)} "
            f"min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f}"
        )

    return result