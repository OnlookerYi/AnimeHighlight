# colors/builder.py
import numpy as np
import cv2
import yaml
import os

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "config")
PRESETS_PATH = os.path.join(BASE_DIR, "colors.yaml")


def build_color_fn(cfg: dict):
    name = cfg["name"]
    h_range = cfg.get("h", [0.0, 1.0])
    s_range = cfg.get("s", [0.0, 1.0])
    v_range = cfg.get("v", [0.0, 1.0])
    center_ratio = cfg.get("center_ratio", 0.6)

    h_ranges = h_range if isinstance(h_range[0], list) else [h_range]

    def color_fn(frame_bgr):
        h, w = frame_bgr.shape[:2]
        margin = int((1 - center_ratio) / 2 * min(h, w))
        crop = frame_bgr[margin:h - margin, margin:w - margin]

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
        hn = hsv[..., 0] / 179.0
        sn = hsv[..., 1] / 255.0
        vn = hsv[..., 2] / 255.0

        h_mask = np.zeros_like(hn, dtype=bool)
        for hr in h_ranges:
            h_mask |= (hn >= hr[0]) & (hn <= hr[1])

        mask = (
            h_mask &
            (sn >= s_range[0]) & (sn <= s_range[1]) &
            (vn >= v_range[0]) & (vn <= v_range[1])
        )

        return {name: float(np.mean(mask))}

    return color_fn


def build_all(active_names=None, center_ratio=None):
    if not os.path.exists(PRESETS_PATH):
        return {}

    with open(PRESETS_PATH, "r", encoding="utf-8") as f:
        configs = yaml.safe_load(f) or {}

    if active_names is not None:
        configs = {
            k: v for k, v in configs.items()
            if k in active_names
        }

    result = {}
    for name, cfg in configs.items():
        # ✅ name 来自 YAML 的 key
        cfg = dict(cfg, name=name)

        if center_ratio is not None:
            cfg = dict(cfg, center_ratio=center_ratio)

        result[name] = build_color_fn(cfg)

    return result