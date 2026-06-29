# colors/sampler.py
import numpy as np
import cv2
import yaml
import os

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "config")
PRESETS_PATH = os.path.join(BASE_DIR, "colors.yaml")


def sample_color_from_image(image_path, center_ratio=0.6, name="custom_color"):
    f = cv2.imread(image_path)
    if f is None:
        raise ValueError(f"无法读取图片: {image_path}")

    h, w = f.shape[:2]
    margin = int((1 - center_ratio) / 2 * min(h, w))
    crop = f[margin:h - margin, margin:w - margin]

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    hn = hsv[..., 0] / 179.0
    sn = hsv[..., 1] / 255.0
    vn = hsv[..., 2] / 255.0

    h_min, h_max = float(hn.min()), float(hn.max())
    s_min, s_max = float(sn.min()), float(sn.max())
    v_min, v_max = float(vn.min()), float(vn.max())

    pad = 0.02
    return {
        "name": name,
        "h": [max(0.0, h_min - pad), min(1.0, h_max + pad)],
        "s": [max(0.0, s_min - pad), min(1.0, s_max + pad)],
        "v": [max(0.0, v_min - pad), min(1.0, v_max + pad)],
        "center_ratio": center_ratio
    }


def save_preset(cfg: dict):
    os.makedirs(os.path.dirname(PRESETS_PATH), exist_ok=True)

    if os.path.exists(PRESETS_PATH):
        with open(PRESETS_PATH, "r", encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}
    else:
        existing = {}

    existing[cfg["name"]] = {
        k: v for k, v in cfg.items() if k != "name"
    }

    with open(PRESETS_PATH, "w", encoding="utf-8") as f:
        yaml.dump(existing, f, allow_unicode=True, sort_keys=False)


def register_from_image(image_path, name, center_ratio=0.6):
    """
    用户唯一需要调用的接口。
    """
    cfg = sample_color_from_image(image_path, center_ratio, name)
    save_preset(cfg)
    return cfg