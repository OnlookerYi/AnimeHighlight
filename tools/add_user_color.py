#!/usr/bin/env python3
"""
用户上传图片 → 覆盖 user_color
"""
import argparse
import sys
import os

# ✅ 把项目根目录加入 Python 搜索路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from colors.sampler import sample_color_from_image, save_preset


def main():
    parser = argparse.ArgumentParser(description="上传图片覆盖 user_color")
    parser.add_argument("image", help="参考图片路径")
    parser.add_argument("--ratio", type=float, default=0.6)
    args = parser.parse_args()

    cfg = sample_color_from_image(
        args.image,
        center_ratio=args.ratio,
        name="user_color"
    )
    save_preset(cfg)
    print("✅ user_color 已更新")
    print("👉 使用方式：--user-color 2.0")


if __name__ == "__main__":
    main()