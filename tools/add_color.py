# tools/add_color.py
import argparse
from colors.sampler import register_from_image
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="【开发者】新增系统色彩配置")
    parser.add_argument("image", help="参考图片路径")
    parser.add_argument("name", help="色彩名称（如 boss_flash / heal_green）")
    parser.add_argument("--ratio", type=float, default=0.6)
    args = parser.parse_args()

    register_from_image(args.image, args.name, args.ratio)
    print(f"✅ 系统色彩已添加：{args.name}")
    print(f"👉 在 pipeline 中启用：--{args.name} 2.0")


if __name__ == "__main__":
    main()