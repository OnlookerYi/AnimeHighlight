# motion_patterns.py
import numpy as np

def detect_motion_patterns(motion_feat):
    print('检测动作标签')
    """
    输入：motion.py 输出的特征 dict
    输出：pattern 标签 list
    """
    patterns = []

    if is_mechanical(motion_feat):
        patterns.append("mechanical")   # 俯卧撑 / 开合跳
    if is_intense(motion_feat):
        patterns.append("intense")      # 战斗 / 跳舞
    if is_flash(motion_feat):
        patterns.append("flash")        # 转场 / 特效

    return patterns


# ---------- 判断函数 ----------
def is_mechanical(f):
    return (
        f["jitter_local"].mean() > 0.6 and
        f["jitter_freq"].mean() > 0.5 and
        f["jitter_stable"].mean() > 0.7
    )


def is_intense(f):
    """
    剧烈运动：高能 + 高方差
    """
    return (
        f["energy"].mean() > 0.6 and
        f["variance"].mean() > 0.4
    )


def is_flash(f):
    """
    画面突变：边缘剧烈变化
    """
    return f["cut"].max() > 0.8