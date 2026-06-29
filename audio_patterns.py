# audio_patterns.py
import numpy as np

def detect_audio_patterns(audio_feat):
    """
    输入:audio.py 输出的特征 dict
    输出:pattern 标签 list
    """
    patterns = []
    if is_loud(audio_feat):
        patterns.append("loud")
    if is_cheer(audio_feat):
        patterns.append("cheer")       # 啦啦队 / 喊麦
    if is_scream(audio_feat):
        patterns.append("scream")      # 尖叫
    if is_shout(audio_feat):
        patterns.append("shout")        # 大吼
    if is_shy_or_soft(audio_feat):
        patterns.append("soft")         # 害羞 / 日常

    return patterns


# ---------- 判断函数 ----------
def is_loud(f):
    return (
        f["rms"].mean() > 0.8
    )
def is_cheer(f):
    return (
        f["voice_ratio"].mean() > 0.2 and
        f["energy_var"].mean() > 0.2 and
        f["zcr"].mean() > 0.3
    )


def is_scream(f):
    return (
        f["zcr"].mean() > 0.4 and
        f["spectral_contrast"].mean() > 0.5 and
        f["silence"].mean() < 0.1
    )


def is_shout(f):
    return (
        f["rms"].mean() > 0.5 and
        f["voice_ratio"].mean() > 0.4 and
        f["flatness"].mean() < 0.5
    )


def is_shy_or_soft(f):
    return (
        f["rms"].mean() < 0.3 and
        f["energy_var"].mean() < 0.2
    )