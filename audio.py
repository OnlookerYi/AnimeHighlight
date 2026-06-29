import numpy as np
import librosa
import subprocess
from scipy.signal import resample

from logger import info, debug, warn, error


def load_audio(video, sr=16000):
    info("AUDIO", f"load {video}")

    cmd = [
        "ffmpeg", "-i", video,
        "-vn", "-ac", "1", "-ar", str(sr),
        "-f", "wav", "-"
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    wav = np.frombuffer(p.stdout.read(), dtype=np.int16).astype(np.float32)

    debug("AUDIO", f"wav samples={len(wav)} sr={sr}")
    return wav


def align_to_rms(x, target_len):
    """统一对齐到 RMS 时间轴"""
    if isinstance(x, np.ndarray) and len(x) != target_len:
        debug("AUDIO", f"resample {len(x)} → {target_len}")
        return resample(x, target_len)
    return x


def extract_audio_features(video, sr=16000):
    info("AUDIO", "开始提取音频特征")

    wav = load_audio(video, sr=sr)

    # ---------- 1. RMS（时间轴基准） ----------
    rms = rms_energy(wav, sr)
    target_len = len(rms)

    debug("AUDIO", f"rms len={target_len}")
    debug("AUDIO", f"rms min={rms.min():.4f} max={rms.max():.4f} mean={rms.mean():.4f}")
    debug("AUDIO", f"rms > 0: {(rms > 0).sum()} / {target_len}")
    debug("AUDIO", f"rms > 0.1: {(rms > 0.1).sum()}")

    if rms.max() - rms.min() < 1e-3:
        warn("AUDIO", "RMS 几乎为常数，高光将极不稳定")

    if np.isnan(rms).any() or np.isinf(rms).any():
        error("AUDIO", "RMS 包含 NaN / Inf")

    # ---------- 2. 其他特征 ----------
    zcr = zero_cross_rate(wav, sr)
    contrast = spectral_contrast(wav, sr, rms_len=target_len)
    flatness = spectral_flatness(wav, sr, rms_len=target_len)
    voice_ratio = voice_to_music_ratio(wav, sr, rms_len=target_len)
    energy_var = energy_variance(rms)
    impulse = impulse_peaks(rms, align_to_rms(zcr, target_len))
    silence = silence_ratio(rms)

    # ---------- 3. 强制对齐 ----------
    feats = {
        "rms": rms,
        "energy_var": align_to_rms(energy_var, target_len),
        "impulse": impulse,
        "zcr": align_to_rms(zcr, target_len),
        "spectral_contrast": align_to_rms(contrast, target_len),
        "flatness": align_to_rms(flatness, target_len),
        "voice_ratio": align_to_rms(voice_ratio, target_len),
        "silence": silence,
        "duration": len(wav) / sr
    }

    # ---------- 4. 特征完整性检查 ----------
    for k, v in feats.items():
        if isinstance(v, np.ndarray):
            debug("AUDIO", f"{k} len={len(v)}")

    debug("AUDIO", f"silence ratio={silence:.3f}")
    debug("AUDIO", f"impulse peaks={len(impulse)}")

    for name in ["energy_var", "voice_ratio", "flatness"]:
        v = feats[name]
        if np.isnan(v).any():
            warn("AUDIO", f"{name} contains NaN")

    debug("AUDIO", "音频特征提取完成")
    return feats


# =======================
# 特征计算函数（完整）
# =======================

def rms_energy(wav, sr, win_sec=1):
    win = int(sr * win_sec)
    return np.array([
        np.sqrt(np.mean(wav[i:i + win] ** 2))
        for i in range(0, len(wav) - win, win)
    ])


def energy_variance(rms, win=3):
    diff = np.diff(rms)
    var = np.convolve(diff ** 2, np.ones(win) / win, mode="same")
    return np.concatenate([[var[0]], var])


def impulse_peaks(rms, zcr, min_gap=2):
    diff = np.diff(rms)
    peaks = np.where(
        (diff > np.percentile(diff, 90)) &
        (zcr[:-1] > np.percentile(zcr, 70))
    )[0]
    return peaks


def zero_cross_rate(wav, sr, win_sec=1):
    hop = int(sr * win_sec)
    zcr = librosa.feature.zero_crossing_rate(wav, hop_length=hop)
    return zcr.flatten()


def spectral_contrast(wav, sr, rms_len):
    spec = librosa.feature.spectral_contrast(y=wav, sr=sr)
    if spec.shape[1] != rms_len:
        spec = resample(spec, rms_len, axis=1)
    return spec.mean(axis=0)


def spectral_flatness(wav, sr, rms_len):
    flat = librosa.feature.spectral_flatness(y=wav)
    if flat.shape[1] != rms_len:
        flat = resample(flat, rms_len, axis=1)
    return flat.flatten()


def voice_to_music_ratio(wav, sr, rms_len):
    spec = np.abs(librosa.stft(wav))
    voice = spec[10:40, :].sum(axis=0)
    music = spec[40:, :].sum(axis=0)
    ratio = voice / (music + 1e-6)
    if len(ratio) != rms_len:
        ratio = resample(ratio, rms_len)
    return ratio


def silence_ratio(rms, th=0.01):
    return np.mean(rms < th)