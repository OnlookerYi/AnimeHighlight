import cv2
import numpy as np
from logger import info, debug, warn, error


def extract_motion_features(
    video,
    sample_rate=4,
    flow_scale=0.5,
    mode="fast"
):
    info("MOTION", f"开始提取动作特征: {video} mode={mode}")

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    info("MOTION", f"fps={fps}")

    ret, prev = cap.read()
    if not ret:
        error("MOTION", "无法读取视频帧")
        cap.release()
        return {}

    h, w = prev.shape[:2]
    x1, x2 = w // 2 - 200, w // 2 + 200
    y1, y2 = int(h * 0.25), int(h * 0.6)

    prev_g = gray(prev, y1, y2, x1, x2)
    prev_g = resize(prev_g, flow_scale)

    frame_energy = []
    frame_variance = []
    frame_cuts = []
    frame_diffs = []
    frame_osc = []
    frame_jitter_local = []

    frame_idx = 0
    last_log = 0

    # ✅ 动漫振荡：维护质心历史
    osc_history = []

    while True:
        ret, f = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % sample_rate != 0:
            continue

        g = gray(f, y1, y2, x1, x2)
        g = resize(g, flow_scale)
        diff = np.abs(g.astype(np.float32) - prev_g.astype(np.float32))

        frame_energy.append(diff.mean())
        frame_variance.append(local_variance(diff))
        frame_cuts.append(edge_change(g, prev_g))
        frame_diffs.append(diff.mean())
        frame_jitter_local.append(local_variance(diff))

        if mode == "fast":
            # ✅ 动漫专用振荡函数
            osc, osc_history = motion_oscillation_anime(
                prev_g, g, history=osc_history
            )
            frame_osc.append(osc)
        else:
            frame_osc.append(motion_oscillation_accurate(prev_g, g))

        if frame_idx - last_log >= 100 * sample_rate:
            debug("MOTION", f"processed frames={frame_idx}")
            last_log = frame_idx

        prev_g = g

    cap.release()
    info("MOTION", f"总处理帧数={frame_idx}")

    if frame_idx == 0:
        warn("MOTION", "未提取到任何帧")
        return {}

    energy_sec = aggregate_to_seconds(frame_energy, fps, sample_rate)
    variance_sec = aggregate_to_seconds(frame_variance, fps, sample_rate)
    cuts_sec = aggregate_to_seconds(frame_cuts, fps, sample_rate)
    diffs_sec = aggregate_to_seconds(frame_diffs, fps, sample_rate)
    osc_sec = aggregate_to_seconds(frame_osc, fps, sample_rate)
    jitter_local_sec = aggregate_to_seconds(frame_jitter_local, fps, sample_rate)

    jitter_freq_sec = jitter_frequency_sliding(diffs_sec, win=5)
    jitter_stable_sec = jitter_stability_sliding(diffs_sec, win=5)

    info("MOTION", f"jitter_freq_mean={np.mean(jitter_freq_sec):.4f}")
    info("MOTION", f"jitter_stable_mean={np.mean(jitter_stable_sec):.4f}")

    feats = smooth({
        "energy": energy_sec,
        "variance": variance_sec,
        "cut": cuts_sec,
        "jitter_local": jitter_local_sec,
        "jitter_freq": jitter_freq_sec,
        "jitter_stable": jitter_stable_sec,
        "oscillation": osc_sec
    })

    return feats


# =======================
# ✅ 动漫专用振荡函数
# =======================

def motion_oscillation_anime(
    prev_g, curr_g,
    history=None,
    center_ratio=0.7,   # 中心 70% 区域
    diff_thresh_ratio=0.3  # 差分阈值（相对）
):
    """
    适用于二次元动漫、人物居中、深蹲/跳绳/摇头等往复动作。
    核心思路：跟踪"运动区域质心"的位置交替，不假设连续位移。
    """
    h, w = prev_g.shape

    # 1️⃣ 只取中心区域（人物通常在这里）
    margin = int((1 - center_ratio) / 2 * min(h, w))
    y1 = margin
    y2 = h - margin
    x1 = margin
    x2 = w - margin

    p = prev_g[y1:y2, x1:x2].astype(np.float32)
    c = curr_g[y1:y2, x1:x2].astype(np.float32)

    # 2️⃣ 帧间差分（定位"哪些像素变了"）
    delta = np.abs(c - p)
    d_mean = delta.mean()
    d_std = delta.std()
    motion_mask = delta > (d_mean + diff_thresh_ratio * d_std)

    if motion_mask.sum() < 20:
        # 没足够运动像素，返回 0 并保留 history
        if history is not None:
            history.append((w / 2, h / 2))  # 默认中心
            if len(history) > 8:
                history.pop(0)
        return 0.0, history if history is not None else []

    # 3️⃣ 算"运动区域的质心"
    ys, xs = np.where(motion_mask)
    cx = xs.mean() / (x2 - x1)   # 归一化到 [0, 1]
    cy = ys.mean() / (y2 - y1)

    energy = delta[motion_mask].mean()

    if history is None:
        history = []

    history.append((cx, cy))
    if len(history) > 8:
        history.pop(0)

    if len(history) < 3:
        return 0.0, history

    # 4️⃣ 检测质心在 x / y 方向的"来回"
    hist_arr = np.array(history)
    x_s = hist_arr[:, 0]
    y_s = hist_arr[:, 1]

    # 用 abs diff > 0.05 代替 sign flip，避免被静止帧打断
    x_osc = np.sum(np.abs(np.diff(x_s)) > 0.05)
    y_osc = np.sum(np.abs(np.diff(y_s)) > 0.05)

    score = energy * (1.0 + x_osc + y_osc)
    return score, history


# =======================
# 原版 oscillation（保留，用于真人视频）
# =======================

def motion_oscillation_fast(prev_g, curr_g):
    delta = curr_g.astype(np.float32) - prev_g.astype(np.float32)
    energy = np.mean(np.abs(delta))
    sign_changes = np.sum(np.diff(np.sign(delta.ravel())) != 0)
    flip_ratio = sign_changes / delta.size
    return energy * (1.0 + flip_ratio)


def motion_oscillation_accurate(prev_g, curr_g):
    flow = cv2.calcOpticalFlowFarneback(
        prev_g.astype(np.uint8),
        curr_g.astype(np.uint8),
        None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    dx = flow[..., 0].ravel()
    dy = flow[..., 1].ravel()
    mag = np.sqrt(dx**2 + dy**2)

    moving = mag > 0.3  # ✅ 降低阈值适配更多场景
    if moving.sum() < 10:
        return 0.0

    dx_m = dx[moving]
    dy_m = dy[moving]

    # ✅ 用 abs diff 代替 sign flip，对跳变更鲁棒
    dx_diff = np.diff(np.abs(dx_m))
    dy_diff = np.diff(np.abs(dy_m))
    flips = (np.sum(np.abs(dx_diff) > 0.3) +
             np.sum(np.abs(dy_diff) > 0.3))

    return flips / (len(dx_m) * 2) * np.mean(mag[moving])


# =======================
# 工具函数
# =======================

def aggregate_to_seconds(values, fps, sample_rate=1):
    step = int(fps / sample_rate)
    secs = len(values) // step
    if secs == 0:
        return np.array([np.mean(values)])
    return np.array([
        np.mean(values[i * step:(i + 1) * step])
        for i in range(secs)
    ])


def resize(img, scale):
    if scale == 1.0:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)


def gray(f, y1, y2, x1, x2):
    return cv2.cvtColor(f[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(float)


def edge_change(g, prev_g):
    return cv2.Canny(g.astype(np.uint8), 50, 150).mean()


def local_variance(diff, grid=2):
    h, w = diff.shape
    vars_ = []
    for i in range(grid):
        for j in range(grid):
            vars_.append(diff[
                i * h // grid:(i + 1) * h // grid,
                j * w // grid:(j + 1) * w // grid
            ].var())
    return np.mean(vars_)


def jitter_frequency_sliding(diff_series, win=5):
    out = np.zeros_like(diff_series)
    for i in range(len(diff_series)):
        s = max(0, i - win // 2)
        e = min(len(diff_series), i + win // 2 + 1)
        window = diff_series[s:e]
        if len(window) < 2:
            out[i] = 0.0
            continue
        fft = np.fft.rfft(window)
        freqs = np.fft.rfftfreq(len(window), d=1.0)
        mask = (freqs >= 0.5) & (freqs <= 5)
        out[i] = np.mean(np.abs(fft)[mask]) if mask.any() else 0.0
    return out


def jitter_stability_sliding(diff_series, win=5):
    out = np.zeros_like(diff_series)
    for i in range(len(diff_series)):
        s = max(0, i - win // 2)
        e = min(len(diff_series), i + win // 2 + 1)
        window = diff_series[s:e]
        out[i] = 1.0 / (np.std(window) + 1e-6)
    return out


def smooth(feats, win=3):
    return {k: np.convolve(v, np.ones(win) / win, mode="same")
            for k, v in feats.items()}