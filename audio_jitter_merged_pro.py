import cv2
import numpy as np
import subprocess
import os
import argparse

# ============================================================
# 音频
# ============================================================
def extract_audio_rms(video, win_sec=1):
    print("🎵 正在提取音频 RMS ...")
    cmd = ["ffmpeg", "-i", video, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    audio = np.frombuffer(p.stdout.read(), dtype=np.int16)

    win = int(16000 * win_sec)
    rms = []
    for i in range(0, len(audio), win):
        seg = audio[i:i+win]
        if len(seg) == 0:
            break
        rms.append(np.sqrt(np.mean(seg.astype(float) ** 2)))

    print("✅ 音频 RMS 完成")
    return np.array(rms)


# ============================================================
# 抖动
# ============================================================
def extract_jitter(
    video,
    roi_ratio=0.18,
    head_ratio=0.25,
    foot_ratio=0.55,
    win_sec=0.5,
    stride_sec=0.25,
    freq_emphasis=2.5,
    low_freq_filter=1.8
):
    print("🎞 开始人物抖动检测 ...")
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, prev = cap.read()
    h, w = prev.shape[:2]

    cx = w // 2
    rw = int(w * roi_ratio)
    x1 = max(0, cx - rw)
    x2 = min(w, cx + rw)
    y1 = int(h * head_ratio)
    y2 = int(h * foot_ratio)

    prev_g = cv2.cvtColor(prev[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(float)

    win = max(2, int(win_sec * fps))
    stride = max(1, int(stride_sec * fps))

    scores = []
    frame_count = 0

    while True:
        ret, f = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 300 == 0 or frame_count == total_frames:
            print(f"🎞 抖动检测进度: {frame_count/total_frames*100:.1f}%")

        g = cv2.cvtColor(f[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(float)
        diff = np.abs(g - prev_g)

        mean_d = np.mean(diff)
        var_d = np.var(diff)

        score = var_d * (mean_d ** freq_emphasis)
        if mean_d < low_freq_filter:
            score *= 0.1

        scores.append(score)
        prev_g = g

    cap.release()
    print("✅ 人物抖动检测完成")

    scores = np.array(scores)
    out = []
    for i in range(0, len(scores), stride):
        seg = scores[i:i+win]
        if len(seg) == 0:
            break
        out.append(np.mean(seg))

    return np.array(out), fps


# ============================================================
# 特征颜色（角色锁定）
# ============================================================
def extract_color_activity(
    video,
    hsv_ranges,
    win_sec=0.5,
    stride_sec=0.25,
    min_area=0.001
):
    print("🌈 正在提取特征颜色活动 ...")
    cap = cv2.VideoCapture(video)

    win = max(1, int(win_sec * 30))
    stride = max(1, int(stride_sec * 30))

    prev_mask = None
    scores = []

    while True:
        ret, f = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        mask = np.zeros((hsv.shape[0], hsv.shape[1]), dtype=np.uint8)

        for lower, upper in hsv_ranges:
            m = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_or(mask, m)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        if np.count_nonzero(mask) / mask.size < min_area:
            mask[:] = 0

        if prev_mask is not None:
            diff = cv2.absdiff(mask, prev_mask)
            scores.append(np.sum(diff) / mask.size)

        prev_mask = mask

    cap.release()
    print("✅ 特征颜色活动完成")

    out = []
    for i in range(0, len(scores), stride):
        seg = scores[i:i+win]
        if len(seg) == 0:
            break
        out.append(np.mean(seg))

    return np.array(out)


# ============================================================
# 高光检测
# ============================================================
def find_highlights(
    score,
    threshold_percentile=65,
    min_duration=2,
    max_duration=30,
    pad=1,
    merge_gap=1.5,
    min_interval=0,
    smooth=0,
    op=0,
    ed=0,
    peak_only=False
):
    if smooth > 1:
        score = np.convolve(score, np.ones(smooth)/smooth, mode="same")

    norm = (score - score.min()) / (score.max() - score.min() + 1e-6)
    active = norm > np.percentile(norm, threshold_percentile)

    segs = []
    start = None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(active)))

    merged = []
    cur_s, cur_e = segs[0]
    for s, e in segs[1:]:
        if s - cur_e <= merge_gap:
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    if min_interval > 0:
        filtered = [merged[0]]
        for s, e in merged[1:]:
            if s - filtered[-1][1] >= min_interval:
                filtered.append((s, e))
        merged = filtered

    total = len(score)
    final = []
    for s, e in merged:
        dur = e - s
        if min_duration <= dur <= max_duration:
            seg_start = max(0, s - pad)
            seg_end = e + pad
            if seg_end <= op or seg_start >= total - ed:
                continue
            final.append((seg_start, seg_end))

    if peak_only:
        strengths = [e - s for s, e in final]
        mean = np.mean(strengths)
        final = [x for x, d in zip(final, strengths) if d >= mean]

    return sorted(final, key=lambda x: x[0])


# ============================================================
# 切割 & 合并
# ============================================================
def cut_and_concat(video, segments, output, keep_parts=False):
    print("✂️ 正在切割并合并视频 ...")
    base = os.path.splitext(os.path.basename(video))[0]
    out_dir = f"output_{base}"
    os.makedirs(out_dir, exist_ok=True)
    parts_dir = os.path.join(out_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)

    output_path = os.path.join(out_dir, output)
    concat_path = os.path.join(out_dir, "concat.txt")

    with open(concat_path, "w") as f:
        for i, (s, e) in enumerate(segments):
            part = os.path.join(parts_dir, f"part{i}.mp4")
            subprocess.run([
                "ffmpeg", "-y", "-ss", str(s), "-to", str(e),
                "-i", video,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                part
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            f.write(f"file '{os.path.abspath(part)}'\n")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not keep_parts:
        for i in range(len(segments)):
            try:
                os.remove(os.path.join(parts_dir, f"part{i}.mp4"))
            except:
                pass
        os.rmdir(parts_dir)

    os.remove(concat_path)
    print(f"📂 输出目录: {out_dir}")


# ============================================================
# CLI
# ============================================================
def main():
    p = argparse.ArgumentParser(description="Audio + Jitter + Color highlight extractor")

    p.add_argument("video")
    p.add_argument("-o", "--output", default="highlights.mp4")

    # 音频
    p.add_argument("--w-audio", type=float, default=0.4)
    p.add_argument("--smooth-audio", type=int, default=0)

    # 抖动
    p.add_argument("--w-jitter", type=float, default=0.3)
    p.add_argument("--roi", type=float, default=0.25)
    p.add_argument("--win", type=float, default=0.5)
    p.add_argument("--stride", type=float, default=0.25)
    p.add_argument("--freq", type=float, default=2.0)

    # 颜色（角色）
    p.add_argument("--w-color", type=float, default=0.3)
    p.add_argument("--color-win", type=float, default=0.5)
    p.add_argument("--color-stride", type=float, default=0.25)

    p.add_argument("--color-h-min", type=int, default=125)
    p.add_argument("--color-h-max", type=int, default=155)
    p.add_argument("--color-s-min", type=int, default=80)
    p.add_argument("--color-s-max", type=int, default=255)
    p.add_argument("--color-v-min", type=int, default=80)
    p.add_argument("--color-v-max", type=int, default=255)

    # 高光
    p.add_argument("--vol", type=float, default=65)
    p.add_argument("--min", type=int, default=2)
    p.add_argument("--max", type=int, default=30)
    p.add_argument("--pad", type=int, default=1)
    p.add_argument("--gap", type=float, default=1.5)
    p.add_argument("--interval", type=float, default=0)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--smooth", type=int, default=0)
    p.add_argument("--op", type=int, default=0)
    p.add_argument("--ed", type=int, default=0)
    p.add_argument("--peak-only", action="store_true")
    p.add_argument("--keep-parts", action="store_true")

    args = p.parse_args()

    # 音频
    rms = extract_audio_rms(args.video)
    rms = (rms - rms.min()) / (rms.max() - rms.min() + 1e-6)
    if args.smooth_audio > 1:
        rms = np.convolve(rms, np.ones(args.smooth_audio)/args.smooth_audio, mode="same")

    # 抖动
    if args.w_jitter > 0:
        jitter, _ = extract_jitter(
            args.video,
            roi_ratio=args.roi,
            win_sec=args.win,
            stride_sec=args.stride,
            freq_emphasis=args.freq
        )
        jitter = (jitter - jitter.min()) / (jitter.max() - jitter.min() + 1e-6)
    else:
        jitter = np.zeros_like(rms)

    # 颜色
    if args.w_color > 0:
        hsv_ranges = [(
            np.array([args.color_h_min, args.color_s_min, args.color_v_min]),
            np.array([args.color_h_max, args.color_s_max, args.color_v_max])
        )]
        color = extract_color_activity(
            args.video,
            hsv_ranges=hsv_ranges,
            win_sec=args.color_win,
            stride_sec=args.color_stride
        )
        color = (color - color.min()) / (color.max() - color.min() + 1e-6)
    else:
        color = np.zeros_like(rms)

    # 对齐
    min_len = min(len(rms), len(jitter), len(color))
    rms = rms[:min_len]
    jitter = jitter[:min_len]
    color = color[:min_len]

    # 融合
    score = (
        args.w_audio * rms +
        args.w_jitter * jitter +
        args.w_color * color
    )

    segs = find_highlights(
        score,
        threshold_percentile=args.vol,
        min_duration=args.min,
        max_duration=args.max,
        pad=args.pad,
        merge_gap=args.gap,
        min_interval=args.interval,
        smooth=args.smooth,
        op=args.op,
        ed=args.ed,
        peak_only=args.peak_only
    )[:args.top_k]

    if not segs:
        print("⚠️ 未检测到高光片段")
        return

    cut_and_concat(args.video, segs, args.output, keep_parts=args.keep_parts)
    print(f"✅ 生成 {len(segs)} 个高光片段")


if __name__ == "__main__":
    main()