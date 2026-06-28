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
    roi_ratio=0.18,     # 人物宽度占比（越窄越准）
    head_ratio=0.25,     # 头顶位置（跳过头部/嘴）
    foot_ratio=0.55,     # 脚底位置
    win_sec=0.5,
    stride_sec=0.25,
    freq_emphasis=2.5,   # 高频强调（人物抖动）
    low_freq_filter=1.8   # 低于此=背景/对话
):
    print("🎞 开始人物抖动检测 ...")
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, prev = cap.read()
    h, w = prev.shape[:2]

    # ✅ 自适应中轴
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

        # ✅ 高频强调
        score = var_d * (mean_d ** freq_emphasis)

        # ✅ 低频频谱过滤（背景/对话）
        if mean_d < low_freq_filter:
            score *= 0.1

        scores.append(score)
        prev_g = g

    cap.release()
    print("✅ 人物抖动检测完成")

    # 转成每秒
    scores = np.array(scores)
    out = []
    for i in range(0, len(scores), stride):
        seg = scores[i:i+win]
        if len(seg) == 0:
            break
        out.append(np.mean(seg))

    return np.array(out), fps

# ============================================================
# 高光检测（通用）
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
    threshold = np.percentile(norm, threshold_percentile)

    active = norm > threshold
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

    if not segs:
        return []

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
# 切割
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
            try: os.remove(os.path.join(parts_dir, f"part{i}.mp4"))
            except: pass
        os.rmdir(parts_dir)

    os.remove(concat_path)
    print(f"📂 输出目录: {out_dir}")

# ============================================================
# CLI
# ============================================================
def main():
    p = argparse.ArgumentParser(description="Audio + Jitter highlight extractor")
    p.add_argument("video")
    p.add_argument("-o", "--output", default="highlights.mp4")

    # 音频
    p.add_argument("--vol-audio", type=float, default=60, help="音频门槛")
    p.add_argument("--smooth-audio", type=int, default=0, help="音频平滑")

    # 抖动
    p.add_argument("--vol-jitter", type=float, default=65, help="抖动门槛")
    p.add_argument("--roi", type=float, default=0.25, help="抖动区域")
    p.add_argument("--win", type=float, default=0.5, help="抖动窗口（秒）")
    p.add_argument("--stride", type=float, default=0.25, help="抖动步长（秒）")
    p.add_argument("--freq", type=float, default=2.0, help="高频强调")

    # 融合
    p.add_argument("--w-audio", type=float, default=0.5, help="音频权重")
    p.add_argument("--w-jitter", type=float, default=0.5, help="抖动权重")

    # 片段
    p.add_argument("--min", type=int, default=2, help="最短秒数")
    p.add_argument("--max", type=int, default=30, help="最长秒数")
    p.add_argument("--pad", type=int, default=1, help="前后缓冲")
    p.add_argument("--gap", type=float, default=1.5, help="合并间隔")
    p.add_argument("--interval", type=float, default=0, help="段间最小间隔")
    p.add_argument("--top-k", type=int, default=10, help="最多片段数")

    # OP/ED
    p.add_argument("--op", type=int, default=0, help="OP 秒数")
    p.add_argument("--ed", type=int, default=0, help="ED 秒数")

    # 高级
    p.add_argument("--peak-only", action="store_true", help="只保留冲击点")
    p.add_argument("--keep-parts", action="store_true", help="保留 parts")

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

    # 对齐长度
    min_len = min(len(rms), len(jitter))
    rms = rms[:min_len]
    jitter = jitter[:min_len]

    # 融合
    score = args.w_audio * rms + args.w_jitter * jitter

    segs = find_highlights(
        score,
        threshold_percentile=args.vol_jitter,
        min_duration=args.min,
        max_duration=args.max,
        pad=args.pad,
        merge_gap=args.gap,
        min_interval=args.interval,
        smooth=0,
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