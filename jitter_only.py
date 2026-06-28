import cv2
import numpy as np
import subprocess
import os
import argparse

# ---------- 抖动检测 ----------
def extract_jitter(
    video,
    roi_ratio=0.25,
    win_sec=0.5,
    stride_sec=0.25,
    freq_emphasis=2.0
):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    ret, prev = cap.read()
    h, w = prev.shape[:2]

    # ROI：上半身中心
    x1 = int(w * (0.5 - roi_ratio))
    x2 = int(w * (0.5 + roi_ratio))
    y1 = int(h * 0.1)
    y2 = int(h * 0.55)

    prev_g = cv2.cvtColor(prev[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(float)

    win = int(win_sec * fps)
    stride = int(stride_sec * fps)
    if win < 2:
        win = 2
    if stride < 1:
        stride = 1

    scores = []

    while True:
        ret, f = cap.read()
        if not ret:
            break
        g = cv2.cvtColor(f[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(float)
        diff = np.abs(g - prev_g)

        # 高频强调
        var = np.var(diff)
        mean = np.mean(diff)
        score = var * (mean ** freq_emphasis)

        scores.append(score)
        prev_g = g

    cap.release()

    # 转成每秒
    scores = np.array(scores)
    out = []
    for i in range(0, len(scores), stride):
        seg = scores[i:i+win]
        if len(seg) == 0:
            break
        out.append(np.mean(seg))

    return np.array(out), fps

# ---------- 高光检测 ----------
def find_jitter_highlights(
    jitter,
    threshold_percentile=65,
    min_duration=2,
    max_duration=30,
    pad=1,
    merge_gap=1.5,
    min_interval=0,
    smooth=0
):
    if smooth > 1:
        jitter = np.convolve(jitter, np.ones(smooth)/smooth, mode="same")

    norm = (jitter - jitter.min()) / (jitter.max() - jitter.min() + 1e-6)
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

    # 合并
    merged = []
    cur_s, cur_e = segs[0]
    for s, e in segs[1:]:
        if s - cur_e <= merge_gap:
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    # 最小间隔
    if min_interval > 0 and merged:
        filtered = [merged[0]]
        for s, e in merged[1:]:
            if s - filtered[-1][1] >= min_interval:
                filtered.append((s, e))
        merged = filtered

    # 时长过滤
    final = []
    for s, e in merged:
        dur = e - s
        if min_duration <= dur <= max_duration:
            final.append((max(0, s - pad), e + pad))

    return sorted(final, key=lambda x: x[0])

# ---------- 切割 ----------
def cut_and_concat(video, segments, output, keep_parts=False):
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

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser(description="Jitter-based anime highlight extractor")
    p.add_argument("video")
    p.add_argument("-o", "--output", default="highlights.mp4")

    # 抖动
    p.add_argument("--roi", type=float, default=0.25, help="抖动区域大小")
    p.add_argument("--win", type=float, default=0.5, help="抖动窗口（秒）")
    p.add_argument("--stride", type=float, default=0.25, help="滑动步长（秒）")
    p.add_argument("--freq", type=float, default=2.0, help="高频强调程度")
    p.add_argument("--vol", type=float, default=65, help="抖动门槛（百分位数）")

    # 片段
    p.add_argument("--min", type=int, default=2, help="最短秒数")
    p.add_argument("--max", type=int, default=30, help="最长秒数")
    p.add_argument("--pad", type=int, default=1, help="前后缓冲")
    p.add_argument("--gap", type=float, default=1.5, help="合并间隔")
    p.add_argument("--interval", type=float, default=0, help="段间最小间隔")
    p.add_argument("--top-k", type=int, default=10, help="最多片段数")

    # 高级
    p.add_argument("--smooth", type=int, default=2, help="平滑窗口")
    p.add_argument("--keep-parts", action="store_true", help="保留 parts")

    args = p.parse_args()

    jitter, fps = extract_jitter(
        args.video,
        roi_ratio=args.roi,
        win_sec=args.win,
        stride_sec=args.stride,
        freq_emphasis=args.freq
    )

    segs = find_jitter_highlights(
        jitter,
        threshold_percentile=args.vol,
        min_duration=args.min,
        max_duration=args.max,
        pad=args.pad,
        merge_gap=args.gap,
        min_interval=args.interval,
        smooth=args.smooth
    )[:args.top_k]

    if not segs:
        print("⚠️ 未检测到抖动高光")
        return

    cut_and_concat(args.video, segs, args.output, keep_parts=args.keep_parts)
    print(f"✅ 生成 {len(segs)} 个抖动高光片段")

if __name__ == "__main__":
    main()