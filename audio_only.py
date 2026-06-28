import subprocess
import numpy as np
import os
import argparse

# ---------- 音频 ----------
def extract_audio_rms(video, win_sec=1):
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
    return np.array(rms)

# ---------- 高光检测 ----------
def find_audio_highlights(
    rms,
    threshold_percentile=60,
    min_duration=3,
    max_duration=30,
    pad=1,
    merge_gap=1.5,
    min_interval=0,
    smooth=0,
    peak_only=False,
    op=0,   # ← OP 秒数
    ed=0,   # ← ED 秒数
):
    # 平滑
    if smooth > 1:
        rms = np.convolve(rms, np.ones(smooth)/smooth, mode="same")

    norm = (rms - rms.min()) / (rms.max() - rms.min() + 1e-6)
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
    if min_interval > 0:
        filtered = [merged[0]]
        for s, e in merged[1:]:
            if s - filtered[-1][1] >= min_interval:
                filtered.append((s, e))
        merged = filtered

    total = len(rms)

    # 时长过滤 + OP/ED 过滤
    final = []
    for s, e in merged:
        dur = e - s
        if min_duration <= dur <= max_duration:
            seg_start = max(0, s - pad)
            seg_end = e + pad

            # 完全在 OP 或 ED 内则丢弃
            if seg_end <= op or seg_start >= total - ed:
                continue

            final.append((seg_start, seg_end))

    # 只保留冲击点
    if peak_only:
        strengths = [e - s for s, e in final]
        mean = np.mean(strengths)
        final = [x for x, d in zip(final, strengths) if d >= mean]

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
    p = argparse.ArgumentParser(description="Professional audio highlight extractor")
    p.add_argument("video")
    p.add_argument("-o", "--output", default="highlights.mp4")

    # 高光
    p.add_argument("--vol", type=float, default=60, help="音量门槛（百分位数）")
    p.add_argument("--min", type=int, default=3, help="最短秒数")
    p.add_argument("--max", type=int, default=30, help="最长秒数")
    p.add_argument("--pad", type=int, default=1, help="前后缓冲")
    p.add_argument("--gap", type=float, default=1.5, help="合并间隔")
    p.add_argument("--interval", type=float, default=0, help="段间最小间隔")
    p.add_argument("--top-k", type=int, default=10, help="最多片段数")

    # OP/ED
    p.add_argument("--op", type=int, default=0, help="OP 秒数")
    p.add_argument("--ed", type=int, default=0, help="ED 秒数")

    # 高级
    p.add_argument("--smooth", type=int, default=0, help="平滑窗口（0=关闭）")
    p.add_argument("--peak-only", action="store_true", help="只保留冲击点")
    p.add_argument("--keep-parts", action="store_true", help="保留 part 文件")

    args = p.parse_args()

    rms = extract_audio_rms(args.video)
    segs = find_audio_highlights(
        rms,
        threshold_percentile=args.vol,
        min_duration=args.min,
        max_duration=args.max,
        pad=args.pad,
        merge_gap=args.gap,
        min_interval=args.interval,
        smooth=args.smooth,
        peak_only=args.peak_only,
        op=args.op,
        ed=args.ed,
    )[:args.top_k]

    if not segs:
        print("⚠️ 未检测到任何高光片段")
        return

    cut_and_concat(args.video, segs, args.output, keep_parts=args.keep_parts)
    print(f"✅ 生成 {len(segs)} 个音频高光片段")

if __name__ == "__main__":
    main()