import os
import subprocess
import numpy as np
from logger import info, debug, warn, error

SECOND_PER_SCORE = 1.0  # 每个 score 代表 1 秒


def idx_to_sec(idx):
    return idx * SECOND_PER_SCORE


def sec_to_hms(seconds):
    s = int(seconds)
    h, m = divmod(s, 3600)
    m, s = divmod(m, 60)
    return f"{h:02}:{m:02}:{s:02}"


def find_highlights(
    score,
    threshold_percentile=65,
    min_duration=2,
    max_duration=30,
    pad=1,
    merge_gap=1.5,
    min_interval=0,
    op=0,
    ed=0,
    peak_only=False
):
    if not isinstance(score, np.ndarray):
        score = np.array(score)

    total = len(score)
    total_sec = total * SECOND_PER_SCORE
    debug("HIGHLIGHT", f"score len={total}  ({sec_to_hms(total_sec)})")
    debug("HIGHLIGHT", f"score min={score.min():.4f} max={score.max():.4f} mean={score.mean():.4f}")

    if total == 0:
        warn("HIGHLIGHT", "score 为空，直接返回")
        return []

    # ---------- 1. normalize ----------
    norm = (score - score.min()) / (score.max() - score.min() + 1e-6)
    debug("HIGHLIGHT", f"norm min={norm.min():.4f} max={norm.max():.4f}")

    # ---------- 2. threshold ----------
    threshold = np.percentile(norm, threshold_percentile)
    debug("HIGHLIGHT", f"threshold={threshold:.4f} (percentile={threshold_percentile})")

    active = norm > threshold
    debug("HIGHLIGHT", f"active_frames={active.sum()} / {total}")

    if not np.any(active):
        warn("HIGHLIGHT", "active 全为 False，无高光候选")
        return []

    # ---------- 3. 初分段 ----------
    segs = []
    start = None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, total))

    debug("HIGHLIGHT", f"raw segments={len(segs)}")
    for i, (s, e) in enumerate(segs):
        s_sec = idx_to_sec(s)
        e_sec = idx_to_sec(e)
        debug("HIGHLIGHT",
              f"raw_seg[{i:02d}] {sec_to_hms(s_sec)}–{sec_to_hms(e_sec)}  "
              f"({e_sec-s_sec:.1f}s)")

    if not segs:
        return []

    # ---------- 4. merge ----------
    merged = []
    cur_s, cur_e = segs[0]
    for s, e in segs[1:]:
        if s - cur_e <= merge_gap:
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    debug("HIGHLIGHT", f"merge_gap={merge_gap}, before={len(segs)}, after={len(merged)}")

    # ---------- 5. min_interval ----------
    if min_interval > 0:
        filtered = [merged[0]]
        for s, e in merged[1:]:
            if s - filtered[-1][1] >= min_interval:
                filtered.append((s, e))
        merged = filtered

    debug("HIGHLIGHT", f"after min_interval={len(merged)}")

    # ---------- 6. pad + 边界裁剪 ----------
    final = []
    for s, e in merged:
        dur = e - s
        if min_duration <= dur <= max_duration:
            seg_start = max(0, s - pad)
            seg_end = min(total, e + pad)

            if seg_end <= op or seg_start >= total - ed:
                continue

            final.append((seg_start, seg_end))
            debug("HIGHLIGHT",
                  f"pad {sec_to_hms(idx_to_sec(s))}–{sec_to_hms(idx_to_sec(e))}  →  "
                  f"{sec_to_hms(idx_to_sec(seg_start))}–{sec_to_hms(idx_to_sec(seg_end))}")

    if not final:
        warn("HIGHLIGHT", "最终高光片段为空")
        return []

    # ---------- 7. peak_only ----------
    if peak_only:
        strengths = [e - s for s, e in final]
        mean_dur = np.mean(strengths)
        before = len(final)
        final = [x for x, d in zip(final, strengths) if d >= mean_dur]
        debug("HIGHLIGHT", f"peak_only mean_dur={mean_dur:.2f}, before={before}, after={len(final)}")

    # ---------- 8. 输出 ----------
    for i, (s, e) in enumerate(final):
        s_sec = idx_to_sec(s)
        e_sec = idx_to_sec(e)
        debug("HIGHLIGHT",
              f"final[{i:02d}] {sec_to_hms(s_sec)}–{sec_to_hms(e_sec)}  "
              f"({e_sec-s_sec:.1f}s)")

    return sorted(final, key=lambda x: x[0])


def cut_and_concat(video, segments, output, keep_parts=False):
    info("CUT", "开始切割并合并视频")

    base = os.path.splitext(os.path.basename(video))[0]
    out_dir = f"output_{base}"
    os.makedirs(out_dir, exist_ok=True)

    parts_dir = os.path.join(out_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)

    output_path = os.path.join(out_dir, output)
    concat_path = os.path.join(out_dir, "concat.txt")

    info("CUT", f"input segments={len(segments)}")
    for i, (s, e) in enumerate(segments):
        s_sec = idx_to_sec(s)
        e_sec = idx_to_sec(e)
        info("CUT", f"seg[{i:02d}] {sec_to_hms(s_sec)}–{sec_to_hms(e_sec)}")

    with open(concat_path, "w") as f:
        for i, (s, e) in enumerate(segments):
            part = os.path.join(parts_dir, f"part{i:02d}.mp4")
            debug("CUT", f"cutting part {i:02d}: {sec_to_hms(idx_to_sec(s))}–{sec_to_hms(idx_to_sec(e))}")

            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(idx_to_sec(s)),
                "-to", str(idx_to_sec(e)),
                "-i", video,
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-c:a", "aac", "-b:a", "128k",
                part
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            f.write(f"file '{os.path.abspath(part)}'\n")

    info("CUT", "开始 concat 合并")
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
                os.remove(os.path.join(parts_dir, f"part{i:02d}.mp4"))
            except Exception:
                pass
        try:
            os.rmdir(parts_dir)
        except Exception:
            pass

    os.remove(concat_path)
    info("CUT", f"输出完成: {output_path}")