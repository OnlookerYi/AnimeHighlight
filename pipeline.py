import numpy as np
import yaml
import os
import argparse

from audio import extract_audio_features
from motion import extract_motion_features
from color import extract_color_features
from audio_patterns import detect_audio_patterns
from motion_patterns import detect_motion_patterns
from highlight_utils import find_highlights, cut_and_concat
from logger import info, debug, warn, error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =======================
# 工具函数
# =======================

def normalize_feature(x):
    x = np.asarray(x, dtype=float)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-6:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def normalize(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


# =======================
# Config & Weights
# =======================

def load_config(path=None):
    info("CONFIG", "加载配置")
    if path is None:
        path = os.path.join(BASE_DIR, "config", "highlight.yaml")
    if not os.path.exists(path):
        error("CONFIG", f"配置文件不存在: {path}")
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_weights(profile_name, patterns, cfg, args):
    profile = cfg["profiles"].get(profile_name, {})
    weights = {}

    for p in patterns:
        if p in profile:
            weights.update(profile[p])

    cli_map = {
        "rms": args.rms,
        "energy_var": args.energy_var,
        "voice": args.voice,
        "motion_energy": args.motion_energy,
        "motion_var": args.motion_var,
        "motion_cut": args.motion_cut,
        "jitter_local": args.jitter_local,
        "jitter_freq": args.jitter_freq,
        "jitter_stability": args.jitter_stability,
        "oscillation": args.oscillation,
        "user_color": args.user_color,   # ✅ 修复
    }

    for k, v in cli_map.items():
        if v > 0:
            weights[k] = v

    return weights


# =======================
# 长度对齐（三路）
# =======================

def align_length(audio, motion, color):
    lengths = []
    if audio is not None:
        lengths.append(len(next(iter(audio.values()))))
    if motion is not None:
        lengths.append(len(next(iter(motion.values()))))
    if color is not None:
        lengths.append(len(next(iter(color.values()))))

    if not lengths:
        error("ALIGN", "audio / motion / color 同时为空")
        return None, None, None

    min_len = min(lengths)

    def truncate(feats):
        if feats is None:
            return None
        return {
            k: v[:min_len] if isinstance(v, np.ndarray) else v
            for k, v in feats.items()
        }

    return truncate(audio), truncate(motion), truncate(color)


# =======================
# Score Building
# =======================

def build_raw_score(audio, motion, color, weights):
    audio, motion, color = align_length(audio, motion, color)

    if audio is not None:
        length = len(next(iter(audio.values())))
    elif motion is not None:
        length = len(next(iter(motion.values())))
    elif color is not None:
        length = len(next(iter(color.values())))
    else:
        error("SCORE", "无可用特征，无法计算 score")
        return np.array([])

    score = np.zeros(length)

    if audio is not None:
        score += weights.get("rms", 0) * normalize_feature(audio["rms"])
        score += weights.get("energy_var", 0) * normalize_feature(audio["energy_var"])
        score += weights.get("voice", 0) * normalize_feature(audio["voice_ratio"])

    if motion is not None:
        score += weights.get("motion_energy", 0) * normalize_feature(motion["energy"])
        score += weights.get("motion_var", 0) * normalize_feature(motion["variance"])
        score += weights.get("motion_cut", 0) * normalize_feature(motion["cut"])
        score += weights.get("jitter_local", 0) * normalize_feature(motion["jitter_local"])
        score += weights.get("jitter_freq", 0) * normalize_feature(motion["jitter_freq"])
        score += weights.get("jitter_stability", 0) * normalize_feature(motion["jitter_stable"])
        score += weights.get("oscillation", 0) * normalize_feature(motion["oscillation"])

    if color is not None:
        for name in color:
            score += weights.get(name, 0) * normalize_feature(color[name])

    return score


# =======================
# Debug 导出
# =======================

def dump_debug_txt(score_raw, score_norm, audio, motion, color, video_path, out_dir="debug"):
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(video_path))[0]
    path = os.path.join(out_dir, f"{name}_debug.txt")

    length = len(score_raw)

    def nf(feats, key, i):
        if feats is None or key not in feats:
            return 0.0
        return float(normalize_feature(feats[key])[i])

    def fmt_time(sec):
        h, rem = divmod(int(sec), 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    w = {
        "sec": 10, "rms": 12, "energy_var": 14, "voice_ratio": 14,
        "motion_energy": 16, "oscillation": 14,
        "jitter_local": 16, "jitter_freq": 14, "jitter_stable": 16,
        "score_raw": 12, "score": 12,
    }

    with open(path, "w", encoding="utf-8") as f:
        header = (
            f"{'sec':<{w['sec']}}\t"
            f"{'rms':<{w['rms']}}\t"
            f"{'energy_var':<{w['energy_var']}}\t"
            f"{'voice_ratio':<{w['voice_ratio']}}\t"
            f"{'motion_energy':<{w['motion_energy']}}\t"
            f"{'oscillation':<{w['oscillation']}}\t"
            f"{'jitter_local':<{w['jitter_local']}}\t"
            f"{'jitter_freq':<{w['jitter_freq']}}\t"
            f"{'jitter_stable':<{w['jitter_stable']}}\t"
        )

        if color is not None:
            for k in color:
                header += f"{k:<14}\t"

        header += f"{'score_raw':<{w['score_raw']}}\t{'score':<{w['score']}}\n"
        f.write(header)

        for i in range(length):
            line = (
                f"{fmt_time(i):<{w['sec']}}\t"
                f"{nf(audio, 'rms', i):<{w['rms']}.4f}\t"
                f"{nf(audio, 'energy_var', i):<{w['energy_var']}.4f}\t"
                f"{nf(audio, 'voice_ratio', i):<{w['voice_ratio']}.4f}\t"
                f"{nf(motion, 'energy', i):<{w['motion_energy']}.4f}\t"
                f"{nf(motion, 'oscillation', i):<{w['oscillation']}.4f}\t"
                f"{nf(motion, 'jitter_local', i):<{w['jitter_local']}.4f}\t"
                f"{nf(motion, 'jitter_freq', i):<{w['jitter_freq']}.4f}\t"
                f"{nf(motion, 'jitter_stable', i):<{w['jitter_stable']}.4f}\t"
            )

            if color is not None:
                for k in color:
                    line += f"{nf(color, k, i):<14.4f}\t"

            line += f"{score_raw[i]:<{w['score_raw']}.4f}\t{score_norm[i]:<{w['score']}.4f}\n"
            f.write(line)

    info("DEBUG", f"已导出调试信息: {path}")


# =======================
# Pipeline（✅ 已修复）
# =======================

def run(video_path, args):
    info("RUN", f"video={video_path}")
    info("RUN", f"use_audio={args.use_audio}, use_motion={args.use_motion}, use_color={args.use_color}")

    if args.output == "highlights.mp4":
        src_name = os.path.splitext(os.path.basename(video_path))[0]
        args.output = f"HL-{src_name}.mp4"

    cfg = load_config(args.config)
    if not cfg:
        error("RUN", "配置为空，终止运行")
        return

    use_audio = args.use_audio
    use_motion = args.use_motion
    use_color = args.use_color

    if not use_audio and not use_motion and not use_color:
        error("RUN", "至少需要开启 audio / motion / color 之一")
        return

    audio = extract_audio_features(video_path) if use_audio else None
    motion = extract_motion_features(
        video_path,
        sample_rate=args.sample_rate,
        flow_scale=args.flow_scale,
        mode=args.motion_mode
    ) if use_motion else None

    ap = detect_audio_patterns(audio) if use_audio else []
    mp = detect_motion_patterns(motion) if use_motion else []
    patterns = set(ap + mp)

    info("PATTERN", f"detected={patterns}")

    weights = resolve_weights(args.profile, patterns, cfg, args)
    info("WEIGHT", f"final weights={weights}")

    # ✅ 关键修复：color 必须在所有路径下定义
    color = None

    if use_color:
        from colors.builder import build_all
        all_colors = set(build_all().keys())

        active_colors = {
            k for k in weights
            if k in all_colors and weights[k] > 0
        }

        if not active_colors:
            warn("COLOR", "use_color 已开启，但未给任何色彩权重，跳过色彩")
        else:
            color = extract_color_features(
                video_path,
                sample_rate=args.sample_rate,
                center_ratio=args.center_ratio,
                active_colors=active_colors
            )

    score_raw = build_raw_score(audio, motion, color, weights)
    if score_raw.size == 0:
        error("RUN", "score 为空，无法生成高光")
        return

    score = normalize(score_raw)

    if args.smooth > 1:
        score = np.convolve(score, np.ones(args.smooth) / args.smooth, mode="same")

    dump_debug_txt(score_raw, score, audio, motion, color, video_path)

    segs = find_highlights(
        score,
        threshold_percentile=args.threshold,
        min_duration=args.min,
        max_duration=args.max,
        pad=args.pad,
        merge_gap=args.gap,
        min_interval=args.interval,
        op=args.op,
        ed=args.ed
    )[:args.top_k]

    info("RESULT", f"highlights={len(segs)}")
    cut_and_concat(video_path, segs, args.output, keep_parts=args.keep_parts)


# =======================
# CLI
# =======================

def main():
    info("MAIN", "start pipeline")

    p = argparse.ArgumentParser(description="Auto Highlight Extractor")
    p.add_argument("video")
    p.add_argument("-o", "--output", default="highlights.mp4")
    p.add_argument("--profile", default="default")
    p.add_argument("--config", default=None)

    p.add_argument("--min", type=int, default=3)
    p.add_argument("--max", type=int, default=30)
    p.add_argument("--pad", type=int, default=0)
    p.add_argument("--gap", type=float, default=1.5)
    p.add_argument("--threshold", type=int, default=70)
    p.add_argument("--interval", type=float, default=2)
    p.add_argument("--smooth", type=int, default=0)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--op", type=int, default=0)
    p.add_argument("--ed", type=int, default=0)

    p.add_argument("--sample-rate", type=int, default=4)
    p.add_argument("--flow-scale", type=float, default=0.5)
    p.add_argument("--motion-mode", choices=["fast", "accurate"], default="fast")

    p.add_argument("--use-audio", action="store_true")
    p.add_argument("--use-motion", action="store_true")
    p.add_argument("--use-color", action="store_true")
    p.add_argument("--keep-parts", action="store_true")

    # Audio
    p.add_argument("--rms", type=float, default=0.0)
    p.add_argument("--energy-var", type=float, default=0.0)
    p.add_argument("--voice", type=float, default=0.0)

    # Motion
    p.add_argument("--motion-energy", type=float, default=0.0)
    p.add_argument("--motion-var", type=float, default=0.0)
    p.add_argument("--motion-cut", type=float, default=0.0)
    p.add_argument("--jitter-local", type=float, default=0.0)
    p.add_argument("--jitter-freq", type=float, default=0.0)
    p.add_argument("--jitter-stability", type=float, default=0.0)
    p.add_argument("--oscillation", type=float, default=0.0)

    # Color
    p.add_argument("--milk-white", type=float, default=0.0)
    p.add_argument("--dark-shadow", type=float, default=0.0)
    p.add_argument("--boss-flash", type=float, default=0.0)
    p.add_argument("--user-color", type=float, default=0.0)
    p.add_argument("--center-ratio", type=float, default=0.6)

    args = p.parse_args()
    run(args.video, args)


if __name__ == "__main__":
    main()