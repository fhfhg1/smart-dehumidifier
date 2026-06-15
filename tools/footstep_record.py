#!/usr/bin/env python3
"""脚步声分类器 —— 训练样本录制器(也是可行性探针)。

依赖 sox(brew install sox)。把麦克风录成定长小段 wav,存到带标签的目录,
并打印每段的 RMS/峰值——用来判断脚步声到底有没有冒出房间底噪(若冒不出,
分类器再准也没用,早点止损)。

用法:
  录正样本(录的时候你来回走动 / 走向书桌):
      python3 tools/footstep_record.py footstep --count 40
  录负样本(分多批:安静、打字、说话、关门、空调……越杂越好):
      python3 tools/footstep_record.py other --count 40

样本存到仓库根的 footstep_data/<label>/(已 gitignore,音频不入库)。
追加录制:再次运行同一 label 会接着已有编号往后存。
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

DATADIR = Path(__file__).resolve().parent.parent / "footstep_data"


def rms_peak(wav: Path) -> tuple[float, float]:
    """用 sox stat 读 RMS / 峰值幅度(0~1)。"""
    out = subprocess.run(
        ["sox", str(wav), "-n", "stat"], capture_output=True, text=True
    ).stderr
    def g(key: str) -> float:
        m = re.search(re.escape(key) + r"\s*:\s*([-0-9.]+)", out)
        return float(m.group(1)) if m else 0.0
    return g("RMS     amplitude"), g("Maximum amplitude")


def main() -> int:
    ap = argparse.ArgumentParser(description="录脚步声训练样本")
    ap.add_argument("label", help="标签目录名,如 footstep / other")
    ap.add_argument("--count", type=int, default=30, help="录多少段")
    ap.add_argument("--seconds", type=float, default=1.5, help="每段时长(秒)")
    ap.add_argument("--rate", type=int, default=16000, help="采样率")
    a = ap.parse_args()

    if not subprocess.run(["which", "sox"], capture_output=True).stdout.strip():
        print("找不到 sox,请先 brew install sox", file=sys.stderr)
        return 1

    d = DATADIR / a.label
    d.mkdir(parents=True, exist_ok=True)
    start = len(list(d.glob("*.wav")))
    print(f"→ 录 {a.count} 段({a.seconds}s/段)到 {d}(已有 {start} 段)")
    print(f"  label='{a.label}':", "走动/走向书桌" if a.label == "footstep" else "请制造对应的负样本声音")
    print("  3 秒后开始……"); time.sleep(3)

    rms_vals = []
    for i in range(a.count):
        f = d / f"{a.label}_{start + i:04d}.wav"
        print(f"  [{i + 1}/{a.count}] 录…", end=" ", flush=True)
        subprocess.run(
            ["sox", "-d", "-r", str(a.rate), "-c", "1", str(f), "trim", "0", str(a.seconds)],
            stderr=subprocess.DEVNULL,
        )
        r, p = rms_peak(f)
        rms_vals.append(r)
        print(f"RMS={r:.5f}  peak={p:.5f}")
        time.sleep(0.3)

    if rms_vals:
        avg = sum(rms_vals) / len(rms_vals)
        print(f"\n完成。{a.label} 平均 RMS={avg:.5f}(房间底噪约 0.0004 做参照)。")
        if a.label == "footstep" and avg < 0.0010:
            print("⚠️ 脚步样本平均 RMS 很接近底噪——麦克风可能采不到清晰脚步声,"
                  "分类器大概率不可靠。建议先看这个数,再决定要不要继续训模型。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
