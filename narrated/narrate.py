"""
逐句合成旁白（微软 Edge 神经语音 · 云健），生成：
  narration.wav   128 秒旁白轨（44.1kHz 单声道）
  timeline.json   每句字幕的起止时间 + 每个场景的起止时间

运行：python3 narrate.py            （需要联网；已合成的句子会缓存到 .tts/）
     python3 narrate.py --rate +5%  调语速
"""
import asyncio
import hashlib
import json
import os
import ssl
import subprocess
import sys

import numpy as np
from scipy.io import wavfile

import edge_tts.communicate as ec
from script import SCENES

# 沙箱里出网要走代理，edge-tts 默认只信任 certifi 证书，这里换成系统 CA 包
CA = "/root/.ccr/ca-bundle.crt"
if os.path.exists(CA):
    ec._SSL_CTX = ssl.create_default_context(cafile=CA)

VOICE = "zh-CN-YunjianNeural"
RATE = sys.argv[sys.argv.index("--rate") + 1] if "--rate" in sys.argv else "+0%"
SR = 44100
LENGTH = 128.0
LEAD_IN = 3.0        # 开头先让标题和音乐走 3 秒
TAIL = 4.0           # 结尾留 4 秒给音乐收尾
GAP_LINE = 0.5      # 同一场景内句间停顿
GAP_SCENE = 1.1      # 场景之间停顿（画面转场）
CACHE = ".tts"


async def synth(text, path):
    await ec.Communicate(text, VOICE, rate=RATE).save(path)


def load(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "s16le",
                          "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    # 去掉首尾静音，时间轴更准
    idx = np.where(np.abs(x) > 0.01)[0]
    return x[max(idx[0] - 400, 0): idx[-1] + 2000]


def main():
    os.makedirs(CACHE, exist_ok=True)
    clips = []
    for si, (sid, lines) in enumerate(SCENES):
        for li, (zh, en) in enumerate(lines):
            key = hashlib.md5(f"{VOICE}|{RATE}|{zh}".encode()).hexdigest()[:12]
            p = f"{CACHE}/{key}.mp3"
            if not os.path.exists(p) or os.path.getsize(p) == 0:
                asyncio.run(synth(zh, p))
            clips.append((si, li, zh, en, load(p)))

    speech = sum(len(c[4]) for c in clips) / SR
    n_scene_gaps = len(SCENES) - 1
    n_line_gaps = len(clips) - len(SCENES)
    budget = LENGTH - LEAD_IN - TAIL
    fixed = speech + n_line_gaps * GAP_LINE + n_scene_gaps * GAP_SCENE
    print(f"语音 {speech:.1f}s，含停顿 {fixed:.1f}s，可用 {budget:.1f}s")
    if fixed > budget:
        sys.exit("超时：请加快语速（--rate +5%）或删减文案")
    # 富余时间平均分给场景间停顿
    gap_scene = GAP_SCENE + (budget - fixed) / n_scene_gaps

    out = np.zeros(int(LENGTH * SR), np.float32)
    subs, scenes = [], []
    t = LEAD_IN
    for k, (si, li, zh, en, x) in enumerate(clips):
        if k > 0:
            t += gap_scene if li == 0 else GAP_LINE
        if li == 0:
            scenes.append({"id": SCENES[si][0], "voice_start": round(t, 3)})
        i = int(t * SR)
        out[i:i + len(x)] += x
        subs.append({"scene": si, "start": round(t, 3),
                     "end": round(t + len(x) / SR, 3), "zh": zh, "en": en})
        t += len(x) / SR
    # 场景切换点：落在两段旁白中间
    for k, s in enumerate(scenes):
        s["start"] = 0.0 if k == 0 else round(
            (subs_end(subs, k - 1) + s["voice_start"]) / 2, 3)
    for k, s in enumerate(scenes):
        s["end"] = scenes[k + 1]["start"] if k + 1 < len(scenes) else LENGTH

    out /= max(np.abs(out).max(), 1e-6) / 0.9
    wavfile.write("narration.wav", SR, (out * 32767).astype(np.int16))
    with open("timeline.json", "w") as f:
        json.dump({"length": LENGTH, "scenes": scenes, "subs": subs}, f,
                  ensure_ascii=False, indent=1)
    print(f"最后一句结束于 {subs[-1]['end']:.1f}s，场景间停顿 {gap_scene:.2f}s")
    for s in scenes:
        print(f"  {s['id']:<11} {s['start']:6.1f} – {s['end']:6.1f}")


def subs_end(subs, scene_idx):
    return max(s["end"] for s in subs if s["scene"] == scene_idx)


if __name__ == "__main__":
    main()
