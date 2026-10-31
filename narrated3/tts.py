"""
旁白生成：script.json -> narration.wav + timings.json + 字幕(.srt)
声音：微软 edge-tts「云健」(zh-CN-YunjianNeural)，正常语速。需联网。
时间轴：8 段 × 16 秒（对齐 120 BPM 配乐的 8 小节）；每段旁白从段首 +1.0s 开始，句间停 0.8s。
运行：python3 tts.py
"""
import asyncio, json, os, ssl, subprocess
import numpy as np
from scipy.io import wavfile
import edge_tts.communicate as comm

# 沙箱里走代理，代理 CA 不在 certifi 里：替换 edge-tts 的 SSL 上下文
CA = "/root/.ccr/ca-bundle.crt"
if os.path.exists(CA):
    comm._SSL_CTX = ssl.create_default_context(cafile=CA)

VOICE, RATE = "zh-CN-YunjianNeural", "+0%"
SR = 44100
SCENE = 16.0
LENGTH = 128.0
CACHE = ".tts_cache"
S = json.load(open("script.json", encoding="utf-8"))
os.makedirs(CACHE, exist_ok=True)


async def synth(text, fn):
    await comm.Communicate(text, VOICE, rate=RATE,
                           proxy=os.environ.get("HTTPS_PROXY")).save(fn)


def load_trim(mp3):
    """解码成 44.1k 单声道并裁掉首尾静音"""
    raw = subprocess.check_output(["ffmpeg", "-loglevel", "error", "-i", mp3, "-ac", "1",
                                   "-ar", str(SR), "-f", "f32le", "-"])
    a = np.frombuffer(raw, np.float32).astype(np.float64)
    env = np.convolve(np.abs(a), np.ones(441) / 441, "same")
    idx = np.where(env > 0.01)[0]
    i, j = max(0, idx[0] - 1300), min(len(a), idx[-1] + 2600)
    return a[i:j]


clips = []
for k, sc in enumerate(S["scenes"]):
    for j, (zh, en) in enumerate(sc["lines"]):
        fn = f"{CACHE}/{k}_{j}.mp3"
        meta = fn + ".txt"
        if not (os.path.exists(fn) and open(meta, encoding="utf-8").read() == VOICE + RATE + zh):
            asyncio.run(synth(zh, fn))
            open(meta, "w", encoding="utf-8").write(VOICE + RATE + zh)
        clips.append((k, j, zh, en, load_trim(fn)))

# 排时间轴
out = np.zeros(int(LENGTH * SR))
cues, t = [], 0.0
for k, j, zh, en, a in clips:
    if j == 0:
        start = 2.4 if k == 0 else k * SCENE + 1.0
        if k == len(S["scenes"]) - 1:      # 结尾三句较长，提前开始，给片尾留时间
            start = k * SCENE - 0.4
        t = max(start, t + 0.6)
    else:
        t += 0.8
    d = len(a) / SR
    i = int(t * SR)
    out[i:i + len(a)] += a
    cues.append({"scene": k, "start": round(t, 3), "end": round(t + d, 3), "zh": zh, "en": en})
    t += d
assert t < LENGTH - 2, f"旁白超长：结束于 {t:.2f}s"

out = out / np.abs(out).max() * 0.89
wavfile.write("narration.wav", SR, (out * 32767).astype(np.int16))
json.dump(cues, open("timings.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def ts(x):
    ms = int(round(x * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


for name, f in [("zh", lambda c: c["zh"]), ("en", lambda c: c["en"]),
                ("zh_en", lambda c: c["zh"] + "\n" + c["en"])]:
    with open(f"subtitles_{name}.srt", "w", encoding="utf-8") as fh:
        for n, c in enumerate(cues, 1):
            fh.write(f"{n}\n{ts(c['start'])} --> {ts(c['end'] + 0.3)}\n{f(c)}\n\n")

for c in cues:
    print(f"{c['start']:7.2f} – {c['end']:7.2f}  {c['zh']}")
print(f"旁白结束于 {t:.2f}s")
