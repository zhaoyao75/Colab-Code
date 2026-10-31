"""混音：旁白 + 背景音乐。旁白说话时音乐自动压低（按 timeline.json 的句子时间做闪避）。

运行：python3 mix.py   ->  .cache/final_audio.wav、audio_preview.mp3
"""
import json, os, subprocess
import numpy as np
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 44100
MUSIC_GAIN = 0.24      # 无旁白时的音乐音量
DUCK = 0.38            # 旁白时再乘上这个系数（约 -8.4 dB）

tl = json.load(open(os.path.join(HERE, "timeline.json"), encoding="utf-8"))
_, v = wavfile.read(os.path.join(HERE, ".cache", "narration.wav"))
_, m = wavfile.read(os.path.join(HERE, ".cache", "music.wav"))
v = v.astype(np.float32)
v = v / 32768 if np.abs(v).max() > 2 else v
m = m.astype(np.float32) / 32768
n = int(tl["total"] * SR)
v, m = np.pad(v, (0, max(0, n - len(v))))[:n], np.pad(m, ((0, max(0, n - len(m))), (0, 0)))[:n]

t = np.arange(n) / SR
talk = np.zeros(n, np.float32)
for ln in tl["lines"]:
    talk[(t > ln["t0"] - 0.25) & (t < ln["t1"] + 0.45)] = 1
# 平滑：约 0.3 秒过渡
k = int(0.3 * SR)
talk = np.convolve(talk, np.ones(k) / k, "same")
g = MUSIC_GAIN * (1 - (1 - DUCK) * talk)

out = m * g[:, None] + v[:, None] * 0.92
out = np.tanh(out * 1.1) / 1.1
out *= 0.95 / np.abs(out).max()

wav = os.path.join(HERE, ".cache", "final_audio.wav")
wavfile.write(wav, SR, (out * 32767).astype(np.int16))
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", wav, "-b:a", "160k",
                os.path.join(HERE, "audio_preview.mp3")], check=True)

on, off = talk > 0.9, talk < 0.1
db = lambda x: 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-9)
print(f"旁白段：人声 {db(v[on]):.1f} dB，音乐 {db(m[on] * g[on, None]):.1f} dB；"
      f"空隙段音乐 {db(m[off] * g[off, None]):.1f} dB")
