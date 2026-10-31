"""
旁白 + 配乐混音（配乐自动闪避：有人声时再压低）
输入 narration.wav、music.wav -> 输出 mix.wav，并导出试听用 mix_preview.mp3
"""
import subprocess

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import maximum_filter1d, uniform_filter1d

SR = 44100
MUSIC_GAIN_DB = -14.0     # 配乐整体相对旁白峰值
DUCK_DB = -7.0            # 有人声时再降多少

_, v = wavfile.read("narration.wav")
_, m = wavfile.read("music.wav")
v = v.astype(np.float32) / 32768
m = m.astype(np.float32) / 32768
n = min(len(v), len(m))
v, m = v[:n], m[:n]

# 人声包络：保持 0.4 秒再平滑，避免句中小停顿时配乐忽大忽小
env = np.abs(v) > 0.02
env = maximum_filter1d(env.astype(np.float32), int(0.4 * SR))
env = uniform_filter1d(env, int(0.3 * SR))
gain = 10 ** ((MUSIC_GAIN_DB + DUCK_DB * env) / 20)

out = v * 0.95 + m * gain
out *= 0.95 / np.abs(out).max()
wavfile.write("mix.wav", SR, (out * 32767).astype(np.int16))
subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", "mix.wav", "-b:a", "160k",
                "mix_preview.mp3"], check=True)

speech = env > 0.5
def db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9)
print(f"人声 RMS {db(v[speech]):.1f} dB，人声段配乐 {db((m * gain)[speech]):.1f} dB，"
      f"间隙配乐 {db((m * gain)[~speech]):.1f} dB")
