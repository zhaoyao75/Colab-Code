"""
混音：narration.wav + bgm.wav -> mix.wav（旁白说话时音乐自动压低）
旁白段音乐比人声低约 15dB，空隙处低约 8dB；最后整体响度标准化到 -14 LUFS（抖音/B站常用）。
"""
import json, subprocess
import numpy as np
from scipy.io import wavfile
from scipy.ndimage import maximum_filter1d, uniform_filter1d

SR = 44100
_, v = wavfile.read("narration.wav")
_, m = wavfile.read("bgm.wav")
v = v.astype(np.float64) / 32768
m = m.astype(np.float64).T / 32768
n = min(len(v), m.shape[1])
v, m = v[:n], m[:, :n]


def rms_db(x):
    return 20 * np.log10(np.sqrt((x ** 2).mean()) + 1e-9)


env = uniform_filter1d(np.abs(v), int(0.03 * SR))
active = env > 0.02
active = maximum_filter1d(active.astype(float), int(0.35 * SR))       # 句中短停顿不回弹
v_db = rms_db(v[env > 0.02])
m_db = rms_db(m)
g_voice = 10 ** ((v_db - 15 - m_db) / 20)
g_gap = 10 ** ((v_db - 8 - m_db) / 20)
g = g_gap + (g_voice - g_gap) * active
g = uniform_filter1d(g, int(0.25 * SR))                              # 平滑进出
out = m * g + v[None, :] * 1.0
out /= np.abs(out).max() / 0.9
wavfile.write(".mix_raw.wav", SR, (out.T * 32767).astype(np.int16))

# 两遍 loudnorm
r = subprocess.run(["ffmpeg", "-hide_banner", "-i", ".mix_raw.wav", "-af",
                    "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json", "-f", "null", "-"],
                   capture_output=True, text=True).stderr
js = json.loads(r[r.rindex("{"):r.rindex("}") + 1])
af = ("loudnorm=I=-14:TP=-1.5:LRA=11:linear=true:measured_I={input_i}:measured_TP={input_tp}:"
      "measured_LRA={input_lra}:measured_thresh={input_thresh}:offset={target_offset}").format(**js)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", ".mix_raw.wav", "-af", af, "-ar", str(SR), "mix.wav"], check=True)
print(f"人声 {v_db:.1f} dBFS | 音乐增益 说话时 {20*np.log10(g_voice):.1f} dB / 空隙 {20*np.log10(g_gap):.1f} dB | 输入响度 {js['input_i']} LUFS")
