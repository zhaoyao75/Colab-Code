"""
《郑州：前世今生》背景音乐 —— 128 秒原创古风配乐（淡淡地垫在旁白下面）

D 羽调五声音阶（D F G A C），60 BPM，4/4，32 小节 = 128 秒，每 8 小节一段。
乐器（纯 numpy 合成）：古琴（Karplus-Strong 拨弦）、古筝（明亮拨弦）、箫（气声 + 颤音）、
编钟（非谐泛音）、低音铺底、堂鼓。编钟落在 timeline.json 的场景切换点上。

运行：python3 compose_history.py   ->  music.wav
"""
import json
import os

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, fftconvolve, lfilter, sosfilt

SR = 44100
LENGTH = 128.0
BEAT = 1.0
BAR = 4 * BEAT
N = int((LENGTH + 4) * SR)
rng = np.random.default_rng(1954)

NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def hz(name):
    midi = NOTE[name[0]] + 12 * (int(name[-1]) + 1)
    return 440.0 * 2 ** ((midi - 69) / 12)


def lowpass(x, fc, order=2):
    return sosfilt(butter(order, fc, fs=SR, output="sos"), x)


# ---------------------------------------------------------------- 乐器
def pluck(f, dur, bright=0.5, decay=0.996):
    """Karplus-Strong 拨弦：用 lfilter 实现梳状反馈，速度快。"""
    n = int(dur * SR)
    L = int(SR / f)
    burst = rng.uniform(-1, 1, L)
    burst = lowpass(burst, 600 + 6000 * bright)
    x = np.zeros(n)
    x[:L] = burst
    a = np.zeros(L + 2)
    a[0] = 1
    a[L] = a[L + 1] = -decay / 2
    y = lfilter([1.0], a, x)
    y *= np.clip(np.arange(n) / (0.003 * SR), 0, 1)
    return y / (np.abs(y).max() + 1e-9)


def guqin(f, dur):
    y = pluck(f, dur + 1.5, bright=0.25, decay=0.998)
    # 琴体共鸣：叠一点低八度正弦，温润一些
    t = np.arange(len(y)) / SR
    y += 0.25 * np.sin(2 * np.pi * f * t) * np.exp(-t / 1.2)
    return lowpass(y, 2500) * 0.9


def guzheng(f, dur):
    return pluck(f, dur + 1.0, bright=0.7, decay=0.995) * 0.6


def xiao(f, dur, att=0.25, rel=0.6):
    n = int((dur + rel) * SR)
    t = np.arange(n) / SR
    vib = 0.005 * np.sin(2 * np.pi * 4.5 * t) * np.clip((t - 0.6) / 0.8, 0, 1)
    ph = 2 * np.pi * f * np.cumsum(1 + vib) / SR
    sig = np.sin(ph) + 0.12 * np.sin(2 * ph) + 0.03 * np.sin(3 * ph)
    breath = lowpass(rng.standard_normal(n), 3000) * 0.12
    sig += breath * (0.5 + 0.5 * np.sin(ph))
    env = np.clip(t / att, 0, 1)
    ri = int(dur * SR)
    env[ri:] *= np.exp(-(t[ri:] - dur) / (rel / 4))
    return sig * env * 0.5


def bell(f, dur=6.0):
    """编钟：非谐泛音 + 不同衰减速度"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for ratio, amp, tau in [(1, 1.0, 2.5), (2.76, 0.5, 1.2), (5.4, 0.25, 0.6),
                            (8.93, 0.12, 0.3), (0.5, 0.3, 3.0)]:
        sig += amp * np.sin(2 * np.pi * f * ratio * t) * np.exp(-t / tau)
    sig *= np.clip(t / 0.002, 0, 1)
    return sig * 0.35


def drone(f, dur, att=4.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for det in (0.997, 1.0, 1.003):
        ph = (f * det * t + rng.random()) % 1
        sig += 2 * ph - 1
    sig = lowpass(sig, 500, 4)
    env = np.clip(t / att, 0, 1) * np.clip((dur - t) / att, 0, 1)
    return sig * env * 0.15


def drum(dur=1.5):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 70 + 50 * np.exp(-t / 0.05)
    sig = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.35)
    sig += lowpass(rng.standard_normal(n), 800) * np.exp(-t / 0.03) * 0.3
    return sig * 0.7


# ---------------------------------------------------------------- 旋律（原创）
# (音名, 拍数)；None 为休止
PHRASE_A = [("A4", 2), ("G4", 1), ("A4", 1), ("D5", 3), ("C5", 1),
            ("A4", 2), ("G4", 1), ("F4", 1), ("G4", 4),
            ("F4", 2), ("G4", 1), ("A4", 1), ("C5", 2), ("A4", 2),
            ("G4", 1), ("F4", 1), ("D4", 2), ("D4", 4)]
PHRASE_B = [("D5", 2), ("C5", 1), ("D5", 1), ("A4", 3), ("G4", 1),
            ("C5", 2), ("D5", 2), ("A4", 4),
            ("G4", 2), ("A4", 1), ("G4", 1), ("F4", 2), ("D4", 2),
            ("F4", 1), ("G4", 1), ("A4", 2), ("D4", 4)]


def octave(name, k):
    return name[:-1] + str(int(name[-1]) + k)


tracks = {k: np.zeros(N) for k in ("qin", "zheng", "xiao", "bell", "drone", "drum")}


def put(track, sig, t, gain=1.0):
    i = int(t * SR)
    j = min(i + len(sig), N)
    tracks[track][i:j] += sig[: j - i] * gain


def play(phrase, start, inst, track, k=0, gain=1.0, legato=1.0):
    t = start
    for name, beats in phrase:
        if name:
            put(track, inst(hz(octave(name, k)), beats * BEAT * legato), t, gain)
        t += beats * BEAT


S = 8 * BAR  # 一段 32 秒

# 低音铺底：D2 + A2 贯穿全曲，每段换一次气
for sec in range(4):
    put("drone", drone(hz("D2"), S + 2), sec * S - (1 if sec else 0))
    put("drone", drone(hz("A2"), S + 2), sec * S - (1 if sec else 0), 0.6)

# 第 1 段：古琴低八度独奏 A，非常稀疏
play(PHRASE_A, 0.0, guqin, "qin", k=-1, gain=0.8)
# 第 2 段：古琴 B + 箫长音对位
play(PHRASE_B, S, guqin, "qin", k=-1, gain=0.8)
play([(None, 4), ("A4", 8), ("G4", 8), ("F4", 8), ("D4", 4)], S, xiao, "xiao", k=-1, gain=0.7)
# 第 3 段：箫吹 A，古筝分解和弦轻轻点缀
play(PHRASE_A, 2 * S, xiao, "xiao", k=-1, gain=0.9, legato=0.95)
for bar in range(8):
    root = ["D3", "F3", "G3", "A3", "D3", "C3", "G3", "D3"][bar]
    for i, name in enumerate([root, "A3", "D4", "A3"]):
        put("zheng", guzheng(hz(name), 1.5), 2 * S + bar * BAR + i * 1.0, 0.35)
# 第 4 段：箫 B + 古琴 A 交织，最后两小节收住
play(PHRASE_B, 3 * S, xiao, "xiao", k=-1, gain=0.9, legato=0.95)
play(PHRASE_A[:12], 3 * S, guqin, "qin", k=-1, gain=0.6)
put("qin", guqin(hz("D3"), 6), LENGTH - 6, 0.9)

# 堂鼓：第 3、4 段每两小节一下
for bar in range(16, 31, 2):
    put("drum", drum(), bar * BAR, 0.5)

# 编钟：开头、每个场景切换点、结尾
bell_times = [0.0]
if os.path.exists("timeline.json"):
    tl = json.load(open("timeline.json"))
    bell_times += [s["start"] for s in tl["scenes"][1:]]
for k, bt in enumerate(bell_times):
    put("bell", bell(hz(["D4", "A3"][k % 2])), bt, 0.8)
put("bell", bell(hz("D3"), 8), LENGTH - 7, 0.9)


# ---------------------------------------------------------------- 混音
def reverb(x, secs=3.0, wet=0.35):
    n = int(secs * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-t / (secs / 6))
    ir = lowpass(ir, 5000)
    ir /= np.sqrt((ir ** 2).sum())
    return (1 - wet) * x + wet * fftconvolve(x, ir)[: len(x)]


levels = {"qin": 0.9, "zheng": 1.4, "xiao": 0.55, "bell": 0.8, "drone": 0.45, "drum": 1.0}
sends = {"qin": 0.35, "zheng": 0.4, "xiao": 0.45, "bell": 0.5, "drone": 0.2, "drum": 0.3}
dry = sum(tracks[k] * levels[k] for k in tracks)
wet_in = sum(tracks[k] * levels[k] * sends[k] for k in tracks)
mix = dry + reverb(wet_in, wet=1.0) * 0.8
mix = mix[: int(LENGTH * SR)]
# 首尾淡入淡出
t = np.arange(len(mix)) / SR
mix *= np.clip(t / 1.0, 0, 1) * np.clip((LENGTH - t) / 3.0, 0, 1)
mix = np.tanh(mix / (np.abs(mix).max() + 1e-9) * 1.2)
mix *= 0.89 / np.abs(mix).max()

wavfile.write("music.wav", SR, (mix * 32767).astype(np.int16))
rms = [np.sqrt(np.mean(mix[i * 8 * SR:(i + 1) * 8 * SR] ** 2)) for i in range(16)]
print("music.wav", len(mix) / SR, "s  每 8 秒 RMS:", " ".join(f"{r:.2f}" for r in rms))
