"""
童趣背景音乐：128 秒，120 BPM，4/4，64 小节，C 大调，旋律原创。
8 段 × 8 小节，每段对应一个画面场景；配器轻、音区偏高，给旁白让出中频。
运行：python3 music.py  ->  bgm.wav
"""
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
from scipy.io import wavfile

SR = 44100
BEAT, BAR, BARS, LENGTH = 0.5, 2.0, 64, 128.0
N = int((LENGTH + 4) * SR)
rng = np.random.default_rng(11)
NOTE = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7,
        "G#": 8, "A": 9, "A#": 10, "B": 11}


def hz(name):
    midi = NOTE[name[:-1]] + 12 * (int(name[-1]) + 1)
    return 440.0 * 2 ** ((midi - 69) / 12)


def T(d):
    return np.arange(int(d * SR)) / SR


# ---------------------------------------------------------------- 乐器
def celesta(f, d=0):
    t = T(2.0)
    s = (np.sin(2 * np.pi * f * t) * np.exp(-t / 0.8)
         + 0.25 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.35)
         + 0.3 * np.sin(2 * np.pi * 4.2 * f * t) * np.exp(-t / 0.07))
    return s * np.clip(t / 0.002, 0, 1) / 1.5


def glock(f, d=0):
    t = T(2.5)
    s = (np.sin(2 * np.pi * f * t) * np.exp(-t / 1.0)
         + 0.4 * np.sin(2 * np.pi * 2.76 * f * t) * np.exp(-t / 0.3)
         + 0.2 * np.sin(2 * np.pi * 5.4 * f * t) * np.exp(-t / 0.1))
    return s * np.clip(t / 0.001, 0, 1) / 1.6


def marimba(f, d=0):
    t = T(1.2)
    s = (np.sin(2 * np.pi * f * t) * np.exp(-t / 0.35)
         + 0.35 * np.sin(2 * np.pi * 3.93 * f * t) * np.exp(-t / 0.06)
         + 0.15 * np.sin(2 * np.pi * 9.2 * f * t) * np.exp(-t / 0.015))
    return s * np.clip(t / 0.003, 0, 1)


def pluck(f, d=0):
    """拨弦（尤克里里/拨奏弦乐感）：衰减泛音，高次泛音衰减更快"""
    t = T(1.0)
    s = np.zeros_like(t)
    for h in range(1, 9):
        if f * h > 12000:
            break
        s += np.sin(2 * np.pi * f * h * t + rng.random()) / h ** 1.1 * np.exp(-t * (3 + 2.5 * h))
    return s * np.clip(t / 0.002, 0, 1) / 1.3


def toy_piano(f, d=0):
    t = T(1.5)
    s = (np.sin(2 * np.pi * f * t) * np.exp(-t / 0.5)
         + 0.5 * np.sin(2 * np.pi * 3.01 * f * t) * np.exp(-t / 0.2)
         + 0.3 * np.sin(2 * np.pi * 6.3 * f * t) * np.exp(-t / 0.05))
    return s * np.clip(t / 0.001, 0, 1) / 1.5


def flute(f, d):
    t = T(d + 0.25)
    vib = 0.004 * np.sin(2 * np.pi * 5 * t) * np.clip((t - 0.2) / 0.3, 0, 1)
    ph = 2 * np.pi * f * np.cumsum(1 + vib) / SR
    s = np.sin(ph) + 0.25 * np.sin(2 * ph) + 0.015 * rng.standard_normal(len(t))
    env = np.clip(t / 0.06, 0, 1) * np.clip((d + 0.25 - t) / 0.25, 0, 1)
    return s * env / 1.3


def woodblock(hi=True):
    t = T(0.08)
    f = 1250 if hi else 880
    return np.sin(2 * np.pi * f * t) * np.exp(-t / 0.012) + 0.3 * rng.standard_normal(len(t)) * np.exp(-t / 0.002)


def shaker():
    n = rng.standard_normal(int(0.06 * SR))
    n = sosfilt(butter(2, 5000 / (SR / 2), "high", output="sos"), n)
    t = np.arange(len(n)) / SR
    return n * np.exp(-t / 0.015) * np.clip(t / 0.005, 0, 1)


# ---------------------------------------------------------------- 混音
tracks = {}
SEND = {"mel": 0.35, "pluck": 0.2, "bass": 0.05, "perc": 0.12, "pad": 0.4}


def add(tr, start, s, gain=1.0, pan=0.0):
    tracks.setdefault(tr, np.zeros((2, N)))
    i = int(start * SR)
    j = min(N, i + len(s))
    s = s[: j - i] * gain
    a = (pan + 1) * np.pi / 4
    tracks[tr][0, i:j] += s * np.cos(a)
    tracks[tr][1, i:j] += s * np.sin(a)


# 和声：每小节一个和弦 (低音, 和弦音)
CH = {"C": ("C3", ["E4", "G4", "C5"]), "G": ("G2", ["D4", "G4", "B4"]),
      "F": ("F2", ["F4", "A4", "C5"]), "Am": ("A2", ["E4", "A4", "C5"]),
      "Dm": ("D3", ["D4", "F4", "A4"]), "Em": ("E3", ["E4", "G4", "B4"])}
PROG8 = ["C", "G", "F", "C", "C", "Am", "G", "C"]
PROG_B = ["F", "G", "Em", "Am", "Dm", "G", "C", "C"]      # 第 6 段（荷花）换个色彩

# 原创主题：蹦蹦跳跳的八分音符 (音名/None, 拍)
THEME = [
    [("E5", .5), ("G5", .5), ("C6", 1), ("G5", .5), ("E5", .5), ("C5", 1)],
    [("D5", .5), ("E5", .5), ("F5", .5), ("E5", .5), ("D5", 2)],
    [("F5", .5), ("A5", .5), ("C6", 1), ("A5", .5), ("F5", .5), ("D5", 1)],
    [("E5", 1), ("G5", 1), ("C5", 2)],
    [("E5", .5), ("G5", .5), ("C6", .5), ("D6", .5), ("E6", 1), ("C6", 1)],
    [("A5", .5), ("G5", .5), ("F5", .5), ("E5", .5), ("D5", 1), ("E5", 1)],
    [("F5", .5), ("E5", .5), ("D5", .5), ("F5", .5), ("E5", .5), ("D5", .5), ("B4", 1)],
    [("C5", 1), (None, .5), ("G4", .5), ("C5", 2)],
]
THEME_B = [
    [("A5", 1), ("C6", 1), ("A5", 2)],
    [("B5", 1), ("G5", 1), ("D5", 2)],
    [("G5", 1), ("B5", 1), ("E5", 2)],
    [("A5", 1), ("E5", 1), ("C5", 2)],
    [("D5", 1), ("F5", 1), ("A5", 2)],
    [("G5", 1), ("B5", .5), ("A5", .5), ("G5", 2)],
    [("E5", 1), ("G5", 1), ("C6", 2)],
    [("C6", 3), (None, 1)],
]

# 每段：(旋律乐器, 旋律音量, 八度移位, 伴奏强度, 打击乐)
SECTIONS = [
    (celesta, 0.55, 0, 0.7, None),        # 0 凤凰：清亮开场
    (marimba, 0.6, 0, 0.8, "wood"),       # 1 木刻：木质敲击感
    (celesta, 0.5, 0, 0.8, "shaker"),     # 2 北京
    (flute, 0.35, 0, 0.5, None),          # 3 猫头鹰：夜里轻一点
    (glock, 0.5, 0, 1.0, "wood"),         # 4 猴票：最活泼
    (marimba, 0.55, 0, 0.8, "shaker"),    # 5 荷花（换和声）
    (toy_piano, 0.5, 0, 1.0, "wood"),     # 6 老顽童：玩具钢琴
    (celesta, 0.5, 0, 0.6, None),         # 7 结尾：渐弱
]

for sec, (inst, mg, octv, acc, perc) in enumerate(SECTIONS):
    prog = PROG_B if sec == 5 else PROG8
    theme = THEME_B if sec == 5 else THEME
    for b in range(8):
        t0 = (sec * 8 + b) * BAR
        fade = 1.0
        if sec == 7:
            fade = 1 - 0.55 * b / 7                    # 结尾逐渐变淡
        bass_n, chord = CH[prog[b]]
        # 低音：马林巴，第 1、3 拍
        for beat in (0, 2):
            if sec == 3 and beat == 2:
                continue
            add("bass", t0 + beat * BEAT, marimba(hz(bass_n)), 0.55 * acc * fade)
        # 拨弦和弦：第 2、4 拍反拍（oom-pah）
        for beat in (1, 3):
            for k, n in enumerate(chord):
                add("pluck", t0 + beat * BEAT + k * 0.012, pluck(hz(n)), 0.18 * acc * fade, pan=-0.3 + 0.3 * k)
        # 打击乐
        if perc == "wood":
            for beat in range(4):
                add("perc", t0 + beat * BEAT, woodblock(beat % 2 == 0), 0.12 * fade, pan=0.4)
        elif perc == "shaker":
            for e in range(8):
                add("perc", t0 + e * BEAT / 2, shaker(), (0.10 if e % 2 else 0.06) * fade, pan=-0.4)
        # 旋律
        pos = 0.0
        for n, d in theme[b]:
            if n:
                f = hz(n) * 2 ** octv
                if inst is flute:
                    add("mel", t0 + pos * BEAT, flute(f / 2, d * BEAT * 0.95), mg * fade, pan=0.15)
                else:
                    add("mel", t0 + pos * BEAT, inst(f), mg * fade, pan=0.15)
                    # 第 4、7 段加一层高八度的钢片琴点缀
                    if sec in (4, 6) and d >= 1:
                        add("mel", t0 + pos * BEAT, celesta(f * 2), 0.15 * fade, pan=-0.2)
            pos += d
        # 段首铺一层很轻的长音（长笛低八度）作为过渡
        if b == 0 and sec in (0, 3, 7):
            add("pad", t0, flute(hz(chord[0]) / 2, BAR * 2), 0.18 * fade)

# 结尾最后一个和弦：钢片琴琶音 + 长音
end = 126.0
for k, n in enumerate(["C5", "E5", "G5", "C6", "E6"]):
    add("mel", end - 1.5 + k * 0.12, celesta(hz(n)), 0.35, pan=-0.3 + 0.15 * k)

# 混响 + 总线
rt = T(2.2)
ir = rng.standard_normal((2, len(rt))) * np.exp(-rt / 0.45)
ir = sosfilt(butter(1, 6000 / (SR / 2), output="sos"), ir)
ir /= np.sqrt((ir ** 2).sum(axis=1, keepdims=True))
dry = sum(tracks.values())
send = sum(tracks[k] * SEND[k] for k in tracks)
wet = np.stack([fftconvolve(send[c], ir[c])[:N] for c in range(2)])
mix = dry + 0.6 * wet
# 去掉 120Hz 以下的轰鸣、轻微压低 1–3kHz 给人声让位
mix = sosfilt(butter(2, 120 / (SR / 2), "high", output="sos"), mix)
mix = mix[:, : int(LENGTH * SR)]
fo = int(1.5 * SR)
mix[:, -fo:] *= np.linspace(1, 0, fo) ** 2
mix = np.tanh(mix / np.abs(mix).max() * 1.2)
mix = mix / np.abs(mix).max() * 0.85
wavfile.write("bgm.wav", SR, (mix.T * 32767).astype(np.int16))

# 自检：每 8 秒 RMS
rms = [np.sqrt((mix[:, int(i * SR):int((i + 8) * SR)] ** 2).mean()) for i in range(0, 128, 8)]
print("RMS/8s:", " ".join(f"{r:.3f}" for r in rms), f"| 时长 {mix.shape[1] / SR:.2f}s")
