"""
《星际回响》(Stellar Echo) —— 128 秒原创配乐
风格参考：Hans Zimmer《星际穿越》配乐（管风琴、钟表滴答、重复琶音、层层推进）
主旋律为原创，未使用原片任何旋律。

120 BPM，4/4 拍，64 小节 = 128 秒。
运行：python3 compose.py          ->  小调版 stellar_echo.wav / .mp3
     python3 compose.py major    ->  大调欢快版 stellar_echo_major.wav / .mp3
     python3 compose.py lyrical  ->  慢速抒情版 stellar_echo_lyrical.wav / .mp3（60 BPM，32 小节）
"""
import subprocess
import sys
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve
from scipy.io import wavfile

SR = 44100
MODE = sys.argv[1] if len(sys.argv) > 1 else "minor"   # minor / major / lyrical
MAJOR = MODE == "major"
LYRICAL = MODE == "lyrical"
OUT = {"minor": "stellar_echo", "major": "stellar_echo_major",
       "lyrical": "stellar_echo_lyrical"}[MODE]
BPM = 60 if LYRICAL else 120
BEAT = 60 / BPM          # 0.5 s（抒情版 1 s）
BAR = 4 * BEAT           # 2 s（抒情版 4 s）
LENGTH = 128.0
BARS = int(LENGTH / BAR)  # 64（抒情版 32）
N = int(LENGTH * SR) + SR * 4
rng = np.random.default_rng(7)

NOTE = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7,
        "G#": 8, "A": 9, "A#": 10, "B": 11}


def m(name):
    """'A4' -> MIDI 编号"""
    return NOTE[name[:-1]] + 12 * (int(name[-1]) + 1)


def hz(midi):
    return 440.0 * 2 ** ((midi - 69) / 12)


# ---------------------------------------------------------------- 乐器
def organ(f, dur, att=0.06, rel=0.35, bright=1.0):
    n = int((dur + rel) * SR)
    t = np.arange(n) / SR
    harm = [(1, 1.0), (2, 0.55), (3, 0.3 * bright), (4, 0.25 * bright),
            (6, 0.12 * bright), (8, 0.08 * bright), (0.5, 0.35)]
    sig = np.zeros(n)
    for det in (0.9985, 1.0015):
        for h, a in harm:
            if f * h < 16000:
                sig += a * np.sin(2 * np.pi * f * h * det * t + rng.random() * 6.28)
    sig *= 1 + 0.03 * np.sin(2 * np.pi * 5.5 * t)
    env = np.clip(t / att, 0, 1)
    rel_i = int(dur * SR)
    env[rel_i:] *= np.exp(-(t[rel_i:] - dur) / (rel / 4))
    return sig * env / 4


def flute(f, dur, att=0.05, rel=0.3):
    n = int((dur + rel) * SR)
    t = np.arange(n) / SR
    vib = 0.004 * np.sin(2 * np.pi * 5 * t) * np.clip((t - 0.25) / 0.3, 0, 1)
    ph = 2 * np.pi * f * np.cumsum(1 + vib) / SR
    sig = np.sin(ph) + 0.35 * np.sin(2 * ph) + 0.08 * np.sin(3 * ph)
    sig += 0.02 * rng.standard_normal(n) * np.exp(-t / 0.08)
    env = np.clip(t / att, 0, 1)
    rel_i = int(dur * SR)
    env[rel_i:] *= np.exp(-(t[rel_i:] - dur) / (rel / 4))
    return sig * env / 1.4


def piano(f, dur, rel=0.4):
    n = int((dur + rel) * SR)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    B = 0.00015
    for h in range(1, 10):
        fh = f * h * np.sqrt(1 + B * h * h)
        if fh > 16000:
            break
        tau = 1.6 / (1 + 0.5 * h) * (330 / f) ** 0.35
        sig += (1 / h ** 1.3) * np.sin(2 * np.pi * fh * t) * np.exp(-t / tau)
    env = np.clip(t / 0.003, 0, 1)
    rel_i = int(dur * SR)
    env[rel_i:] *= np.exp(-(t[rel_i:] - dur) / 0.12)
    return sig * env / 1.8


def celesta(f, dur):
    n = int(2.2 * SR)
    t = np.arange(n) / SR
    sig = (np.sin(2 * np.pi * f * t) * np.exp(-t / 0.9)
           + 0.25 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.4)
           + 0.3 * np.sin(2 * np.pi * 4.2 * f * t) * np.exp(-t / 0.08))
    return sig * np.clip(t / 0.002, 0, 1) / 1.5


def strings(f, dur, att=0.45, rel=0.6):
    n = int((dur + rel) * SR)
    t = np.arange(n) / SR
    sig = np.zeros(n)
    for det in (0.997, 1.0, 1.003):
        vib = 0.003 * np.sin(2 * np.pi * (5 + det) * t + rng.random() * 6)
        ph = f * det * np.cumsum(1 + vib) / SR + rng.random()
        sig += 2 * (ph % 1) - 1
    env = np.clip(t / att, 0, 1)
    rel_i = int(dur * SR)
    env[rel_i:] *= np.exp(-(t[rel_i:] - dur) / (rel / 4))
    return sig * env / 3


def bass(f, dur, rel=0.25):
    n = int((dur + rel) * SR)
    t = np.arange(n) / SR
    sig = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t) + 0.08 * np.sin(6 * np.pi * f * t)
    env = np.clip(t / 0.01, 0, 1) * (0.7 + 0.3 * np.exp(-t / 0.3))
    rel_i = int(dur * SR)
    env[rel_i:] *= np.exp(-(t[rel_i:] - dur) / (rel / 4))
    return sig * env


def tick(accent=False):
    n = int(0.05 * SR)
    t = np.arange(n) / SR
    f1, f2 = (3200, 5100) if accent else (3600, 5600)
    sig = (np.sin(2 * np.pi * f1 * t) * np.exp(-t / 0.006)
           + 0.6 * np.sin(2 * np.pi * f2 * t) * np.exp(-t / 0.004)
           + 0.5 * rng.standard_normal(n) * np.exp(-t / 0.0015))
    return sig


def boom():
    n = int(2.0 * SR)
    t = np.arange(n) / SR
    f = 45 + 35 * np.exp(-t / 0.08)
    sig = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.55)
    noise = rng.standard_normal(n) * np.exp(-t / 0.03)
    noise = sosfilt(butter(2, 300 / (SR / 2), output="sos"), noise)
    return sig + 0.8 * noise


# ---------------------------------------------------------------- 混音轨道
tracks = {}
SEND = {"piano": 0.25, "celesta": 0.45, "organ": 0.4, "flute": 0.35, "strings": 0.4,
        "bass": 0.05, "shaker": 0.1, "tick": 0.12, "boom": 0.25, "arp_organ": 0.4}


def add(track, start, sig, gain=1.0, pan=0.0):
    if track not in tracks:
        tracks[track] = np.zeros((2, N))
    i = int(start * SR)
    j = min(N, i + len(sig))
    sig = sig[: j - i] * gain
    a = (pan + 1) * np.pi / 4
    tracks[track][0, i:j] += sig * np.cos(a)
    tracks[track][1, i:j] += sig * np.sin(a)


# ---------------------------------------------------------------- 和声
CHORDS = {   # (低音, [和弦音], 琶音根音)
    "Am": (m("A2"), [m("A3"), m("C4"), m("E4")], m("A3")),
    "F":  (m("F2"), [m("F3"), m("A3"), m("C4")], m("F3")),
    "C":  (m("C3"), [m("G3"), m("C4"), m("E4")], m("C4")),
    "G":  (m("G2"), [m("G3"), m("B3"), m("D4")], m("G3")),
    "Em": (m("E2"), [m("G3"), m("B3"), m("E4")], m("E3")),
    "E":  (m("E2"), [m("G#3"), m("B3"), m("E4")], m("E3")),
}
THIRD = {"Am": 3, "F": 4, "C": 4, "G": 4, "Em": 3, "E": 4}

CYCLE = ["Am", "F", "C", "G", "Am", "F", "G", "Am"]
B_CHORDS = ["F", "G", "Em", "Am", "F", "G", "E", "E"]
PROG = (["Am", "F", "C", "G"] * 2 + CYCLE * 2 + B_CHORDS + CYCLE * 3 + CYCLE)

if MAJOR:   # C 大调：I–V–vi–IV 明亮进行
    CYCLE = ["C", "G", "Am", "F", "C", "F", "G", "C"]
    B_CHORDS = ["F", "G", "Em", "Am", "F", "G", "G", "G"]
    PROG = (["C", "G", "Am", "F"] * 2 + CYCLE * 2 + B_CHORDS + CYCLE * 3 + CYCLE)
assert LYRICAL or len(PROG) == BARS

# ---------------------------------------------------------------- 旋律（原创主题）
# (音名 或 None 休止, 拍数)
THEME = [
    [("A4", 1), ("C5", 1), ("E5", 1), ("D5", .5), ("E5", .5)],
    [("F5", 2), ("E5", 1), ("C5", 1)],
    [("G4", 1), ("C5", 1), ("E5", 1), ("D5", .5), ("C5", .5)],
    [("D5", 3), (None, 1)],
    [("A4", 1), ("C5", 1), ("E5", 1), ("D5", .5), ("E5", .5)],
    [("F5", 1), ("A5", 1), ("G5", 1), ("F5", 1)],
    [("G5", 1), ("F5", .5), ("E5", .5), ("D5", 1), ("B4", 1)],
    [("A4", 3), (None, 1)],
]
B_THEME = [
    [("C5", 1), ("F5", 1), ("A5", 2)],
    [("B5", 1), ("A5", .5), ("G5", .5), ("D5", 2)],
    [("E5", 1), ("G5", 1), ("B5", 1), ("A5", .5), ("G5", .5)],
    [("A5", 2), ("E5", 2)],
    [("F5", 1), ("A5", 1), ("C6", 2)],
    [("B5", 1), ("A5", .5), ("G5", .5), ("D6", 2)],
    [("G#5", 2), ("B5", 2)],
    [("B5", 3), ("E5", 1)],
]


if MAJOR:
    THEME = [
        [("C5", 1), ("E5", 1), ("G5", 1), ("F5", .5), ("G5", .5)],
        [("G5", 2), ("D5", 1), ("B4", 1)],
        [("A4", 1), ("C5", 1), ("E5", 1), ("D5", .5), ("C5", .5)],
        [("C5", 3), (None, 1)],
        [("C5", 1), ("E5", 1), ("G5", 1), ("F5", .5), ("G5", .5)],
        [("F5", 1), ("A5", 1), ("G5", 1), ("F5", 1)],
        [("G5", 1), ("F5", .5), ("E5", .5), ("D5", 1), ("B4", 1)],
        [("C5", 3), (None, 1)],
    ]
    B_THEME = B_THEME[:6] + [
        [("B5", 2), ("D6", 2)],
        [("D6", 3), ("G5", 1)],
    ]


def melody_notes(phrase, first_bar):
    out = []
    for k, bar in enumerate(phrase):
        beat = 0
        for name, length in bar:
            if name:
                out.append(((first_bar + k) * BAR + beat * BEAT, m(name), length * BEAT))
            beat += length
    return out


def section(b):
    for name, lo, hi in [("intro", 0, 8), ("A1", 8, 16), ("A2", 16, 24), ("B", 24, 32),
                         ("A3", 32, 40), ("A4", 40, 48), ("climax", 48, 56), ("outro", 56, 64)]:
        if lo <= b < hi:
            return name


# ---------------------------------------------------------------- 编排
if not LYRICAL:
    for b in range(BARS):
        sec = section(b)
        ch = PROG[b]
        low, tones, aroot = CHORDS[ch]
        t0 = b * BAR

        # 钟表滴答：贯穿全曲（像《星际穿越》里的时间流逝）
        tg = {"intro": .30, "A1": .22, "A2": .2, "B": .18, "A3": .16, "A4": .16,
              "climax": .14, "outro": .26}[sec]
        if sec == "outro":
            tg *= max(0.0, 1 - (b - 56) / 9)
        for k in range(4):
            add("tick", t0 + k * BEAT, tick(k == 0), tg * (1.2 if k == 0 else 1), 0.35)

        # 琶音 ostinato
        third = THIRD[ch]
        ext = [aroot, aroot + third, aroot + 7, aroot + 12, aroot + 12 + third, aroot + 19, aroot + 24]
        pat8 = [0, 2, 3, 4, 5, 4, 3, 2]
        sixteenth = sec in ("B", "A3", "A4", "climax") or (MAJOR and sec == "A2")
        steps = 16 if sixteenth else 8
        step_len = BAR / steps
        pat = pat8 * 2 if sixteenth else pat8
        ag = {"intro": .32, "A1": .28, "A2": .28, "B": .24, "A3": .22, "A4": .22,
              "climax": .22, "outro": .25}[sec]
        if sec == "intro":
            ag *= 0.5 + 0.5 * b / 7
        if sec == "outro":
            ag *= max(0.0, 1 - (b - 56) / 8.5)
        for k in range(steps):
            note = ext[pat[k]]
            acc = 1.15 if k % (steps // 4) == 0 else 0.9
            add("piano", t0 + k * step_len, piano(hz(note), step_len * 0.9), ag * acc,
                -0.3 + 0.6 * (pat[k] / 6))
            if sec in ("A3", "A4", "climax") and k % 2 == 0:
                add("arp_organ", t0 + k * step_len, organ(hz(note + 12), step_len * 1.6, att=0.01, rel=0.2, bright=0.5),
                    0.07, 0.4 - 0.8 * (pat[k] / 6))

        # 大调版：反拍沙锤，增加律动感
        if MAJOR and sec in ("A2", "B", "A3", "A4", "climax"):
            for k in range(8):
                n = int(0.06 * SR)
                tt = np.arange(n) / SR
                sh = rng.standard_normal(n) * np.clip(tt / 0.008, 0, 1) * np.exp(-tt / 0.02)
                add("shaker", t0 + k * BEAT / 2, sh, 0.11 if k % 2 else 0.05, -0.4)

        # 管风琴和弦铺底
        pg = {"intro": .0, "A1": .16, "A2": .2, "B": .24, "A3": .3, "A4": .32,
              "climax": .42, "outro": .18}[sec]
        if sec == "intro" and b >= 4:
            pg = 0.06 * (b - 3)
        if sec == "B":
            pg += 0.02 * (b - 24)
        if sec == "outro":
            pg *= max(0.0, 1 - (b - 56) / 8)
        if pg > 0:
            for nte in tones:
                add("organ", t0, organ(hz(nte), BAR * 0.98, att=0.5, bright=0.6), pg * 0.5, 0)
            add("organ", t0, organ(hz(low), BAR * 0.98, att=0.4, bright=0.4), pg * 0.5, 0)
            if sec in ("climax",) or (sec == "B" and b >= 28):
                add("organ", t0, organ(hz(low - 12), BAR * 0.98, att=0.4, bright=0.3), pg * 0.45, 0)

        # 低音
        if sec not in ("intro",):
            bg = {"A1": .22, "A2": .28, "B": .32, "A3": .36, "A4": .38, "climax": .42, "outro": .2}[sec]
            if sec == "outro":
                bg *= max(0.0, 1 - (b - 56) / 8)
            add("bass", t0, bass(hz(low), BEAT * 1.9), bg)
            add("bass", t0 + 2 * BEAT, bass(hz(low), BEAT * 1.9), bg * 0.85)

        # 弦乐长音
        if sec in ("A2", "B", "A4", "climax"):
            sg = {"A2": .1, "B": .14, "A4": .14, "climax": .18}[sec]
            for nte in tones:
                add("strings", t0, strings(hz(nte + 12), BAR), sg, 0.3 if nte == tones[0] else -0.3)

        # 低鼓（定音鼓感）
        if sec == "B" and b >= 28:
            for k in range(0, 4, 2 if b < 30 else 1):
                add("boom", t0 + k * BEAT, boom(), 0.25 + 0.06 * (b - 28))
        if sec in ("A3", "A4"):
            add("boom", t0, boom(), 0.38)
        if sec == "climax":
            add("boom", t0, boom(), 0.5)
            add("boom", t0 + 2.5 * BEAT, boom(), 0.3)
        if b == 56:
            add("boom", t0, boom(), 0.55)

    # 主旋律
    for start, sec_name in [(8, "A1"), (16, "A2"), (32, "A3"), (40, "A4"), (48, "climax"), (56, "outro")]:
        for t, note, length in melody_notes(THEME, start):
            f = hz(note)
            if sec_name == "A1":
                add("celesta", t, celesta(f, length), 0.42, 0.1)
                add("piano", t, piano(f, length * 0.95), 0.4, 0.1)
            elif sec_name == "A2":
                add("flute", t, flute(f, length * 0.95), 0.28, 0.05)
                add("piano", t, piano(f, length * 0.95), 0.32, 0.1)
            elif sec_name == "A3":
                add("organ", t, organ(f, length * 0.95, att=0.03, bright=1.0), 0.32, 0)
                add("celesta", t, celesta(f * 2, length), 0.2, 0.25)
            elif sec_name == "A4":
                add("organ", t, organ(f, length * 0.95, att=0.03, bright=1.0), 0.32, 0)
                add("strings", t, strings(f / 2, length * 0.95, att=0.08), 0.18, -0.2)
                add("flute", t, flute(f * 2, length * 0.95), 0.12, 0.3)
            elif sec_name == "climax":
                add("organ", t, organ(f, length * 0.95, att=0.03, bright=1.2), 0.4, 0)
                add("organ", t, organ(f / 2, length * 0.95, att=0.03, bright=0.8), 0.28, 0)
                add("strings", t, strings(f, length * 0.95, att=0.06), 0.2, 0)
                add("celesta", t, celesta(f * 2, length), 0.18, 0.3)
            elif sec_name == "outro":
                fade = max(0.15, 1 - (t / BAR - 56) / 9)
                add("celesta", t, celesta(f, length), 0.42 * fade, 0.1)
                add("piano", t, piano(f, length * 0.95), 0.32 * fade, 0.1)

    for t, note, length in melody_notes(B_THEME, 24):
        f = hz(note)
        add("strings", t, strings(f, length * 0.97, att=0.12), 0.28, 0)
        add("flute", t, flute(f, length * 0.95), 0.18, 0.1)


# ---------------------------------------------------------------- 抒情版编排（60 BPM，每小节 4 秒）
def lsec(b):
    return "intro" if b < 4 else "A1" if b < 12 else "B" if b < 20 else "A2" if b < 28 else "outro"


if LYRICAL:
    L_PROG = ["Am", "F", "C", "G"] + CYCLE + B_CHORDS + CYCLE + ["Am", "F", "G", "Am"]
    assert len(L_PROG) == BARS
    for b in range(BARS):
        sec = lsec(b)
        ch = L_PROG[b]
        low, tones, aroot = CHORDS[ch]
        t0 = b * BAR
        out_fade = max(0.0, 1 - (b - 28) / 4.5) if sec == "outro" else 1.0

        # 钟表滴答：每秒一下
        tg = {"intro": .28, "A1": .2, "B": .16, "A2": .13, "outro": .24}[sec] * out_fade
        for k in range(4):
            add("tick", t0 + k * BEAT, tick(k == 0), tg * (1.2 if k == 0 else 1), 0.35)

        # 慢速钢琴琶音（踏板延音）
        third = THIRD[ch]
        ext = [aroot, aroot + third, aroot + 7, aroot + 12, aroot + 12 + third, aroot + 19, aroot + 24]
        steps = 16 if (sec == "A2" and b >= 24) else 8
        step_len = BAR / steps
        pat = [0, 2, 3, 4, 5, 4, 3, 2] * (steps // 8)
        ag = {"intro": .14 + .04 * b, "A1": .2, "B": .2, "A2": .16, "outro": .2}[sec] * out_fade
        for k in range(steps):
            note = ext[pat[k]]
            acc = 1.1 if k % (steps // 4) == 0 else 0.9
            add("piano", t0 + k * step_len, piano(hz(note), step_len * 1.8, rel=0.8), ag * acc,
                -0.3 + 0.6 * (pat[k] / 6))

        # 管风琴铺底：缓慢渐强
        pg = {"intro": .05 * (b + 1), "A1": .18, "B": .22 + .025 * (b - 12), "A2": .42,
              "outro": .2}[sec] * out_fade
        for nte in tones:
            add("organ", t0, organ(hz(nte), BAR * 0.99, att=1.2, rel=0.8, bright=0.5), pg * 0.5)
        add("organ", t0, organ(hz(low), BAR * 0.99, att=1.0, rel=0.8, bright=0.4), pg * 0.5)
        if sec == "A2" or (sec == "B" and b >= 16):
            add("organ", t0, organ(hz(low - 12), BAR * 0.99, att=1.0, rel=0.8, bright=0.3), pg * 0.45)

        # 低音：整小节长音
        if sec != "intro":
            bg = {"A1": .22, "B": .28, "A2": .38, "outro": .2}[sec] * out_fade
            add("bass", t0, bass(hz(low), BAR * 0.97, rel=0.6), bg)

        # 弦乐和弦
        if sec in ("B", "A2"):
            sg = {"B": .12, "A2": .17}[sec]
            for nte in tones:
                add("strings", t0, strings(hz(nte + 12), BAR, att=1.0, rel=1.0), sg,
                    0.3 if nte == tones[0] else -0.3)

        # 低鼓：只在推进和高潮处
        if sec == "B" and b >= 18:
            for k in (0, 2):
                add("boom", t0 + k * BEAT, boom(), 0.3 + 0.08 * (b - 18))
        if sec == "A2":
            add("boom", t0, boom(), 0.42)
        if b == 28:
            add("boom", t0, boom(), 0.45)

    # 主旋律：同一主题，放慢一倍、连奏
    for t, note, length in melody_notes(THEME, 4):
        f = hz(note)
        add("piano", t, piano(f, length * 1.05, rel=0.9), 0.48, 0.1)
        add("celesta", t, celesta(f * 2, length), 0.1, 0.3)
    for t, note, length in melody_notes(B_THEME, 12):
        f = hz(note)
        add("strings", t, strings(f, length, att=0.35, rel=0.8), 0.3, 0)
        add("flute", t, flute(f, length * 0.97, att=0.15, rel=0.6), 0.14, 0.1)
    for t, note, length in melody_notes(THEME, 20):
        f = hz(note)
        add("organ", t, organ(f, length * 0.98, att=0.12, rel=0.6, bright=1.0), 0.36)
        add("organ", t, organ(f / 2, length * 0.98, att=0.12, rel=0.6, bright=0.7), 0.22)
        add("strings", t, strings(f, length, att=0.25, rel=0.8), 0.22)
    for t, note, length in melody_notes(THEME[4:], 28):
        fade = max(0.25, 1 - (t / BAR - 28) / 5)
        f = hz(note)
        add("piano", t, piano(f, length * 1.05, rel=1.0), 0.42 * fade, 0.1)
        add("celesta", t, celesta(f * 2, length), 0.1 * fade, 0.3)


# ---------------------------------------------------------------- 后期
tracks["strings"] = sosfilt(butter(2, 3200 / (SR / 2), output="sos"), tracks["strings"], axis=1)
if "shaker" in tracks:
    tracks["shaker"] = sosfilt(butter(2, 6000 / (SR / 2), btype="high", output="sos"), tracks["shaker"], axis=1)
tracks["tick"] = sosfilt(butter(2, 1500 / (SR / 2), btype="high", output="sos"), tracks["tick"], axis=1)

dry = sum(tracks.values())
send = sum(tracks[k] * SEND[k] for k in tracks)

ir_len = int(3.2 * SR)
ti = np.arange(ir_len) / SR
ir = rng.standard_normal((2, ir_len)) * np.exp(-ti / (1.1 if LYRICAL else 0.75))
ir = sosfilt(butter(1, 5000 / (SR / 2), output="sos"), ir, axis=1)
ir[:, : int(0.02 * SR)] = 0
ir /= np.sqrt((ir ** 2).sum(axis=1, keepdims=True))
wet = np.stack([fftconvolve(send[c], ir[c])[:N] for c in range(2)])

mix = dry + 0.9 * wet
mix = sosfilt(butter(2, 28 / (SR / 2), btype="high", output="sos"), mix, axis=1)
mix = mix[:, : int(LENGTH * SR)]
mix /= np.max(np.abs(mix)) + 1e-9
mix = np.tanh(1.4 * mix) / np.tanh(1.4)
fade = np.ones(mix.shape[1])
fade[: int(0.01 * SR)] = np.linspace(0, 1, int(0.01 * SR))
fl = int(2.5 * SR)
fade[-fl:] = np.linspace(1, 0, fl) ** 2
mix *= fade * 0.89

wavfile.write(OUT + ".wav", SR, (mix.T * 32767).astype(np.int16))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", OUT + ".wav",
                "-codec:a", "libmp3lame", "-b:a", "256k",
                "-metadata", "title=Stellar Echo 星际回响" + {"minor": "", "major": " (Major)", "lyrical": " (Lyrical)"}[MODE],
                "-metadata", "comment=Original theme inspired by the style of Interstellar",
                OUT + ".mp3"], check=True)
print("done: %.2f s" % (mix.shape[1] / SR))
