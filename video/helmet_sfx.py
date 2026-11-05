"""
头盔视角 4 秒的音效：引擎 + 风噪（不含音乐）。纯 numpy 合成，立体声 wav。
时间点和画面对应（race.py --helmet，HELMET_T0 起 4 秒）：
  0–1.8 秒  高转速引擎，转速缓升
  1.8 秒    路面颠簸：一声闷响，引擎声卡一下
  2.3–2.65  撕膜：声音由「闷」变「清脆」（低通截止频率从 2.2k 抬到 7k），风噪不变
  2.65–4.0  引擎继续升高，收在最高点
运行：python3 helmet_sfx.py [输出.wav]
"""
import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

SR = 44100
DUR = 4.0


def smooth(t, a, b):
    x = np.clip((t - a) / (b - a), 0, 1)
    return x * x * (3 - 2 * x)


def engine(t):
    """V6 涡轮混动：点火频率 = 转速/60 × 3；转速 9600 → 13200，颠簸处掉一下"""
    rpm = 9600 + 3600 * (t / DUR) ** 0.9
    rpm = rpm - 900 * np.exp(-((t - 1.82) / 0.07) ** 2)                  # 颠簸：节气门一顿
    f0 = rpm / 60 * 3
    ph = np.cumsum(f0) / SR
    sig = sum(np.sin(2 * np.pi * h * ph + 0.7 * h) / h ** 0.75 for h in range(1, 13))
    sig += 0.35 * np.sin(2 * np.pi * (ph * 0.5))                         # 次谐波：厚度
    rng = np.random.default_rng(5)
    sig += 0.22 * sosfilt(butter(2, [200 / (SR / 2), 1800 / (SR / 2)], btype="band", output="sos"), rng.standard_normal(len(t)))   # 燃烧噪声
    sig = np.tanh(1.5 * sig)
    turbo = 0.06 * np.sin(2 * np.pi * np.cumsum(2400 + 1800 * (t / DUR)) / SR) * (0.6 + 0.4 * np.sin(2 * np.pi * 9 * t))     # 涡轮啸叫
    sig = sig + turbo
    # 撕膜前后：「闷」→「清脆」。用两种滤波的交叉渐变，不是改单个滤波器的截止频率（那样要逐样本滤波）
    muffled = sosfilt(butter(3, 2200 / (SR / 2), output="sos"), sig)
    clear = sosfilt(butter(2, 7500 / (SR / 2), output="sos"), sig)
    mix = smooth(t, 2.3, 2.65)
    sig = muffled * (1 - mix) + clear * mix
    amp = 0.55 + 0.3 * (t / DUR) - 0.25 * np.exp(-((t - 1.82) / 0.07) ** 2)
    return sig * amp


def wind(t):
    """150 米/秒的风噪：宽带噪声 + 头盔抖动造成的忽强忽弱（低频调幅）+ 左右声道不相关"""
    rng = np.random.default_rng(11)
    out = []
    for c in range(2):
        n = rng.standard_normal(len(t))
        n = sosfilt(butter(2, [350 / (SR / 2), 4200 / (SR / 2)], btype="band", output="sos"), n)
        buf = sosfilt(butter(2, 11 / (SR / 2), output="sos"), rng.standard_normal(len(t)))
        buf = buf / (np.abs(buf).max() + 1e-9)
        out.append(n * (0.55 + 0.3 * buf + 0.1 * (t / DUR)))
    return np.stack(out)


def bump(t):
    """1.82 秒的路面颠簸：低频闷响 + 一点噪声撞击"""
    tt = np.clip(t - 1.82, 0, None)
    e = np.exp(-tt / 0.09) * (t >= 1.82)
    rng = np.random.default_rng(2)
    thump = np.sin(2 * np.pi * (70 - 25 * np.clip(tt / 0.3, 0, 1)) * tt) * e
    knock = sosfilt(butter(2, 900 / (SR / 2), output="sos"), rng.standard_normal(len(t))) * np.exp(-tt / 0.03) * (t >= 1.82)
    return 0.9 * thump + 0.35 * knock


def make(path):
    t = np.arange(int(DUR * SR)) / SR
    eng = engine(t)
    w = wind(t)
    b = bump(t)
    # 立体声：引擎在车尾后方，略偏中；风噪左右分开；颠簸居中
    L = eng * 0.95 + w[0] * 0.33 + b
    R = eng * 1.0 + w[1] * 0.33 + b
    out = np.stack([L, R])
    out = np.tanh(1.1 * out)
    out /= np.max(np.abs(out)) + 1e-9
    fade = np.minimum(np.clip(t / 0.04, 0, 1), np.clip((DUR - t) / 0.25, 0, 1))
    out = out * fade * 0.86
    wavfile.write(path, SR, (out.T * 32767).astype(np.int16))


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "helmet_sfx.wav")
    print("done")
