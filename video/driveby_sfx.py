"""
赛车呼啸而过的音效（预告片 14.4 秒后的赛道边分镜）：两辆车先后闯过，左→右声像 + 多普勒音高下滑。
纯 numpy 合成，输出 16 秒立体声 wav，之后在 teaser_text.py 里与成片音乐混合。
运行：python3 driveby_sfx.py [输出.wav]
"""
import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

SR = 44100
TOTAL = 16.0
CUT = 14.4                  # 分镜起点（成片秒数）
PASSES = [(CUT + 0.38, 190.0, 1.0), (CUT + 0.58, 235.0, 0.85)]   # (车心过画面中央的时刻, 基频, 音量)；对应 race.py 的 DB_T0 / DB_GAP


def engine_pass(center, f0, win=0.9, narrow=0.07):
    n = int(win * SR)
    t = np.arange(n) / SR
    u = (t - win / 2) / narrow                                    # 过中央前后各约 0.1 秒内音高和音量剧变
    f = f0 * (1 + 0.28 * np.tanh(-u)) * (1 + 0.1 * t / win)
    ph = np.cumsum(f) / SR
    sig = sum(np.sin(2 * np.pi * h * ph) / h ** 0.8 for h in range(1, 14))
    rng = np.random.default_rng(3)
    sig = np.tanh(1.6 * sig) + 0.45 * rng.standard_normal(n)
    sig = sosfilt(butter(2, 5200 / (SR / 2), output="sos"), sig)
    amp = 1 / (1 + (u * 0.55) ** 2) ** 0.9
    pan = np.tanh(u * 0.6)                                        # 左 → 右
    a = (pan + 1) * np.pi / 4
    return np.stack([sig * amp * np.cos(a), sig * amp * np.sin(a)])


def make(path):
    out = np.zeros((2, int(TOTAL * SR)), np.float32)
    for center, f0, g in PASSES:
        sig = engine_pass(center, f0)
        i = int((center - 0.45) * SR)
        out[:, i:i + sig.shape[1]] += (sig * g).astype(np.float32)
    out /= np.max(np.abs(out)) + 1e-9
    out *= 0.55
    wavfile.write(path, SR, (out.T * 32767).astype(np.int16))


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "driveby_sfx.wav")
    print("done")
