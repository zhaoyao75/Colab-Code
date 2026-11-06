"""合集版接缝音效：逆向镲（0.9 秒上升的噪声）+ 接缝处的低频重击 + 镲尾。运行：python3 seam_fx.py 输出.wav [接缝秒数=8.4] [总长=12]"""
import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt
SR = 44100
out_path = sys.argv[1]; T0 = float(sys.argv[2]) if len(sys.argv) > 2 else 8.4; DUR = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
n = int(DUR * SR); t = np.arange(n) / SR
rng = np.random.default_rng(3)
x = np.zeros((2, n))
for c in range(2):
    nz = sosfilt(butter(2, 3500 / (SR / 2), btype="high", output="sos"), rng.standard_normal(n))
    rise = np.clip((t - (T0 - 0.9)) / 0.9, 0, 1) ** 2.2 * (t < T0)                      # 逆向镲：在接缝前 0.9 秒越来越响，到接缝处突然切断
    tail = np.exp(-np.clip(t - T0, 0, None) / 0.55) * (t >= T0) * 0.55                  # 接缝后的镲尾
    x[c] = nz * (rise * 0.55 + tail)
tt = np.clip(t - T0, 0, None)
thump = np.sin(2 * np.pi * (62 - 22 * np.clip(tt / 0.35, 0, 1)) * tt) * np.exp(-tt / 0.22) * (t >= T0)    # 重击
x += 0.9 * thump
x = np.tanh(x)
wavfile.write(out_path, SR, (x.T * 0.8 * 32767).astype(np.int16))
