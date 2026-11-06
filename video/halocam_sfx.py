"""halo 视角 4 秒音效：沿用 helmet_sfx 的引擎/风噪/颠簸，但声源在镜头后方（halo 在车手前面）：
引擎更靠后、更闷，风噪更大（镜头暴露在气流里）。运行：python3 halocam_sfx.py [输出.wav]"""
import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt
import helmet_sfx as h

SR, DUR = h.SR, h.DUR
t = np.arange(int(DUR * SR)) / SR
eng = h.engine(t)
eng = sosfilt(butter(2, 3800 / (SR / 2), output="sos"), eng) * 0.8
w, b = h.wind(t), h.bump(t) * 0.8
out = np.stack([eng * 0.95 + w[0] * 0.62 + b, eng + w[1] * 0.62 + b])
out = np.tanh(1.1 * out)
out /= np.max(np.abs(out)) + 1e-9
out *= np.minimum(np.clip(t / 0.04, 0, 1), np.clip((DUR - t) / 0.25, 0, 1))
wavfile.write(sys.argv[1] if len(sys.argv) > 1 else "halocam_sfx.wav", SR, (out.T * 0.9 * 32767).astype(np.int16))
