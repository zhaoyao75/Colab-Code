"""
抖音短版（约 53 秒，竖屏）：从完整竖屏版剪辑，开头 0 秒就是鼓点进入 + 霓虹点亮（抓住前 3 秒）。
  片段 A：10.67–42.67 秒（鼓点进入、霓虹闪亮、主旋律第一遍，正好 12 小节）
  片段 B：106.67–128 秒（尾声：主旋律 + 灯火渐熄 + 片尾标题）
两段都切在小节线上，音频硬接（50ms 交叉淡化），节拍不乱；画面 0.6 秒叠化，且与音频严格对齐。
开头 0.3–3.8 秒叠加标题。

运行：cd music_video/douyin && python3 make_short.py   -> neon_rain_douyin_53s.mp4
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.argv = [sys.argv[0], "--portrait"]
import render as rv  # noqa: E402

SRC = os.path.join(HERE, "..", "neon_rain_9x16.mp4")
OUT = os.path.join(HERE, "neon_rain_douyin_53s.mp4")
BAR = 8 / 3
A0, A1 = 4 * BAR, 16 * BAR          # 10.667 – 42.667
B0, B1 = 40 * BAR, 128.0            # 106.667 – 128
XF = 0.6


def main():
    title, _ = rv.overlay_layers()
    png = os.path.join(HERE, ".title.png")
    Image.fromarray((np.clip(title, 0, 1) * 255).astype(np.uint8), "RGBA").save(png)
    la = A1 - A0
    fc = (
        f"[0:v]trim={A0}:{A1},setpts=PTS-STARTPTS[va];"
        f"[0:v]trim={B0 - XF}:{B1},setpts=PTS-STARTPTS[vb];"
        f"[va][vb]xfade=transition=fade:duration={XF}:offset={la - XF}[vx];"
        f"[1:v]format=rgba,fade=t=in:st=0.3:d=0.7:alpha=1,fade=t=out:st=3.1:d=0.7:alpha=1[tt];"
        f"[vx][tt]overlay=0:0:enable='lt(t,4)'[v];"
        f"[0:a]atrim={A0}:{A1},asetpts=PTS-STARTPTS,afade=t=in:d=0.05[aa];"
        f"[0:a]atrim={B0}:{B1},asetpts=PTS-STARTPTS[ab];"
        f"[aa][ab]acrossfade=d=0.05[a]"
    )
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", SRC,
                    "-loop", "1", "-t", "4", "-framerate", "30", "-i", png,
                    "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "22", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", OUT], check=True)
    os.remove(png)
    print("saved", OUT)


if __name__ == "__main__":
    main()
