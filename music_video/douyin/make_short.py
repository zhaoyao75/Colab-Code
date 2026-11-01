"""
抖音短版（约 53 秒，竖屏）：只渲染用到的两段，不再先出整条 128 秒完整版。
  片段 A：10.67–42.67 秒（鼓点进入、招牌逐块亮起、主旋律第一遍，正好 12 小节）
  片段 B：106.07–128 秒（尾声：主旋律 + 灯火渐熄 + 片尾标题；前 0.6 秒只用来做叠化）
两段都切在小节线上，音频硬接（50ms 交叉淡化），节拍不乱；画面 0.6 秒叠化，且与音频严格对齐。
开头 0.3–3.8 秒叠加标题。

运行：cd music_video/douyin && python3 make_short.py   -> neon_rain_douyin_53s.mp4（约 6 分钟，4 进程）
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

OUT = os.path.join(HERE, "neon_rain_douyin_53s.mp4")
BAR = 8 / 3
FPS = rv.FPS
A0, A1 = 4 * BAR, 16 * BAR          # 10.667 – 42.667
B1 = 128.0
XF = 0.6
B0 = 40 * BAR                       # 106.667：片段 B 真正的起点
BS = B0 - XF                        # 106.067：片段 B 实际渲染的起点（提前 XF 秒用来叠化）


def main():
    rv.init()
    pa, pb = os.path.join(HERE, ".part_A.mp4"), os.path.join(HERE, ".part_B.mp4")
    ia0, ia1 = round(A0 * FPS), round(A1 * FPS)
    ib0, ib1 = round(BS * FPS), rv.NF
    rv.render(ia0, ia1, pa, ia0 / FPS)
    rv.render(ib0, ib1, pb, ib0 / FPS)
    la = (ia1 - ia0) / FPS
    title, _ = rv.overlay_layers()
    png = os.path.join(HERE, ".title.png")
    Image.fromarray((np.clip(title, 0, 1) * 255).astype(np.uint8), "RGBA").save(png)
    fc = (
        f"[0:v]setpts=PTS-STARTPTS[va];[1:v]setpts=PTS-STARTPTS[vb];"
        f"[va][vb]xfade=transition=fade:duration={XF}:offset={la - XF}[vx];"
        f"[2:v]format=rgba,fade=t=in:st=0.3:d=0.7:alpha=1,fade=t=out:st=3.1:d=0.7:alpha=1[tt];"
        f"[vx][tt]overlay=0:0:enable='lt(t,4)'[v];"
        f"[0:a]asetpts=PTS-STARTPTS,afade=t=in:d=0.05[aa];"
        f"[1:a]atrim=start={XF},asetpts=PTS-STARTPTS[ab];"
        f"[aa][ab]acrossfade=d=0.05[a]"
    )
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", pa, "-i", pb,
                    "-loop", "1", "-t", "4", "-framerate", str(FPS), "-i", png,
                    "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
                    "-c:v", "libx264", "-preset", "slow", "-crf", "22", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", OUT], check=True)
    for f in (pa, pb, png):
        os.remove(f)
    print("saved", OUT)


if __name__ == "__main__":
    main()
