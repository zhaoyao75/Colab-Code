"""
抖音 16 秒版（竖屏，正好 6 个小节 = 16.0 秒）：只渲染用到的两段。
  片段 A：42.67–53.33 秒（第 16–19 小节：主旋律前 4 小节，招牌已亮大半、还在陆续点亮）
  片段 B：122.67–128 秒（第 46–47 小节：旋律收束，灯火熄灭、标题再现、淡出；前 0.6 秒只用来做叠化）
和弦衔接：C7 → Gm7·C7 → Fmaj7（V–ii–V–I，一个完整的收束）。
两段都切在小节线上，音频只做 50ms 交叉淡化；画面 0.6 秒叠化，与音频严格对齐。开头 0.2–3.3 秒叠加标题。

运行：cd music_video/douyin && python3 make_short16.py   -> neon_rain_douyin_16s.mp4（约 3 分钟）
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

OUT = os.path.join(HERE, "neon_rain_douyin_16s.mp4")
BAR = 8 / 3
FPS = rv.FPS
A0, A1 = 16 * BAR, 20 * BAR         # 42.667 – 53.333
B0 = 46 * BAR                       # 122.667：片段 B 真正的起点
XF = 0.6
BS = B0 - XF                        # 122.067：片段 B 实际渲染的起点（提前 XF 秒用来叠化）


def main():
    rv.init()
    pa, pb = os.path.join(HERE, ".part_A16.mp4"), os.path.join(HERE, ".part_B16.mp4")
    ia0, ia1 = round(A0 * FPS), round(A1 * FPS)
    ib0, ib1 = round(BS * FPS), rv.NF
    rv.render(ia0, ia1, pa, ia0 / FPS)
    rv.render(ib0, ib1, pb, ib0 / FPS)
    la = (ia1 - ia0) / FPS
    title, _ = rv.overlay_layers()
    # 16 秒版开头就是满屏招牌：在标题后面垫一条柔和的暗带，标题才读得清
    yy = np.arange(rv.H, dtype=np.float32)[:, None]
    band = np.exp(-((yy - 0.31 * rv.H) / (0.06 * rv.H)) ** 2) * 0.62
    ta = title[..., 3]
    na = ta + band * (1 - ta)
    title = np.dstack([title[..., :3] * (ta / np.maximum(na, 1e-4))[..., None], na])
    png = os.path.join(HERE, ".title16.png")
    Image.fromarray((np.clip(title, 0, 1) * 255).astype(np.uint8), "RGBA").save(png)
    fc = (
        f"[0:v]setpts=PTS-STARTPTS[va];[1:v]setpts=PTS-STARTPTS[vb];"
        f"[va][vb]xfade=transition=fade:duration={XF}:offset={la - XF}[vx];"
        f"[2:v]format=rgba,fade=t=in:st=0.2:d=0.5:alpha=1,fade=t=out:st=2.8:d=0.5:alpha=1[tt];"
        f"[vx][tt]overlay=0:0:enable='lt(t,3.5)'[v];"
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
