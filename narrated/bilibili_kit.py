"""
B 站发布材料：封面（16:9 / 4:3，两款）+ 字幕（中 / 英 / 双语 SRT）
运行：python3 bilibili_kit.py   ->  bilibili/
"""
import json
import os

import numpy as np
from PIL import Image

import render as R

OUT = "bilibili"
os.makedirs(OUT, exist_ok=True)
TL = json.load(open("timeline.json"))


# ---------------------------------------------------------------- 字幕
def ts(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_srt(name, fn):
    subs = TL["subs"]
    with open(f"{OUT}/{name}", "w", encoding="utf-8") as f:
        for i, s in enumerate(subs):
            end = min(s["end"] + 0.3, subs[i + 1]["start"] - 0.05) if i + 1 < len(subs) else s["end"] + 0.5
            f.write(f"{i + 1}\n{ts(s['start'])} --> {ts(end)}\n{fn(s)}\n\n")


write_srt("subtitles_zh.srt", lambda s: s["zh"])
write_srt("subtitles_en.srt", lambda s: s["en"])
write_srt("subtitles_bilingual.srt", lambda s: s["zh"] + "\n" + s["en"])


# ---------------------------------------------------------------- 封面
def tag_sprite(txt, size=40):
    t = R.text_sprite(txt, R.SANS_BOLD, size, fill=(255, 245, 235))
    pad_x, pad_y = 28, 12
    sp = R.Sprite.__new__(R.Sprite)
    h, w = t.h + 2 * pad_y, t.w + 2 * pad_x
    sp.rgb = np.ones((h, w, 3), np.float32) * R.RED
    yy, xx = np.mgrid[0:h, 0:w]
    r = 14
    corner = np.minimum.reduce([xx, w - 1 - xx, yy, h - 1 - yy])
    sp.a = np.ones((h, w), np.float32)
    for cx, cy in ((r, r), (w - 1 - r, r), (r, h - 1 - r), (w - 1 - r, h - 1 - r)):
        m = ((xx - cx) * (1 if cx > w / 2 else -1) > 0) & ((yy - cy) * (1 if cy > h / 2 else -1) > 0)
        sp.a[m & (np.hypot(xx - cx, yy - cy) > r)] = 0
    sp.h, sp.w = h, w
    reg = sp.rgb[pad_y:pad_y + t.h, pad_x:pad_x + t.w]
    reg[:] = reg * (1 - t.a[..., None]) + t.rgb * t.a[..., None]
    return sp


def cover(t, name, ink, sub_col, shade=None):
    img = R.scene_mix(t)
    if shade is not None:   # 文字区压暗 / 提亮，保证缩略图里也读得清
        xx = np.linspace(0, 1, R.W, dtype=np.float32)[None, :, None]
        yy = np.linspace(0, 1, R.H, dtype=np.float32)[:, None, None]
        img = img * (1 - shade * np.clip(1.2 - xx * 1.6, 0, 1) * np.clip(1.3 - yy * 1.5, 0, 1))
    big = R.text_sprite(R.TITLE_ZH, R.SERIF_BLACK, 170, fill=ink, shadow=6)
    line2 = R.text_sprite("从商朝都城到「米」字形高铁", R.SERIF_BOLD, 70, fill=sub_col, shadow=4)
    seal = R.seal_sprite("中原", 92)
    tag = tag_sprite("2 分钟 · 水墨动画 · 中英字幕")
    R.paste(img, big, 110, 110)
    R.paste(img, seal, 110 + big.w + 10, 125)
    R.paste(img, line2, 120, 110 + big.h + 10)
    R.paste(img, tag, 128, 110 + big.h + line2.h + 40)
    im = Image.fromarray(R.to_u8(img))
    im.save(f"{OUT}/{name}_16x9.png")
    im.crop((0, 0, 1440, 1080)).save(f"{OUT}/{name}_4x3.png")
    print("封面", name)


cover(9.0, "cover_A_ink", ink=(28, 24, 22), sub_col=(70, 58, 50))
cover(118.0, "cover_B_night", ink=(255, 214, 140), sub_col=(245, 238, 228), shade=0.25)
print("字幕与封面已输出到", OUT)
