"""
微信封面：视频号 3:4（1080×1440）、公众号头图 2.35:1（900×383）、公众号小图 1:1（500×500）
每种两款：A 水墨 / B 夜景。运行：python3 wechat_kit.py -> wechat/
"""
import os

import numpy as np
from PIL import Image

import render as R

OUT = "wechat"
os.makedirs(OUT, exist_ok=True)
SUB = "从商朝都城到「米」字形高铁"


def crop_frame(img, cx, aspect):
    """从 1920×1080 画面里按 aspect(宽/高) 裁一块，cx 为水平中心（0~1）"""
    H, W = img.shape[:2]
    ch = H
    cw = int(ch * aspect)
    if cw > W:
        cw, ch = W, int(W / aspect)
    x0 = int(np.clip(cx * W - cw / 2, 0, W - cw))
    y0 = 100 if aspect > 1.6 else 0
    return img[y0:y0 + ch, x0:x0 + cw]


def resize(img, w, h):
    im = Image.fromarray(R.to_u8(img)).resize((w, h), Image.LANCZOS)
    return np.asarray(im, np.float32) / 255


def shade(img, strength, top=True):
    h = img.shape[0]
    v = np.linspace(1, 0, h, dtype=np.float32) if top else np.linspace(0, 1, h, dtype=np.float32)
    return img * (1 - strength * v ** 1.4)[:, None, None]


def build(name, t, cx, ink, sub_col, dark, W, H, layout, ss=1):
    w, h = W * ss, H * ss
    frame = R.scene_mix(t)
    img = resize(crop_frame(frame, cx, W / H), w, h)
    if dark:
        img = shade(img, 0.55, top=True)
    k = ss
    if layout == "portrait":        # 3:4
        t1 = R.text_sprite("天地之中", R.SERIF_BLACK, 190 * k, fill=ink, shadow=6)
        t2 = R.text_sprite("· 郑州", R.SERIF_BLACK, 190 * k, fill=ink, shadow=6)
        sub = R.text_sprite(SUB, R.SERIF_BOLD, 52 * k, fill=sub_col, shadow=4)
        seal = R.seal_sprite("中原", 96 * k)
        x = 90 * k
        R.paste(img, t1, x, 90 * k)
        R.paste(img, t2, x, (90 + 195) * k)
        R.paste(img, sub, x + 8 * k, (90 + 195 * 2 + 12) * k)
        R.paste(img, seal, x + t2.w + 24 * k, (90 + 195 + 40) * k)
        tag = R.text_sprite("2 分钟 · 水墨动画 · 中英字幕", R.SANS_BOLD, 40 * k, fill=(255, 255, 255), shadow=4)
        R.paste(img, tag, 90 * k, (H - 110) * k)
    elif layout == "banner":        # 2.35:1
        t1 = R.text_sprite(R.TITLE_ZH, R.SERIF_BLACK, 56 * k, fill=ink, shadow=3)
        sub = R.text_sprite(SUB, R.SERIF_BOLD, 24 * k, fill=sub_col, shadow=2)
        seal = R.seal_sprite("中原", 42 * k)
        R.paste(img, t1, 30 * k, 40 * k)
        R.paste(img, seal, 30 * k + t1.w + 2 * k, 46 * k)
        R.paste(img, sub, 36 * k, 40 * k + t1.h + 2 * k)
    else:                           # 1:1
        t1 = R.text_sprite("天地之中", R.SERIF_BLACK, 92 * k, fill=ink, shadow=4)
        t2 = R.text_sprite("· 郑州", R.SERIF_BLACK, 92 * k, fill=ink, shadow=4)
        seal = R.seal_sprite("中原", 50 * k)
        R.paste(img, t1, 34 * k, 30 * k)
        R.paste(img, t2, 34 * k, 30 * k + 96 * k)
        R.paste(img, seal, 34 * k + t2.w + 14 * k, 30 * k + 106 * k)
    out = Image.fromarray(R.to_u8(img))
    if ss > 1:
        out = out.resize((W, H), Image.LANCZOS)
    out.save(f"{OUT}/{name}.png")
    print(name)


A = dict(t=9.0, cx=0.42, ink=(28, 24, 22), sub_col=(70, 58, 50), dark=False)
B = dict(t=118.0, cx=0.50, ink=(255, 214, 140), sub_col=(245, 238, 228), dark=True)
for tag, cfg in (("A_ink", A), ("B_night", B)):
    build(f"视频号封面_{tag}_3x4_1080x1440", W=1080, H=1440, layout="portrait", **cfg)
    build(f"公众号头图_{tag}_900x383", W=900, H=383, layout="banner", ss=2, **cfg)
    build(f"公众号小图_{tag}_500x500", W=500, H=500, layout="square", ss=2, **cfg)
