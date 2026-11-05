"""
合集封面「AI 原创音乐」：上半《霓虹雨夜》招牌街，下半《星际回响》黑洞，中间是标题。
输出：collection_cover_3x4.jpg（1080×1440，抖音合集封面）、collection_cover_9x16.jpg（1080×1920）
运行：cd music_video/douyin && python3 make_collection_cover.py
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.argv = [sys.argv[0], "--portrait"]
import render as rv  # noqa: E402

ROOT = os.path.join(HERE, "..", "..")
W = 1080


def spaced(dr, text, font, cx, y, spacing, fill):
    widths = [dr.textlength(c, font=font) for c in text]
    x = cx - (sum(widths) + spacing * (len(text) - 1)) / 2
    for c, w in zip(text, widths):
        dr.text((x, y), c, font=font, fill=fill)
        x += w + spacing


def claude_mark(dr, cx, cy, R, size):
    orange = (217, 119, 87, 255)
    lr = np.random.default_rng(11)
    for i in range(12):
        ang = i * 2 * np.pi / 12 + lr.uniform(-0.12, 0.12)
        ln = R * lr.uniform(0.75, 1.0)
        x2, y2 = cx + ln * np.cos(ang), cy + ln * np.sin(ang)
        dr.line([(cx, cy), (x2, y2)], fill=orange, width=max(3, int(R / 4.5)))
        dr.ellipse([x2 - R / 9, y2 - R / 9, x2 + R / 9, y2 + R / 9], fill=orange)
    dr.text((cx + R * 1.6, cy - size * 0.62), "Claude", font=ImageFont.truetype(rv.EN, size), fill=(240, 236, 230, 255))


def panel(img, h, y_off=0.0):
    """把图片等比放大到宽 W 后，取高 h 的一条（y_off 为取景中心的相对位置）"""
    iw, ih = img.size
    sc = max(W / iw, h / ih)
    img = img.resize((int(iw * sc + .5), int(ih * sc + .5)), Image.LANCZOS)
    x0 = (img.width - W) // 2
    y0 = int((img.height - h) * y_off)
    return img.crop((x0, y0, x0 + W, y0 + h))


def build(H):
    rv.init()
    neon = Image.fromarray(rv.frame(int(100 * rv.FPS)))                       # 1080×1920
    hole = Image.open(os.path.join(ROOT, "video", "still.png")).convert("RGB").crop((0, 0, 1280, 560))   # 去掉素材里自带的标题和标志
    ph = H // 2
    top = np.asarray(panel(neon, ph, 0.30)).astype(np.float32)
    bot = np.asarray(panel(hole, ph, 0.5)).astype(np.float32)
    canvas = np.concatenate([top, bot], axis=0)
    yy = np.arange(H, dtype=np.float32)[:, None, None]
    mid = H / 2
    canvas *= 1 - 0.80 * np.exp(-((yy - mid) / (H * 0.12)) ** 2)               # 中间标题带压暗
    canvas *= 1 - 0.30 * np.clip((yy - H * 0.86) / (H * 0.14), 0, 1)           # 底部略压暗
    out = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8)).convert("RGBA")
    s = W / 1080
    zh = ImageFont.truetype(rv.ZH, int(150 * s))
    sub = ImageFont.truetype(rv.ZH, int(40 * s))
    tag = ImageFont.truetype(rv.ZH, int(34 * s))
    ty = mid - 110 * s
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    spaced(ImageDraw.Draw(glow), "AI原创音乐", zh, W / 2, ty, int(14 * s), (255, 90, 170, 255))
    glow = glow.filter(ImageFilter.GaussianBlur(16 * s))
    out.alpha_composite(glow)
    out.alpha_composite(glow)
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for d, dark in ((ImageDraw.Draw(shadow), True), (ImageDraw.Draw(layer), False)):
        spaced(d, "AI原创音乐", zh, W / 2, ty, int(14 * s), (0, 0, 0, 255) if dark else (255, 242, 250, 255))
        spaced(d, "旋律 · 画面 · 全部由代码生成", sub, W / 2, ty + 190 * s, int(8 * s),
               (0, 0, 0, 255) if dark else (232, 226, 240, 255))
    shadow = shadow.filter(ImageFilter.GaussianBlur(6 * s))
    out.alpha_composite(shadow)
    out.alpha_composite(layer)
    dr = ImageDraw.Draw(out)
    spaced(dr, "霓虹雨夜", tag, W / 2, H * 0.06, int(10 * s), (255, 214, 236, 235))      # 上下各标一部作品名
    spaced(dr, "星际回响", tag, W / 2, H * 0.905, int(10 * s), (214, 226, 255, 235))
    lay2 = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    claude_mark(ImageDraw.Draw(lay2), W - 235 * s, H * 0.962, 20 * s, int(32 * s))
    out.alpha_composite(lay2)
    return out.convert("RGB")


if __name__ == "__main__":
    build(1440).save(os.path.join(HERE, "collection_cover_3x4.jpg"), quality=93)
    build(1920).save(os.path.join(HERE, "collection_cover_9x16.jpg"), quality=93)
    print("saved")
