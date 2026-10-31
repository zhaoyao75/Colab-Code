"""
B 站封面：用黑白版高潮时刻的黑洞画面，按 1920x1080 原分辨率渲染，
输出 cover_16x9.jpg（1920x1080）和 cover_4x3.jpg（1440x1080）。
运行：cd bilibili && python3 make_cover.py
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import gaussian_filter, zoom

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "video"))
import render_demo as rd  # noqa: E402

rd.BW = True
RW, RH = rd.RW, rd.RH
ZH = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
EN = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def black_hole(t=6.0, bright=1.3):
    hr, hp, esc, edir = rd.trace()
    tex = rd.disk_texture()
    bg = rd.star_background(esc, edir).reshape(-1, 3) @ np.array([0.299, 0.587, 0.114], np.float32)
    NR, NP = tex.shape
    trans = np.ones(RW * RH, np.float32)
    acc = np.zeros(RW * RH, np.float32)
    for k in range(hr.shape[0]):
        idx = np.nonzero(~np.isnan(hr[k]))[0]
        r = hr[k][idx]
        ir = np.clip((r - rd.RIN) / (rd.ROUT - rd.RIN) * (NR - 1), 0, NR - 1).astype(np.int32)
        x = np.clip((r - rd.RIN) / (rd.ROUT - rd.RIN), 0, 1)
        prof = (rd.RIN / np.maximum(r, rd.RIN)) ** 1.6
        prof *= np.clip((r - rd.RIN * 0.9) / (rd.RIN * 0.25), 0, 1) * np.clip((rd.ROUT * 1.05 - r) / 3.5, 0, 1)
        omega = 0.55 * (rd.RIN / np.maximum(r, rd.RIN)) ** 1.5
        ip = (((hp[k][idx] + omega * t) % (2 * np.pi)) / (2 * np.pi) * NP).astype(np.int32) % NP
        inten = prof * tex[ir, ip]
        alpha = np.clip(inten * 1.3, 0, 0.97)
        acc[idx] += (1 - 0.25 * x) * inten * 2.2 * bright * trans[idx] * alpha
        trans[idx] *= 1 - alpha
    hdr = (acc + bg * trans).reshape(RH, RW)
    bloom = zoom(gaussian_filter(np.maximum(hdr[::4, ::4] - 0.6, 0), 10), 4, order=1)[:RH, :RW]
    hdr = hdr + 0.6 * bloom
    ldr = np.clip(1 - np.exp(-hdr * 1.25), 0, 1) ** (1 / 1.15)
    return Image.fromarray((ldr * 255).astype(np.uint8)).convert("RGB")


def spaced(dr, text, font, cx, y, spacing, fill, shadow=None):
    widths = [dr.textlength(c, font=font) for c in text]
    x = cx - (sum(widths) + spacing * (len(text) - 1)) / 2
    for c, w in zip(text, widths):
        if shadow:
            dr.text((x + 3, y + 3), c, font=font, fill=shadow)
        dr.text((x, y), c, font=font, fill=fill)
        x += w + spacing


def claude_mark(dr, cx, cy, R, label_size):
    orange = (217, 119, 87)
    lr = np.random.default_rng(11)
    for i in range(12):
        ang = i * 2 * np.pi / 12 + lr.uniform(-0.12, 0.12)
        ln = R * lr.uniform(0.75, 1.0)
        x2, y2 = cx + ln * np.cos(ang), cy + ln * np.sin(ang)
        dr.line([(cx, cy), (x2, y2)], fill=orange, width=max(3, int(R / 4.5)))
        e = R / 9
        dr.ellipse([x2 - e, y2 - e, x2 + e, y2 + e], fill=orange)
    f = ImageFont.truetype(EN, label_size)
    dr.text((cx + R * 1.6, cy - label_size * 0.62), "Claude", font=f, fill=(238, 238, 238))


def compose(img, w, h):
    """img: 已裁好尺寸的黑洞底图。下方压暗渐变，放大标题。"""
    img = img.copy()
    grad = np.linspace(0, 1, h)[:, None]
    shade = 1 - 0.75 * np.clip((grad - 0.58) / 0.42, 0, 1) ** 1.5
    arr = np.asarray(img).astype(np.float32) * shade[..., None]
    img = Image.fromarray(arr.astype(np.uint8))
    # 黑洞整体上移一点，给标题留空间
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    canvas.paste(img, (0, -int(h * 0.08)))
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    s = h / 1080
    zh = ImageFont.truetype(ZH, int(176 * s))
    spaced(dr, "星际回响", zh, w / 2, h * 0.62, int(40 * s), (250, 250, 250, 255), shadow=(0, 0, 0, 200))
    en = ImageFont.truetype(EN, int(38 * s))
    spaced(dr, "STELLAR ECHO", en, w / 2, h * 0.835, int(16 * s), (205, 205, 205, 255))
    tag = ImageFont.truetype(ZH, int(40 * s))
    spaced(dr, "致敬《星际穿越》", tag, w / 2, h * 0.905, int(10 * s), (170, 170, 170, 255))
    claude_mark(dr, w - 250 * s, 70 * s, 26 * s, int(40 * s))   # 右上角，避开 B 站右下角时长角标
    glow = layer.filter(ImageFilter.GaussianBlur(6 * s))
    out = canvas.convert("RGBA")
    out.alpha_composite(Image.fromarray((np.asarray(glow).astype(np.float32) * [1, 1, 1, 0.35]).astype(np.uint8)))
    out.alpha_composite(layer)
    return out.convert("RGB")


def main():
    base = black_hole()
    compose(base, 1920, 1080).save(os.path.join(HERE, "cover_16x9.jpg"), quality=93)
    crop = base.crop(((1920 - 1440) // 2, 0, (1920 + 1440) // 2, 1080))
    compose(crop, 1440, 1080).save(os.path.join(HERE, "cover_4x3.jpg"), quality=93)
    print("covers saved")


if __name__ == "__main__":
    main()
