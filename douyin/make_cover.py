"""
抖音封面：cover_9x16.jpg（1080x1920，全屏）和 cover_3x4.jpg（1080x1440，个人主页格子）。
个人主页只显示封面中间 3:4 的部分（y 240~1680），所以文字都放在这个范围内。
横封面 4:3 直接复用 ../bilibili/cover_4x3.jpg。
运行：cd douyin && python3 make_cover.py
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "video"))
sys.path.insert(0, os.path.join(HERE, "..", "bilibili"))
import make_cover as bili          # noqa: E402  B 站封面里的黑洞渲染与 Claude 标志
import render_vertical as rv       # noqa: E402  竖版星空

W, H = 1080, 1920
BAND_Y, BAND_H = 560, 810


def main():
    base = bili.black_hole()                                   # 1920x1080 灰度黑洞（高潮亮度）
    x0, y0 = (1920 - 1260) / 2, (1080 - 945) / 2
    band = np.asarray(base.resize((W, BAND_H), Image.LANCZOS, box=(x0, y0, x0 + 1260, y0 + 945)).convert("L"))
    band = band.astype(np.float32) / 255
    canvas = rv.star_canvas()
    y = np.arange(BAND_H, dtype=np.float32)
    f = np.minimum(np.clip(y / 110, 0, 1), np.clip((BAND_H - 1 - y) / 110, 0, 1))
    f = (f * f * (3 - 2 * f))[:, None]
    canvas[BAND_Y:BAND_Y + BAND_H] = canvas[BAND_Y:BAND_Y + BAND_H] * (1 - f) + band * f
    img = Image.fromarray((np.repeat(canvas[..., None], 3, axis=2) * 255).astype(np.uint8)).convert("RGBA")

    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    bili.spaced(dr, "星际回响", ImageFont.truetype(bili.ZH, 168), W / 2, 300, 22,
                (250, 250, 250, 255), shadow=(0, 0, 0, 200))
    bili.spaced(dr, "STELLAR ECHO", ImageFont.truetype(bili.EN, 34), W / 2, 500, 15, (200, 200, 200, 255))
    bili.spaced(dr, "致敬《星际穿越》", ImageFont.truetype(bili.ZH, 44), W / 2, 1420, 8, (175, 175, 175, 255))
    f_lbl = 36
    lw = dr.textlength("Claude", font=ImageFont.truetype(bili.EN, f_lbl))
    R = 20
    bili.claude_mark(dr, (W - (R * 2.6 + lw)) / 2 + R, 1525, R, f_lbl)
    glow = layer.filter(ImageFilter.GaussianBlur(6))
    img.alpha_composite(Image.fromarray((np.asarray(glow).astype(np.float32) * [1, 1, 1, 0.35]).astype(np.uint8)))
    img.alpha_composite(layer)
    img = img.convert("RGB")
    img.save(os.path.join(HERE, "cover_9x16.jpg"), quality=93)
    img.crop((0, 240, W, 1680)).save(os.path.join(HERE, "cover_3x4.jpg"), quality=93)
    print("covers saved")


if __name__ == "__main__":
    main()
