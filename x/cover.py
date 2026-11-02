"""
X（Twitter）封面：cover_16x9.jpg（1920x1080）。纯英文，海报风标题，
与英文版视频一致（Inter Display ExtraLight + 大字距）。
黑洞复用 ../bilibili/make_cover.py 的高潮时刻渲染。
运行：cd x && python3 cover.py
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "video"))
sys.path.insert(0, os.path.join(HERE, "..", "bilibili"))
import make_cover as bili  # noqa: E402

TITLE = "/usr/share/fonts/opentype/inter/InterDisplay-ExtraLight.otf"
SMALL = "/usr/share/fonts/opentype/inter/Inter-Light.otf"
W, H = 1920, 1080


def main():
    base = bili.black_hole()
    grad = np.linspace(0, 1, H)[:, None]
    shade = 1 - 0.7 * np.clip((grad - 0.62) / 0.38, 0, 1) ** 1.5
    arr = np.asarray(base).astype(np.float32) * shade[..., None]
    canvas = Image.new("RGB", (W, H))
    canvas.paste(Image.fromarray(arr.astype(np.uint8)), (0, -int(H * 0.06)))
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    bili.spaced(dr, "STELLAR ECHO", ImageFont.truetype(TITLE, 124), W / 2, 830, 46, (244, 244, 244, 255))
    bili.spaced(dr, "AN ORIGINAL SCORE  ·  INSPIRED BY INTERSTELLAR", ImageFont.truetype(SMALL, 27),
                W / 2, 985, 11, (178, 178, 178, 255))
    bili.claude_mark(dr, W - 250, 70, 26, 40)
    glow = layer.filter(ImageFilter.GaussianBlur(6))
    out = canvas.convert("RGBA")
    out.alpha_composite(Image.fromarray((np.asarray(glow).astype(np.float32) * [1, 1, 1, 0.3]).astype(np.uint8)))
    out.alpha_composite(layer)
    out.convert("RGB").save(os.path.join(HERE, "cover_16x9.jpg"), quality=93)
    print("cover saved")


if __name__ == "__main__":
    main()
