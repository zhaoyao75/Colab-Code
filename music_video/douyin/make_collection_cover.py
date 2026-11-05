"""
合集封面「AI原创音乐」：1080×1080。文字在上，《星际回响》的黑洞缩小放在下方。
字体（均为 Google Fonts 开源字体，放在 fonts/ 里）：zcool = 站酷小薇，noto = 思源宋体粗体，mashan = 马善政毛笔
运行：cd music_video/douyin && python3 make_collection_cover.py [zcool|noto|mashan]  -> collection_cover_1080_title.jpg
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
N = 1080
FONTS = {"zcool": "ZCOOLXiaoWei-Regular.ttf", "noto": "NotoSerifSC-VF.ttf", "mashan": "MaShanZheng-Regular.ttf"}
TEXT = "AI原创音乐"


def load_font(key, size):
    f = ImageFont.truetype(os.path.join(HERE, "fonts", FONTS[key]), size)
    if key == "noto":
        f.set_variation_by_axes([800])        # 粗体
    return f


def build(key, out_path):
    still = Image.open(os.path.join(HERE, "..", "..", "video", "still.png")).convert("RGB")   # 1280×720
    hole = still.crop((0, 70, 1280, 540))                                      # 去掉素材下部自带的标题
    w = 1000
    hole = hole.resize((w, int(w * hole.height / hole.width)), Image.LANCZOS)  # 黑洞缩小
    arr = np.asarray(hole).astype(np.float32)
    arr = np.clip((arr - 14) * 1.06, 0, 255)                                   # 压掉素材背景的底色，和画布的黑融在一起
    yy = np.linspace(-1, 1, arr.shape[0])[:, None, None]
    xx = np.linspace(-1, 1, arr.shape[1])[None, :, None]
    fade = np.clip((1.0 - np.sqrt((xx / 0.98) ** 2 + (yy / 0.95) ** 2)) / 0.22, 0, 1) ** 1.2   # 椭圆渐隐，没有矩形边
    hole = Image.fromarray(np.clip(arr * fade, 0, 255).astype(np.uint8))
    canvas = Image.new("RGB", (N, N), (2, 2, 4))
    cv = np.asarray(canvas).astype(np.float32)                                 # 叠加（而不是粘贴），素材的纯黑背景不会在画布上留下框
    x0, y0 = (N - w) // 2, 540
    cv[y0:y0 + hole.height, x0:x0 + w] += np.asarray(hole).astype(np.float32)
    canvas = Image.fromarray(np.clip(cv, 0, 255).astype(np.uint8))
    out = canvas.convert("RGBA")
    size = 190 if key != "mashan" else 200
    f = load_font(key, size)
    d0 = ImageDraw.Draw(Image.new("L", (1, 1)))
    tw = d0.textlength(TEXT, font=f)
    x, y = (N - tw) / 2, 150
    for layer_fill, blur, gain in (((255, 170, 90, 255), 22, 2), ((0, 0, 0, 255), 7, 2), ((255, 246, 232, 255), 0, 1)):
        layer = Image.new("RGBA", (N, N), (0, 0, 0, 0))
        ImageDraw.Draw(layer).text((x, y), TEXT, font=f, fill=layer_fill)
        if blur:
            layer = layer.filter(ImageFilter.GaussianBlur(blur))
        for _ in range(gain):
            out.alpha_composite(layer)
    out.convert("RGB").save(out_path, quality=95)


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else "noto"
    build(key, os.path.join(HERE, "collection_cover_1080_title.jpg"))
    print("saved", key)
