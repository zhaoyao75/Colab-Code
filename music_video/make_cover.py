"""
封面：取高潮段（约 100 秒）的雨夜城市原分辨率画面，去掉视频里的小标题和标志，叠加大标题。
运行：python3 make_cover.py             -> cover_16x9.jpg（1920x1080，B 站）+ cover_4x3.jpg（1440x1080）
     python3 make_cover.py --portrait  -> cover_9x16.jpg（1080x1920，抖音/视频号）+ cover_3x4.jpg（1080x1440）
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import render as rv   # 读取同一套场景（--portrait 参数由 render 模块解析）

T = 100.0


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
        e = R / 9
        dr.ellipse([x2 - e, y2 - e, x2 + e, y2 + e], fill=orange)
    dr.text((cx + R * 1.6, cy - size * 0.62), "Claude", font=ImageFont.truetype(rv.EN, size),
            fill=(240, 236, 230, 255))


def compose(base, w, h, ty, mark_xy):
    """base: 裁好尺寸的底图；ty: 标题的垂直位置（占画面高度的比例）"""
    s = min(w, h) / 1080
    arr = np.asarray(base).astype(np.float32)
    yy = np.linspace(0, 1, h)[:, None, None]
    arr *= 1 - 0.78 * np.exp(-((yy - ty - 0.13 * 1080 / h) / (0.2 * 1080 / h)) ** 2)    # 标题区域压暗
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGBA")
    zh = ImageFont.truetype(rv.ZH, int(190 * s))
    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    spaced(ImageDraw.Draw(glow), "霓虹雨夜", zh, w / 2, h * ty, int(40 * s), (255, 60, 160, 255))
    glow = glow.filter(ImageFilter.GaussianBlur(16 * s))
    out.alpha_composite(glow)
    out.alpha_composite(glow)
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(layer)
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    en = ImageFont.truetype(rv.EN, int(44 * s))
    sub = ImageFont.truetype(rv.ZH, int(42 * s))
    for d_, dark in ((sd, True), (dr, False)):   # 先画一层模糊黑影，再画文字
        spaced(d_, "霓虹雨夜", zh, w / 2, h * ty, int(40 * s), (0, 0, 0, 160) if dark else (255, 240, 248, 255))
        spaced(d_, "NEON RAIN", en, w / 2, h * ty + 235 * s, int(18 * s),
               (0, 0, 0, 255) if dark else (185, 225, 255, 255))
        spaced(d_, "Lo-fi 原创 · 旋律与画面全部由代码生成", sub, w / 2, h * ty + 320 * s, int(4 * s),
               (0, 0, 0, 255) if dark else (235, 228, 242, 255))
    shadow = shadow.filter(ImageFilter.GaussianBlur(6 * s))
    out.alpha_composite(shadow)
    out.alpha_composite(shadow)
    claude_mark(dr, mark_xy[0], mark_xy[1], 26 * s, int(40 * s))
    out.alpha_composite(layer)
    return out.convert("RGB")


def main():
    rv.init()
    rv.SC["logo"][..., 3] = 0      # 去掉视频里的小标志，封面上另画大的
    base = Image.fromarray(rv.frame(int(T * rv.FPS)))
    W, H = rv.W, rv.H
    if rv.PORTRAIT:
        compose(base, W, H, 0.12, (90, 120)).save("cover_9x16.jpg", quality=93)
        crop = base.crop((0, H - 1440 - 160, W, H - 160))
        compose(crop, 1080, 1440, 0.12, (90, 80)).save("cover_3x4.jpg", quality=93)
    else:
        # 右上角放标志，避开 B 站右下角的时长角标
        compose(base, W, H, 0.12, (W - 250, 60)).save("cover_16x9.jpg", quality=93)
        crop = base.crop(((W - 1440) // 2, 0, (W + 1440) // 2, H))
        compose(crop, 1440, 1080, 0.12, (1440 - 250, 60)).save("cover_4x3.jpg", quality=93)
    print("covers saved")


if __name__ == "__main__":
    main()
