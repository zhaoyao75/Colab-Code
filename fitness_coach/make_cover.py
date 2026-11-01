"""
抖音封面：cover_9x16.jpg（1080x1920，全屏）、cover_3x4.jpg（1080x1440，个人主页格子，取 9:16 的 y 240~1680）、
cover_4x3.jpg（1440x1080，网页端要求的横封面，专门按横版重新构图：标题在左、角色在右）。
文字都放在 3:4 范围内。复用 render.py 的背景和星星头角色（挥手姿势）。
运行：cd fitness_coach && python3 make_cover.py
"""
import math
import os
from PIL import Image, ImageDraw
import render as rd

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = rd.W, rd.H


def main(t=1.0):
    im = rd.background().convert("RGBA")
    rd.sun_pulse(t, im)
    im.alpha_composite(rd.draw_coach(rd.pose(t), t, 0.3), (rd.LAY[0], rd.LAY[1]))      # 挥手 + 张嘴打招呼
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
    rd.stext(dr, (W / 2, 342), "16秒挑战", 186, rd.YEL, stroke=13)
    rd.stext(dr, (W / 2, 488), "久坐的你，敢接吗？", 56, (255, 255, 255), stroke=8)
    label, f36 = "小燃 · AI虚拟教练", rd.font(36); lw_ = dr.textlength(label, font=f36); w_ = lw_ + 116; nx0 = W / 2 - w_ / 2; ny0, nh = 1352, 62      # 与成片一致的 AI 名牌
    dr.rounded_rectangle([nx0, ny0, nx0 + w_, ny0 + nh], radius=31, fill=(255, 248, 232, 255), outline=rd.INK, width=4)
    ix, iy = nx0 + 40, ny0 + nh / 2
    rd.draw_star(dr, ix, iy, 29, 13, 4, fill=rd.STAR); dr.text((ix, iy + 2), "AI", font=rd.font(16), fill=rd.INK, anchor="mm")
    dr.text((nx0 + 76, iy + 1), label, font=f36, fill=rd.INK, anchor="lm")
    rd.stext(dr, (W / 2, 1478), "居家小挑战 · 量力而行", 52, (255, 255, 255), stroke=8)
    im.alpha_composite(ov)
    img = im.convert("RGB")
    img.save(os.path.join(HERE, "cover_9x16.jpg"), quality=93)
    img.crop((0, 240, W, 1680)).save(os.path.join(HERE, "cover_3x4.jpg"), quality=93)
    print("covers saved")



def cover_4x3(t=1.0):
    """横封面 1440x1080：自己画横版居家客厅背景；角色用 render.draw_coach（临时改全局：位置/缩放/图层范围）。"""
    Wl, Hl = 1440, 1080
    im = Image.new("RGB", (Wl, Hl)); d = ImageDraw.Draw(im)
    for y in range(Hl):                                                   # 墙面 / 木地板渐变
        a, b = ((255, 244, 228), (252, 232, 208)) if y < 700 else ((226, 178, 124), (196, 146, 96))
        k = y / 700 if y < 700 else (y - 700) / 380
        d.line([(0, y), (Wl, y)], fill=tuple(int(a[i] + (b[i] - a[i]) * k) for i in range(3)))
    d.rectangle([0, 690, Wl, 712], fill=(255, 255, 255))
    for k in range(0, 8): d.line([(0, 712 + k * k * 8), (Wl, 712 + k * k * 8)], fill=(196, 146, 100), width=3)
    im = im.convert("RGBA")
    g = Image.new("RGBA", (Wl, Hl), (0, 0, 0, 0)); gd = ImageDraw.Draw(g)
    for x in range(0, Wl, 90): gd.rectangle([x, 0, x + 45, 690], fill=(255, 255, 255, 46))
    im.alpha_composite(g); d = ImageDraw.Draw(im)
    fy = lambda x: 14 + 70 * (1 - (2 * x / Wl - 1) ** 2)                 # 彩旗
    d.line([(x, fy(x)) for x in range(-20, Wl + 40, 20)], fill=rd.INK, width=4)
    for i, c in zip(range(1, 14), [(255, 170, 170), (255, 224, 120), (150, 220, 200), (160, 190, 255), (230, 170, 255)] * 3):
        x = i * Wl / 14; y = fy(x); d.polygon([(x - 30, y), (x + 30, y), (x, y + 70)], fill=c, outline=rd.INK)
    wx0, wy0, wx1, wy1 = 770, 190, 1230, 650                              # 窗户（角色在它前面）
    d.rounded_rectangle([wx0 - 20, wy0 - 20, wx1 + 20, wy1 + 20], radius=22, fill=(255, 255, 255), outline=rd.INK, width=4)
    win = Image.new("RGB", (wx1 - wx0, wy1 - wy0)); wd = ImageDraw.Draw(win)
    for y in range(wy1 - wy0): wd.line([(0, y), (wx1 - wx0, y)], fill=(int(110 + 104 * y / 460), int(182 + 56 * y / 460), int(246 + 6 * y / 460)))
    rd.cloud(wd, 110, 100, .55); rd.cloud(wd, 290, 190, .4); wd.ellipse([-100, 290, 300, 560], fill=(150, 214, 128)); wd.ellipse([120, 320, 520, 590], fill=(112, 190, 102))
    im.paste(win, (wx0, wy0)); d = ImageDraw.Draw(im)
    cx_, cy_ = (wx0 + wx1) // 2, (wy0 + wy1) // 2
    d.rectangle([cx_ - 7, wy0, cx_ + 7, wy1], fill=(255, 255, 255)); d.rectangle([wx0, cy_ - 7, wx1, cy_ + 7], fill=(255, 255, 255))
    d.rounded_rectangle([wx0 - 34, wy1 + 12, wx1 + 34, wy1 + 36], radius=10, fill=(255, 255, 255), outline=rd.INK, width=3)
    d.rounded_rectangle([40, 560, 150, 690], radius=16, fill=(255, 170, 160), outline=rd.INK, width=4)           # 绿植
    for ang, ln in ((50, 120), (80, 150), (110, 135), (135, 110), (20, 95), (155, 90)): rd.leaf(d, 95, 560, ln, ang, (96, 186, 100, 255), .34)
    CXl, GRl = 1060, 950
    d.ellipse([CXl - 300, GRl - 36, CXl + 300, GRl + 46], fill=(255, 183, 196), outline=rd.INK, width=4)           # 粉色瑜伽垫
    old = (rd.CX, rd.GROUND, rd.K, rd.LAY)
    rd.CX, rd.GROUND, rd.K, rd.LAY = CXl, GRl, 1.5, (0, 0, Wl, Hl)
    try: im.alpha_composite(rd.draw_coach(rd.pose(t), t, 0.3), (0, 0))
    finally: rd.CX, rd.GROUND, rd.K, rd.LAY = old
    ov = Image.new("RGBA", (Wl, Hl), (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
    rd.stext(dr, (400, 252), "16秒挑战", 158, rd.YEL, stroke=11)
    rd.stext(dr, (400, 388), "久坐的你，敢接吗？", 58, (255, 255, 255), stroke=8)
    label, f36 = "小燃 · AI虚拟教练", rd.font(36); lw_ = dr.textlength(label, font=f36); w_ = lw_ + 116; nx0 = 410 - w_ / 2; ny0, nh = 464, 62
    dr.rounded_rectangle([nx0, ny0, nx0 + w_, ny0 + nh], radius=31, fill=(255, 248, 232, 255), outline=rd.INK, width=4)
    ix, iy = nx0 + 40, ny0 + nh / 2
    rd.draw_star(dr, ix, iy, 29, 13, 4, fill=rd.STAR); dr.text((ix, iy + 2), "AI", font=rd.font(16), fill=rd.INK, anchor="mm"); dr.text((nx0 + 76, iy + 1), label, font=f36, fill=rd.INK, anchor="lm")
    rd.stext(dr, (440, 640), "居家小挑战 · 量力而行", 48, (255, 255, 255), stroke=8)
    im.alpha_composite(ov)
    im.convert("RGB").save(os.path.join(HERE, "cover_4x3.jpg"), quality=93)
    print("cover_4x3 saved")


if __name__ == "__main__":
    main()
    cover_4x3()
