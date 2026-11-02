"""
第二支（办公室不出汗）抖音封面：cover_9x16.jpg（1080x1920）、cover_3x4.jpg（1080x1440，取 9:16 的 y 240~1680）、
cover_4x3.jpg（1440x1080，横版重新构图：标题在左、老板居中、小燃在右）。取 2.25 秒：老板路过+小燃瞟他+问号气泡。
运行：cd fitness_coach/ep2_office && python3 make_cover.py
"""
import math, os
from PIL import Image, ImageDraw
import render as rd

HERE = os.path.dirname(os.path.abspath(__file__)); W, H = rd.W, rd.H
T = 2.25


def nameplate(dr, cx, ny0):
    label, f36 = "小燃 · AI虚拟教练", rd.font(36); lw_ = dr.textlength(label, font=f36); w_ = lw_ + 116; nx0 = cx - w_ / 2; nh = 62
    dr.rounded_rectangle([nx0, ny0, nx0 + w_, ny0 + nh], radius=31, fill=(255, 248, 232, 255), outline=rd.INK, width=4)
    ix, iy = nx0 + 40, ny0 + nh / 2
    rd.draw_star(dr, ix, iy, 29, 13, 4, fill=rd.STAR); dr.text((ix, iy + 2), "AI", font=rd.font(16), fill=rd.INK, anchor="mm"); dr.text((nx0 + 76, iy + 1), label, font=f36, fill=rd.INK, anchor="lm")


def main():
    im = rd.background().convert("RGBA"); rd.sun_pulse(T, im); rd.boss(im, T)
    im.alpha_composite(rd.draw_coach(rd.pose(T), T, 0.3), (rd.LAY[0], rd.LAY[1]))
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
    rd.stext(dr, (W / 2, 378), "老板路过", 186, rd.YEL, stroke=13)
    rd.stext(dr, (W / 2, 514), "都看不出你在运动", 62, (255, 255, 255), stroke=8)
    nameplate(dr, W / 2, 1352)
    rd.stext(dr, (W / 2, 1478), "办公室3个坐姿小动作 · 量力而行", 46, (255, 255, 255), stroke=8)
    im.alpha_composite(ov); img = im.convert("RGB")
    img.save(os.path.join(HERE, "cover_9x16.jpg"), quality=93); img.crop((0, 240, W, 1680)).save(os.path.join(HERE, "cover_3x4.jpg"), quality=93)
    print("covers saved")


def cover_4x3():
    Wl, Hl = 1440, 1080
    im = Image.new("RGB", (Wl, Hl)); d = ImageDraw.Draw(im)
    for y in range(Hl):
        a, b = ((206, 226, 250), (244, 240, 236)) if y < 700 else ((226, 178, 124), (196, 146, 96))
        k = y / 700 if y < 700 else (y - 700) / 380
        d.line([(0, y), (Wl, y)], fill=tuple(int(a[i] + (b[i] - a[i]) * k) for i in range(3)))
    d.rectangle([0, 690, Wl, 712], fill=(255, 255, 255))
    for k in range(8): d.line([(0, 712 + k * k * 8), (Wl, 712 + k * k * 8)], fill=(190, 140, 94), width=3)
    fy = lambda x: 14 + 70 * (1 - (2 * x / Wl - 1) ** 2)
    d.line([(x, fy(x)) for x in range(-20, Wl + 40, 20)], fill=rd.INK, width=4)
    for i, c in zip(range(1, 14), [(255, 170, 170), (255, 224, 120), (150, 220, 200), (160, 190, 255), (230, 170, 255)] * 3):
        x = i * Wl / 14; y = fy(x); d.polygon([(x - 30, y), (x + 30, y), (x, y + 70)], fill=c, outline=rd.INK)
    wx0, wy0, wx1, wy1 = 960, 200, 1320, 600                                            # 窗外天际线
    d.rounded_rectangle([wx0 - 16, wy0 - 16, wx1 + 16, wy1 + 16], radius=18, fill=(255, 255, 255), outline=rd.INK, width=4)
    win = Image.new("RGB", (wx1 - wx0, wy1 - wy0)); wd = ImageDraw.Draw(win)
    for y in range(wy1 - wy0): wd.line([(0, y), (wx1 - wx0, y)], fill=(int(120 + 100 * y / 400), int(190 + 50 * y / 400), int(248 + 4 * y / 400)))
    rd.cloud(wd, 100, 80, .4)
    for bx, bw, bh, cc in ((20, 80, 200, (150, 170, 208)), (110, 70, 150, (176, 190, 224)), (190, 90, 240, (130, 152, 196)), (290, 60, 170, (168, 184, 220))):
        wd.rectangle([bx, 400 - bh, bx + bw, 400], fill=cc)
        for yy in range(400 - bh + 14, 390, 26): wd.rectangle([bx + 12, yy, bx + 22, yy + 11], fill=(255, 238, 170)); wd.rectangle([bx + bw - 24, yy, bx + bw - 14, yy + 11], fill=(255, 238, 170))
    im.paste(win, (wx0, wy0)); d = ImageDraw.Draw(im); d.rectangle([(wx0 + wx1) // 2 - 6, wy0, (wx0 + wx1) // 2 + 6, wy1], fill=(255, 255, 255))
    im = im.convert("RGBA")
    CXl, GRl = 1130, 960
    old = (rd.CX, rd.GROUND, rd.K, rd.LAY); rd.CX, rd.GROUND, rd.K, rd.LAY = CXl, GRl, 1.55, (0, 0, Wl, Hl)
    try:
        d = ImageDraw.Draw(im); S = rd.S
        d.ellipse([CXl - 300, GRl - 40, CXl + 300, GRl + 50], fill=(150, 168, 214), outline=rd.INK, width=4)                  # 椅垫
        dz0 = int(S(0, rd.DESK_H, 0)[1])                                                                                      # 桌子（右侧，压住窗户下沿）
        d.rounded_rectangle([CXl + 80, dz0, Wl + 40, dz0 + 26], radius=10, fill=(244, 208, 164), outline=rd.INK, width=4)
        d.rounded_rectangle([CXl + 150, dz0 - 96, CXl + 270, dz0 - 8], radius=10, fill=(214, 218, 230), outline=rd.INK, width=4); d.polygon(rd.star_pts(CXl + 210, dz0 - 52, 18, 8), fill=rd.STAR, outline=rd.INK)
        CH, CH2 = (104, 116, 164), (84, 94, 138)
        for ex, ez in ((-52, 0), (52, 0), (-26, -46), (26, 46), (-26, 46)):
            p0, p1 = S(0, 16, 0), S(ex, 10, ez); d.line([p0, p1], fill=rd.INK, width=22); d.line([p0, p1], fill=CH2, width=12)
        a, c = S(0, 62, 0), S(0, 16, 0); d.line([a, c], fill=rd.INK, width=20); d.line([a, c], fill=(176, 180, 200), width=10)
        rd.box3(d, -36, 36, 100, 206, -46, -34, CH); rd.box3(d, -38, 38, 62, 78, -36, 66, CH)
        im.alpha_composite(rd.draw_coach(rd.pose(T), T, 0.3), (0, 0))
    finally: rd.CX, rd.GROUND, rd.K, rd.LAY = old
    cv = Image.new("RGBA", (W, H), (0, 0, 0, 0)); rd.boss(cv, T)                         # 老板：在竖版画布上画好，再裁剪缩放贴到横版
    X0 = rd.boss_x(T); bs = cv.crop((int(X0 - 230), 440, int(X0 + 230), 1110)); bs = bs.resize((int(bs.width * .8), int(bs.height * .8)), Image.LANCZOS)
    im.alpha_composite(bs, (790 - bs.width // 2, 700 - bs.height + 20))
    ov = Image.new("RGBA", (Wl, Hl), (0, 0, 0, 0)); dr = ImageDraw.Draw(ov)
    rd.stext(dr, (370, 250), "老板路过", 160, rd.YEL, stroke=11)
    rd.stext(dr, (370, 385), "都看不出你在运动", 58, (255, 255, 255), stroke=8)
    nameplate(dr, 370, 464); rd.stext(dr, (380, 640), "办公室3个坐姿小动作", 46, (255, 255, 255), stroke=8)
    im.alpha_composite(ov); im.convert("RGB").save(os.path.join(HERE, "cover_4x3.jpg"), quality=93); print("cover_4x3 saved")


if __name__ == "__main__":
    main(); cover_4x3()
