"""
画风对比静帧（镜 1：跑车车灯 + 背小包袱的少年），三种风格，同一构图：
  A 现版（宣纸 + 原创卡通 + 水墨）   B 木刻 + 蜡笔涂鸦混合   E 拼贴绘本（撕纸 + 报纸 + 邮票）
运行：python3 styles_demo.py  ->  styles/A_现版.jpg  styles/BC_木刻加蜡笔.jpg  styles/E_拼贴绘本.jpg
"""
import math, os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import gaussian_filter, map_coordinates
import render as R

W, H = R.W, R.H
ZH = "你见过九十多岁还开跑车的老头吗？他叫黄永玉。"
EN = "Ever met a man in his nineties who still drove a sports car? His name is Huang Yongyu."
GROUND = 800


def subtitle(img, dark=True):
    fz, fe = R.font(R.F_FUN, 52), R.font(R.F_EN, 32)
    bw = max(fz.getlength(ZH), fe.getlength(EN)) + 70
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    y0 = H - 150
    d.rounded_rectangle([(W - bw) / 2, y0, (W + bw) / 2, y0 + 112], radius=22, fill=(20, 16, 12, 130))
    d.text((W / 2, y0 + 38), ZH, font=fz, fill=(255, 252, 240, 255), anchor="mm", stroke_width=3, stroke_fill=(30, 22, 15, 255))
    d.text((W / 2, y0 + 88), EN, font=fe, fill=(235, 228, 210, 255), anchor="mm", stroke_width=2, stroke_fill=(30, 22, 15, 255))
    return Image.alpha_composite(img.convert("RGBA"), lay).convert("RGB")



def P_(pts, cx, base, s=1.0, dx=0, dy=0):
    return [(cx + (x + dx) * s, base + (y + dy) * s) for x, y in pts]


CAR_BODY = [(-270, -30), (-268, -70), (-190, -88), (-95, -100), (-30, -142), (70, -150), (135, -102), (255, -84), (272, -60), (268, -30)]
CAR_WIN = [(-18, -134), (62, -140), (110, -104), (-62, -100)]


def draw_car(d, cx, base, fill=(222, 38, 44, 255), line=(40, 34, 44, 255), win=(160, 214, 242, 255), s=1.0, off=0, lw=8):
    """低矮长车头的跑车：车身 + 车窗 + 尾翼 + 车轮 + 大灯 + 白色赛车条纹"""
    d.polygon(P_(CAR_BODY, cx, base, s, off, off * 0.7), fill=fill)
    d.polygon(P_(CAR_WIN, cx, base, s, off, off * 0.7), fill=win)
    for pts in (CAR_BODY, CAR_WIN):
        d.line(P_(pts + [pts[0]], cx, base, s), fill=line, width=int(lw * s), joint="curve")
    d.line(P_([(-262, -62), (262, -64)], cx, base, s), fill=(250, 246, 236, 255), width=int(9 * s))
    d.polygon(P_([(-292, -108), (-212, -112), (-212, -98), (-292, -94)], cx, base, s), fill=line)      # 尾翼
    d.line(P_([(-262, -98), (-262, -72)], cx, base, s), fill=line, width=int(7 * s))
    for wx in (-165, 165):
        x0, y0 = cx + wx * s, base - 36 * s
        d.ellipse([x0 - 40 * s, y0 - 40 * s, x0 + 40 * s, y0 + 40 * s], fill=(48, 46, 52, 255))
        d.ellipse([x0 - 17 * s, y0 - 17 * s, x0 + 17 * s, y0 + 17 * s], fill=(205, 205, 212, 255))
    hx, hy = cx + 258 * s, base - 72 * s
    d.ellipse([hx - 17 * s, hy - 15 * s, hx + 17 * s, hy + 15 * s], fill=(255, 226, 90, 255), outline=line, width=int(4 * s))


def car_sprite_cartoon():
    S = R.SS
    im = Image.new("RGBA", (640 * S, 250 * S), (0, 0, 0, 0))
    draw_car(ImageDraw.Draw(im), 320 * S, 232 * S, s=S)
    return R.fin(im)


def boy_sprite(mode="color"):
    """背小包袱的少年（朝右走）：mode='color' 彩色，'sil' 纯黑剪影（包袱另用蜡笔画）"""
    S = R.SS
    w, h = 240, 330
    im = Image.new("RGBA", (w * S, h * S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    sil = mode == "sil"
    K = (22, 20, 20, 255)
    skin, tunic, pants, hair, shoe = ((K,) * 5) if sil else ((246, 208, 168, 255), (84, 112, 150, 255), (62, 58, 70, 255), (34, 28, 28, 255), (40, 34, 34, 255))
    line = K
    def L(pts, fill, wd):
        d.line([(x * S, y * S) for x, y in pts], fill=fill, width=int(wd * S), joint="curve")
    # 腿
    L([(100, 210), (86, 300)], pants, 22); L([(124, 210), (150, 296)], pants, 22)
    d.ellipse([60 * S, 292 * S, 108 * S, 314 * S], fill=shoe); d.ellipse([130 * S, 288 * S, 180 * S, 310 * S], fill=shoe)
    # 身体
    d.polygon([(x * S, y * S) for x, y in [(78, 118), (146, 118), (158, 226), (68, 226)]], fill=tunic)
    d.line([(112 * S, 120 * S), (112 * S, 226 * S)], fill=(255, 255, 255, 60) if not sil else K, width=2 * S)
    # 木棍 + 包袱
    L([(120, 160), (228, 34)], (120, 80, 44, 255) if not sil else K, 8)
    if not sil:
        d.ellipse([196 * S, 6 * S, 250 * S, 62 * S], fill=(214, 70, 60, 255), outline=(120, 30, 30, 255), width=3 * S)
        for a, b in ((214, 22), (232, 30), (220, 44)):
            d.ellipse([(a - 4) * S, (b - 4) * S, (a + 4) * S, (b + 4) * S], fill=(250, 240, 230, 255))
        d.polygon([(x * S, y * S) for x, y in [(206, 56), (240, 56), (226, 70)]], fill=(214, 70, 60, 255))
    # 手臂搭在棍上
    L([(138, 130), (172, 96)], skin if not sil else K, 16)
    # 头：圆脸 + 短发 + 耳朵 + 眼睛
    d.ellipse([72 * S, 34 * S, 152 * S, 118 * S], fill=skin)
    d.pieslice([70 * S, 28 * S, 154 * S, 112 * S], 180, 360, fill=hair)
    d.ellipse([68 * S, 70 * S, 84 * S, 92 * S], fill=skin)
    if not sil:
        d.ellipse([124 * S, 76 * S, 134 * S, 90 * S], fill=(30, 24, 24, 255))
        d.arc([118 * S, 88 * S, 146 * S, 108 * S], 20, 150, fill=(150, 80, 60, 255), width=3 * S)
        d.ellipse([132 * S, 90 * S, 146 * S, 100 * S], fill=(244, 160, 150, 160))
    return R.fin(im)


def boy_bundle_pos(cx, cy, s):
    """boy_sprite 里包袱中心 (223,34) 相对精灵中心 (120,165)"""
    return cx + (223 - 120) * s, cy + (34 - 165) * s


def beams(img, x, y, length=520, spread=95, alpha=90, color=(255, 238, 160)):
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    d.polygon([(x, y), (x + length, y - spread), (x + length, y + spread * 0.7)], fill=color + (alpha,))
    lay = lay.filter(ImageFilter.GaussianBlur(6))
    return Image.alpha_composite(img.convert("RGBA"), lay).convert("RGB")


# ===================================================================== A 现版
def style_A():
    fh = R.Fenghuang()
    img = fh.frame(9.0).convert("RGBA")
    d = ImageDraw.Draw(img)
    r = np.random.default_rng(6)
    ridge = GROUND + 6 * np.sin(np.arange(W) / 70) + r.normal(0, 1.5, W).cumsum() * 0.2
    d.polygon([(0, H)] + [(x, ridge[x]) for x in range(0, W, 4)] + [(W, H)], fill=(58, 48, 42, 255))
    img = img.convert("RGB")
    car = car_sprite_cartoon()
    img = beams(img, 800, GROUND - 78, length=380, spread=70)
    R.place(img, R.shadowed(car, off=(8, 10), blur=8, op=0.3), 500, GROUND - 100)
    boy = boy_sprite("color")
    R.place(img, R.shadowed(boy, off=(8, 10), blur=8, op=0.25), 1300, GROUND - 145 * 1.25, scale=1.25)
    R.place(img, R.seal("童心", 96), 1780, 170)
    return subtitle(img)


# ===================================================================== B+C 木刻 + 蜡笔
def crayon_finish(layer, seed=1, wobble=3.0, grain=0.4):
    """给图层加手绘抖动和蜡笔颗粒"""
    r = np.random.default_rng(seed)
    a = np.asarray(layer, np.float32)
    dx = gaussian_filter(r.standard_normal((H, W)), 22) * wobble * 22
    dy = gaussian_filter(r.standard_normal((H, W)), 22) * wobble * 22
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = np.stack([map_coordinates(a[..., c], [yy + dy, xx + dx], order=1, mode="nearest") for c in range(4)], -1)
    g = gaussian_filter(r.random((H, W)), (1.0, 3.5))
    g = (g - g.min()) / (g.max() - g.min())
    out[..., 3] *= (1 - grain) + grain * np.clip((g - 0.25) * 2.2, 0, 1)
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGBA")


def with_halo(layer, pad=14):
    a = layer.getchannel("A").point(lambda v: 255 if v > 40 else 0).filter(ImageFilter.MaxFilter(pad * 2 + 1))
    halo = Image.new("RGBA", layer.size, (250, 246, 236, 0))
    halo.putalpha(a.filter(ImageFilter.GaussianBlur(2)))
    return Image.alpha_composite(halo, layer)


def doodle_car(cx, base):
    lay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(lay)
    draw_car(d, cx, base, fill=(232, 50, 52, 255), line=(40, 34, 44, 255), s=1.0, off=7, lw=10)
    for k in range(4):                                  # 速度线
        d.line([(cx - 300 - k * 18, base - 150 + k * 28), (cx - 440 - k * 30, base - 150 + k * 28)], fill=(40, 34, 44, 255), width=6)
    return lay


def style_BC():
    wc = R.Woodcut()
    ink = 1 - wc.carved
    tex = np.asarray(wc.bg, np.float32) / 255
    out = tex * (1 - ink * 0.92)[..., None]
    img = Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8)).convert("RGBA")
    # 木刻少年（黑色剪影，放大）；包袱稍后用蜡笔上色
    boy = boy_sprite("sil")
    ha = boy.getchannel("A").point(lambda v: 255 if v > 30 else 0).filter(ImageFilter.MaxFilter(15))
    white = Image.new("RGBA", boy.size, (244, 238, 224, 255))
    white.putalpha(ha)
    boy = Image.alpha_composite(white, boy)          # 白边：像木刻刀刻出的轮廓
    bx, by, bs = 1070, 735, 1.5
    R.place(img, boy, bx, by, scale=bs)
    img = img.convert("RGB")
    lay = doodle_car(480, 900)
    d = ImageDraw.Draw(lay)
    for k in range(18):                                 # 车灯光束：蜡笔黄排线
        d.line([(780, 828 + (k - 9) * 1.2), (1020, 700 + k * 11)], fill=(255, 226, 90, 255), width=6)
    hx, hy = boy_bundle_pos(bx, by, bs)
    d.ellipse([hx - 46, hy - 42, hx + 46, hy + 42], fill=(250, 150, 40, 255), outline=(60, 40, 30, 255), width=6)
    for dx, dy in ((-20, -8), (12, -20), (18, 14), (-8, 18)):
        d.ellipse([hx + dx - 8, hy + dy - 8, hx + dx + 8, hy + dy + 8], fill=(220, 40, 50, 255))
    # 星星
    for sx, sy, s in ((300, 160, 34), (620, 90, 26), (1000, 200, 30), (1500, 120, 36), (1740, 330, 26)):
        pts = [(sx + (s if k % 2 == 0 else s * 0.42) * math.sin(k * math.pi / 5),
                sy - (s if k % 2 == 0 else s * 0.42) * math.cos(k * math.pi / 5)) for k in range(10)]
        d.polygon(pts, fill=(255, 220, 70, 255), outline=(60, 40, 30, 255))
    # 片名：白色毛笔字 + 蜡笔下划线
    d.text((330, 330), "黄永玉", font=R.font(R.F_BRUSH, 150), fill=(250, 246, 236, 255), anchor="mm")
    d.line([(150, 440), (520, 436), (500, 450), (160, 452)], fill=(232, 50, 52, 255), width=10)
    lay = crayon_finish(lay, seed=3, wobble=2.2, grain=0.35)
    img = Image.alpha_composite(img.convert("RGBA"), with_halo(lay, 6)).convert("RGB")
    R.place(img, R.seal("童心", 96), 1780, 960)
    return subtitle(img)


# ===================================================================== E 拼贴绘本
def torn(color, ridge, seed):
    r = np.random.default_rng(seed)
    yy = np.arange(H)[:, None]
    dd = yy - ridge[None, :]
    nz = 1 + 0.08 * R.noise2d(H, W, 40, seed) + 0.04 * R.noise2d(H, W, 4, seed + 1)
    border = np.zeros((H, W, 4), np.uint8)
    border[..., :3] = (247, 243, 232)
    border[..., 3] = (np.clip(dd + 11, 0, 1) * 255).astype(np.uint8)
    fill = np.zeros((H, W, 4), np.uint8)
    fill[..., :3] = np.clip(np.array(color, np.float32)[None, None] * nz[..., None], 0, 255)
    fill[..., 3] = (np.clip(dd, 0, 1) * 255).astype(np.uint8)
    sh = np.zeros((H, W, 4), np.uint8)
    sh[..., :3] = (40, 30, 20)
    sh[..., 3] = (np.clip((dd + 26) / 26, 0, 1) * 70).astype(np.uint8)
    base = Image.alpha_composite(Image.fromarray(sh, "RGBA").filter(ImageFilter.GaussianBlur(7)), Image.fromarray(border, "RGBA"))
    return Image.alpha_composite(base, Image.fromarray(fill, "RGBA"))


def ridge_line(base, amp, scale, seed, rough=5):
    return base - amp * (0.5 + 0.5 * R.fbm1d(W, scale, 4, seed)) + rough * R.fbm1d(W, 22, 3, seed + 5)


def cutout(spr, border=10):
    a = spr.getchannel("A").point(lambda v: 255 if v > 30 else 0).filter(ImageFilter.MaxFilter(border * 2 + 1))
    white = Image.new("RGBA", spr.size, (248, 244, 236, 0))
    white.putalpha(a)
    return R.shadowed(Image.alpha_composite(white, spr), off=(9, 13), blur=8, op=0.42)


def newspaper():
    w, h = 520, 340
    im = R.paper(w, h, 88, base=(228, 220, 198), vig=False).convert("RGBA")
    d = ImageDraw.Draw(im)
    d.text((w / 2, 40), "湘 西 日 报", font=R.font(R.F_BRUSH, 46), fill=(50, 45, 40, 255), anchor="mm")
    d.line([(20, 72), (w - 20, 72)], fill=(60, 55, 50, 255), width=3)
    r = np.random.default_rng(5)
    chars = "天地人山水日月风云花鸟鱼虫木石江河湖海春夏秋冬东西南北"
    f = R.font(R.F_FUN, 15)
    for row in range(14):
        for col in range(3):
            x0 = 24 + col * 160
            txt = "".join(r.choice(list(chars), 9))
            d.text((x0, 86 + row * 17), txt, font=f, fill=(90, 84, 76, 255))
    d.rectangle([24, 250, 170, 322], fill=(196, 190, 170, 255))
    d.ellipse([60, 258, 130, 312], fill=(110, 104, 94, 255))
    return im


def stamp_small():
    im, d = R.perforated(170, 210, (250, 246, 238, 255))
    S = R.SS
    d.rectangle([22 * S, 22 * S, 148 * S, 188 * S], fill=(196, 30, 36, 255))
    d.text((85 * S, 105 * S), "邮", font=R.font(R.F_BRUSH, 90 * S), fill=(255, 225, 120, 255), anchor="mm")
    return R.fin(im)


def style_E():
    img = R.paper(W, H, 91, base=(206, 172, 128)).convert("RGBA")
    for col, base, amp, sc, sd in (((236, 228, 206), 150, 0, 900, 11),           # 淡奶油天空
                                    ((156, 166, 164), 560, 150, 620, 21),          # 远山：灰绿纸
                                    ((108, 132, 130), 600, 90, 380, 31),           # 近山：青灰纸
                                    ((86, 146, 168), 700, 20, 260, 41),            # 河：蓝纸
                                    ((52, 40, 36), GROUND, 8, 300, 51)):           # 河岸：深褐纸
        img = Image.alpha_composite(img, torn(col, ridge_line(base, amp, sc, sd), sd))
    img = img.convert("RGB")
    # 河面上的撕纸小浪
    d = ImageDraw.Draw(img, "RGBA")
    r = np.random.default_rng(12)
    for _ in range(26):
        x, y = r.random() * W, 730 + r.random() * 60
        d.line([(x, y), (x + 40 + r.random() * 50, y)], fill=(236, 244, 246, 200), width=5)
    # 太阳：红纸圆
    sun = Image.new("RGBA", (220, 220), (0, 0, 0, 0))
    ImageDraw.Draw(sun).ellipse([10, 10, 210, 210], fill=(214, 70, 52, 255))
    R.place(img, cutout(sun, 8), 1500, 260)
    # 报纸、邮票、片名
    R.place(img, cutout(newspaper(), 6), 960, 170, angle=-4, scale=0.62)
    R.place(img, cutout(stamp_small(), 6), 1700, 120, angle=9)
    title = Image.new("RGBA", (520, 190), (0, 0, 0, 0))
    td = ImageDraw.Draw(title)
    td.rectangle([0, 0, 520, 190], fill=(250, 246, 236, 255))
    td.text((260, 80), "黄永玉", font=R.font(R.F_BRUSH, 120), fill=R.INK + (255,), anchor="mm")
    td.line([(60, 150), (460, 146)], fill=R.RED + (255,), width=8)
    R.place(img, cutout(title, 6), 400, 190, angle=-3)
    # 跑车（红纸剪贴）+ 车灯光束（黄色半透明纸）
    img = beams(img, 760, GROUND - 78, alpha=110, color=(255, 226, 110))
    R.place(img, cutout(car_sprite_cartoon(), 11), 500, GROUND - 100)
    boy = boy_sprite("color")
    boy = boy.resize((int(boy.width * 1.25), int(boy.height * 1.25)), Image.BICUBIC)
    R.place(img, cutout(boy, 9), 1300, GROUND - 145 * 1.25)
    return subtitle(img)


if __name__ == "__main__":
    os.chdir(R.HERE)
    os.makedirs("styles", exist_ok=True)
    for name, fn in (("A_现版", style_A), ("BC_木刻加蜡笔", style_BC), ("E_拼贴绘本", style_E)):
        fn().save(f"styles/{name}.jpg", quality=92)
        print("->", f"styles/{name}.jpg")
