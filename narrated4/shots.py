"""《后悔按钮》v2：有人物、有故事的 5 镜头分镜。黑白，1080x1920 竖版，16 秒 @30fps。
① 深夜房间（0–3.5）→ ② 手指悬停（3.5–7）→ ③ 倒带闪回（7–8.2）→ 站台雨夜 ④（8.2–11.9）→ ⑤ 回到房间（12.1–16）
用法：python3 shots.py --still 1 3 5 6.9 7.3 8.7 10 11.2 13 14.6
时间点集中在下面的 T_*。
"""
import json, sys, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS, DUR = 1080, 1920, 30, 16.0
SUB_Y = 1540
T_S2, T_PRESS, T_FB, T_S4, T_NUM, T_DIM, T_S5 = 3.4, 6.95, 7.12, 8.2, 10.5, 10.6, 12.1

FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
F_SUB = ImageFont.truetype(FONT, 58)
F_CAL = ImageFont.truetype(FONT, 190)
F_YEAR = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 96)
F_HUD = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 90)
F_SM = ImageFont.truetype(FONT, 44)
CUES = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "timings.json"), encoding="utf-8"))

Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
VIG = (1 - 0.5 * (np.hypot((X - W / 2) / (W / 2), (Y - H / 2) / (H / 2)) ** 2)).clip(0, 1)
rng0 = np.random.default_rng(3)
DROPS = rng0.uniform(0, 1, (320, 4))           # x, y0, 长度, 速度


def ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def glow(cx, cy, r, amp):
    return amp * np.exp(-np.hypot(X - cx, Y - cy) / r).astype(np.float32)


def to_img(a): return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "L")


def soft(img, r=0.9): return np.asarray(img.filter(ImageFilter.GaussianBlur(r))).astype(np.float32)


def zoom(a, z, cx, cy):
    if z <= 1.001: return a
    w, h = W / z, H / z
    x0 = min(max(cx - w / 2, 0), W - w); y0 = min(max(cy - h / 2, 0), H - h)
    return np.asarray(to_img(a).resize((W, H), Image.BICUBIC, box=(x0, y0, x0 + w, y0 + h))).astype(np.float32)


def rain(t, alpha, slant=0.25, val=190):
    im = Image.new("L", (W, H), 0); d = ImageDraw.Draw(im)
    for x, y0, ln, sp in DROPS:
        yy = ((y0 + t * (0.9 + sp * 0.8)) % 1.0) * (H + 200) - 100
        xx = x * (W + 300) - 150 - yy * slant * 0.0
        L = 40 + ln * 70
        d.line([(xx, yy), (xx - L * slant, yy - L)], fill=int(val * (0.35 + sp * 0.65)), width=2)
    return np.asarray(im).astype(np.float32) * alpha


# ============ ① ⑤ 房间 ============
def room(t, lit, hx=0, hy=0, face=1, win_rain=0.0):
    base = np.full((H, W), 10, np.float32)
    base += glow(265, 540, 520, 30)                       # 窗外月光铺在墙上
    base += glow(800, 1190, 240, 95 * lit)                # 按钮的光
    im = Image.new("L", (W, H), 0); d = ImageDraw.Draw(im)
    d.rectangle([110, 260, 420, 820], fill=52)            # 窗
    for y in range(284, 820, 46): d.rectangle([110, y, 420, y + 17], fill=16)
    d.rectangle([258, 260, 272, 820], fill=70); d.rectangle([110, 260, 420, 820], outline=80, width=6)
    d.rectangle([0, 1185, 600, 1465], fill=22); d.line([(0, 1185), (600, 1185)], fill=62, width=3)   # 床
    d.rectangle([0, 1465, W, H], fill=6)                                                              # 地板
    d.rectangle([660, 1215, 1080, 1245], fill=26); d.line([(660, 1215), (1080, 1215)], fill=75, width=3)  # 桌
    d.rectangle([700, 1245, 730, 1465], fill=18); d.rectangle([1010, 1245, 1040, 1465], fill=18)
    layer = np.asarray(im).astype(np.float32)
    m = layer > 0
    out = np.where(m, layer, base)
    out += np.where(m, 0, 0)
    if win_rain > 0:
        rr = rain(t * 0.8, win_rain * 0.7, 0.2, 150)
        wm = np.zeros((H, W), bool); wm[270:810, 118:412] = True
        out = np.where(wm, np.maximum(out, rr * 0.8), out)
    out += glow(800, 1190, 240, 60 * lit) * (1 - m * 0)    # 按钮光再叠一次（家具也被照亮）
    # 按钮
    im = Image.fromarray(out.clip(0, 255).astype(np.uint8), "L"); d = ImageDraw.Draw(im)
    d.ellipse([752, 1200, 848, 1226], fill=44)
    d.rectangle([752, 1212, 848, 1220], fill=44); d.ellipse([752, 1198, 848, 1224], fill=40)
    top = int(40 + 215 * lit)
    d.ellipse([760, 1183, 840, 1207], fill=top)
    # 人物：侧面剪影，先画偏移的亮边，再画黑色本体
    def person(ox, oy, fill):
        h_x, h_y = hx, hy
        d.polygon([(280 + ox, 1180 + oy), (298 + ox, 1010 + oy), (360 + ox + h_x * .4, 955 + oy + h_y * .6),
                   (425 + ox + h_x * .4, 985 + oy + h_y * .6), (402 + ox, 1070 + oy), (385 + ox, 1180 + oy)], fill=fill)
        d.polygon([(372 + ox + h_x * .5, 945 + oy + h_y), (412 + ox + h_x * .5, 945 + oy + h_y),
                   (418 + ox, 985 + oy), (376 + ox, 995 + oy)], fill=fill)
        cx, cy = 400 + h_x, 900 + h_y
        d.ellipse([cx - 46 + ox, cy - 46 + oy, cx + 46 + ox, cy + 46 + oy], fill=fill)
        d.polygon([(cx + face * 40 + ox, cy - 8 + oy), (cx + face * 62 + ox, cy + 14 + oy), (cx + face * 38 + ox, cy + 22 + oy)], fill=fill)
        d.polygon([(290 + ox, 1135 + oy), (520 + ox, 1122 + oy), (526 + ox, 1180 + oy), (300 + ox, 1190 + oy)], fill=fill)
        d.polygon([(490 + ox, 1125 + oy), (545 + ox, 1125 + oy), (558 + ox, 1405 + oy), (505 + ox, 1405 + oy)], fill=fill)
        d.polygon([(500 + ox, 1385 + oy), (610 + ox, 1385 + oy), (610 + ox, 1418 + oy), (500 + ox, 1418 + oy)], fill=fill)
        d.line([(385 + ox, 1010 + oy), (445 + ox, 1105 + oy), (535 + ox, 1118 + oy)], fill=fill, width=36)
    person(4, -3, 58 + int(50 * lit)); person(0, 0, 3)
    return soft(im, 0.9)


def s1(t):                                      # ① 深夜，缓推向按钮
    lit = 0.8 + 0.2 * np.sin(t * 2.4)
    a = room(t, lit, hx=18, hy=26, face=1)
    return zoom(a, 1 + 0.42 * ss(0, 3.7, t), 700, 1130)


def s5(t):                                      # ⑤ 回到现在：按钮熄灭，抬头看窗外
    up = ss(13.2, 14.4, t)
    a = room(t, 0.0, hx=18 - 30 * up, hy=26 - 40 * up, face=1 - 2 * up, win_rain=0.9 * (1 - ss(14.6, 15.4, t)))
    return zoom(a, 1.0 + 0.1 * ss(12.1, 16, t), 480, 1000)


# ============ ② 手指悬停 ============
def s2(t):
    breath = 0.85 + 0.15 * np.sin(t * 3)
    pr = ss(T_PRESS - 0.05, T_PRESS + 0.08, t)
    base = np.full((H, W), 8, np.float32)
    base += glow(540, 1020, 420, 95 * breath * (1 - 0.3 * pr))
    im = Image.new("L", (W, H), 0); d = ImageDraw.Draw(im)
    d.rectangle([0, 1130, W, H], fill=14); d.line([(0, 1130), (W, 1130)], fill=70, width=3)       # 桌面
    # 日历页（翻动）
    flip = (t * 2.6) % 1.0
    sq = abs(np.cos(flip * np.pi)) * 0.9 + 0.1
    cw = 300 * sq
    d.rectangle([250 - cw * .6, 260, 250 + cw * .6, 520], outline=80, width=4, fill=18)
    if cw > 80:
        d.text((250, 395), f"{int(11 - (t * 2.6) % 9 if t < 7 else 3)}", font=F_CAL, anchor="mm", fill=int(95 * min(1, sq * 1.4)))
    # 按钮：圆柱体
    dep = 26 * pr
    d.ellipse([320, 1075 + 20, 760, 1075 + 150], fill=34)
    d.rectangle([320, 1075 + 80, 760, 1075 + 100], fill=34)
    d.ellipse([320, 1060 + dep, 760, 1060 + 150 - 0 + dep - 40], fill=0)
    top = int(235 - 70 * pr)
    d.ellipse([336, 1000 + dep, 744, 1120 + dep], fill=top)
    # 手指（被按钮光从下方照亮）
    tx = 548 + 2.5 * np.sin(t * 31) * (1 - pr); ty = 760 + 120 * pr + 4 * np.sin(t * 23) * (1 - pr) + 90 * ss(5.4, 6.9, t) - 90
    def finger(ox, oy, fill):
        d.rounded_rectangle([tx - 46 + ox, ty - 560 + oy, tx + 46 + ox, ty + oy], 46, fill=fill)       # 食指
        d.rounded_rectangle([tx - 60 + ox, ty - 700 + oy, tx + 250 + ox, ty - 330 + oy], 70, fill=fill)  # 手背
        d.rounded_rectangle([tx + 36 + ox, ty - 420 + oy, tx + 118 + ox, ty - 120 + oy], 40, fill=fill)  # 蜷起的中指
        d.rounded_rectangle([tx + 112 + ox, ty - 430 + oy, tx + 188 + ox, ty - 170 + oy], 38, fill=fill)
    finger(0, 6, 150); finger(0, 0, 14)
    a = soft(im, 1.1)
    out = np.where(a > 0, a, base) + glow(548, 1000, 160, 40 * pr)
    # 手指的影子投在按钮上
    return out


# ============ ③ 倒带闪回 ============
def icon(k, t):
    im = Image.new("L", (W, H), 0); d = ImageDraw.Draw(im)
    d.ellipse([180, 560, 900, 1280], outline=60, width=4)
    if k == 0:                                                      # 学士帽
        d.polygon([(540, 780), (790, 880), (540, 980), (290, 880)], fill=235)
        d.polygon([(400, 940), (400, 1040), (540, 1090), (680, 1040), (680, 940), (540, 1000)], fill=170)
        d.line([(760, 895), (760, 1040)], fill=235, width=6); d.ellipse([746, 1030, 774, 1062], fill=235)
    elif k == 1:                                                    # 半掩的门
        d.polygon([(560, 640), (760, 640), (760, 1190), (560, 1190)], fill=230)    # 门缝里的光
        d.polygon([(400, 660), (560, 640), (560, 1190), (400, 1170)], fill=90)
        d.rectangle([380, 640, 400, 1190], fill=150)
        d.ellipse([515, 900, 545, 930], fill=250)
    elif k == 2:                                                    # 没发出去的消息
        d.rounded_rectangle([390, 620, 690, 1200], 40, outline=210, width=8)
        d.rounded_rectangle([425, 840, 655, 940], 30, fill=235)
        d.polygon([(455, 930), (445, 975), (500, 935)], fill=235)
        for i in range(3): d.ellipse([480 + i * 50, 880, 504 + i * 50, 904], fill=20)
        d.text((540, 1050), "未发送", font=F_SM, anchor="mm", fill=200)
    else:                                                           # 车票
        d.rounded_rectangle([300, 780, 780, 1020], 24, fill=225)
        d.ellipse([280, 880, 320, 920], fill=0); d.ellipse([760, 880, 800, 920], fill=0)
        for yy in range(800, 1010, 24): d.line([(610, yy), (610, yy + 12)], fill=30, width=4)
        d.text((455, 900), "末班车", font=F_SM, anchor="mm", fill=30)
    return soft(im, 0.8)


def s3(t):
    u = t - T_FB
    k = min(3, int(u / 0.27))
    a = icon(k, t)
    # 倒带：错位的扫描条 + 年份倒着数
    rng = np.random.default_rng(int(t * 60))
    for _ in range(5):
        y0 = int(rng.integers(0, H - 60)); h = int(rng.integers(8, 50)); dx = int(rng.integers(-70, 70))
        a[y0:y0 + h] = np.roll(a[y0:y0 + h], dx, axis=1)
    im = Image.new("L", (W, H), 0); d = ImageDraw.Draw(im)
    d.text((540, 1420), str(int(2026 - 12 * ss(0, 1.0, u))), font=F_YEAR, anchor="mm", fill=210)
    d.text((540, 1340), "⏪", font=F_SM, anchor="mm", fill=0)
    return np.maximum(a, soft(im, 0.8)) + glow(540, 920, 400, 18)


# ============ ④ 站台雨夜 ============
def stand(d, x, facing, fill, reach=0.0, away=False, ox=0, oy=0):
    x += ox
    b = 1400 + oy
    d.rectangle([x - 28 + 0, b - 220, x - 4, b], fill=fill); d.rectangle([x + 4, b - 220, x + 28, b], fill=fill)
    d.polygon([(x - 50, b - 520), (x + 50, b - 520), (x + 68, b - 170), (x - 68, b - 170)], fill=fill)
    d.ellipse([x - 56, b - 540, x + 56, b - 500], fill=fill)
    hx = x + facing * (6 if not away else -10)
    d.ellipse([hx - 40, b - 600, hx + 40, b - 520], fill=fill)
    sh = (x + facing * 34, b - 500)
    if reach > 0:
        hand = (x + facing * (50 + reach), b - 440 - reach * 0.04)
        el = ((sh[0] + hand[0]) / 2, (sh[1] + hand[1]) / 2 + 40)
        d.line([sh, el, hand], fill=fill, width=30, joint="curve")
        d.ellipse([hand[0] - 20, hand[1] - 20, hand[0] + 20, hand[1] + 20], fill=fill)
    else:
        d.line([sh, (sh[0] + facing * 6, b - 360)], fill=fill, width=30)


def s4(t):
    lamp = 1.0 - 0.65 * ss(T_DIM, T_DIM + 0.5, t) + 0.04 * np.sin(t * 40) * (t > 9.2)
    rain_a = 1.0 - ss(T_NUM, T_NUM + 0.4, t)
    base = np.full((H, W), 9, np.float32)
    hw = (Y - 500) * 0.55 + 60
    cone = np.where(Y > 500, np.exp(-((X - 860) / hw) ** 4) * np.exp(-(Y - 500) / 1300), 0).astype(np.float32)
    base += cone * 70 * lamp + glow(860, 500, 110, 110 * lamp)
    im = Image.new("L", (W, H), 0); d = ImageDraw.Draw(im)
    d.rectangle([0, 1400, W, H], fill=20); d.rectangle([0, 1488, W, 1500], fill=int(140 * (0.5 + 0.5 * lamp)))
    d.rectangle([0, 1500, W, H], fill=3)
    d.rectangle([852, 505, 868, 1400], fill=60); d.ellipse([834, 478, 886, 520], fill=int(255 * lamp))
    layer = np.asarray(im).astype(np.float32)
    out = np.where(layer > 0, np.maximum(layer, base * (layer > 0) * 0 + layer), base)
    out = np.where(layer > 0, layer + cone * 40 * lamp * (Y < 1500), out)
    out = out + rain(t, 0.55 * rain_a, 0.2)
    im = Image.fromarray(out.clip(0, 255).astype(np.uint8), "L"); d = ImageDraw.Draw(im)
    reach = 175 * ss(T_S4 + 0.8, T_NUM, t)
    drift = 60 * ss(T_NUM - 0.2, T_NUM + 1.6, t)
    def figs(ox, oy, fill):
        stand(d, 470, 1, fill, reach, False, ox, oy)
        stand(d, 790 + drift, -1, fill, 0, drift > 6, ox, oy)
    figs(3, -2, 70); figs(0, 0, 2)
    a = soft(im, 0.9)
    # 按钮的次数 HUD
    hud = Image.new("L", (W, H), 0); h = ImageDraw.Draw(hud)
    one = 1 - ss(T_NUM - 0.1, T_NUM + 0.2, t); zero = ss(T_NUM + 0.05, T_NUM + 0.4, t)
    ha = ss(T_S4 + 0.3, T_S4 + 0.8, t)
    r = 80
    h.ellipse([540 - r, 300 - r, 540 + r, 300 + r], outline=int(220 - 140 * zero), width=4)
    if one > 0.01: h.text((540, 296), "1", font=F_HUD, anchor="mm", fill=int(240 * one * ha))
    if zero > 0.01: h.text((540, 296), "0", font=F_HUD, anchor="mm", fill=int(150 * zero * ha))
    a = np.maximum(a, np.asarray(hud).astype(np.float32) * ha)
    return zoom(a, 1.0 + 0.2 * ss(T_S4, T_S5, t), 620, 1000)


def frame(t):
    if t < T_S2 + 0.3:
        a = s1(t)
        if t > T_S2:
            k = ss(T_S2, T_S2 + 0.3, t); a = a * (1 - k) + s2(t) * k
    elif t < T_PRESS + 0.16:
        a = s2(t)
        a = a + 230 * (ss(T_PRESS, T_PRESS + 0.03, t) * (1 - ss(T_PRESS + 0.03, T_PRESS + 0.16, t)))
    elif t < T_S4:
        a = s3(t)
        if t > T_S4 - 0.35:
            k = ss(T_S4 - 0.35, T_S4 + 0.1, t); a = a * (1 - k) + s4(t) * k
    elif t < T_S5 - 0.3:
        a = s4(t) * (1 - ss(T_S5 - 0.7, T_S5 - 0.3, t))
    else:
        a = s5(t) * ss(T_S5 - 0.3, T_S5 + 0.4, t)
    a = a * (0.55 + 0.45 * VIG)
    fade = ss(0, 0.8, t) * (1 - ss(15.2, 15.95, t))
    rng = np.random.default_rng(int(t * FPS))
    a = a * fade + rng.normal(0, 2.0, (H, W)).astype(np.float32) * fade
    rgb = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "L").convert("RGB")
    dr = ImageDraw.Draw(rgb, "RGBA")
    for c in CUES:
        al = ss(c["start"] - 0.05, c["start"] + 0.25, t) * (1 - ss(c["end"] + 0.1, c["end"] + 0.4, t))
        if al > 0.01:
            txt = c["zh"].replace("，", " ").replace("。", "")
            dr.text((W / 2 + 2, SUB_Y + 3), txt, font=F_SUB, anchor="mm", fill=(0, 0, 0, int(200 * al)), stroke_width=3, stroke_fill=(0, 0, 0, int(200 * al)))
            dr.text((W / 2, SUB_Y), txt, font=F_SUB, anchor="mm", fill=(255, 255, 255, int(255 * al)))
    return np.asarray(rgb)


if __name__ == "__main__":
    if "--still" in sys.argv:
        ts = [float(x) for x in sys.argv[sys.argv.index("--still") + 1:]]
        os.makedirs("stills", exist_ok=True)
        for t in ts:
            Image.fromarray(frame(t)).save(f"stills/t{t:05.2f}.png"); print("ok", t)
