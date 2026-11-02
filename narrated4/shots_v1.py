"""《后悔按钮》画面：黑白极简，1080x1920 竖版，16 秒 @30fps。
用法：python3 shots.py --still 1 3 6 7.2 10 13   # 出静帧到 stills/
时间点全部写在下面的 T_* 里，对应 timings.json 的旁白起点。
"""
import json, sys, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS, DUR = 1080, 1920, 30, 16.0
CX, CY, R = 540, 700, 130            # 按钮
TL_Y, TL_X0, TL_X1, NT = 1130, 120, 960, 37   # 时间线
SUB_Y = 1330                         # 字幕中心（避开抖音底部界面）
# ---- 剧本时间点 ----
T_LINE0, T_LINE1 = 2.6, 4.4          # 时间线从左到右画出
T_PRESS = 5.3                        # 按下按钮
T_BACK0, T_BACK1 = 5.45, 7.2         # 播放头回溯
TARGET = 0.30                        # 回到的位置（占整条时间线）
T_ONE = 8.7                          # 出现「1」
T_ZERO = 10.35                       # 「1」变「0」，按钮熄灭
T_ASK0 = 12.2                        # 犹豫地扫过各个时刻
T_FADE = 15.2

FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
F_SUB = ImageFont.truetype(FONT, 58)
F_NUM = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 150)
CUES = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "timings.json"), encoding="utf-8"))

Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
D = np.hypot(X - CX, Y - CY)
VIG = (1 - 0.55 * (np.hypot((X - W / 2) / (W / 2), (Y - H / 2) / (H / 2)) ** 2)).clip(0, 1) * 14


def ss(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def eo(x):  # ease out cubic
    x = min(max(x, 0), 1)
    return 1 - (1 - x) ** 3


def ring(r, w, soft=1.5):
    return np.clip(1 - (np.abs(D - r) - w / 2) / soft, 0, 1)


def tl_x(f):
    return TL_X0 + (TL_X1 - TL_X0) * f


def frame(t):
    img = VIG.copy()
    # ---- 按钮状态 ----
    press = ss(T_PRESS - 0.02, T_PRESS + 0.1, t) - ss(T_PRESS + 0.12, T_PRESS + 0.4, t)
    dead = ss(T_ZERO, T_ZERO + 0.5, t)                   # 熄灭
    breath = 0.5 + 0.5 * np.sin(t * 2.2)
    r = R * (1 - 0.07 * press)
    live = 1 - dead
    # 光晕
    glow = np.exp(-np.clip(D - r, 0, None) / 90) * (0.16 + 0.1 * breath) * live
    glow += np.exp(-np.clip(D - r, 0, None) / 40) * 0.5 * press
    img += glow * 255
    # 实心圆盘（上亮下暗的微弱立体感）
    disc = np.clip(r - D + 0.7, 0, 1)
    shade = 0.93 - 0.18 * ((Y - CY) / R).clip(-1, 1) * 0.5 - 0.22 * press
    img = img * (1 - disc * live) + disc * live * shade * 255
    # 外环
    rr = ring(R + 38, 3) * (0.8 * live + 0.5 * dead)
    img = np.maximum(img, rr * 255 * (0.9 if dead < 0.5 else 0.5))
    # 熄灭后留一个空心圆
    img = np.maximum(img, ring(R - 2, 4) * dead * 90)
    # 涟漪
    for tp, mx in ((T_PRESS, 1000), (T_BACK1, 560)):
        a = t - tp
        if 0 <= a < 1.8:
            rad = R + 40 + 520 * eo(a / 1.8)
            img = np.maximum(img, ring(rad, 3 + 4 * a, 2) * 255 * (1 - a / 1.8) ** 1.5 * 0.7)
    # 回到那一刻的闪光
    fl = ss(T_BACK1 - 0.05, T_BACK1 + 0.04, t) * (1 - ss(T_BACK1 + 0.04, T_BACK1 + 0.5, t))
    img += fl * 38

    # ---- 数字 ----
    num_a = ss(T_ONE, T_ONE + 0.35, t)
    im = Image.fromarray(np.zeros((H, W), np.uint8))
    pil_layers = []

    base = Image.fromarray(img.clip(0, 255).astype(np.uint8)).convert("RGB")
    dr = ImageDraw.Draw(base, "RGBA")

    def gray(v, a=255):
        v = int(max(0, min(255, v)))
        return (v, v, v, int(max(0, min(255, a))))

    # 按钮上的数字
    if num_a > 0:
        one = 1 - ss(T_ZERO - 0.1, T_ZERO + 0.25, t)
        zero = ss(T_ZERO + 0.1, T_ZERO + 0.45, t)
        for ch, al, col in (("1", one, 20), ("0", zero, 150)):
            if al * num_a > 0.01:
                dr.text((CX, CY - 4), ch, font=F_NUM, anchor="mm",
                        fill=gray(col if ch == "1" else 170, 255 * al * num_a))
        # 「×」小字提示次数
    # ---- 时间线 ----
    reveal = ss(T_LINE0, T_LINE1, t)
    dim = 1 - 0.65 * dead * (1 - ss(T_ASK0, T_ASK0 + 0.6, t))
    fade = 1 - ss(T_FADE, DUR - 0.05, t)
    if reveal > 0:
        xe = tl_x(reveal)
        dr.line([(TL_X0, TL_Y), (xe, TL_Y)], fill=gray(150 * dim), width=3)
        for i in range(NT):
            f = i / (NT - 1)
            x = tl_x(f)
            if x > xe:
                break
            major = i % 6 == 0
            h = 30 if major else 14
            v = (200 if major else 110) * dim
            # 问句阶段：整条逐个亮起
            if t > T_ASK0:
                v = max(v, 255 * ss(T_ASK0 + f * 1.4, T_ASK0 + f * 1.4 + 0.5, t) * 0.55 * dim ** 0.5)
            dr.line([(x, TL_Y - h), (x, TL_Y + h)], fill=gray(v), width=3)
    # 播放头
    if reveal > 0:
        if t < T_BACK0:
            pf = reveal                                   # 「现在」跟着画线端点
        elif t < T_BACK1:
            pf = 1 + (TARGET - 1) * eo((t - T_BACK0) / (T_BACK1 - T_BACK0))
        else:
            pf = TARGET
            if t > T_ASK0:                                # 犹豫：在几个时刻之间游移
                u = t - T_ASK0
                pf = TARGET + 0.07 * np.sin(u * 2.6) + 0.22 * ss(0, 2, u) * np.sin(u * 1.1) - 0.04 * u
        px = tl_x(min(max(pf, 0), 1))
        # 拖尾
        if T_BACK0 < t < T_BACK1 + 0.4:
            for k in range(1, 14):
                pfk = pf + k * 0.012 * (1 - 0.0)
                if pfk <= 1 and t < T_BACK1:
                    dr.ellipse([tl_x(pfk) - 7 + k * 0.2, TL_Y - 7, tl_x(pfk) + 7, TL_Y + 7],
                               fill=gray(255, 90 * (1 - k / 14)))
        pv = 255 * dim ** 0.6
        rad = 13 + (4 * np.sin(t * 6) if t > T_ASK0 else 0)
        dr.ellipse([px - rad - 14, TL_Y - rad - 14, px + rad + 14, TL_Y + rad + 14], fill=gray(255, 28 * dim))
        dr.ellipse([px - rad, TL_Y - rad, px + rad, TL_Y + rad], fill=gray(pv))
        # 落点外圈
        if t >= T_BACK1:
            a = t - T_BACK1
            rr_ = 30 + 8 * np.sin(a * 3)
            dr.ellipse([px - rr_, TL_Y - rr_, px + rr_, TL_Y + rr_], outline=gray(230 * dim, 200 * ss(0, 0.3, a)), width=2)
        # 「现在」标记：右端小字
    # ---- 字幕 ----
    for c in CUES:
        a = ss(c["start"] - 0.05, c["start"] + 0.25, t) * (1 - ss(c["end"] + 0.1, c["end"] + 0.4, t))
        if a > 0.01:
            txt = c["zh"].replace("，", " ").replace("。", "").replace("？", "？")
            dr.text((W / 2 + 2, SUB_Y + 2), txt, font=F_SUB, anchor="mm", fill=(0, 0, 0, int(160 * a)))
            dr.text((W / 2, SUB_Y), txt, font=F_SUB, anchor="mm", fill=(255, 255, 255, int(255 * a)))
    out = np.asarray(base).astype(np.float32)
    # 开场淡入 / 结尾淡出 + 细微胶片颗粒
    g = ss(0, 0.9, t) * fade
    rng = np.random.default_rng(int(t * FPS))
    out = out * g + rng.normal(0, 2.2, (H, W, 1)).astype(np.float32) * g
    return out.clip(0, 255).astype(np.uint8)


if __name__ == "__main__":
    if "--still" in sys.argv:
        ts = [float(x) for x in sys.argv[sys.argv.index("--still") + 1:]]
        os.makedirs("stills", exist_ok=True)
        for t in ts:
            Image.fromarray(frame(t)).save(f"stills/t{t:05.2f}.png")
            print("ok", t)
