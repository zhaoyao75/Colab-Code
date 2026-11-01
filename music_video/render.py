"""
《霓虹雨夜》128 秒动画：程序生成的雨夜城市，配 neon_rain.mp3。
- 三层视差楼群，窗灯随音乐段落一盏盏亮起、尾声逐渐熄灭
- 霓虹招牌在鼓点进入时闪烁点亮，随低鼓脉动
- 湿地面倒影（翻转 + 纵向模糊 + 波纹扰动），雨丝、涟漪、路灯光锥、撑伞的人
- 横屏 1920x1080 与竖屏 1080x1920 各自原生构图

运行：python3 render.py --still 3 30 70 100 122             -> 横屏关键帧预览 preview_16x9.png
     python3 render.py --portrait --still 3 30 70 100 122  -> 竖屏 preview_9x16.png
     python3 render.py --demo 18                           -> 从第 18 秒起的 10 秒 demo
     python3 render.py [--portrait]                        -> 完整 128 秒（4 进程并行）
"""
import multiprocessing as mp
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, sosfilt
from scipy.io import wavfile

HERE = os.path.dirname(os.path.abspath(__file__))
ZH = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
EN = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FPS = 30
DUR = 128.0
NF = int(DUR * FPS)
WORKERS = 4
CRF = "25"          # 雨丝每帧都在动，码率很高；25 画质仍好，128 秒约 80MB（GitHub 单文件上限 100MB）
AUDIO = os.path.join(HERE, "neon_rain.mp3")
PORTRAIT = "--portrait" in sys.argv
STYLE = sys.argv[sys.argv.index("--style") + 1] if "--style" in sys.argv else "hk"   # hk（香港招牌峡谷，默认）/ tower / street / close / alley
TAG = "9x16" if PORTRAIT else "16x9"
W, H = (1080, 1920) if PORTRAIT else (1920, 1080)
S = min(W, H) / 1080
YH = int(H * ((0.78 if PORTRAIT else 0.80) if STYLE == "hk" else (0.60 if PORTRAIT else 0.64)))     # 地平线（hk 场景里路面更窄）
GH = H - YH
BAR = 8 / 3


def interp(keys, t):
    ks = np.array(keys, float)
    return float(np.interp(t, ks[:, 0], ks[:, 1]))


def ramp(t, a, b):
    x = float(np.clip((t - a) / (b - a), 0, 1))
    return x * x * (3 - 2 * x)


# 音乐段落：前奏 0-21.3（10.7 鼓点进入）/ A1 21.3-42.7 / A2 -64 / B -85.3 / A3 -106.7 / 尾声 -128
LIGHTS = [(0, .05), (10.7, .1), (21.3, .26), (42.7, .4), (64, .48), (85.3, .62), (106.7, .64),
          (114, .4), (122, .1), (128, .04)]                     # 窗灯点亮比例
HUE = [(0, 0), (62, 0), (65, 1), (84, 1), (87, 0), (128, 0)]   # B 段转为冷蓝色调
RAIN = [(0, .8), (21.3, 1), (85.3, 1.15), (106.7, 1.15), (128, .9)]
PAN = {"cloud": .05, "far": .03, "mid": .07, "near": .12, "fg": .12}   # 128 秒内的平移量（× 画面宽）


def col(*c):
    return np.array(c, np.float32)


# ------------------------------------------------------------------ 场景生成
def fractal_noise(rng, h, w, octaves=((60, 1), (25, .5), (10, .25))):
    out = np.zeros((h, w), np.float32)
    for sig, a in octaves:
        out += a * gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), sig * S)
    out -= out.min()
    return out / out.max()


def gradient(stops, n):
    ys = np.linspace(0, 1, n)
    return np.stack([np.interp(ys, [p for p, _ in stops], [c[k] for _, c in stops]) for k in range(3)],
                    axis=1).astype(np.float32)[:, None, :]


WIN_COLORS = [(col(1.0, .70, .40), .45), (col(1.0, .86, .66), .25), (col(.62, .78, 1.0), .18),
              (col(.45, .95, .9), .06), (col(1.0, .5, .78), .06)]


def gen_layer(rng, name, hmin, hmax, wmin, wmax, win, inten, top, bot, gap=(-0.15, .5)):
    Ww = W + int(PAN[name] * W) + 4
    sil = np.zeros((YH, Ww), np.float32)
    wid = np.zeros((YH, Ww), np.int32)
    wrgb = np.zeros((YH, Ww, 3), np.float32)
    blds, antennas = [], []
    nid = 1
    us, vs = [np.inf], [0.0]
    x = -rng.uniform(0, wmax * S)
    pcol = np.array([c for c, _ in WIN_COLORS])
    pp = np.array([p for _, p in WIN_COLORS])
    while x < Ww:
        bw = rng.uniform(wmin, wmax) * S
        bh = rng.uniform(hmin, hmax) * YH
        x0, x1, y0 = max(0, int(x)), min(Ww, int(x + bw)), int(YH - bh)
        if x1 - x0 > 4:
            sil[y0:, x0:x1] = 1
            if rng.random() < 0.35:   # 退台式楼顶
                w2 = bw * rng.uniform(.35, .65)
                h2 = rng.uniform(.03, .09) * YH
                cx = (x0 + x1) / 2
                sil[max(0, int(y0 - h2)):y0, max(0, int(cx - w2 / 2)):min(Ww, int(cx + w2 / 2))] = 1
                ytop = int(y0 - h2)
            else:
                ytop = y0
            if rng.random() < 0.3:    # 天线 + 航空障碍灯
                ah = rng.uniform(.04, .09) * YH
                ax = int((x0 + x1) / 2 + rng.uniform(-.2, .2) * bw)
                aw = max(1, int(2 * S))
                if 0 <= ax < Ww - aw:
                    sil[max(0, int(ytop - ah)):ytop, ax:ax + aw] = 1
                    antennas.append((ax + aw / 2, ytop - ah, rng.uniform(0, 6.28)))
            # 窗户：网格或横向条窗
            ww, wh, gx, gy = [v * S for v in win]
            strip = rng.random() < 0.12
            c_bld = pcol[rng.choice(len(pp), p=pp)]
            mx = max(2, int((x1 - x0) % gx / 2 + gx * .35))
            yy = y0 + gy * 0.8
            while yy < YH - gy * 0.6:
                if strip:   # 横向条窗：分成几段，亮度更低
                    segs = np.linspace(x0 + mx, x1 - mx, rng.integers(2, 5) + 1)
                    ids_row = [(a + 2 * S, b_ - 2 * S) for a, b_ in zip(segs[:-1], segs[1:])]
                else:
                    ids_row = [(xx, xx + ww) for xx in np.arange(x0 + mx, x1 - mx - ww + 1, gx)]
                for a, b_ in ids_row:
                    a, b_ = int(a), int(b_)
                    if b_ - a < 1:
                        continue
                    c = c_bld if rng.random() < 0.7 else pcol[rng.choice(len(pp), p=pp)]
                    wid[int(yy):int(yy + wh), a:b_] = nid
                    wrgb[int(yy):int(yy + wh), a:b_] = c * inten * rng.uniform(.6, 1.0) * (.55 if strip else 1)
                    us.append(rng.uniform(0, 1.25))
                    vs.append(1.0)
                    nid += 1
                yy += gy
            blds.append((x0, x1, y0))
        x += bw + rng.uniform(*gap) * wmax * S
    return dict(name=name, sil=sil, wid=wid, wrgb=wrgb, u=np.array(us, np.float32),
                col=gradient([(0, top), (1, bot)], YH), blds=blds, antennas=antennas)


def glow_patch(mask, color, s1=5, s2=16, core=1.0):
    """把文字/线条蒙版做成霓虹灯：白色灯芯 + 两层彩色光晕"""
    g1 = gaussian_filter(mask, s1 * S)
    g2 = gaussian_filter(mask, s2 * S)
    c = np.asarray(color, np.float32)
    corec = c * 0.4 + 0.6
    return (mask[..., None] * corec * core + g1[..., None] * c * 1.6 + g2[..., None] * c * 0.9).astype(np.float32)


def text_mask(text, size, vertical, pad, border=False):
    """英文霓虹字：横排，或竖排的「招牌式」逐字母叠放；border=True 时外加圆角灯管边框"""
    f = ImageFont.truetype(EN, int(size))
    tmp = ImageDraw.Draw(Image.new("L", (1, 1)))
    if vertical:
        cw = int(size * 1.0)
        im = Image.new("L", (cw + 2 * pad, int(size * 1.12 * len(text)) + 2 * pad))
        d = ImageDraw.Draw(im)
        for k, ch in enumerate(text):
            d.text((pad + (cw - d.textlength(ch, font=f)) / 2, pad + k * size * 1.12), ch, font=f, fill=255)
    else:
        tw = int(tmp.textlength(text, font=f))
        bx = int(size * 0.35) if border else 0
        im = Image.new("L", (tw + 2 * pad + 2 * bx, int(size * 1.2) + 2 * pad + 2 * bx))
        d = ImageDraw.Draw(im)
        d.text((pad + bx, pad + bx), text, font=f, fill=255)
        if border:
            d.rounded_rectangle([pad, pad, pad + tw + 2 * bx, pad + int(size * 1.2) + 2 * bx],
                                radius=int(size * .3), outline=255, width=max(1, int(2 * S)))
    return np.asarray(im).astype(np.float32) / 255


def icon_mask(kind, size, pad, lw=None):
    """图形霓虹：鸡尾酒杯 / 音符 / 咖啡杯，用细线勾勒成灯管"""
    sz = int(size)
    im = Image.new("L", (sz + 2 * pad, sz + 2 * pad))
    d = ImageDraw.Draw(im)
    lw = lw or max(2, int(2.5 * S))
    o = pad
    if kind == "cocktail":
        d.line([(o, o + sz * .1), (o + sz, o + sz * .1), (o + sz / 2, o + sz * .58), (o, o + sz * .1)], fill=255, width=lw)
        d.line([(o + sz / 2, o + sz * .58), (o + sz / 2, o + sz * .92)], fill=255, width=lw)
        d.line([(o + sz * .25, o + sz * .95), (o + sz * .75, o + sz * .95)], fill=255, width=lw)
        r = sz * .08
        d.ellipse([o + sz * .62 - r, o + sz * .22 - r, o + sz * .62 + r, o + sz * .22 + r], outline=255, width=lw)
        d.line([(o + sz * .62, o + sz * .22), (o + sz * .85, o - sz * .02)], fill=255, width=lw)
    elif kind == "note":
        r = sz * .13
        for cx in (o + sz * .25, o + sz * .78):
            d.ellipse([cx - r * 1.3, o + sz * .8 - r, cx + r * 1.3, o + sz * .8 + r], outline=255, width=lw)
            d.line([(cx + r * 1.25, o + sz * .8), (cx + r * 1.25, o + sz * .12)], fill=255, width=lw)
        d.line([(o + sz * .25 + r * 1.25, o + sz * .12), (o + sz * .78 + r * 1.25, o + sz * .04)], fill=255, width=lw * 2)
    elif kind == "cup":
        d.line([(o + sz * .1, o + sz * .35), (o + sz * .2, o + sz * .9), (o + sz * .7, o + sz * .9),
                (o + sz * .8, o + sz * .35), (o + sz * .1, o + sz * .35)], fill=255, width=lw)
        d.arc([o + sz * .68, o + sz * .45, o + sz * 1.0, o + sz * .75], 270, 90, fill=255, width=lw)
        for k in range(3):   # 热气
            x = o + sz * (.28 + .17 * k)
            d.arc([x - sz * .05, o, x + sz * .05, o + sz * .14], 90, 270, fill=255, width=lw)
            d.arc([x - sz * .05, o + sz * .12, x + sz * .05, o + sz * .26], 270, 90, fill=255, width=lw)
    elif kind == "fish":
        d.ellipse([o + sz * .05, o + sz * .28, o + sz * .72, o + sz * .72], outline=255, width=lw)
        d.line([(o + sz * .7, o + sz * .5), (o + sz * .96, o + sz * .26), (o + sz * .96, o + sz * .74),
                (o + sz * .7, o + sz * .5)], fill=255, width=lw)
        d.ellipse([o + sz * .17, o + sz * .42, o + sz * .25, o + sz * .5], fill=255)
        d.arc([o + sz * .35, o + sz * .35, o + sz * .5, o + sz * .65], 300, 60, fill=255, width=lw)
    elif kind == "crown":
        pts = [(.08, .78), (.08, .3), (.3, .52), (.5, .18), (.7, .52), (.92, .3), (.92, .78), (.08, .78)]
        d.line([(o + sz * x, o + sz * y) for x, y in pts], fill=255, width=lw, joint="curve")
        d.line([(o + sz * .08, o + sz * .9), (o + sz * .92, o + sz * .9)], fill=255, width=lw)
    elif kind == "star":
        pts = []
        for k in range(10):
            r, a = (sz * .48 if k % 2 == 0 else sz * .2), -np.pi / 2 + k * np.pi / 5
            pts.append((o + sz * .5 + r * np.cos(a), o + sz * .53 + r * np.sin(a)))
        d.line(pts + [pts[0]], fill=255, width=lw, joint="curve")
    elif kind == "heart":
        pts = [(o + sz * (.5 + .03 * 16 * np.sin(a) ** 3),
                o + sz * (.45 - .03 * (13 * np.cos(a) - 5 * np.cos(2 * a) - 2 * np.cos(3 * a) - np.cos(4 * a))))
               for a in np.linspace(0, 2 * np.pi, 80)]
        d.line(pts, fill=255, width=lw, joint="curve")
    elif kind == "bolt":
        pts = [(.58, .04), (.22, .56), (.46, .56), (.36, .96), (.78, .4), (.54, .4), (.58, .04)]
        d.line([(o + sz * x, o + sz * y) for x, y in pts], fill=255, width=lw, joint="curve")
    elif kind == "moon":
        outer = [(o + sz * .5 + sz * .45 * np.cos(a), o + sz * .5 + sz * .45 * np.sin(a))
                 for a in np.linspace(np.radians(50), np.radians(310), 50)]
        inner = [(o + sz * .6 + sz * .38 * np.cos(a), o + sz * .5 + sz * .38 * np.sin(a))
                 for a in np.linspace(np.radians(299), np.radians(61), 50)]
        d.line(outer + inner + [outer[0]], fill=255, width=lw, joint="curve")
    elif kind == "beer":
        d.rounded_rectangle([o + sz * .12, o + sz * .2, o + sz * .6, o + sz * .92], radius=int(sz * .05), outline=255, width=lw)
        d.arc([o + sz * .5, o + sz * .34, o + sz * .9, o + sz * .74], 270, 90, fill=255, width=lw)
        for cx in (.22, .36, .5):
            d.ellipse([o + sz * (cx - .09), o + sz * .08, o + sz * (cx + .09), o + sz * .28], outline=255, width=max(1, lw - 1))
    return np.asarray(im).astype(np.float32) / 255


SF = {"tower": 1.0, "street": 1.4, "close": 1.9, "alley": 1.5, "hk": 1.0}[STYLE]      # 招牌放大倍数
PINK, CYAN, RED, AMBER, GREEN, BLUE, ORANGE, PURPLE = ((1.0, .25, .62), (.2, .85, 1.0), (1.0, .3, .3),
                                                       (1.0, .72, .25), (.3, 1.0, .75), (.45, .7, 1.0),
                                                       (1.0, .55, .2), (.7, .35, 1.0))
# 横招牌（门头）：(类型, 内容, 颜色, 是否加灯管边框)；只用英文和图形
HSIGNS = [("text", "CAFE", PINK, True), ("text", "BOOKS", CYAN, False), ("text", "OPEN", RED, True),
          ("text", "24H", AMBER, True), ("icon", "cocktail", GREEN, False), ("icon", "note", BLUE, False),
          ("icon", "cup", ORANGE, False), ("text", "RECORDS", GREEN, False), ("text", "RAMEN", ORANGE, False),
          ("text", "PIZZA", RED, True), ("text", "DINER", PINK, True), ("text", "VINYL", CYAN, False)]
# 竖招牌（挂在楼角）
VSIGNS = [("HOTEL", PURPLE), ("BAR", PINK), ("JAZZ", CYAN), ("MOTEL", GREEN), ("LIVE", AMBER)]
# 点亮 / 熄灭时刻：第一块在鼓点进入（10.7s），其余在主歌开头按拍子依次亮起，竖招牌在 B 段亮
H_ON = [10.67, 21.33, 22.0, 22.67, 23.33, 24.0, 24.67, 25.33, 42.67, 43.33, 44.0, 44.67]
H_OFF = [124.5, 118.0, 121.5, 120.0, 116.8, 117.5, 119.0, 122.0, 116.0, 118.5, 120.5, 115.5]
V_ON = [64.0, 44.0, 65.33, 45.33, 66.0]
V_OFF = [115.0, 117.5, 119.5, 116.2, 118.2]


def shop_patch(w, h, c, awn):
    """一楼橱窗：暖光玻璃 + 窗框 + 条纹遮阳篷，带光晕"""
    pad = int(30 * S)
    ah = max(6, int(h * 0.2))
    m = np.zeros((h + 2 * pad, w + 2 * pad), np.float32)
    aw = np.zeros_like(m)
    m[pad + ah:pad + h, pad:pad + w] = 1
    step = max(8, int(40 * S))
    for xx in range(pad + step, pad + w - 4, step):
        m[pad + ah:pad + h, xx:xx + max(2, int(2 * S))] = 0.15
    sw = max(6, int(16 * S))
    for k, xx in enumerate(range(pad, pad + w, sw)):
        aw[pad:pad + ah, xx:min(pad + w, xx + sw)] = 1.0 if k % 2 == 0 else 0.25
    glow = gaussian_filter(m, 14 * S)
    c, awn = np.asarray(c, np.float32), np.asarray(awn, np.float32)
    patch = (m[..., None] * c * 0.30 + glow[..., None] * c * 0.38 + aw[..., None] * awn * 0.42
             + gaussian_filter(aw, 6 * S)[..., None] * awn * 0.3)
    return patch.astype(np.float32), pad


def build_signs(rng, near):
    """街面店铺：一楼橱窗 + 遮阳篷，门头横招牌挂在橱窗上方，竖招牌挂在楼角"""
    signs, hi, vi, k = [], 0, 0, 0
    pad = int(40 * S)
    SH = int(44 * S * SF ** 0.8)                      # 一楼橱窗高度
    shop_cols = [(1.0, .72, .4), (1.0, .72, .4), (.5, .8, 1.0), (1.0, .5, .75)]
    awn_cols = [PINK, CYAN, ORANGE, GREEN, PURPLE, RED]
    blds = sorted([b for b in near["blds"] if b[1] - b[0] > 90 * S], key=lambda b: b[0])
    for x0, x1, y0 in blds:
        bw = x1 - x0
        if YH - y0 < SH + 20 * S:
            continue
        # 橱窗
        sx, sw_ = x0 + int(8 * S), int(bw - 16 * S)
        patch, sp = shop_patch(sw_, SH, shop_cols[k % 4], awn_cols[k % 6])
        near["wid"][int(YH - SH - 4 * S):, x0 + int(4 * S):x1 - int(4 * S)] = 0
        signs.append(dict(patch=patch, x=sx - sp, y=int(YH - SH - sp - 2 * S), on=21.33 + 0.4 * k,
                          off=119.0 + (k % 3) * 0.8, seed=int(rng.integers(1 << 30)), steady=True))
        k += 1
        if x1 < 60 * S:
            continue
        # 门头横招牌
        fx0 = fx1 = None
        if hi < len(HSIGNS) and x0 > -bw * 0.3:
            kind, txt, c, border = HSIGNS[hi]
            m = icon_mask(txt, 34 * S * SF, pad) if kind == "icon" else text_mask(txt, 24 * S * SF, False, pad, border)
            ph, pw = m.shape
            top_ok = YH - SH - 12 * S - (ph - 2 * pad) >= y0 + 8 * S
            if pw - 2 * pad <= bw - 12 * S and top_ok:
                jit = rng.uniform(-.2, .2) * max(0, bw - (pw - 2 * pad))
                px = (x0 + x1) / 2 - pw / 2 + jit
                py = YH - SH - 12 * S - (ph - 2 * pad) - pad
                near["wid"][int(py + pad - 4 * S):int(YH - SH - 4 * S), int(px + pad - 4 * S):int(px + pw - pad + 4 * S)] = 0
                signs.append(dict(patch=glow_patch(m, c), x=int(px), y=int(py), on=H_ON[hi % len(H_ON)],
                                  off=H_OFF[hi % len(H_OFF)], seed=int(rng.integers(1 << 30))))
                fx0, fx1 = px + pad, px + pw - pad
                hi += 1
        # 竖招牌：挂在门头招牌旁边空位较大的一侧
        if vi < len(VSIGNS) and fx0 is not None and rng.random() < .75:
            txt, c = VSIGNS[vi]
            m = text_mask(txt, 20 * S * SF, True, pad)
            ph, pw = m.shape
            room_l, room_r = fx0 - x0, x1 - fx1
            bwid = pw - 2 * pad
            side = 0 if room_l >= room_r else 1
            if max(room_l, room_r) >= bwid + 10 * S:
                px = (x0 + 5 * S if side == 0 else x1 - 5 * S - bwid) - pad
                py = YH - SH - 10 * S - (ph - 2 * pad) - pad
                if py + pad >= y0 + 6 * S:
                    near["wid"][int(py + pad - 4 * S):int(YH - SH - 4 * S), int(px + pad - 4 * S):int(px + pw - pad + 4 * S)] = 0
                    signs.append(dict(patch=glow_patch(m, c), x=int(px), y=int(py), on=V_ON[vi % len(V_ON)],
                                      off=V_OFF[vi % len(V_OFF)], seed=int(rng.integers(1 << 30))))
                    vi += 1
    return signs


def sign_level(sg, t, pulse):
    """霓虹灯开关：点亮瞬间不规则闪几下，平时轻微电流抖动"""
    if sg.get("steady"):
        if t < sg["on"]:
            return 0.0
        if t > sg["off"]:
            return max(0.0, 1 - (t - sg["off"]) / 1.5)
        return min(1.0, (t - sg["on"]) / 0.6) * (1 + 0.08 * pulse)
    if t < sg["on"] or t > sg["off"] + 0.9:
        return 0.0
    r = np.random.default_rng(sg["seed"] + int(t * 20))
    if t - sg["on"] < 0.9:
        return 1.0 if r.random() < (t - sg["on"]) / 0.9 + 0.1 else 0.08
    if t > sg["off"]:
        return 0.6 if r.random() > (t - sg["off"]) / 0.9 + 0.2 else 0.0
    hum = 0.95 + 0.05 * r.random()
    if r.random() < 0.004:
        hum = 0.3
    return hum * (1 + 0.28 * pulse)


def build_fg(rng):
    """前景：路灯 + 撑红伞的人（含倒影），以及路灯的光（叠加层）和光锥蒙版"""
    Wf = W + int(PAN["fg"] * W) + 4
    yg = int(YH + 0.55 * GH)
    lx = int(W * (0.80 if PORTRAIT else 0.78))
    top = int(yg - 520 * S)
    hx, hy = lx - int(62 * S), top + int(26 * S)        # 灯头发光点
    sil = Image.new("L", (Wf, H))
    colim = Image.new("RGB", (Wf, H))
    d, dc = ImageDraw.Draw(sil), ImageDraw.Draw(colim)
    dark = (5, 4, 9)

    def poly(pts, fill_c=dark):
        d.polygon(pts, fill=255)
        dc.polygon(pts, fill=fill_c)

    pw = 5 * S
    poly([(lx - pw * 1.8, yg), (lx + pw * 1.8, yg), (lx + pw * 1.2, yg - 40 * S), (lx - pw * 1.2, yg - 40 * S)])
    poly([(lx - pw, yg - 40 * S), (lx + pw, yg - 40 * S), (lx + pw * .8, top), (lx - pw * .8, top)])
    if STYLE != "tower":   # 拉过街面的电线
        for sag, dy in ((60, 0), (95, 40), (45, -28)):
            for xa, xb in ((0, lx), (lx, Wf)):
                pts = [(xa + (xb - xa) * u, top + (10 + dy) * S + sag * S * 4 * u * (1 - u)) for u in np.linspace(0, 1, 40)]
                d.line(pts, fill=255, width=max(1, int(1.5 * S)))
                dc.line(pts, fill=dark, width=max(1, int(1.5 * S)))
    for k in range(12):   # 弯曲的灯臂
        a0, a1 = k / 12 * np.pi / 2, (k + 1) / 12 * np.pi / 2
        r = 62 * S
        p0 = (lx - r + r * np.cos(a0), top - r * np.sin(a0) + r)
        p1 = (lx - r + r * np.cos(a1), top - r * np.sin(a1) + r)
        d.line([(p0[0], p0[1] - 24 * S), (p1[0], p1[1] - 24 * S)], fill=255, width=int(5 * S))
        dc.line([(p0[0], p0[1] - 24 * S), (p1[0], p1[1] - 24 * S)], fill=dark, width=int(5 * S))
    poly([(hx - 26 * S, hy - 4 * S), (hx + 26 * S, hy - 4 * S), (hx + 14 * S, hy - 22 * S), (hx - 14 * S, hy - 22 * S)])
    # 撑红伞的人，站在灯下
    fx, fh = hx - int(95 * S), 178 * S
    poly([(fx - 7 * S, yg), (fx - 3 * S, yg), (fx - 4 * S, yg - fh * .42), (fx - 10 * S, yg - fh * .42)])
    poly([(fx + 3 * S, yg), (fx + 7 * S, yg), (fx + 10 * S, yg - fh * .42), (fx + 4 * S, yg - fh * .42)])
    poly([(fx - 19 * S, yg - fh * .38), (fx + 19 * S, yg - fh * .38), (fx + 13 * S, yg - fh * .82),
          (fx - 13 * S, yg - fh * .82)], (12, 9, 16))
    d.ellipse([fx - 10 * S, yg - fh * .95, fx + 10 * S, yg - fh * .82], fill=255)
    dc.ellipse([fx - 10 * S, yg - fh * .95, fx + 10 * S, yg - fh * .82], fill=dark)
    d.line([(fx + 12 * S, yg - fh * .6), (fx + 4 * S, yg - fh * 1.08)], fill=255, width=max(1, int(2 * S)))
    dc.line([(fx + 12 * S, yg - fh * .6), (fx + 4 * S, yg - fh * 1.08)], fill=dark, width=max(1, int(2 * S)))
    ur, uy = 66 * S, yg - fh * 1.04
    box = [fx + 4 * S - ur, uy - ur * .75, fx + 4 * S + ur, uy + ur * .75]
    d.pieslice(box, 180, 360, fill=255)
    dc.pieslice(box, 180, 360, fill=(120, 14, 26))

    sil_a = np.asarray(sil).astype(np.float32) / 255
    fcol = np.asarray(colim).astype(np.float32) / 255
    # 伞面被路灯照亮的渐变（左上亮）
    yy, xx = np.mgrid[0:H, 0:Wf].astype(np.float32)
    lit = np.exp(-(((xx - hx) / (160 * S)) ** 2 + ((yy - hy) / (160 * S)) ** 2))
    umb = (fcol[..., 0] > 0.3).astype(np.float32)
    fcol = fcol * (1 + umb[..., None] * lit[..., None] * 2.2)
    # 倒影：以地面为轴翻转，纵向拉长模糊
    ref_a = np.zeros_like(sil_a)
    n = H - yg
    ref_a[yg:] = sil_a[yg - 1::-1][:n]
    ref_a = gaussian_filter(ref_a, (6 * S, 1.5 * S)) * 0.55
    ref_c = np.zeros_like(fcol)
    ref_c[yg:] = fcol[yg - 1::-1][:n] * 0.5
    ref_c = gaussian_filter(ref_c, (6 * S, 1.5 * S, 0))
    alpha = np.maximum(sil_a, ref_a)
    fcolor = np.where(sil_a[..., None] > 0.01, fcol, ref_c / np.maximum(ref_a[..., None], 1e-3) * 0.55)

    warm = col(1.0, .72, .42)
    r2 = ((xx - hx) / S) ** 2 + ((yy - hy) / S) ** 2
    add = (1.6 * np.exp(-r2 / (2 * 10 ** 2)) + .45 * np.exp(-r2 / (2 * 45 ** 2))
           + .16 * np.exp(-r2 / (2 * 160 ** 2)))[..., None] * warm
    # 光锥：从灯头向下展开
    dy = np.clip((yy - hy) / (yg - hy), 0, None)
    half = (14 + 200 * dy) * S
    cone = np.clip(1 - np.abs(xx - hx) / half, 0, 1) ** 1.5 * (dy > 0) * (dy < 1.02) * (1 - 0.55 * dy)
    cone = gaussian_filter(cone.astype(np.float32), 6 * S)
    add += 0.13 * cone[..., None] * warm
    # 地面光池 + 湿地面上的纵向反光
    pool = np.exp(-(((xx - hx) / (230 * S)) ** 2 + ((yy - yg) / (34 * S)) ** 2))
    streak = np.exp(-(((xx - hx) / (16 * S)) ** 2)) * np.exp(-np.clip(yy - yg, 0, None) / (150 * S)) * (yy > yg)
    add += (0.32 * pool + 0.35 * streak)[..., None] * warm
    return dict(alpha=alpha.astype(np.float32), color=fcolor.astype(np.float32), add=add.astype(np.float32),
                cone=(cone + 0.6 * np.exp(-r2 / (2 * 70 ** 2))).astype(np.float32))


def overlay_layers():
    title = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dr = ImageDraw.Draw(title)
    zh = ImageFont.truetype(ZH, int(104 * S))
    en = ImageFont.truetype(EN, int(26 * S))

    def spaced(d, text, font, y, spacing, fill):
        widths = [d.textlength(c, font=font) for c in text]
        x = (W - sum(widths) - spacing * (len(text) - 1)) / 2
        for c, w in zip(text, widths):
            d.text((x, y), c, font=font, fill=fill)
            x += w + spacing

    ty = H * (0.27 if PORTRAIT else 0.30)
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    spaced(gd, "霓虹雨夜", zh, ty, 34 * S, (255, 70, 160, 255))
    glow = glow.filter(ImageFilter.GaussianBlur(10 * S))
    glow = Image.alpha_composite(glow, glow)
    spaced(dr, "霓虹雨夜", zh, ty, 34 * S, (255, 238, 246, 255))
    spaced(dr, "N E O N    R A I N", en, ty + 140 * S, 6 * S, (190, 225, 255, 235))
    halo = np.asarray(title)[..., 3].astype(np.float32) / 255
    halo = np.clip(gaussian_filter(halo, 28 * S) * 3.0, 0, 0.6)
    title = Image.alpha_composite(glow, title)
    tarr = np.asarray(title).astype(np.float32) / 255
    a = tarr[..., 3]
    na = a + halo * (1 - a)       # 文字背后一圈柔和的暗晕，压在招牌上也看得清
    rgb = tarr[..., :3] * (a / np.maximum(na, 1e-4))[..., None]
    title = Image.fromarray((np.dstack([rgb, na]) * 255).astype(np.uint8), "RGBA")

    logo = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    dl = ImageDraw.Draw(logo)
    if PORTRAIT:
        cx, cy = int(70 * S), int(H * 0.12)   # 避开抖音左上角菜单 / 直播图标
    else:
        cx, cy = W - int(140 * S), H - int(46 * S)
    R = 14 * S
    orange = (217, 119, 87, 255)
    lr = np.random.default_rng(11)
    for i in range(12):   # Claude 星芒标志（简化绘制）
        ang = i * 2 * np.pi / 12 + lr.uniform(-0.12, 0.12)
        ln = R * lr.uniform(0.75, 1.0)
        x2, y2 = cx + ln * np.cos(ang), cy + ln * np.sin(ang)
        dl.line([(cx, cy), (x2, y2)], fill=orange, width=max(2, int(3 * S)))
        dl.ellipse([x2 - 1.5 * S, y2 - 1.5 * S, x2 + 1.5 * S, y2 + 1.5 * S], fill=orange)
    lf = ImageFont.truetype(EN, int(22 * S))
    dl.text((cx + 24 * S, cy - 14 * S), "Claude", font=lf, fill=(240, 236, 230, 255))
    to_np = lambda im: np.asarray(im).astype(np.float32) / 255
    return to_np(title), to_np(logo)


def audio_pulse():
    """低频（低鼓）瞬态包络，每帧一个值，快起慢落"""
    wav = os.path.join(HERE, "neon_rain.wav")
    if not os.path.exists(wav):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", AUDIO, wav], check=True)
    sr, x = wavfile.read(wav)
    x = x.astype(np.float32).mean(axis=1) / 32768
    x = sosfilt(butter(2, 140 / (sr / 2), output="sos"), x)
    hop = sr // FPS
    n = min(NF, len(x) // hop)
    rms = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(axis=1) + 1e-9)
    rms = np.pad(rms, (0, NF - n), mode="edge")
    long = gaussian_filter(rms, FPS * 1.0)
    p = np.clip(rms / (long + 1e-4) - 1.0, 0, 1)
    env = np.zeros_like(p)
    for i in range(NF):
        prev = env[i - 1] if i else 0
        env[i] = p[i] if p[i] > prev else prev * 0.85
    return env


def setup():
    if STYLE == "hk":
        return hk_setup()
    rng = np.random.default_rng(1031)
    hs = 1.0 if not PORTRAIT else 1.08
    lowk = 1080 * 0.64 / YH            # 低层建筑按像素高度（而不是占画面比例）设计，横竖屏楼层数一致
    if STYLE == "tower":
        layers = [
            gen_layer(rng, "far", .22, .5 * hs, 50, 120, (3, 4, 8, 10), .42, col(.09, .055, .16), col(.20, .10, .25)),
            gen_layer(rng, "mid", .28, .66 * hs, 70, 160, (5, 7, 12, 15), .62, col(.045, .032, .09), col(.11, .06, .16)),
            gen_layer(rng, "near", .32, .80 * hs, 100, 230, (8, 11, 17, 22), .85, col(.02, .016, .042), col(.05, .03, .08)),
        ]
    else:
        cfg = {"street": ((.22, .5), (.22, .42), (.24, .40), (160, 300)),
               "close": ((.20, .48), (.2, .38), (.30, .46), (260, 420)),
               "alley": ((.24, .55), (.22, .42), (.34, .56), (190, 300))}[STYLE]
        if STYLE == "alley":
            PAN["near"] = 0.03
        layers = [
            gen_layer(rng, "far", cfg[0][0], cfg[0][1] * hs, 50, 120, (3, 4, 8, 10), .42, col(.09, .055, .16), col(.20, .10, .25)),
            gen_layer(rng, "mid", cfg[1][0] * lowk, cfg[1][1] * lowk, 90, 200, (5, 7, 12, 15), .62, col(.045, .032, .09),
                      col(.11, .06, .16), gap=(-.02, .08)),
            gen_layer(rng, "near", cfg[2][0] * lowk, cfg[2][1] * lowk, cfg[3][0], cfg[3][1], (8, 11, 17, 22), .85,
                      col(.02, .016, .042), col(.05, .03, .08), gap=(-.02, .05)),
        ]
        if STYLE == "alley":   # 中间留出一条缝，看到远处的城市
            L = layers[2]
            a, b = 0.34 * W, 0.66 * W
            keep, gone = [], []
            for (x0, x1, y0) in L["blds"]:
                if x1 > a and x0 < b:
                    L["sil"][:, max(0, x0):x1] = 0
                    L["wid"][:, max(0, x0):x1] = 0
                    gone.append((x0, x1))
                else:
                    keep.append((x0, x1, y0))
            L["blds"] = keep
            L["antennas"] = [an for an in L["antennas"] if not any(g0 <= an[0] <= g1 for g0, g1 in gone)]
    signs = build_signs(rng, layers[2])
    sky_a = gradient([(0, (.015, .015, .06)), (.6, (.08, .045, .16)), (.93, (.30, .12, .27)),
                      (1, (.42, .17, .32))], YH)
    sky_b = gradient([(0, (.008, .02, .07)), (.6, (.035, .07, .19)), (.93, (.10, .17, .38)),
                      (1, (.14, .24, .46))], YH)
    cw = W + int(PAN["cloud"] * W) + int(0.02 * W * DUR / 10) + 8
    clouds = fractal_noise(rng, YH // 2 + 1, cw // 2, ((30, 1), (12, .5), (5, .25)))
    clouds = np.clip(clouds - 0.35, 0, 1) ** 1.3
    clouds = np.repeat(np.repeat(clouds, 2, 0), 2, 1)[:YH]
    clouds *= (0.05 + 0.5 * (np.linspace(0, 1, YH)[:, None] ** 2.5))
    mist = gradient([(0, (0, 0, 0)), (.8, (0, 0, 0)), (1, (1, 1, 1))], YH)[:, :, :1]
    ground = gradient([(0, (.05, .03, .08)), (1, (.012, .01, .02))], GH)
    fres = np.linspace(0.85, 0.5, GH)[:, None]
    puddle = fractal_noise(rng, GH, W, ((40, 1), (14, .4)))
    rmask = (fres * (0.35 + 0.65 * np.clip((puddle - .3) / .3, 0, 1)))[..., None].astype(np.float32)
    yy, xx = np.mgrid[0:GH, 0:W].astype(np.float32)
    persp = (yy + 8 * S) / GH        # 近处波纹大、远处小
    vig_y, vig_x = np.mgrid[0:H, 0:W].astype(np.float32)
    vig = 1 - 0.42 * (((vig_x - W / 2) / (W * .75)) ** 2 + ((vig_y - H / 2) / (H * .75)) ** 2)
    # 雨滴：两层景深
    n_far, n_near = [int(k * W * H / (1920 * 1080)) for k in (900, 380)]
    drops = []
    for n, v, ln, a, wdt in ((n_far, 950, 22, .30, 1), (n_near, 1750, 48, .5, 2)):
        drops.append(dict(x=rng.uniform(0, W + 200 * S, n), y=rng.uniform(0, H + 100 * S, n),
                          v=v * S * rng.uniform(.85, 1.15, n), ln=ln * S * rng.uniform(.7, 1.3, n),
                          a=a * rng.uniform(.5, 1, n), w=max(1, int(wdt * S))))
    # 涟漪：每秒约 70 个，寿命 0.5 秒
    nr = int(DUR * 70)
    ripples = dict(t=np.sort(rng.uniform(-1, DUR, nr)), x=rng.uniform(0, W, nr),
                   y=YH + 4 * S + (GH - 4 * S) * rng.uniform(0, 1, nr) ** 0.7, a=rng.uniform(.4, 1, nr))
    title, logo = overlay_layers()
    grain = [rng.standard_normal((H // 2, W // 2)).astype(np.float32) for _ in range(6)]
    for L in layers:
        L["v"] = rng.uniform(.65, 1.0, len(L["u"])).astype(np.float32)
        L["v"][0] = 0
    return dict(layers=layers, signs=signs, sky_a=sky_a, sky_b=sky_b, clouds=clouds, mist=mist,
                ground=ground, rmask=rmask, xx=xx, yy=yy, persp=persp, vig=vig[..., None], drops=drops,
                ripples=ripples, fg=build_fg(rng), title=title, logo=logo, grain=grain, pulse=audio_pulse())


def add_patch(img, patch, x, y, gain):
    ph, pw = patch.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(img.shape[1], x + pw), min(img.shape[0], y + ph)
    if x1 > x0 and y1 > y0 and gain > 0:
        img[y0:y1, x0:x1] += patch[y0 - y:y1 - y, x0 - x:x1 - x] * gain


SC = None


def frame(i):
    if STYLE == "hk":
        return hk_frame(i)
    sc = SC
    t = i / FPS
    pulse = sc["pulse"][min(i, NF - 1)]
    p = t / DUR
    hue = interp(HUE, t)
    up = sc["sky_a"] * (1 - hue) + sc["sky_b"] * hue
    up = np.broadcast_to(up, (YH, W, 3)).copy()
    hor = up[-1, 0]
    co = int(PAN["cloud"] * W * p + 0.02 * W * t / 10)
    up += sc["clouds"][:, co:co + W, None] * (hor * 1.6 + 0.02)
    lights = interp(LIGHTS, t)
    for L in sc["layers"]:
        o = int(round(PAN[L["name"]] * W * p))
        a = L["sil"][:, o:o + W, None]
        up = up * (1 - a) + L["col"] * a
        lut = L["v"] * np.clip((lights - L["u"]) / 0.004, 0, 1)
        up += L["wrgb"][:, o:o + W] * lut[L["wid"][:, o:o + W]][..., None]
        for ax, ay, ph in L["antennas"]:
            if np.sin(t * 2.4 + ph) > 0.75:
                add_patch(up, sc_red, int(ax - o - sc_red.shape[1] / 2), int(ay - sc_red.shape[0] / 2), 1.0)
    on = int(round(PAN["near"] * W * p))
    for sg in sc["signs"]:
        add_patch(up, sg["patch"], sg["x"] - on, sg["y"], sign_level(sg, t, pulse))
    up += sc["mist"] * (hor * 0.55)

    # 湿地面倒影
    src = up[YH - 1::-1][:GH]
    small = gaussian_filter(src[::2, ::2], (3.5, 0.7, 0))
    refl = np.repeat(np.repeat(small, 2, 0), 2, 1)[:GH, :W]
    xx, yy, persp = sc["xx"], sc["yy"], sc["persp"]
    sx = (persp * 4.5 * S + 1.0) * (np.sin(xx * 0.02 / S + yy * 0.05 / S - t * 2.3)
                                    + 0.6 * np.sin(xx * 0.008 / S - yy * 0.035 / S + t * 1.7))
    xi = np.clip((xx + sx).astype(np.int32), 0, W - 1)
    refl = refl[yy.astype(np.int32), xi]
    ground = sc["ground"] + refl * sc["rmask"]
    img = np.concatenate([up, ground], axis=0)

    fg = sc["fg"]
    of = int(round(PAN["fg"] * W * p))
    a = fg["alpha"][:, of:of + W, None]
    img = img * (1 - a) + fg["color"][:, of:of + W] * a
    img += fg["add"][:, of:of + W]

    # 雨丝 + 涟漪
    rain = Image.new("L", (W, H))
    d = ImageDraw.Draw(rain)
    dens = interp(RAIN, t)
    for L in sc["drops"]:
        n = int(len(L["x"]) * min(1, dens))
        vx = 0.16
        yv = (L["y"][:n] + L["v"][:n] * t) % (H + 100 * S) - 50 * S
        xv = (L["x"][:n] + vx * L["v"][:n] * t) % (W + 200 * S) - 100 * S
        for x, y, ln, al in zip(xv, yv, L["ln"][:n], L["a"][:n] * min(1.3, dens)):
            d.line([(x, y), (x - vx * ln, y - ln)], fill=int(255 * al), width=L["w"])
    rp = sc["ripples"]
    k0, k1 = np.searchsorted(rp["t"], [t - 0.5, t])
    for k in range(k0, k1):
        age = (t - rp["t"][k]) / 0.5
        ps = (rp["y"][k] - YH) / GH
        r = (3 + 26 * ps) * S * (0.3 + age)
        x, y = rp["x"][k], rp["y"][k]
        d.ellipse([x - r, y - r * 0.28, x + r, y + r * 0.28], outline=int(150 * rp["a"][k] * (1 - age)))
    rain = rain.filter(ImageFilter.GaussianBlur(0.6 * S))
    rv = np.asarray(rain).astype(np.float32)[..., None] / 255
    rcol = col(.42, .48, .62) * (1 - hue) + col(.38, .52, .72) * hue
    img += rv * (rcol + fg["cone"][:, of:of + W, None] * col(1.0, .72, .42) * 1.6)

    # 柔光、暗角、颗粒
    small = img[::4, ::4]
    bl = gaussian_filter(np.maximum(small - 0.45, 0), (7, 7, 0))
    img += 0.55 * np.repeat(np.repeat(bl, 4, 0), 4, 1)[:H, :W]
    img *= sc["vig"]
    img = np.where(img > 0.75, 0.75 + 0.25 * np.tanh((img - 0.75) / 0.25), img)
    g = sc["grain"][(i // 2) % len(sc["grain"])]     # 颗粒每两帧换一次、幅度小：好看且不撑大文件
    img += 0.009 * np.repeat(np.repeat(g, 2, 0), 2, 1)[..., None]
    img *= ramp(t, 0, 2.5)

    title_a = ramp(t, 0.6, 1.6) * (1 - ramp(t, 4.6, 6.0)) + ramp(t, 117.5, 119.5)
    for layer, al in ((sc["title"], title_a), (sc["logo"], ramp(t, 1.5, 3.0))):
        if al > 0:
            aa = layer[..., 3:4] * al
            img = img * (1 - aa) + layer[..., :3] * aa
    img *= 1 - ramp(t, DUR - 3, DUR)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


sc_red = None


# ====================================================================== 香港霓虹街景（--style hk）
# 参考：旺角 / 弥敦道的「招牌峡谷」——沿街道往里看，两侧大量招牌层层叠叠、近大远小，
# 中间留一条缝，缝里是一栋亮着窗的高楼；路面有长曝光的车灯光轨。招牌只用英文和图形。
HK_WORDS_V = ["HOTEL", "KARAOKE", "DISCO", "SAUNA", "PAWN", "JAZZ", "LIVE", "MOTEL", "CLUB", "GOLD",
              "NOODLE", "SEAFOOD", "ROOMS", "MASSAGE", "TEA", "BAR", "PUB", "CAFE", "PIZZA", "WINE"]
HK_WORDS_H = ["PUB & DISCO", "RESTAURANT", "PHARMACY", "NEON CLUB", "SEAFOOD", "KARAOKE", "GAME ZONE",
              "LUCKY STAR", "SKY LOUNGE", "MOON BAR", "GOLDEN DRAGON", "NIGHT CLUB", "HOT POT", "DIM SUM",
              "TEA HOUSE", "COFFEE", "BOOKS", "RECORDS", "BILLIARDS", "24H OPEN"]
HK_WORDS_R = ["OPEN", "24H", "VIP", "LIVE", "BAR", "SPA", "FUN", "HOT", "CITY", "GOLD"]
# 中文招牌用语（和英文各占一半，混排）
HK_ZH_V = ["酒店", "茶餐厅", "卡拉OK", "夜总会", "按摩", "当铺", "海鲜酒家", "龙凤茶楼", "金铺", "药房", "旅馆",
           "粥面", "烧味", "咖啡室", "桑拿", "金龙", "星光", "大酒楼"]
HK_ZH_H = ["海鲜酒家", "星光餐厅", "金龙酒家", "幸运茶楼", "茶餐厅", "卡拉OK夜总会", "越南粉面", "龙凤大酒楼",
           "商业大厦", "二十四小时营业", "游戏机中心", "港式烧味", "药房", "百货公司"]
HK_ZH_R = ["福", "酒", "营业", "24小时", "茶", "喜", "龙", "金", "夜"]
HK_ICONS = ["fish", "crown", "star", "cocktail", "note", "cup", "heart", "bolt", "moon", "beer"]
HK_COLS = [PINK, CYAN, RED, AMBER, GREEN, BLUE, ORANGE, PURPLE, (1.0, .15, .15), (.1, 1.0, .5), (1.0, .9, .3)]
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SB = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
FONT_MB = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
HK_PANEL = np.array([.012, .010, .022], np.float32)


def fit_font(text, path, w, h):
    size = max(6, int(h))
    f = ImageFont.truetype(path, size)
    tw = ImageDraw.Draw(Image.new("L", (1, 1))).textlength(text, font=f)
    if tw > w:
        size = max(6, int(size * w / tw))
    return ImageFont.truetype(path, size)


def blur_big(E, sigma):
    """大半径高斯模糊：先缩小 4 倍再放大，快很多"""
    if sigma <= 8:
        return gaussian_filter(E, (sigma, sigma, 0))
    g = gaussian_filter(E[::4, ::4], (sigma / 4, sigma / 4, 0))
    return np.repeat(np.repeat(g, 4, 0), 4, 1)[:E.shape[0], :E.shape[1]]


def hk_make_sign(seed, kind, w, h, c1, c2, word, icon):
    """画一块招牌，返回 (面板不透明度, 发光层, 边距)；发光层 = 灯管 + 两层光晕 + 灯管中心的白热"""
    rng = np.random.default_rng(seed)
    pad = int(max(w, h) * 0.14) + 8
    PW, PH = w + 2 * pad, h + 2 * pad
    em = Image.new("RGB", (PW, PH))
    d = ImageDraw.Draw(em)
    pm = Image.new("L", (PW, PH))
    dp = ImageDraw.Draw(pm)
    C1, C2 = tuple(int(255 * v) for v in c1), tuple(int(255 * v) for v in c2)
    dim = lambda c, k: tuple(int(v * k) for v in c)
    WARM = (255, 238, 214)
    lw = max(2, int(min(w, h) * 0.05))
    box = [pad, pad, pad + w, pad + h]
    rad = max(3, int(min(w, h) * 0.2))
    lightbox = rng.random() < 0.4        # 灯箱（实心色块 + 白字）或 灯管（暗底 + 发光线条）
    cjk = any(ord(ch) > 0x2e80 for ch in word)
    path = ZH if cjk else [FONT_B, FONT_SB, FONT_MB][int(rng.integers(3))]

    def text_c(txt, fpath, cx, cy, bw, bh, fill):
        f = fit_font(txt, fpath, bw, bh)
        bb = d.textbbox((0, 0), txt, font=f)
        sw = max(1, int(lw * .4)) if cjk else 0     # 中文笔画细，加一点描边让灯管更饱满
        d.text((cx - (bb[0] + bb[2]) / 2, cy - (bb[1] + bb[3]) / 2), txt, font=f, fill=fill,
               stroke_width=sw, stroke_fill=fill)

    if kind in ("vblade", "hboard"):
        dp.rounded_rectangle(box, rad, fill=255)
        if lightbox:
            d.rounded_rectangle(box, rad, fill=dim(C1, .36))
            d.rounded_rectangle(box, rad, outline=C2, width=lw)
            tcol = (255, 255, 255)
        else:
            d.rounded_rectangle(box, rad, outline=C1, width=lw)
            ins = lw * 3
            d.rounded_rectangle([pad + ins, pad + ins, pad + w - ins, pad + h - ins], max(2, rad - ins // 2),
                                outline=C2, width=max(1, lw - 1))
            tcol = WARM if rng.random() < .5 else C2
        if kind == "vblade":
            word = word.replace(" ", "")
            n = len(word)
            cell = (h - 6 * lw) / n
            for k, ch in enumerate(word):
                text_c(ch, path, pad + w / 2, pad + 3 * lw + cell * (k + .5), w * .62, cell * .78, tcol)
        else:
            text_c(word, path, pad + w / 2, pad + h / 2, w * .86, h * .56, tcol)
    elif kind == "round":
        dp.ellipse(box, fill=255)
        if lightbox:
            d.ellipse(box, fill=dim(C1, .36))
        d.ellipse(box, outline=C2 if lightbox else C1, width=lw)
        ins = lw * 3
        d.ellipse([pad + ins, pad + ins, pad + w - ins, pad + h - ins], outline=C2, width=max(1, lw - 1))
        if rng.random() < .55:           # 放射光芒
            cx, cy, r0 = pad + w / 2, pad + h / 2, w / 2
            for k in range(24):
                a = k * 2 * np.pi / 24
                r1 = r0 * (1.14 if k % 2 else 1.24)
                d.line([(cx + r0 * 1.04 * np.cos(a), cy + r0 * 1.04 * np.sin(a)),
                        (cx + r1 * np.cos(a), cy + r1 * np.sin(a))], fill=C1, width=max(1, lw - 1))
        text_c(word, ZH if cjk else FONT_B, pad + w / 2, pad + h / 2, w * .62, h * .3, (255, 255, 255) if lightbox else WARM)
    else:                                 # 图形招牌
        m_ = icon_mask(icon, min(w, h) * .8, 0, lw=lw)
        sz = m_.shape[0]
        x, y = pad + (w - sz) // 2, pad + (h - sz) // 2
        mimg = Image.fromarray((m_ * 255).astype(np.uint8))
        em.paste(Image.new("RGB", mimg.size, C1), (x, y), mimg)
        pm.paste(255, (x, y), mimg.filter(ImageFilter.MaxFilter(2 * lw + 3)))
        if rng.random() < .6:
            d.ellipse(box, outline=C2, width=max(1, lw - 1))
            dp.ellipse(box, fill=255)
    E = np.asarray(em).astype(np.float32) / 255
    A = np.asarray(pm).astype(np.float32) / 255
    g1 = blur_big(E, max(1.5, lw * 1.3))
    g2 = blur_big(E, max(4.0, lw * 4.5))
    white = np.clip(E.max(axis=2, keepdims=True) - .55, 0, 1) * .6
    return A, (E + .5 * g1 + .35 * g2 + .8 * white).astype(np.float32), pad


def hk_person():
    """路面上撑红伞的小小人影（含倒影）"""
    yg, fx = int(YH + 0.86 * GH), int(W * .42)
    u = 300 * S / 178                       # 人物高约 300 像素（占竖屏高度的 16%）
    sil, colim = Image.new("L", (W, H)), Image.new("RGB", (W, H))
    d, dc = ImageDraw.Draw(sil), ImageDraw.Draw(colim)
    dark = (5, 4, 9)

    def poly(pts, fc=dark):
        d.polygon(pts, fill=255)
        dc.polygon(pts, fill=fc)
    fh = 178 * u
    poly([(fx - 7 * u, yg), (fx - 3 * u, yg), (fx - 4 * u, yg - fh * .42), (fx - 10 * u, yg - fh * .42)])
    poly([(fx + 3 * u, yg), (fx + 7 * u, yg), (fx + 10 * u, yg - fh * .42), (fx + 4 * u, yg - fh * .42)])
    poly([(fx - 19 * u, yg - fh * .38), (fx + 19 * u, yg - fh * .38), (fx + 13 * u, yg - fh * .82),
          (fx - 13 * u, yg - fh * .82)], (12, 9, 16))
    d.ellipse([fx - 10 * u, yg - fh * .95, fx + 10 * u, yg - fh * .82], fill=255)
    dc.ellipse([fx - 10 * u, yg - fh * .95, fx + 10 * u, yg - fh * .82], fill=dark)
    d.line([(fx + 12 * u, yg - fh * .6), (fx + 4 * u, yg - fh * 1.08)], fill=255, width=max(1, int(2 * u)))
    dc.line([(fx + 12 * u, yg - fh * .6), (fx + 4 * u, yg - fh * 1.08)], fill=dark, width=max(1, int(2 * u)))
    ur, uy = 66 * u, yg - fh * 1.04
    box = [fx + 4 * u - ur, uy - ur * .75, fx + 4 * u + ur, uy + ur * .75]
    d.pieslice(box, 180, 360, fill=255)
    dc.pieslice(box, 180, 360, fill=(150, 18, 34))
    sil_a = np.asarray(sil).astype(np.float32) / 255
    fcol = np.asarray(colim).astype(np.float32) / 255
    ref_a = np.zeros_like(sil_a)
    ref_a[yg:] = sil_a[yg - 1::-1][:H - yg]
    ref_a = gaussian_filter(ref_a, (5 * S, 1.5 * S)) * 0.5
    ref_c = np.zeros_like(fcol)
    ref_c[yg:] = fcol[yg - 1::-1][:H - yg] * 0.5
    ref_c = gaussian_filter(ref_c, (5 * S, 1.5 * S, 0))
    alpha = np.maximum(sil_a, ref_a)
    color = np.where(sil_a[..., None] > 0.01, fcol, ref_c / np.maximum(ref_a[..., None], 1e-3) * 0.5)
    return dict(alpha=alpha.astype(np.float32), color=color.astype(np.float32))


def hk_person_blit(img, t, pulse):
    """路面上缓步行走的撑伞人：每帧现画（腿、大衣、伞都在动），带霓虹边缘光、身后柔光、脚下光斑和模糊倒影"""
    u = 300 * S / 178
    fh = 178 * u
    SS = 2
    CW, CH = int(380 * S), int(540 * S)
    cx0, base = CW // 2, CH - int(90 * S)
    ph = 2 * np.pi * t / (2 * 60 / 90)                    # 两拍一个完整步伐，和音乐同步
    bob = -abs(np.sin(ph)) * 3.5 * u                       # 走路时身体上下起伏
    a = 0.2 * np.sin(ph)                                   # 腿的摆角
    hem = 2.5 * u * np.sin(ph + .8)                        # 大衣下摆摆动
    tilt = .035 * np.sin(2 * np.pi * t / 2.6) + .01 * np.sin(2 * ph)    # 伞轻轻晃
    P = lambda x, y: (x * SS, y * SS)

    def new():
        im = Image.new("L", (CW * SS, CH * SS))
        return im, ImageDraw.Draw(im)
    body, bd = new()
    can, cd = new()
    rib, rd = new()
    hipy, L = base - fh * .40 + bob, fh * .40
    for sg in (1, -1):                                      # 双腿
        fx_, fy_ = cx0 + sg * 4 * u + np.sin(a * sg) * L, hipy + np.cos(a * sg) * L
        bd.line([P(cx0 + sg * 4 * u, hipy), P(fx_, fy_)], fill=255, width=int(11 * u * SS))
        r_ = 5.5 * u * SS
        bd.ellipse([fx_ * SS - r_, fy_ * SS - r_, fx_ * SS + r_, fy_ * SS + r_], fill=255)
    shy, hemy = base - fh * .80 + bob, base - fh * .34 + bob   # 大衣
    bd.polygon([P(cx0 - 18 * u, shy), P(cx0 + 18 * u, shy), P(cx0 + 31 * u + hem, hemy), P(cx0 - 31 * u + hem, hemy)], fill=255)
    hy = base - fh * .89 + bob                              # 头
    bd.ellipse([*P(cx0 - 10 * u, hy - 11 * u), *P(cx0 + 10 * u, hy + 11 * u)], fill=255)
    hand = (cx0 + 11 * u, base - fh * .93 + bob)            # 手握伞柄
    bd.line([P(cx0 + 16 * u, shy + 4 * u), P(*hand)], fill=255, width=int(8 * u * SS))
    cb = (cx0 + 6 * u, base - fh * 1.06 + bob)
    ur, hr = 64 * u, 64 * u * .72
    ct, st = np.cos(tilt), np.sin(tilt)
    rot = lambda p: (hand[0] + (p[0] - hand[0]) * ct - (p[1] - hand[1]) * st, hand[1] + (p[0] - hand[0]) * st + (p[1] - hand[1]) * ct)
    bd.line([P(*hand), P(*rot((cb[0], cb[1] + 2 * u)))], fill=255, width=int(3 * u * SS))
    dome = [rot((cb[0] + ur * np.cos(f), cb[1] + hr * np.sin(f))) for f in np.linspace(np.pi, 2 * np.pi, 40)]
    cd.polygon([P(*p) for p in dome], fill=255)
    for k in range(1, 8):                                   # 伞骨
        rd.line([P(*rot((cb[0], cb[1] - hr))), P(*rot((cb[0] - ur + 2 * ur * k / 8, cb[1])))], fill=255, width=max(1, int(1.6 * u * SS)))
    ds = lambda im: np.asarray(im.resize((CW, CH), Image.LANCZOS)).astype(np.float32) / 255
    A_b, A_c, A_r = ds(body), ds(can), ds(rib)
    A = np.maximum(A_b, A_c)
    xn = np.clip((np.arange(CW, dtype=np.float32)[None, :] - (cx0 - 70 * u)) / (140 * u), 0, 1)[..., None]
    can_col = np.array([.72, .09, .17], np.float32) * (1.15 - .6 * xn) + np.array([.35, .08, .14], np.float32) * (1 - xn)
    can_col = can_col * (1 - .45 * A_r[..., None])          # 左边被粉色霓虹照亮，伞骨处发暗
    col_ = (np.array([.13, .11, .22], np.float32) * (1.25 - .5 * (np.arange(CH, dtype=np.float32) / CH)[:, None, None])) * (1 - A_c[..., None]) + can_col * A_c[..., None]
    sh = max(2, int(3 * S))
    eL = np.clip(A - np.roll(A, sh, axis=1), 0, 1)
    eR = np.clip(A - np.roll(A, -sh, axis=1), 0, 1)
    eT = np.clip(A - np.roll(A, sh, axis=0), 0, 1)
    rim = (eL[..., None] * np.array([1.0, .35, .65]) + eR[..., None] * np.array([.25, .85, 1.0])
           + eT[..., None] * np.array([1.0, .6, .8]) * .7) * (.9 + .3 * pulse)       # 左粉右青的霓虹边缘光
    halo = gaussian_filter(A, 12 * S)[..., None] * (np.array([1.0, .45, .75]) * .5 + np.array([.3, .8, 1.0]) * .5) * (.22 + .2 * pulse)
    yy, xx = np.mgrid[0:CH, 0:CW].astype(np.float32)
    pool = np.exp(-(((xx - cx0) / (140 * S)) ** 2 + ((yy - base) / (24 * S)) ** 2))[..., None] * np.array([1.0, .55, .4]) * .3
    rr = CH - base                                          # 倒影：翻转 + 模糊
    ra = np.zeros_like(A)
    rc = np.zeros_like(col_)
    ra[base:] = A[base - 1::-1][:rr]
    rc[base:] = (col_ + rim)[base - 1::-1][:rr]
    ra = gaussian_filter(ra, (4 * S, 2 * S)) * .4
    rc = gaussian_filter(rc, (4 * S, 2 * S, 0)) * .6
    oa = np.maximum(A, ra)[..., None]
    oc = np.where(A[..., None] > .01, col_, rc / np.maximum(ra[..., None], 1e-3).clip(.05))
    px = int(W * .42 + 18 * S * np.sin(2 * np.pi * t / 40))   # 缓缓左右漂移
    x0, y0 = px - cx0, int(YH + 0.80 * GH) - base
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + CW), min(H, y0 + CH)
    if xb <= xa or yb <= ya:
        return
    sl = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
    reg = img[ya:yb, xa:xb]
    reg[:] = reg * (1 - oa[sl]) + oc[sl] * oa[sl] + rim[sl] * A[sl][..., None] + halo[sl] + pool[sl]


def hk_setup():
    rng = np.random.default_rng(1101)
    vpx, vpy = W / 2, YH - 0.015 * H            # 灭点：街道尽头，贴着地平线
    N = int(175 * W * H / (1920 * 1080))
    specs, tries = [], 0
    while len(specs) < N and tries < N * 20:
        tries += 1
        dpt = rng.random() ** 0.85              # 0 = 最近，1 = 最远
        s = 1 / (1 + 2.0 * dpt)
        side = rng.choice([-1, 1])
        kind = str(rng.choice(["vblade", "vblade", "vblade", "hboard", "hboard", "round", "icon"]))
        if kind == "vblade":
            bw, bh = rng.uniform(70, 120), rng.uniform(260, 520)
        elif kind == "hboard":
            bw, bh = rng.uniform(240, 460), rng.uniform(64, 120)
        elif kind == "round":
            bw = bh = rng.uniform(120, 210)
        else:
            bw = bh = rng.uniform(90, 190)
        w, h = int(bw * s * S * .95), int(bh * s * S * .95)
        if w < 14 or h < 14:
            continue
        sx = vpx + side * rng.uniform(.14, .62) * W * s
        sy = vpy - rng.uniform(-.12, 1.05) * H * s
        if sy + h / 2 > YH - 0.01 * H or sx < -w * .3 or sx > W + w * .3 or sy + h / 2 < 0:
            continue
        if abs(sx - vpx) < .09 * W + w / 2 and dpt < .78:      # 中间留一条缝给远处的高楼
            continue
        word = {"vblade": HK_WORDS_V + HK_ZH_V, "hboard": HK_WORDS_H + HK_ZH_H,
                "round": HK_WORDS_R + HK_ZH_R}.get(kind, [""])
        word = word[int(rng.integers(len(word)))] if kind != "icon" else ""
        icon = HK_ICONS[int(rng.integers(len(HK_ICONS)))] if kind == "icon" else None
        i1, i2 = rng.choice(len(HK_COLS), 2, replace=False)
        specs.append(dict(s=s, d=dpt, kind=kind, w=w, h=h, sx=sx, sy=sy, word=word, icon=icon,
                          c1=HK_COLS[i1], c2=HK_COLS[i2], seed=int(rng.integers(1 << 30)),
                          alt=rng.random() < .18, c3=HK_COLS[int(rng.integers(len(HK_COLS)))]))
    specs.sort(key=lambda sp: sp["s"])          # 远的先画
    signs = []
    inside = [k for k, sp in enumerate(specs) if sp["sx"] - sp["w"] / 2 >= 0 and sp["sx"] + sp["w"] / 2 <= W
              and sp["sy"] - sp["h"] / 2 >= .08 * H and sp["sy"] + sp["h"] / 2 <= YH]
    first = sorted(inside, key=lambda k: -specs[k]["w"] * specs[k]["h"])[:3]    # 鼓点进入时先亮的三块大招牌（必须完整在画面里）
    for k, sp in enumerate(specs):
        A, add, pad = hk_make_sign(sp["seed"], sp["kind"], sp["w"], sp["h"], sp["c1"], sp["c2"], sp["word"], sp["icon"])
        add_b = None
        if sp["alt"]:
            add_b = hk_make_sign(sp["seed"], sp["kind"], sp["w"], sp["h"], sp["c2"], sp["c3"], sp["word"], sp["icon"])[1]
        u = rng.random()
        on = 10.67 + .67 * first.index(k) if k in first else (21.33 + rng.uniform(0, 21) if u < .5 else
                                                              42.67 + rng.uniform(0, 21) if u < .8 else 64 + rng.uniform(0, 8))
        signs.append(dict(alpha=A, add=add, add_b=add_b, pad=pad, x=int(sp["sx"] - sp["w"] / 2) - pad,
                          y=int(sp["sy"] - sp["h"] / 2) - pad, inten=1.05 - .5 * sp["d"],
                          on=on, off=rng.uniform(111, 124.5), seed=int(rng.integers(1 << 30))))
    # 天空 + 街口暖光
    sky = np.broadcast_to(gradient([(0, (0, 0, .015)), (.55, (.03, .012, .06)), (.9, (.12, .045, .11)),
                                    (1, (.2, .08, .13))], YH), (YH, W, 3)).astype(np.float32).copy()
    yy_ = np.arange(YH, dtype=np.float32)[:, None]
    xx_ = np.arange(W, dtype=np.float32)[None, :]
    band = ((np.exp(-((YH - yy_) / (0.07 * H)) ** 2)) * (np.clip(np.abs(xx_ - W / 2) / (W * .5), 0, 1) ** 1.2 * .8 + .2)
            )[..., None] * np.array([1.0, .55, .32], np.float32) * 0.16
    # 背景高楼：远 / 中 / 近三层高低错落，窗灯按阈值随音乐逐渐点亮；招牌挂在它们前面
    body_a = np.zeros((YH, W), np.float32)
    body_rgb = np.zeros((YH, W, 3), np.float32)
    win_thr = np.full((YH, W), 9.0, np.float32)
    win_rgb = np.zeros((YH, W, 3), np.float32)

    def add_building(x0, x1, y0, colr, br, gx, gy, ww, wh):
        x0, x1 = max(0, x0), min(W, x1)
        if x1 - x0 < 4:
            return
        body_a[y0:, x0:x1] = 1
        body_rgb[y0:, x0:x1] = colr
        win_thr[y0:, x0:x1] = 9.0
        win_rgb[y0:, x0:x1] = 0
        mx = int(rng.uniform(.2, .5) * gx)
        for yy0 in range(y0 + gy // 2, YH - gy, gy):
            for xx0 in range(x0 + mx, x1 - ww - 1, gx):
                if rng.random() < .6:
                    c = np.array([1, .82, .55] if rng.random() < .72 else [.7, .85, 1.0] if rng.random() < .6 else [1, .6, .8],
                                 np.float32) * br * rng.uniform(.6, 1.0)
                    win_thr[yy0:yy0 + wh, xx0:xx0 + ww] = rng.random()
                    win_rgb[yy0:yy0 + wh, xx0:xx0 + ww] = c
        if y0 > 0 and rng.random() < .5:            # 楼顶天线 + 红色航空灯
            ax, ah = (x0 + x1) // 2, int(rng.uniform(.03, .07) * H)
            body_a[max(0, y0 - ah):y0, ax:ax + max(1, int(2 * S))] = 1
            body_rgb[max(0, y0 - ah):y0, ax:ax + max(1, int(2 * S))] = colr
            win_thr[max(0, y0 - ah):max(0, y0 - ah) + int(4 * S), ax - 1:ax + int(3 * S)] = .15
            win_rgb[max(0, y0 - ah):max(0, y0 - ah) + int(4 * S), ax - 1:ax + int(3 * S)] = (1, .1, .1)

    x = int(vpx - .36 * W)                                     # 远景：中间一带，紫色雾气里的摩天楼
    while x < vpx + .36 * W:
        bw, bh = int(rng.uniform(.03, .07) * W), int(rng.uniform(.45, 1.05) * H)
        add_building(x, x + bw, max(0, YH - bh), (.11, .055, .16), .30, int(12 * S), int(16 * S), int(4 * S), int(6 * S))
        x += int(bw * rng.uniform(.7, 1.0))
    x = int(vpx - .55 * W)                                     # 中景
    while x < vpx + .55 * W:
        bw, bh = int(rng.uniform(.05, .09) * W), int(rng.uniform(.3, .9) * H)
        add_building(x, x + bw, max(0, YH - bh), (.055, .03, .09), .38, int(16 * S), int(22 * S), int(6 * S), int(8 * S))
        x += int(bw * rng.uniform(.75, 1.0))
    for sd in (-1, 1):                                         # 近景：街道两侧高楼的立面
        x = -10 if sd < 0 else W + 10
        while (x < .27 * W) if sd < 0 else (x > .73 * W):
            bw, bh = int(rng.uniform(.09, .14) * W), int(rng.uniform(.6, 1.1) * H)
            add_building(x if sd < 0 else x - bw, x + bw if sd < 0 else x, max(0, YH - bh), (.018, .013, .03), .45,
                         int(24 * S), int(32 * S), int(8 * S), int(11 * S))
            x += int(bw * .9) * (1 if sd < 0 else -1)
    # 路面
    ground = np.broadcast_to(gradient([(0, (.03, .02, .045)), (1, (.008, .006, .015))], GH), (GH, W, 3)).astype(np.float32).copy()
    puddle = fractal_noise(rng, GH, W, ((40, 1), (14, .4)))
    frost = fractal_noise(rng, GH, W, ((5, 1), (2, .6)))
    frost2 = fractal_noise(rng, GH, W, ((5, 1), (2, .6)))
    rmask = (np.linspace(1.0, .6, GH)[:, None] * (.55 + .45 * np.clip((puddle - .25) / .35, 0, 1)))[..., None].astype(np.float32)
    yy, xx = np.mgrid[0:GH, 0:W].astype(np.float32)
    # 光轨：左半边是驶离的尾灯（红），右半边是驶来的车头灯（白黄）
    trails = []
    for k in range(0):          # 已去掉车流（光轨）
        away = k < 13
        trails.append(dict(lane=-rng.uniform(.08, .95) if away else rng.uniform(.08, .95), dirn=-1.0 if away else 1.0,
                           speed=rng.uniform(.08, .35), p0=rng.random(), ln=rng.uniform(.18, .5),
                           col=(np.array([1., .07, .04]) if away else np.array([1., .9, .65])) * rng.uniform(.7, 1.1)))
    drops = []
    n_far, n_near = [int(k * W * H / (1920 * 1080)) for k in (900, 380)]
    for n, v, ln, a, wdt in ((n_far, 950, 22, .30, 1), (n_near, 1750, 48, .5, 2)):
        drops.append(dict(x=rng.uniform(0, W + 200 * S, n), y=rng.uniform(0, H + 100 * S, n),
                          v=v * S * rng.uniform(.85, 1.15, n), ln=ln * S * rng.uniform(.7, 1.3, n),
                          a=a * rng.uniform(.5, 1, n), w=max(1, int(wdt * S))))
    nr = int(DUR * 70)
    ripples = dict(t=np.sort(rng.uniform(-1, DUR, nr)), x=rng.uniform(0, W, nr),
                   y=YH + 4 * S + (GH - 4 * S) * rng.uniform(0, 1, nr) ** 0.7, a=rng.uniform(.4, 1, nr))
    vy, vx = np.mgrid[0:H, 0:W].astype(np.float32)
    title, logo = overlay_layers()
    return dict(signs=signs, sky=sky, band=band.astype(np.float32), body_a=body_a, body_rgb=body_rgb,
                win_thr=win_thr, win_rgb=win_rgb, ground=ground, rmask=rmask, xx=xx, yy=yy, yyi=yy.astype(np.int32),
                frost_x=(frost * 2 - 1).astype(np.float32), frost_y=(frost2 * 2 - 1).astype(np.float32), persp=(yy + 8 * S) / GH, rows=np.clip(YH - 1 - (np.arange(GH) * 3.5).astype(np.int32), 0, YH - 1),
                trails=trails, drops=drops, ripples=ripples, title=title, logo=logo,
                vig=(1 - 0.38 * (((vx - W / 2) / (W * .75)) ** 2 + ((vy - H / 2) / (H * .75)) ** 2))[..., None],
                grain=[rng.standard_normal((H // 2, W // 2)).astype(np.float32) for _ in range(6)],
                pulse=audio_pulse(), vp=(vpx, vpy))


def hk_blit(img, sg, level, use_b):
    x, y = sg["x"], sg["y"]
    A = sg["alpha"]
    ph, pw = A.shape
    x0, y0, x1, y1 = max(0, x), max(0, y), min(img.shape[1], x + pw), min(img.shape[0], y + ph)
    if x1 <= x0 or y1 <= y0:
        return
    a = A[y0 - y:y1 - y, x0 - x:x1 - x][..., None]
    reg = img[y0:y1, x0:x1]
    reg *= 1 - a
    reg += a * HK_PANEL
    if level > 0:
        add = sg["add_b"] if (use_b and sg["add_b"] is not None) else sg["add"]
        reg += add[y0 - y:y1 - y, x0 - x:x1 - x] * (level * sg["inten"])


def hk_frame(i):
    sc = SC
    t = i / FPS
    pulse = sc["pulse"][min(i, NF - 1)]
    hue = interp(HUE, t)
    lights = interp(LIGHTS, t)
    up = sc["sky"] + sc["band"] * (0.25 + 1.4 * lights)
    a = sc["body_a"][..., None]
    up = up * (1 - a) + sc["body_rgb"] * a
    up += sc["win_rgb"] * (sc["win_thr"] < lights * 1.6)[..., None]
    beat = int(t / (60 / 90)) % 2 == 1
    for sg in sc["signs"]:
        hk_blit(up, sg, sign_level(sg, t, pulse), beat)
    up += 0.2 * np.repeat(np.repeat(gaussian_filter(up[::8, ::8], (5, 5, 0)), 8, 0), 8, 1)[:YH, :W]

    # 湿地面：把整条招牌峡谷压缩 3.5 倍倒映下来，再加水纹
    refl = up[sc["rows"]]
    xx, yy, persp = sc["xx"], sc["yy"], sc["persp"]
    sx = (persp * 6 * S + 1.5) * (np.sin(xx * 0.02 / S + yy * 0.05 / S - t * 2.3)
                                  + 0.6 * np.sin(xx * 0.008 / S - yy * 0.035 / S + t * 1.7))
    # 磨砂玻璃：细碎的随机折射位移 + 较大半径的模糊，倒影只剩一团团色块和光斑
    xi = np.clip((xx + sx + sc["frost_x"] * 14 * S).astype(np.int32), 0, W - 1)
    yi = np.clip((yy + sc["frost_y"] * 7 * S).astype(np.int32), 0, GH - 1)
    refl = gaussian_filter(refl[yi, xi], (4 * S, 3 * S, 0))
    ground = sc["ground"] + refl * sc["rmask"]
    img = np.concatenate([up, ground], axis=0)
    hk_person_blit(img, t, pulse)

    # 雨丝 + 涟漪
    rain = Image.new("L", (W, H))
    d = ImageDraw.Draw(rain)
    dens = interp(RAIN, t) * 0.75
    for L in sc["drops"]:
        n = int(len(L["x"]) * min(1, dens))
        yv = (L["y"][:n] + L["v"][:n] * t) % (H + 100 * S) - 50 * S
        xv = (L["x"][:n] + 0.16 * L["v"][:n] * t) % (W + 200 * S) - 100 * S
        for x, y, ln, al in zip(xv, yv, L["ln"][:n], L["a"][:n] * min(1.3, dens)):
            d.line([(x, y), (x - 0.16 * ln, y - ln)], fill=int(255 * al), width=L["w"])
    rp = sc["ripples"]
    k0, k1 = np.searchsorted(rp["t"], [t - 0.5, t])
    for k in range(k0, k1):
        age = (t - rp["t"][k]) / 0.5
        r = (3 + 26 * (rp["y"][k] - YH) / GH) * S * (0.3 + age)
        x, y = rp["x"][k], rp["y"][k]
        d.ellipse([x - r, y - r * 0.28, x + r, y + r * 0.28], outline=int(150 * rp["a"][k] * (1 - age)))
    img += np.asarray(rain.filter(ImageFilter.GaussianBlur(0.6 * S))).astype(np.float32)[..., None] / 255 * col(.55, .55, .7)

    # 冷蓝段调色、柔光、暗角、颗粒
    img *= np.array([1 - .22 * hue, 1 - .04 * hue, 1 + .22 * hue], np.float32)
    ga = 0.18 * hue
    img = img * (1 - ga) + img[..., ::-1] * ga
    bl = gaussian_filter(np.maximum(img[::4, ::4] - 0.7, 0), (6, 6, 0))
    img += 0.3 * np.repeat(np.repeat(bl, 4, 0), 4, 1)[:H, :W]
    img *= sc["vig"]
    img = np.where(img > 0.8, 0.8 + 0.2 * np.tanh((img - 0.8) / 0.2), img)
    img += 0.009 * np.repeat(np.repeat(sc["grain"][(i // 2) % 6], 2, 0), 2, 1)[..., None]
    # 镜头缓缓推向街道尽头
    z = 1.0 + 0.07 * ramp(t, 0, DUR)
    vpx, vpy = sc["vp"]
    cw, ch = W / z, H / z
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize(
        (W, H), Image.BILINEAR, box=(vpx - vpx / z, vpy - vpy / z, vpx - vpx / z + cw, vpy - vpy / z + ch))
    img = np.asarray(im).astype(np.float32) / 255 * ramp(t, 0, 2.5)
    title_a = ramp(t, 0.6, 1.6) * (1 - ramp(t, 4.6, 6.0)) + ramp(t, 117.5, 119.5)
    for layer, al in ((sc["title"], title_a), (sc["logo"], ramp(t, 1.5, 3.0))):
        if al > 0:
            aa = layer[..., 3:4] * al
            img = img * (1 - aa) + layer[..., :3] * aa
    img *= 1 - ramp(t, DUR - 3, DUR)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def init():
    global SC, sc_red
    SC = setup()
    r = int(14 * S)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    rr = (xx ** 2 + yy ** 2) / S ** 2
    sc_red = ((1.2 * np.exp(-rr / 3) + 0.35 * np.exp(-rr / 30))[..., None] * col(1.0, .15, .1)).astype(np.float32)


def worker(args):
    k, i0, i1, out = args
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                          "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                          "-preset", "slow", "-crf", CRF, "-pix_fmt", "yuv420p", out],
                         stdin=subprocess.PIPE)
    for i in range(i0, i1):
        p.stdin.write(frame(i).tobytes())
        if (i - i0) % 150 == 0:
            print(f"  worker {k}: {i - i0}/{i1 - i0}", flush=True)
    p.stdin.close()
    p.wait()
    return out


def render(i0, i1, out, t_audio):
    bounds = np.linspace(i0, i1, WORKERS + 1).astype(int)
    segs = [os.path.join(HERE, f".seg_{TAG}_{k}.mp4") for k in range(WORKERS)]
    with mp.get_context("fork").Pool(WORKERS) as pool:
        pool.map(worker, [(k, bounds[k], bounds[k + 1], segs[k]) for k in range(WORKERS)])
    lst = os.path.join(HERE, f".segs_{TAG}.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{s}'\n" for s in segs)
    dur = (i1 - i0) / FPS
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-ss", str(t_audio), "-i", AUDIO, "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-t", str(dur), "-movflags", "+faststart", out], check=True)
    for s in segs + [lst]:
        os.remove(s)
    print("saved", out)


def main():
    init()
    if "--still" in sys.argv:
        ts = []
        for v in sys.argv[sys.argv.index("--still") + 1:]:
            try:
                ts.append(float(v))
            except ValueError:
                pass
        suf = "" if STYLE == "tower" else f"_{STYLE}"
        tw, th = (360, 640) if PORTRAIT else (640, 360)
        cols = len(ts) if PORTRAIT else 2
        ims = [Image.fromarray(frame(min(int(t * FPS), NF - 1))) for t in ts]
        ims[0].save(os.path.join(HERE, f"still_{TAG}{suf}.png"))
        sheet = Image.new("RGB", (tw * cols, th * ((len(ims) + cols - 1) // cols)))
        for n, im in enumerate(ims):
            sheet.paste(im.resize((tw, th), Image.LANCZOS), ((n % cols) * tw, (n // cols) * th))
        sheet.save(os.path.join(HERE, f"preview_{TAG}{suf}.png"))
        print("preview saved")
        return
    if "--demo" in sys.argv:
        t0 = float(sys.argv[sys.argv.index("--demo") + 1])
        i0 = int(t0 * FPS)
        render(i0, i0 + 10 * FPS, os.path.join(HERE, f"neon_rain_demo_{TAG}.mp4"), t0)
        return
    render(0, NF, os.path.join(HERE, f"neon_rain_{TAG}.mp4"), 0)


if __name__ == "__main__":
    main()
