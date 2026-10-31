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
TAG = "9x16" if PORTRAIT else "16x9"
W, H = (1080, 1920) if PORTRAIT else (1920, 1080)
S = min(W, H) / 1080
YH = int(H * (0.60 if PORTRAIT else 0.64))     # 地平线
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


def gen_layer(rng, name, hmin, hmax, wmin, wmax, win, inten, top, bot):
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
        x += bw + rng.uniform(-0.15, 0.5) * wmax * S
    return dict(name=name, sil=sil, wid=wid, wrgb=wrgb, u=np.array(us, np.float32),
                col=gradient([(0, top), (1, bot)], YH), blds=blds, antennas=antennas)


def glow_patch(mask, color, s1=5, s2=16, core=1.0):
    """把文字/线条蒙版做成霓虹灯：白色灯芯 + 两层彩色光晕"""
    g1 = gaussian_filter(mask, s1 * S)
    g2 = gaussian_filter(mask, s2 * S)
    c = np.asarray(color, np.float32)
    corec = c * 0.4 + 0.6
    return (mask[..., None] * corec * core + g1[..., None] * c * 1.6 + g2[..., None] * c * 0.9).astype(np.float32)


def text_mask(text, size, vertical, pad):
    f = ImageFont.truetype(ZH, int(size))
    if vertical:
        cw = int(size * 1.05)
        im = Image.new("L", (cw + 2 * pad, int(size * 1.08 * len(text)) + 2 * pad))
        d = ImageDraw.Draw(im)
        for k, ch in enumerate(text):
            d.text((pad + (cw - d.textlength(ch, font=f)) / 2, pad + k * size * 1.08), ch, font=f, fill=255)
    else:
        tmp = ImageDraw.Draw(Image.new("L", (1, 1)))
        tw = int(tmp.textlength(text, font=f))
        im = Image.new("L", (tw + 2 * pad, int(size * 1.2) + 2 * pad))
        ImageDraw.Draw(im).text((pad, pad), text, font=f, fill=255)
    # 霓虹灯管感：只保留笔画轮廓附近
    m = np.asarray(im).astype(np.float32) / 255
    return m


SIGNS = [("咖啡", (1.0, .25, .62), True), ("书店", (.2, .85, 1.0), True), ("OPEN", (1.0, .3, .3), False),
         ("旅馆", (.7, .35, 1.0), True), ("24H", (1.0, .72, .25), False), ("雨", (.3, 1.0, .75), True),
         ("BAR", (1.0, .25, .62), False), ("拉面", (1.0, .55, .2), True)]
# 点亮时刻：第一块在鼓点进入（10.7s），其余在主歌开头按拍子依次亮起，旅馆在 B 段亮
SIGN_ON = [10.67, 21.33, 22.0, 64.0, 22.67, 23.33, 42.67, 43.33]
SIGN_OFF = [124.5, 118.0, 121.5, 115.0, 120.0, 116.8, 117.5, 119.0]


def build_signs(rng, near):
    signs = []
    blds = sorted([b for b in near["blds"] if b[1] - b[0] > 70 * S], key=lambda b: b[0])
    pad = int(40 * S)
    used = 0
    for x0, x1, y0 in blds:
        if used >= len(SIGNS):
            break
        text, c, vert = SIGNS[used]
        size = (40 if vert else 34) * S
        m = text_mask(text, size, vert, pad)
        ph, pw = m.shape
        if vert:
            px = x0 + (6 * S if rng.random() < .5 else (x1 - x0) - 6 * S - (pw - 2 * pad)) - pad
            py = y0 + rng.uniform(.05, .3) * (YH - y0) - pad
        else:
            px = (x0 + x1) / 2 - pw / 2
            py = y0 + rng.uniform(.15, .45) * (YH - y0) - pad
        if py + ph > YH - 10 * S or py < 0.12 * YH:
            continue
        signs.append(dict(patch=glow_patch(m, c), x=int(px), y=int(py), on=SIGN_ON[used],
                          off=SIGN_OFF[used], seed=rng.integers(1 << 30)))
        used += 1
    # 两栋楼的楼顶霓虹灯带
    for (x0, x1, y0), c in zip(sorted(blds, key=lambda b: b[2])[:2], [(.2, .85, 1.0), (1.0, .25, .62)]):
        pad2 = int(40 * S)
        m = np.zeros((int(8 * S) + 2 * pad2, x1 - x0 + 2 * pad2), np.float32)
        m[pad2:pad2 + max(2, int(3 * S)), pad2:-pad2] = 1
        signs.append(dict(patch=glow_patch(m, c, 4, 12, .8), x=x0 - pad2, y=y0 - pad2 - int(2 * S),
                          on=21.33 + 0.67 * len(signs) % 3, off=117.0, seed=rng.integers(1 << 30)))
    return signs


def sign_level(sg, t, pulse):
    """霓虹灯开关：点亮瞬间不规则闪几下，平时轻微电流抖动"""
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
    rng = np.random.default_rng(1031)
    hs = 1.0 if not PORTRAIT else 1.08
    layers = [
        gen_layer(rng, "far", .22, .5 * hs, 50, 120, (3, 4, 8, 10), .42, col(.09, .055, .16), col(.20, .10, .25)),
        gen_layer(rng, "mid", .28, .66 * hs, 70, 160, (5, 7, 12, 15), .62, col(.045, .032, .09), col(.11, .06, .16)),
        gen_layer(rng, "near", .32, .80 * hs, 100, 230, (8, 11, 17, 22), .85, col(.02, .016, .042), col(.05, .03, .08)),
    ]
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
        ts = [float(v) for v in sys.argv[sys.argv.index("--still") + 1:] if not v.startswith("--")]
        tw, th = (360, 640) if PORTRAIT else (640, 360)
        cols = len(ts) if PORTRAIT else 2
        ims = [Image.fromarray(frame(min(int(t * FPS), NF - 1))) for t in ts]
        ims[0].save(os.path.join(HERE, f"still_{TAG}.png"))
        sheet = Image.new("RGB", (tw * cols, th * ((len(ims) + cols - 1) // cols)))
        for n, im in enumerate(ims):
            sheet.paste(im.resize((tw, th), Image.LANCZOS), ((n % cols) * tw, (n // cols) * th))
        sheet.save(os.path.join(HERE, f"preview_{TAG}.png"))
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
