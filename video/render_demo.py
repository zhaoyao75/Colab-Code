"""
《星际回响》动画 demo（10 秒）
参考《星际穿越》卡冈图雅黑洞：薄吸积盘被引力透镜弯曲成环绕黑洞的光环。
做法：先对每个像素做一次史瓦西度规下的光线追踪（只做一次），
记录光线与吸积盘的交点 (r, φ)；之后每帧只需让盘面纹理按开普勒速度旋转。

运行：python3 render_demo.py [--bw] [--still 秒]
     --bw  黑白版（输出 stellar_echo_demo_bw.mp4）
依赖：numpy、scipy、Pillow、ffmpeg；音频取自 ../music/stellar_echo.mp3
"""
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, zoom

HERE = os.path.dirname(os.path.abspath(__file__))
RW, RH = 1920, 1080          # 追踪分辨率
OW, OH = 1280, 720           # 输出分辨率
FPS = 30
DUR = 10.0
RIN, ROUT = 3.0, 13.0        # 吸积盘内外半径（史瓦西半径 = 1）
CACHE = os.path.join(HERE, ".trace_cache.npz")
rng = np.random.default_rng(3)
BW = "--bw" in sys.argv
EN = "--en" in sys.argv       # 纯英文版：标题和副标题都用英文


# ------------------------------------------------------------------ 光线追踪（一次）
def trace():
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return d["hr"], d["hp"], d["esc"], d["edir"]
    elev = np.radians(7.0)
    dist = 32.0
    cam = dist * np.array([0.0, np.sin(elev), -np.cos(elev)])
    f = -cam / np.linalg.norm(cam)
    right = np.cross([0, 1, 0], f); right /= np.linalg.norm(right)
    up = np.cross(f, right)
    tanh = np.tan(np.radians(50) / 2)
    xs = ((np.arange(RW) + 0.5) / RW * 2 - 1) * tanh
    ys = -((np.arange(RH) + 0.5) / RH * 2 - 1) * tanh * RH / RW
    X, Y = np.meshgrid(xs, ys)
    d = f + X[..., None] * right + Y[..., None] * up
    d = d.reshape(-1, 3)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    n = d.shape[0]
    pos = np.tile(cam, (n, 1))
    vel = d.copy()
    h2 = np.sum(np.cross(pos, vel) ** 2, axis=1)

    K = 3
    hr = np.full((K, n), np.nan, np.float32)
    hp = np.zeros((K, n), np.float32)
    nh = np.zeros(n, np.int8)
    esc = np.zeros(n, bool)
    edir = np.zeros((n, 3), np.float32)
    alive = np.arange(n)

    def acc(p, hh):
        r = np.linalg.norm(p, axis=1, keepdims=True)
        return -1.5 * hh[:, None] * p / r ** 5

    a = acc(pos, h2)
    for it in range(3000):
        if alive.size == 0:
            break
        p, v, aa, hh = pos[alive], vel[alive], a[alive], h2[alive]
        r = np.linalg.norm(p, axis=1)
        dt = np.clip(0.025 * r, 0.004, 2.0)[:, None]
        vh = v + 0.5 * dt * aa
        pn = p + dt * vh
        an = acc(pn, hh)
        vn = vh + 0.5 * dt * an
        # 穿过盘面 (y = 0)
        y0, y1 = p[:, 1], pn[:, 1]
        cr = (y0 * y1 < 0)
        if cr.any():
            ci = np.nonzero(cr)[0]
            fr = (y0[ci] / (y0[ci] - y1[ci]))[:, None]
            pc = p[ci] + fr * (pn[ci] - p[ci])
            rc = np.linalg.norm(pc, axis=1)
            ok = (rc >= RIN * 0.9) & (rc <= ROUT * 1.05)
            gi = alive[ci[ok]]
            k = nh[gi]
            keep = k < K
            gi, k = gi[keep], k[keep]
            hr[k, gi] = rc[ok][keep]
            hp[k, gi] = np.arctan2(pc[ok][keep][:, 2], pc[ok][keep][:, 0])
            nh[gi] += 1
        pos[alive], vel[alive], a[alive] = pn, vn, an
        rn = np.linalg.norm(pn, axis=1)
        out = (rn > 60) & (np.sum(pn * vn, axis=1) > 0)
        cap = rn < 1.01
        if out.any():
            esc[alive[out]] = True
            vv = vn[out]
            edir[alive[out]] = vv / np.linalg.norm(vv, axis=1, keepdims=True)
        alive = alive[~(out | cap | (nh[alive] >= K))]
        if it % 100 == 0:
            print(f"  trace step {it}, rays alive {alive.size}", flush=True)
    np.savez(CACHE, hr=hr, hp=hp, esc=esc, edir=edir)
    return hr, hp, esc, edir


# ------------------------------------------------------------------ 纹理
def disk_texture():
    NR, NP = 512, 4096
    t = np.zeros((NR, NP))
    for sig_r, sig_p, w in [(1.2, 60, 1.0), (3, 200, 0.8), (0.6, 20, 0.35)]:
        t += w * gaussian_filter(rng.standard_normal((NR, NP)), (sig_r, sig_p), mode=("nearest", "wrap"))
    t = (t - t.mean()) / t.std()
    return np.clip(1 + 0.35 * t, 0.25, 2.0).astype(np.float32)


def star_background(esc, edir):
    """逼真星空。恒星在“原始天空”上是点光源；用点质量引力透镜方程
    β = θ (1 - θE² / |θ|²) 把画面像素映射回原始天空，所以远处的星是清晰圆点，
    靠近黑洞的星会被拉成弧线。亮度服从幂律（大量暗星、少量亮星），
    带轻微色温差异、亮星光晕和一条淡银河。"""
    from scipy.spatial import cKDTree
    pix = np.radians(50) / RW                      # 每像素对应的弧度
    xs = (np.arange(RW) - RW / 2 + 0.5) * pix
    ys = (np.arange(RH) - RH / 2 + 0.5) * pix
    X, Y = np.meshgrid(xs, ys)
    th2 = np.maximum(X ** 2 + Y ** 2, 1e-8)
    thE2 = 0.1 ** 2                                # 爱因斯坦环角半径：让透镜变形集中在黑洞附近
    BX, BY = X * (1 - thE2 / th2), Y * (1 - thE2 / th2)
    src = np.stack([BX.ravel(), BY.ravel()], -1)

    srng = np.random.default_rng(21)
    k = 9000
    span = 1.4
    sp = srng.uniform(-span, span, (k, 2))
    lum = np.clip(0.08 * srng.random(k) ** -1.2, 0, 12.0)
    temp = srng.random(k)
    tint = np.stack([0.8 + 0.3 * temp, 0.88 + 0.1 * np.sin(np.pi * temp), 1.08 - 0.36 * temp], -1)
    tint /= tint.mean(axis=1, keepdims=True)

    dist, j = cKDTree(sp).query(src, k=2)
    out = np.zeros((src.shape[0], 3))
    for q in range(2):
        dq, jq = dist[:, q] / pix, j[:, q]
        core = np.exp(-(dq / 0.8) ** 2)
        halo = 0.015 * np.exp(-dq / 3.0) * (lum[jq] > 2.0)
        out += (lum[jq] * (core + halo))[:, None] * tint[jq]

    # 淡银河：一条斜穿画面的宽带，带细碎明暗
    ang = np.radians(-28)
    u = src @ np.array([np.cos(ang), np.sin(ang)])
    v = src @ np.array([-np.sin(ang), np.cos(ang)]) - 0.12
    band = np.exp(-(v / 0.17) ** 2)
    tex = np.zeros(src.shape[0])
    for _ in range(9):
        f = srng.uniform(6, 40, 2) * srng.choice([-1, 1], 2)
        tex += np.sin(u * f[0] + v * f[1] + srng.uniform(0, 6.3))
    band *= np.clip(0.55 + 0.17 * tex, 0, None)
    out += (0.03 * band)[:, None] * np.array([0.95, 0.93, 1.0])

    out *= esc[:, None]
    return out.reshape(RH, RW, 3).astype(np.float32)


# ------------------------------------------------------------------ 文字与标志
def overlay_layers():
    title = Image.new("RGBA", (OW, OH), (0, 0, 0, 0))
    dr = ImageDraw.Draw(title)
    zh = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 58)
    en = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)

    def spaced(text, font, y, spacing, fill):
        widths = [dr.textlength(c, font=font) for c in text]
        total = sum(widths) + spacing * (len(text) - 1)
        x = (OW - total) / 2
        for c, w in zip(text, widths):
            dr.text((x, y), c, font=font, fill=fill)
            x += w + spacing

    main_c = (242, 242, 242, 255) if BW else (245, 240, 232, 255)
    sub_c = (196, 196, 196, 255) if BW else (200, 196, 190, 255)
    if EN:
        big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 50)
        spaced("STELLAR ECHO", big, 576, 20, main_c)
        spaced("A TRIBUTE TO INTERSTELLAR", en, 646, 9, sub_c)
    else:
        spaced("星际回响", zh, 572, 26, main_c)
        spaced("STELLAR  ECHO", en, 646, 9, sub_c)
    title = title.filter(__import__("PIL.ImageFilter", fromlist=["x"]).GaussianBlur(0.3))

    logo = Image.new("RGBA", (OW, OH), (0, 0, 0, 0))
    dl = ImageDraw.Draw(logo)
    cx, cy, R = OW - 132, OH - 44, 13
    orange = (217, 119, 87, 255)
    lr = np.random.default_rng(11)
    for i in range(12):   # Claude 星芒标志（简化绘制）
        ang = i * 2 * np.pi / 12 + lr.uniform(-0.12, 0.12)
        ln = R * lr.uniform(0.75, 1.0)
        x2, y2 = cx + ln * np.cos(ang), cy + ln * np.sin(ang)
        dl.line([(cx, cy), (x2, y2)], fill=orange, width=3)
        dl.ellipse([x2 - 1.5, y2 - 1.5, x2 + 1.5, y2 + 1.5], fill=orange)
    lf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 21)
    dl.text((cx + 22, cy - 13), "Claude", font=lf, fill=(235, 230, 222, 255))
    to_np = lambda im: np.asarray(im).astype(np.float32) / 255
    return to_np(title), to_np(logo)


def ramp(t, a, b):
    return float(np.clip((t - a) / (b - a), 0, 1)) ** 2 * (3 - 2 * float(np.clip((t - a) / (b - a), 0, 1))) \
        if b > a else float(t >= a)


# ------------------------------------------------------------------ 渲染
def main():
    still = None
    if "--still" in sys.argv:
        still = float(sys.argv[sys.argv.index("--still") + 1])
    dur = DUR
    hr, hp, esc, edir = trace()
    tex = disk_texture()
    bg = star_background(esc, edir)
    title, logo = overlay_layers()
    NR, NP = tex.shape

    layers = []
    for k in range(hr.shape[0]):
        m = ~np.isnan(hr[k])
        idx = np.nonzero(m)[0]
        r = hr[k][idx]
        ir = np.clip((r - RIN) / (ROUT - RIN) * (NR - 1), 0, NR - 1).astype(np.int32)
        x = np.clip((r - RIN) / (ROUT - RIN), 0, 1)
        prof = (RIN / np.maximum(r, RIN)) ** 1.6
        prof *= np.clip((r - RIN * 0.9) / (RIN * 0.25), 0, 1) * np.clip((ROUT * 1.05 - r) / 3.5, 0, 1)
        if BW:   # 黑白版：中性银白，外圈略暗
            g = 1.0 - 0.25 * x
            col = np.stack([g, g, g], -1)
        else:
            col = np.stack([np.ones_like(x), 0.84 - 0.36 * x, 0.6 - 0.48 * x], -1)
        omega = 0.55 * (RIN / np.maximum(r, RIN)) ** 1.5
        layers.append((idx, ir, hp[k][idx], prof.astype(np.float32), col.astype(np.float32), omega))

    def frame(t):
        img = bg.reshape(-1, 3).copy()
        trans = np.ones(RW * RH, np.float32)
        acc_c = np.zeros((RW * RH, 3), np.float32)
        for idx, ir, ph, prof, col, omega in layers:
            pr = (ph + omega * t) % (2 * np.pi)
            ip = (pr / (2 * np.pi) * NP).astype(np.int32) % NP
            nz = tex[ir, ip]
            inten = prof * nz
            alpha = np.clip(inten * 1.3, 0, 0.97)
            acc_c[idx] += (col * (inten * 2.2)[:, None]) * (trans[idx] * alpha)[:, None]
            trans[idx] *= 1 - alpha
        hdr = (acc_c + img * trans[:, None]).reshape(RH, RW, 3)
        small = hdr[::4, ::4]
        bloom = gaussian_filter(np.maximum(small - 0.6, 0), (10, 10, 0))
        bloom = zoom(bloom, (4, 4, 1), order=1)[:RH, :RW]
        hdr = hdr + 0.6 * bloom
        ldr = 1 - np.exp(-hdr * 1.25)
        ldr = np.clip(ldr, 0, 1) ** (1 / 1.15)
        # 缓慢推近
        z = 1.0 + 0.06 * (t / DUR)
        cw, ch = RW / z, RH / z
        x0, y0 = (RW - cw) / 2, (RH - ch) / 2
        im = Image.fromarray((ldr * 255).astype(np.uint8))
        im = im.resize((OW, OH), Image.LANCZOS, box=(x0, y0, x0 + cw, y0 + ch))
        out = np.asarray(im).astype(np.float32) / 255
        out *= ramp(t, 0.0, 1.8)
        if BW:   # 画面转灰度；标题和 Claude 标志在之后叠加，标志保留橙色
            lum = out @ np.array([0.299, 0.587, 0.114], np.float32)
            out = np.repeat(lum[..., None], 3, axis=2)
        for layer, a in ((title, ramp(t, 3.0, 4.8)), (logo, ramp(t, 4.2, 5.6))):
            al = layer[..., 3:4] * a
            out = out * (1 - al) + layer[..., :3] * al
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)

    if still is not None:
        Image.fromarray(frame(still)).save(os.path.join(HERE, ("still_bw" if BW else "still") + ("_en" if EN else "") + ".png"))
        print("still saved")
        return

    out = os.path.join(HERE, "stellar_echo_demo_bw.mp4" if BW else "stellar_echo_demo.mp4")
    audio = os.path.join(HERE, "..", "music", "stellar_echo.mp3")
    cmd = ["ffmpeg", "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-",
           "-ss", "0", "-t", str(dur), "-i", audio,
           "-af", f"afade=t=out:st={dur - 1.2}:d=1.2",
           "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    nf = int(dur * FPS)
    for i in range(nf):
        f = frame(i / FPS)
        if i >= nf - 18:   # 结尾淡出
            f = (f * ((nf - i) / 18)).astype(np.uint8)
        p.stdin.write(f.tobytes())
        if i % 30 == 0:
            print(f"  frame {i}/{nf}", flush=True)
    p.stdin.close()
    p.wait()
    print("saved", out)


if __name__ == "__main__":
    main()
