"""
抖音封面：cover_9x16.jpg（1080x1920，全屏）和 cover_3x4.jpg（1080x1440，个人主页格子，取 9:16 的 y 240~1680）。
文字都放在 3:4 范围内。复用 render.py 的背景和星星头角色（挥手姿势）。
运行：cd fitness_coach && python3 make_cover.py
"""
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


if __name__ == "__main__":
    main()
