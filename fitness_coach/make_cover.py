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
    rd.stext(dr, (W / 2, 362), "16秒挑战", 190, rd.YEL, stroke=13)
    rd.stext(dr, (W / 2, 508), "久坐的你，敢接吗？", 60, (255, 255, 255), stroke=8)
    dr.rounded_rectangle([W / 2 - 170, 1318, W / 2 + 170, 1388], radius=35, fill=rd.PANEL, outline=(120, 255, 225, 255), width=3)
    rd.stext(dr, (W / 2, 1353), "AI虚拟教练·小燃", 34, (255, 255, 255), stroke=0)
    rd.stext(dr, (W / 2, 1480), "居家小挑战 · 量力而行", 52, (255, 255, 255), stroke=8)
    im.alpha_composite(ov)
    img = im.convert("RGB")
    img.save(os.path.join(HERE, "cover_9x16.jpg"), quality=93)
    img.crop((0, 240, W, 1680)).save(os.path.join(HERE, "cover_3x4.jpg"), quality=93)
    print("covers saved")


if __name__ == "__main__":
    main()
