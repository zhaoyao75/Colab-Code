"""渲染：frame(t) -> 480 帧 -> ffmpeg 合成 + mix.wav。python3 render.py（约 1–2 分钟）"""
import subprocess, sys, os
from multiprocessing import Pool
from PIL import Image
import shots

os.makedirs(".frames", exist_ok=True)
NF = int(shots.DUR * shots.FPS)

def work(i):
    fn = f".frames/{i:04d}.jpg"
    if not os.path.exists(fn):
        Image.fromarray(shots.frame(i / shots.FPS)).save(fn + ".tmp.jpg", quality=95, subsampling=0)
        os.rename(fn + ".tmp.jpg", fn)
    return i

if __name__ == "__main__":
    with Pool(os.cpu_count()) as p:
        for _ in p.imap_unordered(work, range(NF), chunksize=8): pass
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(shots.FPS), "-i", ".frames/%04d.jpg",
                    "-i", "mix.wav", "-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "128k", "-t", "16", "-movflags", "+faststart", "regret_button_douyin.mp4"], check=True)
