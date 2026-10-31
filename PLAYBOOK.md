# 《星际回响》制作经验手册

用代码从零生成「配乐 + 黑洞动画」的完整流程、参数和踩过的坑，下次做类似作品时直接照着用。

---

## 1. 成品一览

| 文件 | 说明 |
|---|---|
| `music/stellar_echo.mp3` | 小调版（第一版，视频配乐用这个） |
| `music/stellar_echo_major.mp3` | 大调欢快版 |
| `music/stellar_echo_lyrical.mp3` | 慢速抒情版（60 BPM） |
| `video/stellar_echo_demo.mp4` / `_bw.mp4` | 10 秒 demo（彩色 / 黑白） |
| `video/stellar_echo_full_bw.mp4` | 完整 128 秒黑白版（81MB，高画质存档） |
| `video/stellar_echo_full_bw_30mb.mp4` | 压缩版 28.7MB（分享用） |

## 2. 一键复现

```bash
pip install numpy scipy pillow        # 另需系统里有 ffmpeg
```

```bash
cd music && python3 compose.py && python3 compose.py major && python3 compose.py lyrical
```

```bash
cd video && python3 render_demo.py --still 7     # 先出静帧检查构图（首次约 2 分钟，含光线追踪）
```

```bash
cd video && python3 render_demo.py --bw          # 10 秒 demo；去掉 --bw 为彩色
```

```bash
cd video && python3 render_full.py --still 2 8 30 56 100 122   # 完整版关键帧预览
```

```bash
cd video && python3 render_full.py               # 完整 128 秒，4 进程约 6 分钟
```

压缩到 30MB 以内（两遍编码）：

```bash
ffmpeg -y -i stellar_echo_full_bw.mp4 -c:v libx264 -preset slower -b:v 1650k -pass 1 -an -f null /dev/null && ffmpeg -y -i stellar_echo_full_bw.mp4 -c:v libx264 -preset slower -b:v 1650k -maxrate 3500k -bufsize 7000k -pass 2 -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart out_30mb.mp4
```

码率估算：`目标大小(MB) × 8 ÷ 时长(秒) − 音频码率`，再留 5%~10% 余量。128 秒 / 30MB → 视频约 1650k。

---

## 3. 配乐（`music/compose.py`）

### 方法
- 纯 numpy 合成，不依赖音色库：管风琴（加法合成 + 两路微失谐）、钢琴（衰减泛音 + 轻微非谐性）、钢片琴、弦乐（3 路失谐锯齿波 + 低通）、低音、钟表滴答、定音鼓式低鼓。
- 混音：每个乐器一条轨，按比例送入一个公共混响（指数衰减噪声做卷积），最后 tanh 软限幅 + 归一化。
- 同一脚本用参数切换版本：`minor` / `major` / `lyrical`。

### 让时长「刚好」的技巧
- 选一个能整除的速度：**120 BPM × 4/4 → 每小节 2 秒 → 64 小节 = 128 秒**；抒情版 60 BPM → 32 小节。
- 段落按 8 小节划分，结构清楚，画面也好对齐。

### 「星际穿越」风格要素（只借鉴手法，旋律原创）
- 管风琴长音铺底、慢慢渐强；
- 贯穿全曲的钟表滴答（时间流逝感）；
- 重复的琶音 ostinato，后半段从 8 分音符加密到 16 分音符；
- 层层推进到高潮，再收回到最初的单薄配器。

### 好记的主旋律
- 8 小节、节奏简单（四分 + 少量八分），开头 3 个音走和弦分解（A–C–E），全曲重复 5 次以上。
- 和弦用流行进行：小调 Am–F–C–G，大调 C–G–Am–F。
- 改版本时**只换和声与音高，保留节奏型**，听众仍能认出是同一首。

### 自检
我（Claude）听不到声音，只能用数值检查：每 8 秒的 RMS 响度曲线应呈「安静前奏 → 渐强 → 高潮最响 → 回落」，峰值 < 0.9 不削波，时长精确 128.00 秒。最终听感要人来判断。

---

## 4. 黑洞动画（`video/render_demo.py`、`video/render_full.py`）

### 核心思路：光线只追踪一次
1. 对每个像素在史瓦西度规下追踪光线（等效牛顿形式加速度 `a = -1.5·h²·r̂ / r⁵`，史瓦西半径 = 1）。
2. 记录光线与吸积盘平面（y=0）的前 3 次交点 `(r, φ)`，以及逃逸方向。结果缓存到 `.trace_cache.npz`（77MB，已 gitignore）。
3. 每帧只做查表：盘面纹理按开普勒角速度 `ω ∝ r^-1.5` 旋转，多层交点按透明度前后合成。
→ 追踪 1920×1080 约 2 分钟，之后每帧只要零点几秒。

### 关键参数（改这些就能换风格）
| 参数 | 值 | 作用 |
|---|---|---|
| 相机仰角 `elev` | 7° | 越小越像电影里近乎侧视的卡冈图雅 |
| 相机距离 `dist` | 32 | 黑洞在画面中的大小 |
| 吸积盘 `RIN, ROUT` | 3, 13 | 内缘 = 最内稳定圆轨道 |
| 视场 | 50° | 构图松紧 |
| 盘面颜色 | 暖金 / 黑白 `--bw` | `col` 一行 |
| 星空透镜半径 `thE2` | 0.1² | 只让黑洞附近的星星弯曲 |

### 完整版如何「跟着音乐动」
- `BRIGHT`、`SPEED` 两条关键帧曲线对应音乐段落（前奏暗慢 → 高潮最亮最快 → 尾声回落）。
- 旋转用速度曲线的**积分**作为相位，变速时纹理不会跳。
- 从音频算每帧「瞬时响度 ÷ 1 秒平均响度」，快起慢落，作为亮度脉冲 → 低鼓一响光环就闪。
- 标题开头出现约 10 秒后淡出、结尾再现；Claude 标志一直留在右下角。

### 加速
- `multiprocessing` fork 4 个进程，各渲一段编码成独立 mp4，再用 ffmpeg concat 无损拼接并混入音频。
- 3840 帧 4 核约 6 分钟（单进程约 25 分钟）。长任务放后台跑。

### 工作流
**静帧 → 10 秒 demo → 关键帧预览图 → 完整版 → 压缩版**。每一步都先看图再继续，返工成本最低。

---

## 5. 踩过的坑

| 问题 | 原因 | 解决 |
|---|---|---|
| 星星被拉成短线 | 光线追踪远处步长大，逃逸方向有数值误差 | 星空不用追踪结果，改用点质量透镜公式 `β = θ(1 − θE²/θ²)` 在画面上解析映射 |
| 星空「假」 | 星星一样大、一样亮、太密 | 亮度服从幂律（大量暗星 + 少量亮星）、亮星加光晕、轻微色温差、淡银河 |
| 透镜让整个画面的星都弯曲、银河变成环 | 爱因斯坦环半径按物理值 0.25 rad 太大 | 收窄到 0.1 rad，只在黑洞附近变形（牺牲一点物理真实换简洁） |
| 黑白版里标志也变灰 | 灰度转换在叠加文字之后 | 先把画面转灰度，再叠加标题和 Claude 标志 |
| 黑白版标题偏暖 | 标题颜色是暖白 | 黑白版用中性白 |
| Claude 标志 | 没有官方文件，是手绘近似的星芒 | 有官方 logo 时替换 `overlay_layers()` 里的绘制代码 |

---

## 6. GitHub / 会话环境的坑

- **推送 403**：Claude 绑定的 GitHub 账号是 `zhaoyao75`，而 `hello-world` 属于另一个账号 `zhaoyao1975`，所以没有写权限，在 App 设置里也找不到那个仓库。→ 作品放在自己账号下的仓库（本仓库 `zhaoyao75/Colab-Code`）。
- 在会话中添加仓库时要用**推送权限**（push），只读克隆无法推送。
- 会话结束时有「未推送 / 未跟踪文件」检查：临时文件（`.trace_cache.npz`、`.seg_*`、`__pycache__`、`*.wav`）都要写进 `.gitignore`。
- 自动模式下 `git reset --hard` 这类不可逆操作会被拦截；需要撤掉本地提交时，改用「移到备份分支」的方式。
- 单个文件 GitHub 限制 100MB（超过 50MB 会警告）。完整视频 81MB 尚可，更长或更高清时考虑 Git LFS 或只提交压缩版。

---

## 7. 下次可以直接这样提需求

- 「参考 ×× 的风格，生成 N 秒、BPM 为 ××、调性 ×× 的配乐，主旋律要好记」——先说清楚时长、速度、情绪。
- 「先出静帧 / 10 秒 demo」——确认风格后再出完整版，省时间。
- 「彩色 / 黑白 / 标志保留颜色」「标题出现时间」「分享平台的大小限制」——一开始说明，可以少改几轮。
- 想换旋律或音色：改 `compose.py` 里的 `THEME` 和乐器函数；想换画面：改第 4 节的参数表。

---

## 8. 带 AI 旁白的短视频（`narrated3/`《黄永玉 · 永远长不大的老头》）

成品：`narrated3/huang_yongyu_full.mp4`（完整版 48MB，crf 18）、`narrated3/huang_yongyu_30mb.mp4`（压缩版 26MB，两遍 1620k）、`subtitles_*.srt`（中 / 英 / 双语外挂字幕）。

### 流程与命令
```bash
cd narrated3 && python3 tts.py && python3 music.py && python3 mix.py   # 旁白 → 配乐 → 混音（约 1 分钟）
python3 render.py --still 4 24 38 56 70 92 104 120 127                  # 关键帧静帧
python3 render.py --demo 0 16                                           # 16 秒 demo（约 40 秒）
python3 render.py                                                       # 完整 128 秒（4 进程约 5 分钟）
```
文案唯一来源是 `script.json`（每句中文 + 英文）；改文案后重跑整条链路，字幕时间轴自动更新。

### 语音合成（TTS）选型
| 方案 | 结论 |
|---|---|
| **edge-tts**（微软在线，免费、无需 key） | 中文最自然，首选。云健=浑厚讲故事，云希=青年，晓晓=温暖女声 |
| Kokoro（sherpa-onnx，HF `csukuangfj/kokoro-int8-multi-lang-v1_1`） | 离线兜底；sid 3–57 女声、58–102 男声 |
| piper huayan / gTTS / espeak-ng | 依次变差，espeak 只是机器人腔 |

### 坑
| 问题 | 解决 |
|---|---|
| edge-tts 报 `CERTIFICATE_VERIFY_FAILED` | 沙箱代理的 CA 不在 certifi 里：`edge_tts.communicate._SSL_CTX = ssl.create_default_context(cafile="/root/.ccr/ca-bundle.crt")`，并传 `proxy=$HTTPS_PROXY` |
| GitHub releases 403（piper / sherpa 模型） | 改从 HuggingFace 下载（`huggingface_hub.snapshot_download`） |
| 字幕时间对不准 | 逐句合成、裁掉首尾静音、自己排时间轴，比用 TTS 的边界事件更可控 |
| 马善政字体里没有「·」→ 方块 | 标题避开生僻符号；DejaVu 没有中文，中英文分别用不同字体 |
| PIL 在 RGBA 图上用半透明颜色画线不会混合，而是直接改写成半透明像素 | 半透明效果画到单独图层再 `alpha_composite` |
| 纸张纹理自带暗角，拼成书页后中间出现亮带 | `paper(vig=False)` |
| 字幕挡住画面主体 | 主体放在画面上 80%，静帧检查时专门看底部 |

### 混音
- 旁白说话时音乐比人声低约 15–18 dB，空隙处低约 8–12 dB（按人声包络自动压低，0.25 秒平滑）。
- 两遍 `loudnorm` 到 -14 LUFS、真峰值 -1.5 dB（抖音 / B 站常用）。
- 配乐音区偏高（钢片琴、马林巴），低切 120Hz，给人声让出中低频。

### 内容注意
- 讲真实人物时只用公开事实，不编造引语；形象原创，不临摹作者原作。
- 片尾标注「AI 配音 / 代码生成」，符合平台的 AI 内容标识要求。
