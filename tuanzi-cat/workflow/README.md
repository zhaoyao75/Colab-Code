# 团子猫日常 · 一集成片工作流

## 一键运行（阶段: tts→check→frames→audio→mux→covers→pack→qa）
    ./build.sh ep3            # 全流程: tts → frames → audio → mux → covers → pack
    ./build.sh ep3 audio mux  # 只重跑某几个阶段（改了台词时间/配乐，不必重渲染画面）
产物在 out/ep3/：video.mp4、covers/、check/关键帧拼图.png、交付/（成片、音轨 mp3、四种封面 + 2x、发布文案）。

## 目录
- common/        跨集共用：head5.html（猫/字幕/特效函数，不要改）、fonts/（站酷快乐体）、meow/（真实猫叫 A–F）、tools/
- episodes/epN/  每集一份：episode.json（台词、时间、猫叫、封面文字）、tail.html（场景 R(t)）、audio.py（配乐与音效）、发布文案.md
- 新开一集：复制 episodes/ep3 → ep4，改 episode.json，重写 tail.html 的 R(t) 与 audio.py 的音效时间。

## 制作流程（每一步人工把关的点）
1. **选题与分镜**：先给用户 2–3 个方案（0–2 秒要有「软萌×冷淡」的反差钩子），用户选定后再做；骨架不能和前几集相似（偷蛋糕→被抓→奖励；嫌贵窝爱纸箱；嫌水温嘴硬）。下一集题材必须向用户确认。
2. **台词定稿**：全角标点少用「！！」「……」；用户明确要求的除外（第3集用了「。。。」）。
3. **配音**（tts）：edge-tts zh-CN-XiaoyiNeural，需让 certifi 指向 /root/.ccr/ca-bundle.crt，不能关 TLS 校验；每句裁首尾静音，build.sh 会打印每句起止时间，超出总时长会报 ⚠。
4. **场景**（frames）：SVG 页面 R(t)，Playwright 6 进程并行截帧，HTML 必须 <meta charset="utf-8">。
5. **声音**（audio）：开场几乎无声→目标揭晓后起配乐→结尾配乐戛然而止；猫叫每集 1–2 声；配音时 sidechain 压低配乐；两遍 loudnorm 到 -14 LUFS，峰值 < -3 dB。
6. **合成与质检**（mux）：1080×1920、30fps、H.264 High + AAC 48kHz 立体声、faststart。看 check/关键帧拼图.png（遮挡、被边缘切掉、气泡压脸、文字重叠）。
7. **封面**（covers）：9:16、3:4、1:1 合集、4:3 横版，PNG；标题须在中间安全区，9:16 要留出抖音界面遮挡。
8. **交付说明（必须如实）**：只抽查了关键帧、没听过声音；猫叫 ESC-50 为 CC BY-NC，配音为微软合成语音，商用前自行确认；结尾卡的预告语和下集题材需用户确认。

## 环境要求
ffmpeg、python3（numpy、edge-tts、certifi）、node + playwright、能访问 speech.platform.bing.com（走代理）。

## 自动检查（build.sh 内置）
- **台词时间自动排布**（tts 阶段，common/tools/plan.py）：episode.json 的每句台词可写 `at`（固定起点，如对齐动画）或 `after`+`gap`（接在上一句后）；加 `fit:true` 时若会超出总时长，自动加速（最多 1.5 倍，再长就报错让你缩短台词）。结果写入 out/<集>/timeline.json，场景字幕、结尾卡出现时间、配乐收尾、结尾“叮”都从它读，改台词不用手算时间。
- **一致性检查**（check 阶段）：每句配音都有对应字幕；结尾卡含「第N集」「下集预告」和下集题材、不含「关注」；封面标题与开场台词一致；episode.json 的 `banned` 里列的旧词不能残留在场景、台词、文案、文件名里；文案须包含 `copy_must_include` 的关键句；成片和封面文件名符合「第N集_集名_秒数s」命名。
- **自动质检**（qa 阶段）：分辨率 1080×1920、30fps、时长 10–12s、AAC 48kHz 立体声、响度 -14±0.5 LUFS、峰值 ≤ -3 dB、每秒都有声音、配音不超总长、字幕不超出画面、四种封面尺寸（≥640×854）。报告写入 out/<集>/qa报告.md；任何一项不通过，build.sh 以失败退出。
- 这些检查**不能代替人工**：声音没人听过，只能抽查关键帧；猫叫（CC BY-NC）和合成配音的商用许可要自己确认。
- 提示：check 阶段会检查已有的「交付」目录，改了集名或文案后请重新跑 pack 再 qa。

## 更新记录
- 第4集《雨滴赛马》：新增 `episodes/ep4`；`covers.js` 支持每集自定义取景（`cover.focus` / `cover.zoom` / `cover.sy`，zoom 与 sy 可按尺寸 9x16、3x4、4x3、1x1 写成对象）；`plan.py` 自动加速留 0.06s 余量。
- 经验总结见 `../经验总结_第3-4集与工作流.md`，第1、2集那份是 `tuanzi-cat/README_经验总结与第3集交接.md`。
- 交付检查项里「禁用旧词」`banned` 每集都要更新（第4集加了「全体」「落选」）。
- 目前已有 ep3、ep4 两集；第5集《团子猫开快递》待做。
