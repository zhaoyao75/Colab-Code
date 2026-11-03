# 团子猫日常：素材归档（第1、2集）

给后续集数复用的源码与素材。先读 `README_经验总结与第3集交接.md`。

- `scene/head5.html`：猫、表情、字幕、爱心、肥皂泡等公共函数；`scene/tail6.html`：第2集场景（`R(t)` 按时间渲染）；`scene/scene6.html`：两者拼好的成品页。
- `scripts/`：渲染（`run.js`/`run7.js`，Playwright 并行截帧）和各尺寸封面脚本。运行目录需含 `fonts/`，页面里字体路径为 `fonts/ZCOOLKuaiLe-Regular.ttf`。
- `audio/`：切好的真实猫叫（来自 ESC-50）、第2集配音、配乐脚本。
- `episodes/`：第1、2集成片，用来对照猫的造型和节奏。
- `covers/`：已出的封面样例。

## 许可
- 字体：站酷快乐体，OFL 开源，可商用。
- 猫叫：ESC-50 数据集，**CC BY-NC（要署名，不可商用）**，商用前必须更换。
- 配音：微软 edge-tts 合成语音，商用前请自行确认其使用条款。
