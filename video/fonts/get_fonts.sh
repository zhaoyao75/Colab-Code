#!/bin/sh
# 下载预告片字幕用的开源字体（SIL OFL 许可，可商用）。字体文件较大，不提交到仓库。
cd "$(dirname "$0")"
B=https://raw.githubusercontent.com
curl -sSL -o NotoSerifSC-ExtraLight.otf "$B/notofonts/noto-cjk/main/Serif/SubsetOTF/SC/NotoSerifSC-ExtraLight.otf"   # 思源宋体 极细
curl -sSL -o NotoSerifSC-Light.otf      "$B/notofonts/noto-cjk/main/Serif/SubsetOTF/SC/NotoSerifSC-Light.otf"        # 思源宋体 细
curl -sSL -o CormorantGaramond-Italic.ttf "$B/google/fonts/main/ofl/cormorantgaramond/CormorantGaramond-Italic%5Bwght%5D.ttf"   # 英文衬线斜体（可变字重）
curl -sSL -o CormorantGaramond.ttf        "$B/google/fonts/main/ofl/cormorantgaramond/CormorantGaramond%5Bwght%5D.ttf"
curl -sSL -o BarlowCondensed-BlackItalic.ttf "$B/google/fonts/main/ofl/barlowcondensed/BarlowCondensed-BlackItalic.ttf"   # 窄体粗斜体（F1 转播风格）
curl -sSL -o RacingSansOne-Regular.ttf "$B/google/fonts/main/ofl/racingsansone/RacingSansOne-Regular.ttf"
curl -sSL -o Montserrat-Italic.ttf "$B/google/fonts/main/ofl/montserrat/Montserrat-Italic%5Bwght%5D.ttf"
