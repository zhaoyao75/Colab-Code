#!/bin/bash
# 团子猫日常 一集成片工作流
# 用法: ./build.sh <集目录名,如 ep3> [阶段...]   阶段: tts frames audio mux covers pack all(默认)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd); EP=${1:?集目录名}; shift || true; STAGES=${*:-all}
SRC=$ROOT/episodes/$EP; CFG=$SRC/episode.json; OUT=$ROOT/out/$EP; C=$ROOT/common
export NODE_PATH=${NODE_PATH:-$(npm root -g)}
j(){ python3 -I -c "import json,sys;print(json.load(open('$CFG'))$1)"; }
TOTAL=$(j "['total']"); FPS=$(j "['fps']"); NAME=$(j "['name']"); NO=$(j "['no']"); FR=$(python3 -c "print(int($TOTAL*$FPS))")
mkdir -p "$OUT"/{vo,f,covers}
build_scene(){ { cat "$C/head5.html"; echo "window.TL=$(cat "$OUT/timeline.json");"; cat "$SRC/tail.html"; } > "$OUT/scene.html"; ln -sfn "$C/fonts" "$OUT/fonts"; }
FAIL=0
has(){ [[ " $STAGES " == *" all "* || " $STAGES " == *" $1 "* ]]; }

if has tts; then echo "[tts] 配音 + 台词时间排布"
  python3 -I "$C/tools/plan.py" "$CFG" "$OUT/vo" "$C/tools/tts.py" "$OUT/timeline.json"
fi

if has check; then echo "[check] 一致性检查"
  python3 -I "$C/tools/check.py" "$SRC" "$OUT" || FAIL=1
fi

if has frames; then echo "[frames] 渲染 $FR 帧"
  build_scene
  for w in 0 1 2 3 4 5; do node "$C/tools/render_frames.js" "$OUT" "$OUT/f" $FR $FPS $w 6 & done; wait
  echo "  帧数: $(ls "$OUT/f" | wc -l)"
fi

if has audio; then echo "[audio] 配乐/音效/混音/响度"
  python3 -I "$SRC/audio.py" "$OUT" "$CFG" "$C"; "$C/tools/mix.sh" "$OUT"
fi

if has mux; then echo "[mux] 合成成片"
  ffmpeg -y -v error -framerate $FPS -i "$OUT/f/%03d.png" -i "$OUT/final.wav" -c:v libx264 -profile:v high -pix_fmt yuv420p -crf 20 -r $FPS -c:a aac -b:a 192k -ar 48000 -ac 2 -t $TOTAL -movflags +faststart "$OUT/video.mp4"
  echo "  规格: $(ffprobe -v error -select_streams v -show_entries stream=width,height,r_frame_rate -of csv=p=0 "$OUT/video.mp4")"
  ffmpeg -i "$OUT/video.mp4" -af ebur128=peak=true -f null - 2>&1 | grep -E "^\s+I:|Peak:" | tail -2
  mkdir -p "$OUT/check"; for t in 1 3 5 7 9 11; do ffmpeg -v error -y -ss $t -i "$OUT/video.mp4" -frames:v 1 -vf scale=270:-1 "$OUT/check/k$t.png"; done
  ffmpeg -v error -y -i "$OUT/check/k1.png" -i "$OUT/check/k3.png" -i "$OUT/check/k5.png" -i "$OUT/check/k7.png" -i "$OUT/check/k9.png" -i "$OUT/check/k11.png" -filter_complex hstack=6 "$OUT/check/关键帧拼图.png"
  echo "  关键帧拼图: $OUT/check/关键帧拼图.png  (人工目视检查)"
fi

if has covers; then echo "[covers] 封面"
  build_scene
  node "$C/tools/covers.js" "$OUT" "$CFG" "$OUT/covers"
fi

if has pack; then echo "[pack] 交付包"
  P=$OUT/交付; mkdir -p "$P"
  cp "$OUT/video.mp4" "$P/第${NO}集_${NAME}_${TOTAL%.*}s.mp4"
  ffmpeg -v error -y -i "$OUT/video.mp4" -vn -c:a libmp3lame -b:a 192k "$P/第${NO}集_音轨试听.mp3"
  for k in 9x16:竖版9x16 3x4:主页3x4 1x1:合集方形1x1 4x3:横版4x3; do a=${k%%:*}; n=${k##*:}
    cp "$OUT/covers/cover_${a}_2x.png" "$P/第${NO}集_封面_${n}_2x.png"
    ffmpeg -v error -y -i "$OUT/covers/cover_${a}_2x.png" -vf scale=iw/2:ih/2 "$P/第${NO}集_封面_${n}.png"; done
  [ -f "$SRC/发布文案.md" ] && cp "$SRC/发布文案.md" "$P/第${NO}集_发布文案.md"
  if [ -f "$P/第${NO}集_发布文案.md" ] && [ -f "$C/fixed_tags.txt" ]; then   # 固定话题标签（common/fixed_tags.txt）
    python3 -I - "$P/第${NO}集_发布文案.md" "$C/fixed_tags.txt" <<'P'
import re,sys
f,t=sys.argv[1:3];tags=open(t,encoding='utf8').read().strip();s=open(f,encoding='utf8').read()
s=re.sub(r"(## 话题\n).*?(\n\n##)",lambda m:m.group(1)+tags+m.group(2),s,flags=re.S);open(f,'w',encoding='utf8').write(s)
P
  fi
  echo "  交付目录: $P"
fi

if has qa; then echo "[qa] 自动质检"
  build_scene; python3 -I "$C/tools/qa.py" "$SRC" "$OUT" "$C" || FAIL=1
  python3 -I "$C/tools/check.py" "$SRC" "$OUT" > "$OUT/一致性报告.txt" || FAIL=1; tail -1 "$OUT/一致性报告.txt"
fi
[ $FAIL = 0 ] && echo "✅ 全部检查通过（仍需人工：听声音、看拼图、确认许可）" || { echo "❌ 有检查未通过，见上方 ✗ 项"; exit 1; }
