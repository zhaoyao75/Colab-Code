#!/bin/bash
# 用法: mix.sh <工作目录>  需要 bgm.wav sfx.wav vo.wav → final.wav (-14 LUFS 两遍 loudnorm)
set -e; cd "$1"
ffmpeg -v error -y -i bgm.wav -i sfx.wav -i vo.wav -filter_complex "[2]asplit[v1][v2];[0]volume=0.5[b];[b][v1]sidechaincompress=threshold=0.03:ratio=8:attack=5:release=250[bd];[1]volume=0.8[s];[bd][s][v2]amix=inputs=3:normalize=0,aformat=sample_rates=48000:channel_layouts=stereo[o]" -map "[o]" mixraw.wav
J=$(ffmpeg -i mixraw.wav -af loudnorm=I=-14:TP=-3.5:LRA=7:print_format=json -f null - 2>&1 | sed -n '/^{/,/^}/p')
g(){ echo "$J"|python3 -c "import sys,json;print(json.load(sys.stdin)['$1'])"; }
ffmpeg -v error -y -i mixraw.wav -af "loudnorm=I=-14:TP=-3.5:LRA=7:measured_I=$(g input_i):measured_TP=$(g input_tp):measured_LRA=$(g input_lra):measured_thresh=$(g input_thresh):offset=$(g target_offset):linear=true,aformat=sample_rates=48000:channel_layouts=stereo" final.wav
