# 用法: tts.py 输出.mp3 "文本"   (edge-tts 少女音；让 certifi 指向代理 CA，不关闭 TLS 校验)
import sys,asyncio,certifi
certifi.where=lambda:'/root/.ccr/ca-bundle.crt'
import edge_tts
asyncio.run(edge_tts.Communicate(sys.argv[2],'zh-CN-XiaoyiNeural').save(sys.argv[1]))
