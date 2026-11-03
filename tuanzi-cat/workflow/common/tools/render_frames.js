// 用法: node render_frames.js <scene.html所在目录> <帧输出目录> <总帧数> <fps> <进程号> <进程数>
const {chromium}=require('playwright');
const [dir,out,N,fps,W,P]=process.argv.slice(2);
(async()=>{const b=await chromium.launch();const p=await b.newPage({viewport:{width:1080,height:1920}});
p.on('pageerror',e=>console.log('ERR',e.message));
await p.goto('file://'+dir+'/scene.html');await p.evaluate(()=>document.fonts.load('48px KuaiLe'));await p.evaluate(()=>document.fonts.ready);
for(let i=+W;i<+N;i+=+P){await p.evaluate(t=>R(t),i/+fps);await p.screenshot({path:`${out}/${String(i).padStart(3,'0')}.png`});}
await b.close();})();
