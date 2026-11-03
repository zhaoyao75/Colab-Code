const {chromium}=require('playwright');
const W=+process.argv[2],N=+process.argv[3];
(async()=>{const b=await chromium.launch();const p=await b.newPage({viewport:{width:1080,height:1920}});
await p.goto('file://'+process.cwd()+'/scene6.html');await p.evaluate(()=>document.fonts.load('48px KuaiLe'));await p.evaluate(()=>document.fonts.ready);
for(let i=W;i<336;i+=N){await p.evaluate(t=>R(t),i/30);await p.screenshot({path:`f7/${String(i).padStart(3,'0')}.png`});}
await b.close();})();
