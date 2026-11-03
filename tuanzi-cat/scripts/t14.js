const {chromium}=require('playwright');
(async()=>{const b=await chromium.launch();const p=await b.newPage({viewport:{width:1080,height:1920}});
const e=[];p.on('pageerror',x=>e.push(x.message));
await p.goto('file://'+process.cwd()+'/scene6.html');await p.evaluate(()=>document.fonts.load('48px KuaiLe'));await p.evaluate(()=>document.fonts.ready);
const T=process.argv.slice(2).map(Number);
for(const t of T){await p.evaluate(t=>R(t),t);await p.screenshot({path:`e6_${t}.png`});}
console.log(e);await b.close();})();
