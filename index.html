import { useState, useEffect, useRef, useCallback } from "react";

/* ─── ML ENGINE ─────────────────────────────────────────────────────────── */
const DATASET = [
  { text: "I absolutely love this product! It works perfectly.", label: "positive" },
  { text: "Amazing experience, highly recommend to everyone.", label: "positive" },
  { text: "The service was outstanding and the staff were so friendly.", label: "positive" },
  { text: "This is the best purchase I have ever made!", label: "positive" },
  { text: "Fantastic quality, exceeded all my expectations.", label: "positive" },
  { text: "Really happy with the results, will definitely buy again.", label: "positive" },
  { text: "Wonderful product, great value for money spent.", label: "positive" },
  { text: "Super fast delivery and excellent packaging overall.", label: "positive" },
  { text: "Brilliant design and very easy to use daily.", label: "positive" },
  { text: "I am thrilled with this purchase, absolutely perfect.", label: "positive" },
  { text: "Great customer support and top-notch quality always.", label: "positive" },
  { text: "Exceeded expectations, very impressive performance indeed.", label: "positive" },
  { text: "Love the new update, so much better than before.", label: "positive" },
  { text: "Works exactly as described, very satisfied customer.", label: "positive" },
  { text: "Incredible value, would strongly recommend to anyone.", label: "positive" },
  { text: "The craftsmanship is absolutely extraordinary, worth every penny.", label: "positive" },
  { text: "Stunning quality, the attention to detail is exquisite.", label: "positive" },
  { text: "Truly extraordinary product, the craftsmanship blew me away.", label: "positive" },
  { text: "Exquisite finish and premium feel, absolutely delightful.", label: "positive" },
  { text: "The quality is extraordinary, far beyond what I expected.", label: "positive" },
  { text: "Superb craftsmanship, elegant design, and flawless execution.", label: "positive" },
  { text: "Magnificent product, remarkable detail and care throughout.", label: "positive" },
  { text: "Exceptional quality and remarkable attention to every detail.", label: "positive" },
  { text: "Beautifully made, premium materials, and outstanding finish.", label: "positive" },
  { text: "Impressive and well crafted, exceeded every single expectation.", label: "positive" },
  { text: "Absolutely delighted with this purchase, truly outstanding.", label: "positive" },
  { text: "Perfect in every way, flawless quality and beautiful design.", label: "positive" },
  { text: "Remarkable value, superb quality, I am very impressed.", label: "positive" },
  { text: "This product is stunning, the quality is second to none.", label: "positive" },
  { text: "The craftsmanship is absolutely extraordinary, worth every penny.", label: "positive" },
  { text: "Stunning quality, the attention to detail is exquisite.", label: "positive" },
  { text: "Truly extraordinary product, the craftsmanship blew me away.", label: "positive" },
  { text: "Exquisite finish and premium feel, absolutely delightful.", label: "positive" },
  { text: "The quality is extraordinary, far beyond what I expected.", label: "positive" },
  { text: "Superb craftsmanship, elegant design, and flawless execution.", label: "positive" },
  { text: "Magnificent product, remarkable detail and care throughout.", label: "positive" },
  { text: "Exceptional quality and remarkable attention to every detail.", label: "positive" },
  { text: "Beautifully made, premium materials, and outstanding finish.", label: "positive" },
  { text: "Impressive and well crafted, exceeded every single expectation.", label: "positive" },
  { text: "Absolutely delighted with this purchase, truly outstanding.", label: "positive" },
  { text: "Perfect in every way, flawless quality and beautiful design.", label: "positive" },
  { text: "Remarkable value, superb quality, I am very impressed.", label: "positive" },
  { text: "This product is stunning, the quality is second to none.", label: "positive" },
  { text: "This is a terrible product, completely useless.", label: "negative" },
  { text: "Worst experience of my life, do not buy.", label: "negative" },
  { text: "Very disappointed, broke after just one day.", label: "negative" },
  { text: "Absolutely awful, total waste of money.", label: "negative" },
  { text: "Poor quality and terrible customer service.", label: "negative" },
  { text: "Never again, this product is complete garbage.", label: "negative" },
  { text: "Defective item, does not work at all.", label: "negative" },
  { text: "Horrible experience, very frustrating and disappointing.", label: "negative" },
  { text: "Cheap and nasty, not worth the price.", label: "negative" },
  { text: "Complete disappointment, deeply regret buying it.", label: "negative" },
  { text: "Misleading description, very unhappy with purchase.", label: "negative" },
  { text: "Stopped working immediately, terrible quality overall.", label: "negative" },
  { text: "The worst product I have ever encountered.", label: "negative" },
  { text: "Do not waste your money on this awful item.", label: "negative" },
  { text: "Broken on arrival and no support from seller.", label: "negative" },
  { text: "It is okay, nothing particularly special about it.", label: "neutral" },
  { text: "Average product, does what it says nothing more.", label: "neutral" },
  { text: "Neither good nor bad, just barely acceptable.", label: "neutral" },
  { text: "Mediocre quality, could definitely be better.", label: "neutral" },
  { text: "It works as expected, nothing more nothing less.", label: "neutral" },
  { text: "Decent product, average performance across board.", label: "neutral" },
  { text: "Standard quality, meets basic needs adequately.", label: "neutral" },
  { text: "It is fine for the price, nothing extraordinary.", label: "neutral" },
  { text: "Passable quality, not impressed but not disappointed.", label: "neutral" },
  { text: "The product is alright, it does its job.", label: "neutral" },
];

const SW = new Set(["i","me","my","we","our","you","your","he","him","his","she","her","it","its","they","them","their","what","which","who","this","that","these","those","am","is","are","was","were","be","been","have","has","had","do","does","did","a","an","the","and","but","if","or","of","at","by","for","with","to","from","in","on","as","so","too","just","also","very","really","much","even","both","each","some","more","most","other","no","not","only","same","than","now","here","there","when","where","how","all","any","would","could","should","can","will","s","t","d","ll"]);
const clean = t => t.toLowerCase().replace(/[^a-z\s]/g," ").split(/\s+/).filter(w=>w.length>1&&!SW.has(w));

class TFIDF {
  constructor(){this.v={};this.idf={};this.n=0;}
  fit(docs){const N=docs.length,df={};docs.forEach(t=>{new Set(t).forEach(w=>{df[w]=(df[w]||0)+1;});t.forEach(w=>{if(!(w in this.v))this.v[w]=this.n++;});});Object.keys(this.v).forEach(w=>{this.idf[w]=Math.log((N+1)/((df[w]||0)+1))+1;});return this;}
  tr(tok){const tf={};tok.forEach(t=>{tf[t]=(tf[t]||0)+1;});const v=new Array(this.n).fill(0);Object.entries(tf).forEach(([t,c])=>{if(t in this.v)v[this.v[t]]=(c/tok.length)*(this.idf[t]||1);});const n=Math.sqrt(v.reduce((s,x)=>s+x*x,0))||1;return v.map(x=>x/n);}
}

class LR {
  constructor(C,lr=0.1,ep=200){this.C=C;this.lr=lr;this.ep=ep;this.W={};this.B={};}
  sm(s){const m=Math.max(...s),e=s.map(x=>Math.exp(x-m)),t=e.reduce((a,b)=>a+b,0);return e.map(x=>x/t);}
  fit(X,y){const d=X[0].length;this.C.forEach(c=>{this.W[c]=Array.from({length:d},()=>(Math.random()-.5)*.01);this.B[c]=0;});for(let e=0;e<this.ep;e++)X.forEach((x,i)=>{const p=this.sm(this.C.map(c=>x.reduce((s,v,j)=>s+v*this.W[c][j],0)+this.B[c]));this.C.forEach((c,ci)=>{const err=p[ci]-(c===y[i]?1:0);x.forEach((v,j)=>{this.W[c][j]-=this.lr*err*v;});this.B[c]-=this.lr*err;});});return this;}
  prob(x){const s=this.C.map(c=>x.reduce((a,v,j)=>a+v*this.W[c][j],0)+this.B[c]),p=this.sm(s);return Object.fromEntries(this.C.map((c,i)=>[c,p[i]]));}
  pred(x){const p=this.prob(x);return Object.entries(p).sort((a,b)=>b[1]-a[1])[0][0];}
}

class NB {
  constructor(C){this.C=C;}
  fit(X,y){this.pr={};this.mu={};this.va={};this.C.forEach(c=>{const idx=y.map((yi,i)=>yi===c?i:-1).filter(i=>i>=0),rows=idx.map(i=>X[i]),d=X[0].length;this.pr[c]=idx.length/y.length;this.mu[c]=new Array(d).fill(0);this.va[c]=new Array(d).fill(0);rows.forEach(r=>r.forEach((v,j)=>{this.mu[c][j]+=v;}));this.mu[c]=this.mu[c].map(v=>v/rows.length);rows.forEach(r=>r.forEach((v,j)=>{this.va[c][j]+=Math.pow(v-this.mu[c][j],2);}));this.va[c]=this.va[c].map(v=>Math.max(v/rows.length,1e-9));});return this;}
  ll(x,c){return x.reduce((s,v,j)=>{const va=this.va[c][j];return s-.5*Math.log(2*Math.PI*va)-Math.pow(v-this.mu[c][j],2)/(2*va);},Math.log(this.pr[c]));}
  prob(x){const L=Object.fromEntries(this.C.map(c=>[c,this.ll(x,c)])),m=Math.max(...Object.values(L)),E=Object.fromEntries(Object.entries(L).map(([c,l])=>[c,Math.exp(l-m)])),t=Object.values(E).reduce((a,b)=>a+b,0);return Object.fromEntries(Object.entries(E).map(([c,e])=>[c,e/t]));}
  pred(x){const p=this.prob(x);return Object.entries(p).sort((a,b)=>b[1]-a[1])[0][0];}
}

function calcMetrics(yT,yP,C){
  const cm={};C.forEach(c=>{cm[c]={tp:0,fp:0,fn:0};});
  yT.forEach((t,i)=>{const p=yP[i];t===p?cm[t].tp++:(cm[p]&&cm[p].fp++,cm[t].fn++);});
  const acc=yT.filter((t,i)=>t===yP[i]).length/yT.length;
  const pc={};C.forEach(c=>{const pr=cm[c].tp/(cm[c].tp+cm[c].fp)||0,re=cm[c].tp/(cm[c].tp+cm[c].fn)||0,f1=pr+re?2*pr*re/(pr+re):0;pc[c]={precision:pr,recall:re,f1};});
  return{accuracy:acc,perClass:pc,macroF1:C.reduce((s,c)=>s+pc[c].f1,0)/C.length};
}

function buildPipeline(){
  const C=["positive","negative","neutral"];
  const toks=DATASET.map(d=>clean(d.text));
  const vec=new TFIDF().fit(toks);
  const X=toks.map(t=>vec.tr(t)),y=DATASET.map(d=>d.label);
  const sp=Math.floor(X.length*.8);
  const lr=new LR(C).fit(X.slice(0,sp),y.slice(0,sp));
  const nb=new NB(C).fit(X.slice(0,sp),y.slice(0,sp));
  return{vec,lr,nb,C,trainSize:sp,testSize:X.length-sp,vocabSize:vec.n,
    mLR:calcMetrics(y.slice(sp),X.slice(sp).map(x=>lr.pred(x)),C),
    mNB:calcMetrics(y.slice(sp),X.slice(sp).map(x=>nb.pred(x)),C)};
}

/* ─── DESIGN TOKENS ─────────────────────────────────────────────────────── */
const T = {
  bg: "#0a0804",
  cream: "#f5f0e8",
  gold: "#c9a84c",
  goldLight: "#e8c96a",
  goldDim: "rgba(201,168,76,0.15)",
  rust: "#8B2500",
  emerald: "#0d4f3c",
  slate: "#4a5568",
  paper: "#1a1510",
  border: "rgba(201,168,76,0.25)",
};

const SENT = {
  positive:{ color:"#4ade80", glow:"rgba(74,222,128,0.4)", bg:"rgba(74,222,128,0.08)", label:"Positive", symbol:"↑" },
  negative:{ color:"#fb7185", glow:"rgba(251,113,133,0.4)", bg:"rgba(251,113,133,0.08)", label:"Negative", symbol:"↓" },
  neutral:{ color:"#fbbf24", glow:"rgba(251,191,36,0.4)",  bg:"rgba(251,191,36,0.08)",  label:"Neutral",  symbol:"→" },
};

const EXAMPLES = [
  "The craftsmanship is absolutely extraordinary — worth every penny.",
  "Utterly disappointed. Returned it the same day. Avoid.",
  "It's a serviceable product. Does the job, nothing more.",
  "Blown away by how fast the delivery was. Love it!",
  "Not great, not terrible. Somewhere in the middle honestly.",
];

/* ─── 3D CANVAS SCENE ───────────────────────────────────────────────────── */
function Scene3D({ sentiment }) {
  const canvasRef = useRef(null);
  const rafRef = useRef(null);
  const tRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let W = canvas.width = canvas.offsetWidth;
    let H = canvas.height = canvas.offsetHeight;

    const color = sentiment ? SENT[sentiment].color : T.gold;
    const glow  = sentiment ? SENT[sentiment].glow  : "rgba(201,168,76,0.4)";

    const draw = () => {
      tRef.current += 0.008;
      const t = tRef.current;
      ctx.clearRect(0,0,W,H);

      // Background
      ctx.fillStyle = T.bg;
      ctx.fillRect(0,0,W,H);

      const cx = W/2, cy = H/2;
      const R  = Math.min(W,H)*0.32;

      // Draw 3D rotating wireframe sphere
      const latLines = 10, lonLines = 14;
      ctx.strokeStyle = color.replace(")",",0.18)").replace("rgb","rgba").replace("#","");

      // Helper: project 3D to 2D with perspective
      const proj = (x,y,z) => {
        const fov = 600;
        const scale = fov/(fov+z);
        return [cx+x*scale, cy+y*scale, scale];
      };

      // Latitude rings
      for(let i=0;i<=latLines;i++){
        const phi = (i/latLines)*Math.PI;
        const pts = [];
        for(let j=0;j<=60;j++){
          const theta = (j/60)*2*Math.PI;
          const x0 = R*Math.sin(phi)*Math.cos(theta);
          const y0 = R*Math.cos(phi);
          const z0 = R*Math.sin(phi)*Math.sin(theta);
          // Rotate Y
          const rx = x0*Math.cos(t) + z0*Math.sin(t);
          const rz = -x0*Math.sin(t) + z0*Math.cos(t);
          // Rotate X slightly
          const ry2 = y0*Math.cos(0.3) - rz*Math.sin(0.3);
          const rz2 = y0*Math.sin(0.3) + rz*Math.cos(0.3);
          pts.push(proj(rx,ry2,rz2));
        }
        ctx.beginPath();
        const alpha = 0.08+0.12*(1-Math.abs(i/latLines-0.5)*2);
        ctx.strokeStyle = color+(Math.round(alpha*255).toString(16).padStart(2,"0"));
        ctx.lineWidth = 0.8;
        pts.forEach(([px,py],idx)=> idx===0?ctx.moveTo(px,py):ctx.lineTo(px,py));
        ctx.stroke();
      }

      // Longitude arcs
      for(let j=0;j<lonLines;j++){
        const theta = (j/lonLines)*2*Math.PI + t;
        const pts = [];
        for(let i=0;i<=40;i++){
          const phi=(i/40)*Math.PI;
          const x0=R*Math.sin(phi)*Math.cos(theta);
          const y0=R*Math.cos(phi);
          const z0=R*Math.sin(phi)*Math.sin(theta);
          const rx=x0*Math.cos(t)+z0*Math.sin(t);
          const rz=-x0*Math.sin(t)+z0*Math.cos(t);
          const ry2=y0*Math.cos(0.3)-rz*Math.sin(0.3);
          const rz2=y0*Math.sin(0.3)+rz*Math.cos(0.3);
          pts.push(proj(rx,ry2,rz2));
        }
        ctx.beginPath();
        ctx.strokeStyle = color+"22";
        ctx.lineWidth=0.6;
        pts.forEach(([px,py],idx)=>idx===0?ctx.moveTo(px,py):ctx.lineTo(px,py));
        ctx.stroke();
      }

      // Glow core
      const grad = ctx.createRadialGradient(cx,cy,0,cx,cy,R*0.6);
      grad.addColorStop(0, glow.replace("0.4","0.15"));
      grad.addColorStop(1,"transparent");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(cx,cy,R*0.6,0,Math.PI*2);
      ctx.fill();

      // Floating data points orbiting
      for(let k=0;k<18;k++){
        const angle = (k/18)*Math.PI*2 + t*(1+k*0.05);
        const r2    = R*(0.85+0.25*Math.sin(t+k));
        const phi2  = Math.PI/3 + 0.5*Math.sin(t*0.7+k);
        const x0=r2*Math.sin(phi2)*Math.cos(angle);
        const y0=r2*Math.cos(phi2);
        const z0=r2*Math.sin(phi2)*Math.sin(angle);
        const rx=x0*Math.cos(t)+z0*Math.sin(t);
        const rz=-x0*Math.sin(t)+z0*Math.cos(t);
        const ry2=y0*Math.cos(0.3)-rz*Math.sin(0.3);
        const rz2=y0*Math.sin(0.3)+rz*Math.cos(0.3);
        const [px,py,sc]=proj(rx,ry2,rz2);
        const r = (2+k%3)*sc;
        ctx.beginPath();
        ctx.arc(px,py,r,0,Math.PI*2);
        ctx.fillStyle = color+(Math.round((0.3+0.5*sc)*255).toString(16).padStart(2,"0"));
        ctx.fill();
      }

      rafRef.current = requestAnimationFrame(draw);
    };

    rafRef.current = requestAnimationFrame(draw);
    const ro = new ResizeObserver(()=>{ W=canvas.width=canvas.offsetWidth; H=canvas.height=canvas.offsetHeight; });
    ro.observe(canvas);
    return()=>{ cancelAnimationFrame(rafRef.current); ro.disconnect(); };
  }, [sentiment]);

  return <canvas ref={canvasRef} style={{width:"100%",height:"100%",display:"block"}}/>;
}

/* ─── 3D TILT CARD ──────────────────────────────────────────────────────── */
function TiltCard({ children, style, intensity = 12 }) {
  const ref = useRef(null);
  const onMove = e => {
    const r = ref.current.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width  - 0.5;
    const y = (e.clientY - r.top)  / r.height - 0.5;
    ref.current.style.transform = `perspective(800px) rotateY(${x*intensity}deg) rotateX(${-y*intensity}deg) scale(1.02)`;
  };
  const onLeave = () => { ref.current.style.transform = "perspective(800px) rotateY(0deg) rotateX(0deg) scale(1)"; };
  return (
    <div ref={ref} onMouseMove={onMove} onMouseLeave={onLeave}
      style={{ transition:"transform 0.15s ease", transformStyle:"preserve-3d", ...style }}>
      {children}
    </div>
  );
}

/* ─── ANIMATED COUNTER ──────────────────────────────────────────────────── */
function Counter({ to, suffix="" }) {
  const [val, setVal] = useState(0);
  useEffect(() => {
    let start = 0, step = to/60;
    const iv = setInterval(() => { start = Math.min(start+step, to); setVal(Math.round(start)); if(start>=to) clearInterval(iv); }, 16);
    return () => clearInterval(iv);
  }, [to]);
  return <>{val}{suffix}</>;
}

/* ─── MAIN APP ──────────────────────────────────────────────────────────── */
export default function App() {
  const [pipe, setPipe]       = useState(null);
  const [input, setInput]     = useState("");
  const [model, setModel]     = useState("lr");
  const [result, setResult]   = useState(null);
  const [history, setHistory] = useState([]);
  const [tab, setTab]         = useState("home");
  const [busy, setBusy]       = useState(false);
  const [ready, setReady]     = useState(false);
  const [pct, setPct]         = useState(0);
  const [mouseXY, setMouseXY] = useState({x:0,y:0});
  const homeRef    = useRef(null);
  const analyzeRef = useRef(null);
  const metricsRef = useRef(null);

  useEffect(() => {
    let p=0; const iv=setInterval(()=>{ p+=20; setPct(Math.min(p,90)); if(p>=90)clearInterval(iv); },100);
    setTimeout(()=>{ setPipe(buildPipeline()); setPct(100); setTimeout(()=>setReady(true),400); },1200);
  },[]);

  useEffect(() => {
    const h = e => setMouseXY({x:e.clientX/window.innerWidth-.5, y:e.clientY/window.innerHeight-.5});
    window.addEventListener("mousemove", h);
    return () => window.removeEventListener("mousemove", h);
  },[]);

  const scrollTo = (ref, name) => { ref.current?.scrollIntoView({behavior:"smooth"}); setTab(name); };

  const analyze = () => {
    if (!input.trim()||!pipe||busy) return;
    setBusy(true);
    setTimeout(()=>{
      const toks=clean(input), vec=pipe.vec.tr(toks);
      const m = model==="lr"?pipe.lr:pipe.nb;
      const proba=m.prob(vec), sentiment=m.pred(vec);
      const entry={text:input,sentiment,proba,tokens:toks,model,id:Date.now()};
      setResult(entry); setHistory(h=>[entry,...h].slice(0,5));
      setBusy(false);
    },800);
  };

  /* ── BOOT ── */
  if (!ready) return (
    <div style={{minHeight:"100vh",background:T.bg,display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",fontFamily:"'Cormorant Garamond',serif"}}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;0,700;1,300;1,400&family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400&display=swap');`}</style>
      <div style={{textAlign:"center",position:"relative"}}>
        <div style={{fontSize:11,fontFamily:"'Rajdhani',sans-serif",letterSpacing:".4em",color:T.gold,textTransform:"uppercase",marginBottom:48,opacity:.6}}>Lexara</div>
        <div style={{position:"relative",width:320,height:1,background:"rgba(201,168,76,0.1)",margin:"0 auto 20px"}}>
          <div style={{position:"absolute",inset:"0 auto 0 0",height:"100%",background:`linear-gradient(90deg,transparent,${T.gold},${T.goldLight})`,width:`${pct}%`,transition:"width .15s",boxShadow:`0 0 12px ${T.gold}`}}/>
        </div>
        <div style={{fontSize:11,fontFamily:"'JetBrains Mono',monospace",color:"rgba(201,168,76,0.3)",letterSpacing:".1em"}}>
          {pct<35?"Fitting vectorizer…":pct<65?"Training classifiers…":pct<90?"Evaluating metrics…":"Pipeline ready"}
        </div>
      </div>
    </div>
  );

  const M = pipe?.[model==="lr"?"mLR":"mNB"];
  const parallaxStyle = (depth=1) => ({
    transform:`translate(${mouseXY.x*depth*18}px, ${mouseXY.y*depth*18}px)`,
    transition:"transform 0.1s linear"
  });

  return (
    <div style={{background:T.bg, color:T.cream, fontFamily:"'Cormorant Garamond',serif", overflowX:"hidden", minHeight:"100vh"}}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;0,700;1,300;1,400&family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400&display=swap');
        *{box-sizing:border-box;margin:0;padding:0}
        ::-webkit-scrollbar{width:3px}::-webkit-scrollbar-thumb{background:rgba(201,168,76,.2);border-radius:2px}
        @keyframes fadeUp{from{opacity:0;transform:translateY(24px)}to{opacity:1;transform:translateY(0)}}
        @keyframes rotateGlow{0%,100%{box-shadow:0 0 40px rgba(201,168,76,.15),inset 0 0 40px rgba(201,168,76,.04)}50%{box-shadow:0 0 80px rgba(201,168,76,.3),inset 0 0 60px rgba(201,168,76,.08)}}
        @keyframes shimmer{0%{background-position:200% 0}100%{background-position:-200% 0}}
        @keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
        @keyframes scan{0%{top:-100%}100%{top:200%}}
        .fade{animation:fadeUp .6s ease forwards}
        .floating{animation:float 6s ease-in-out infinite}

        /* NAV */
        nav{position:fixed;top:0;left:0;right:0;z-index:200;display:flex;align-items:center;justify-content:space-between;padding:20px 48px;border-bottom:1px solid rgba(201,168,76,.08);background:rgba(10,8,4,.85);backdrop-filter:blur(24px);}
        .nav-a{cursor:pointer;background:none;border:none;font-family:'Rajdhani',sans-serif;font-size:12px;letter-spacing:.25em;text-transform:uppercase;color:rgba(245,240,232,.35);padding:6px 16px;border-radius:2px;transition:all .2s;}
        .nav-a:hover{color:${T.gold};}
        .nav-a.on{color:${T.gold};border-bottom:1px solid ${T.gold};}

        /* GOLD LINE */
        .gold-rule{width:48px;height:1px;background:linear-gradient(90deg,transparent,${T.gold},transparent);margin:16px auto;}

        /* BUTTONS */
        .btn-gold{cursor:pointer;border:1px solid ${T.gold};background:linear-gradient(135deg,rgba(201,168,76,.18),rgba(232,201,106,.08));color:${T.goldLight};font-family:'Rajdhani',sans-serif;font-size:13px;font-weight:600;letter-spacing:.2em;text-transform:uppercase;padding:14px 36px;border-radius:2px;transition:all .3s;position:relative;overflow:hidden;}
        .btn-gold::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,${T.gold},${T.goldLight});opacity:0;transition:opacity .3s;}
        .btn-gold:hover::before{opacity:.15;}
        .btn-gold:hover{box-shadow:0 0 40px rgba(201,168,76,.35);transform:translateY(-1px);}
        .btn-ghost{cursor:pointer;border:1px solid rgba(245,240,232,.15);background:transparent;color:rgba(245,240,232,.5);font-family:'Rajdhani',sans-serif;font-size:12px;font-weight:500;letter-spacing:.15em;text-transform:uppercase;padding:12px 28px;border-radius:2px;transition:all .2s;}
        .btn-ghost:hover{border-color:rgba(245,240,232,.35);color:rgba(245,240,232,.9);}
        .btn-send{cursor:pointer;border:none;background:linear-gradient(135deg,${T.gold},${T.goldLight});color:${T.bg};font-family:'Rajdhani',sans-serif;font-size:13px;font-weight:700;letter-spacing:.2em;text-transform:uppercase;padding:13px 32px;border-radius:2px;transition:all .25s;}
        .btn-send:hover:not(:disabled){transform:translateY(-2px);box-shadow:0 8px 32px rgba(201,168,76,.45);}
        .btn-send:disabled{opacity:.3;cursor:not-allowed;}

        /* CHIP */
        .chip{cursor:pointer;border:1px solid rgba(201,168,76,.2);background:rgba(201,168,76,.04);color:rgba(245,240,232,.4);font-family:'Rajdhani',sans-serif;font-size:11px;letter-spacing:.15em;text-transform:uppercase;padding:5px 14px;border-radius:1px;transition:all .15s;}
        .chip:hover,.chip.on{border-color:${T.gold};color:${T.gold};background:rgba(201,168,76,.1);}

        /* CARD GLASS */
        .card{background:rgba(26,21,16,.75);border:1px solid rgba(201,168,76,.15);border-radius:4px;backdrop-filter:blur(20px);}
        .card-glow{animation:rotateGlow 4s ease-in-out infinite;}

        /* TEXTAREA */
        textarea{background:rgba(245,240,232,.03);border:1px solid rgba(201,168,76,.2);border-radius:2px;padding:18px 20px;font-family:'Cormorant Garamond',serif;font-size:16px;color:${T.cream};resize:none;width:100%;transition:all .25s;line-height:1.7;letter-spacing:.02em;}
        textarea:focus{outline:none;border-color:rgba(201,168,76,.6);background:rgba(245,240,232,.05);box-shadow:0 0 0 1px rgba(201,168,76,.2),0 0 32px rgba(201,168,76,.1);}
        textarea::placeholder{color:rgba(245,240,232,.15);font-style:italic;}

        /* HIST ROW */
        .hist-row{cursor:pointer;padding:12px 16px;border:1px solid transparent;border-radius:2px;transition:all .15s;}
        .hist-row:hover{border-color:rgba(201,168,76,.2);background:rgba(201,168,76,.04);}

        /* SECTION TAG */
        .sec-tag{font-family:'Rajdhani',sans-serif;font-size:10px;letter-spacing:.4em;text-transform:uppercase;color:rgba(201,168,76,.5);}

        /* 3D METRIC BAR */
        .mbar-wrap{height:6px;background:rgba(245,240,232,.05);border-radius:1px;overflow:visible;position:relative;}
        .mbar{height:100%;border-radius:1px;transition:width 1s cubic-bezier(.4,0,.2,1);position:relative;}
        .mbar::after{content:'';position:absolute;top:0;right:0;bottom:0;width:40px;background:inherit;filter:blur(6px);opacity:.8;}

        /* SCAN LINE */
        .scanline{position:absolute;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(201,168,76,.4),transparent);animation:scan 4s linear infinite;pointer-events:none;}
      `}</style>

      {/* NAV */}
      <nav>
        <div style={{display:"flex",alignItems:"center",gap:12}}>
          <div style={{width:32,height:32,border:`1px solid ${T.gold}`,display:"flex",alignItems:"center",justifyContent:"center",transform:"rotate(45deg)",position:"relative"}}>
            <div style={{width:16,height:16,background:T.gold,transform:"rotate(0deg)",clipPath:"polygon(50% 0%,100% 50%,50% 100%,0% 50%)"}}/>
          </div>
          <span style={{fontFamily:"'Cormorant Garamond',serif",fontSize:18,fontWeight:600,letterSpacing:".08em",color:T.cream}}>Lexara</span>
        </div>
        <div style={{display:"flex",gap:4}}>
          {[["home","Home"],[analyzeRef,"Analyse"],[metricsRef,"Metrics"]].map(([ref,label],i)=>(
            <button key={label} className={`nav-a ${tab===label.toLowerCase()?"on":""}`}
              onClick={()=>{ if(typeof ref==="string"){setTab(ref);homeRef.current?.scrollIntoView({behavior:"smooth"});}else{scrollTo(ref,label.toLowerCase());}}}>
              {label}
            </button>
          ))}
        </div>
        <div style={{fontFamily:"'JetBrains Mono',monospace",fontSize:10,color:"rgba(201,168,76,.3)",letterSpacing:".06em"}}>
          {pipe?.vocabSize} TOKEN VOCAB
        </div>
      </nav>

      {/* ═══════════════ HERO ═══════════════ */}
      <section ref={homeRef} style={{minHeight:"100vh",display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",padding:"120px 48px 80px",position:"relative",overflow:"hidden"}}>
        {/* Decorative orbs */}
        <div style={{...parallaxStyle(0.4),position:"absolute",top:"10%",left:"5%",width:"40vw",height:"40vw",borderRadius:"50%",background:"radial-gradient(circle,rgba(201,168,76,.06) 0%,transparent 70%)",pointerEvents:"none"}}/>
        <div style={{...parallaxStyle(0.7),position:"absolute",bottom:"10%",right:"5%",width:"30vw",height:"30vw",borderRadius:"50%",background:"radial-gradient(circle,rgba(139,37,0,.08) 0%,transparent 70%)",pointerEvents:"none"}}/>

        {/* 3D Globe */}
        <div className="floating" style={{...parallaxStyle(0.2),width:"min(420px,70vw)",height:"min(420px,70vw)",position:"absolute",top:"50%",left:"50%",transform:"translate(-50%,-50%)",opacity:.7,zIndex:0}}>
          <Scene3D sentiment={result?.sentiment||null}/>
        </div>

        {/* Hero text */}
        <div className="fade" style={{position:"relative",zIndex:1,textAlign:"center",maxWidth:720}}>
          <div className="sec-tag" style={{marginBottom:28}}>Emotion Intelligence · Real-Time Analysis</div>

          <h1 style={{fontFamily:"'Cormorant Garamond',serif",fontWeight:700,fontSize:"clamp(52px,8vw,100px)",lineHeight:.95,letterSpacing:"-.02em",color:T.cream,marginBottom:8}}>
            <em style={{color:T.gold,fontStyle:"italic"}}>Lexara</em><br/>Reads Every<br/>Emotion
          </h1>

          <div className="gold-rule"/>

          <p style={{fontSize:"clamp(16px,2vw,20px)",fontWeight:300,color:"rgba(245,240,232,.45)",lineHeight:1.7,letterSpacing:".03em",marginBottom:40,fontStyle:"italic"}}>
            Lexara understands what your words truly mean. Paste any text — a review, a message, a comment — and instantly uncover the emotion behind it.
          </p>

          <div style={{display:"flex",justifyContent:"center",gap:12,flexWrap:"wrap"}}>
            <button className="btn-gold" onClick={()=>scrollTo(analyzeRef,"analyse")}>Begin Analysis →</button>
            <button className="btn-ghost" onClick={()=>scrollTo(metricsRef,"metrics")}>View Performance</button>
          </div>
        </div>

        {/* 3D Floating stats */}
        <div className="fade" style={{position:"relative",zIndex:1,display:"flex",gap:0,marginTop:80,border:`1px solid ${T.border}`,borderRadius:4,overflow:"hidden",background:"rgba(10,8,4,.8)",backdropFilter:"blur(20px)",animationDelay:".3s"}}>
          {[
            {n:<Counter to={pipe.trainSize}/>,l:"Training Samples"},
            {n:<Counter to={pipe.vocabSize}/>,l:"Vocabulary Tokens"},
            {n:<Counter to={3}/>,l:"Sentiment Classes"},
            {n:<><Counter to={Math.round(M.accuracy*100)}/><span style={{fontSize:"50%",opacity:.5}}>%</span></>,l:"Model Accuracy"},
          ].map((s,i,a)=>(
            <TiltCard key={i} intensity={6} style={{padding:"20px 28px",borderRight:i<a.length-1?`1px solid ${T.border}`:"none",textAlign:"center",minWidth:140}}>
              <div style={{fontFamily:"'Cormorant Garamond',serif",fontWeight:700,fontSize:36,color:T.goldLight,letterSpacing:"-.02em",lineHeight:1}}>{s.n}</div>
              <div style={{fontFamily:"'Rajdhani',sans-serif",fontSize:10,color:"rgba(201,168,76,.4)",letterSpacing:".2em",textTransform:"uppercase",marginTop:8}}>{s.l}</div>
            </TiltCard>
          ))}
        </div>

        {/* Scroll hint */}
        <div style={{position:"absolute",bottom:32,left:"50%",transform:"translateX(-50%)",display:"flex",flexDirection:"column",alignItems:"center",gap:8,color:"rgba(201,168,76,.25)",animation:"float 3s ease-in-out infinite"}}>
          <span style={{fontFamily:"'Rajdhani',sans-serif",fontSize:10,letterSpacing:".3em",textTransform:"uppercase"}}>Scroll</span>
          <div style={{width:1,height:40,background:`linear-gradient(${T.gold},transparent)`}}/>
        </div>
      </section>

      {/* ═══════════════ ANALYSE ═══════════════ */}
      <section ref={analyzeRef} style={{padding:"100px 48px",position:"relative"}}>
        {/* Decorative vertical line */}
        <div style={{position:"absolute",left:48,top:0,bottom:0,width:1,background:`linear-gradient(${T.bg},${T.gold}33,${T.bg})`}}/>

        <div style={{maxWidth:1100,margin:"0 auto"}}>
          <div className="fade" style={{textAlign:"center",marginBottom:64}}>
            <div className="sec-tag" style={{marginBottom:20}}>§ 01 — Analysis Engine</div>
            <h2 style={{fontFamily:"'Cormorant Garamond',serif",fontWeight:700,fontSize:"clamp(36px,6vw,72px)",letterSpacing:"-.02em",lineHeight:.95,color:T.cream}}>
              Input any text.<br/><em style={{color:T.gold,fontStyle:"italic"}}>Receive the truth.</em>
            </h2>
            <div className="gold-rule"/>
          </div>

          <div style={{display:"grid",gridTemplateColumns:"1fr 300px",gap:24,alignItems:"start"}}>
            {/* LEFT */}
            <div>
              {/* Model selector */}
              <div style={{display:"flex",alignItems:"center",gap:12,marginBottom:20}}>
                <span style={{fontFamily:"'Rajdhani',sans-serif",fontSize:10,color:"rgba(201,168,76,.4)",letterSpacing:".25em",textTransform:"uppercase"}}>Classifier</span>
                {[{id:"lr",l:"Logistic Regression"},{id:"nb",l:"Naïve Bayes"}].map(({id,l})=>(
                  <button key={id} className={`chip ${model===id?"on":""}`} onClick={()=>setModel(id)}>{l}</button>
                ))}
              </div>

              {/* Input card */}
              <TiltCard intensity={4} style={{marginBottom:16}}>
                <div className="card card-glow" style={{padding:28,position:"relative",overflow:"hidden"}}>
                  <div className="scanline"/>
                  <div style={{fontFamily:"'Rajdhani',sans-serif",fontSize:10,color:"rgba(201,168,76,.4)",letterSpacing:".25em",textTransform:"uppercase",marginBottom:14}}>Text Input</div>
                  <textarea value={input} onChange={e=>setInput(e.target.value)}
                    onKeyDown={e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();analyze();}}}
                    placeholder="Enter a review, tweet, or any statement to analyse its sentiment…"
                    rows={5}/>
                  <div style={{marginTop:16,display:"flex",alignItems:"center",justifyContent:"space-between",flexWrap:"wrap",gap:12}}>
                    <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
                      {EXAMPLES.slice(0,2).map((s,i)=>(
                        <button key={i} className="chip" onClick={()=>setInput(s)} style={{maxWidth:200,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{s.slice(0,32)}…</button>
                      ))}
                    </div>
                    <button className="btn-send" onClick={analyze} disabled={!input.trim()||busy}>
                      {busy?"Processing…":"Analyse →"}
                    </button>
                  </div>
                </div>
              </TiltCard>

              {/* Example chips */}
              <div style={{marginBottom:28}}>
                <div style={{fontFamily:"'Rajdhani',sans-serif",fontSize:10,color:"rgba(201,168,76,.3)",letterSpacing:".25em",textTransform:"uppercase",marginBottom:12}}>Try an example</div>
                <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
                  {EXAMPLES.map((s,i)=>(
                    <button key={i} className="chip" onClick={()=>setInput(s)}>{s.slice(0,42)}{s.length>42?"…":""}</button>
                  ))}
                </div>
              </div>

              {/* RESULT */}
              {result && (()=>{
                const c=SENT[result.sentiment], conf=Math.max(...Object.values(result.proba));
                return (
                  <TiltCard intensity={5} className="fade">
                    <div className="card" style={{border:`1px solid ${c.color}44`,background:c.bg,overflow:"hidden",position:"relative"}}>
                      <div className="scanline"/>
                      {/* Header */}
                      <div style={{padding:"24px 28px",borderBottom:`1px solid ${c.color}22`,display:"flex",justifyContent:"space-between",alignItems:"center"}}>
                        <div>
                          <div style={{fontFamily:"'Rajdhani',sans-serif",fontSize:10,color:"rgba(245,240,232,.3)",letterSpacing:".25em",textTransform:"uppercase",marginBottom:8}}>Detected Sentiment</div>
                          <div style={{display:"flex",alignItems:"center",gap:12}}>
                            <div style={{width:14,height:14,borderRadius:"50%",background:c.color,boxShadow:`0 0 20px ${c.glow}`}}/>
                            <span style={{fontFamily:"'Cormorant Garamond',serif",fontWeight:700,fontSize:40,color:c.color,letterSpacing:"-.02em"}}>{c.label}</span>
                            <span style={{fontFamily:"'Cormorant Garamond',serif",fontSize:28,color:"rgba(245,240,232,.2)"}}>{c.symbol}</span>
                          </div>
                        </div>
                        <div style={{textAlign:"right"}}>
                          <div style={{fontFamily:"'Rajdhani',sans-serif",fontSize:10,color:"rgba(245,240,232,.3)",letterSpacing:".2em",textTransform:"uppercase",marginBottom:4}}>Confidence</div>
                          <div style={{fontFamily:"'Cormorant Garamond',serif",fontWeight:700,fontSize:52,color:T.cream,letterSpacing:"-.04em",lineHeight:1}}>
                            {(conf*100).toFixed(1)}<span style={{fontSize:24,color:"rgba(245,240,232,.25)"}}>%</span>
                          </div>
                        </div>
                      </div>

                      {/* Probability breakdown */}
                      <div style={{padding:"20px 28px",display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:16,borderBottom:`1px solid rgba(245,240,232,.05)`}}>
                        {["positive","negative","neutral"].map(cl=>{
                          const cc=SENT[cl], val=result.proba[cl];
                          return (
                            <TiltCard key={cl} intensity={8}>
                              <div style={{padding:"16px",border:`1px solid ${cl===result.sentiment?cc.color+"44":"rgba(245,240,232,.06)"}`,borderRadius:2,background:cl===result.sentiment?cc.bg:"rgba(245,240,232,.02)"}}>
                                <div style={{display:"flex",alignItems:"center",gap:6,marginBottom:10}}>
                                  <div style={{width:6,height:6,borderRadius:"50%",background:cc.color}}/>
                                  <span style={{fontFamily:"'Rajdhani',sans-serif",fontSize:10,letterSpacing:".15em",textTransform:"uppercase",color:"rgba(245,240,232,.4)"}}>{cc.label}</span>
                                </div>
                                <div style={{fontFamily:"'Cormorant Garamond',serif",fontWeight:700,fontSize:28,color:T.cream,marginBottom:10}}>{(val*100).toFixed(1)}%</div>
                                <div className="mbar-wrap">
                                  <div className="mbar" style={{width:`${val*100}%`,background:`linear-gradient(90deg,${cc.color}88,${cc.color})`}}/>
                                </div>
                              </div>
                            </TiltCard>
                          );
                        })}
                      </div>

                      {/* Tokens */}
                      {result.tokens.length>0&&(
                        <div style={{padding:"16px 28px"}}>
                          <div style={{fontFamily:"'Rajdhani',sans-serif",fontSize:10,color:"rgba(201,168,76,.3)",letterSpacing:".25em",textTransform:"uppercase",marginBottom:10}}>Tokens</div>
                          <div style={{display:"flex",flexWrap:"wrap",gap:6}}>
                            {result.tokens.map((t,i)=>(
                              <span key={i} style={{padding:"3px 10px",border:`1px solid rgba(245,240,232,.08)`,borderRadius:1,fontSize:12,fontFamily:"'JetBrains Mono',monospace",color:"rgba(245,240,232,.4)",background:"rgba(245,240,232,.02)"}}>{t}</span>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </TiltCard>
                );
              })()}
            </div>

            {/* SIDEBAR */}
            <div>
              <TiltCard intensity={6}>
                <div className="card" style={{padding:20}}>
                  <div style={{fontFamily:"'Rajdhani',sans-serif",fontSize:10,color:"rgba(201,168,76,.4)",letterSpacing:".25em",textTransform:"uppercase",marginBottom:16}}>History</div>
                  {history.length===0
                    ?<div style={{textAlign:"center",padding:"32px 0",fontStyle:"italic",color:"rgba(245,240,232,.1)",fontSize:14}}>No analyses yet</div>
                    :history.map(h=>{
                      const c=SENT[h.sentiment],conf=Math.max(...Object.values(h.proba));
                      return(
                        <div key={h.id} className="hist-row" onClick={()=>setResult(h)} style={{marginBottom:6}}>
                          <div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}>
                            <div style={{display:"flex",alignItems:"center",gap:6}}>
                              <div style={{width:5,height:5,borderRadius:"50%",background:c.color,boxShadow:`0 0 8px ${c.glow}`}}/>
                              <span style={{fontFamily:"'Rajdhani',sans-serif",fontSize:11,color:c.color,letterSpacing:".1em",textTransform:"uppercase"}}>{c.label}</span>
                            </div>
                            <span style={{fontFamily:"'JetBrains Mono',monospace",fontSize:11,color:"rgba(245,240,232,.3)"}}>{(conf*100).toFixed(0)}%</span>
                          </div>
                          <div style={{fontSize:13,color:"rgba(245,240,232,.35)",overflow:"hidden",whiteSpace:"nowrap",textOverflow:"ellipsis",paddingLeft:11,fontStyle:"italic"}}>{h.text}</div>
                        </div>
                      );
                    })}
                </div>
              </TiltCard>
            </div>
          </div>
        </div>
      </section>

      {/* ═══════════════ METRICS ═══════════════ */}
      <section ref={metricsRef} style={{padding:"100px 48px",position:"relative"}}>
        <div style={{maxWidth:1100,margin:"0 auto"}}>
          <div className="fade" style={{textAlign:"center",marginBottom:64}}>
            <div className="sec-tag" style={{marginBottom:20}}>§ 02 — Performance Metrics</div>
            <h2 style={{fontFamily:"'Cormorant Garamond',serif",fontWeight:700,fontSize:"clamp(36px,6vw,72px)",letterSpacing:"-.02em",lineHeight:.95,color:T.cream}}>
              Numbers,<br/><em style={{color:T.gold,fontStyle:"italic"}}>laid bare.</em>
            </h2>
            <div className="gold-rule"/>
          </div>

          {/* Model toggle */}
          <div style={{display:"flex",justifyContent:"center",gap:8,marginBottom:48}}>
            {[{id:"lr",l:"Logistic Regression"},{id:"nb",l:"Naïve Bayes"}].map(({id,l})=>(
              <button key={id} className={`chip ${model===id?"on":""}`} onClick={()=>setModel(id)} style={{fontSize:12,padding:"8px 22px"}}>{l}</button>
            ))}
          </div>

          {/* Top stat cards */}
          <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:16,marginBottom:32}}>
            {[
              {label:"Overall Accuracy",val:M.accuracy},
              {label:"Macro F1 Score",val:M.macroF1},
              {label:"Positive F1",val:M.perClass.positive.f1,color:SENT.positive.color},
              {label:"Negative F1",val:M.perClass.negative.f1,color:SENT.negative.color},
            ].map((s,i)=>(
              <TiltCard key={i} intensity={8}>
                <div className="card" style={{padding:"22px 20px",textAlign:"center"}}>
                  <div style={{fontFamily:"'Rajdhani',sans-serif",fontSize:10,color:"rgba(201,168,76,.4)",letterSpacing:".2em",textTransform:"uppercase",marginBottom:12}}>{s.label}</div>
                  <div style={{fontFamily:"'Cormorant Garamond',serif",fontWeight:700,fontSize:40,color:s.color||T.goldLight,letterSpacing:"-.03em",lineHeight:1}}>
                    {(s.val*100).toFixed(1)}<span style={{fontSize:18,opacity:.4}}>%</span>
                  </div>
                  <div className="mbar-wrap" style={{marginTop:14}}>
                    <div className="mbar" style={{width:`${s.val*100}%`,background:`linear-gradient(90deg,${s.color||T.gold}88,${s.color||T.goldLight})`}}/>
                  </div>
                </div>
              </TiltCard>
            ))}
          </div>

          {/* Per-class cards */}
          <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:20}}>
            {["positive","negative","neutral"].map(cl=>{
              const cc=SENT[cl],pc=M.perClass[cl];
              return(
                <TiltCard key={cl} intensity={7}>
                  <div className="card" style={{padding:28,borderColor:`${cc.color}33`}}>
                    <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:24}}>
                      <div style={{width:12,height:12,borderRadius:"50%",background:cc.color,boxShadow:`0 0 16px ${cc.glow}`}}/>
                      <span style={{fontFamily:"'Cormorant Garamond',serif",fontWeight:600,fontSize:22,color:T.cream}}>{cc.label}</span>
                    </div>
                    {[["Precision",pc.precision],["Recall",pc.recall],["F1 Score",pc.f1]].map(([lbl,val])=>(
                      <div key={lbl} style={{marginBottom:16}}>
                        <div style={{display:"flex",justifyContent:"space-between",marginBottom:6}}>
                          <span style={{fontFamily:"'Rajdhani',sans-serif",fontSize:11,color:"rgba(245,240,232,.3)",letterSpacing:".15em",textTransform:"uppercase"}}>{lbl}</span>
                          <span style={{fontFamily:"'JetBrains Mono',monospace",fontSize:12,color:"rgba(245,240,232,.55)"}}>{(val*100).toFixed(1)}%</span>
                        </div>
                        <div className="mbar-wrap">
                          <div className="mbar" style={{width:`${val*100}%`,background:`linear-gradient(90deg,${cc.color}66,${cc.color})`}}/>
                        </div>
                      </div>
                    ))}
                  </div>
                </TiltCard>
              );
            })}
          </div>
        </div>
      </section>

      {/* ═══════════════ HOW IT WORKS ═══════════════ */}
      <section style={{padding:"80px 48px",position:"relative"}}>
        <div style={{maxWidth:1100,margin:"0 auto"}}>
          <div className="fade" style={{textAlign:"center",marginBottom:64}}>
            <div className="sec-tag" style={{marginBottom:20}}>§ 03 — Pipeline Architecture</div>
            <h2 style={{fontFamily:"'Cormorant Garamond',serif",fontWeight:700,fontSize:"clamp(36px,6vw,64px)",letterSpacing:"-.02em",lineHeight:.95,color:T.cream}}>
              Built from<br/><em style={{color:T.gold,fontStyle:"italic"}}>first principles.</em>
            </h2>
            <div className="gold-rule"/>
          </div>
          <div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:16}}>
            {[
              {n:"I",t:"Preprocess",b:"Text is lowercased, punctuation stripped, and stopwords removed to surface signal from noise."},
              {n:"II",t:"TF-IDF Vectorise",b:"Token importance is weighted by frequency and corpus rarity, then L2-normalised into dense vectors."},
              {n:"III",t:"Classify",b:"Logistic Regression and Naïve Bayes compete to assign probability mass across the three sentiment classes."},
              {n:"IV",t:"Softmax",b:"Raw logit scores are exponentiated and normalised to produce well-calibrated class probabilities."},
              {n:"V",t:"Evaluate",b:"Precision, recall, and F1 score are computed on a held-out 20% test set for transparent benchmarking."},
              {n:"VI",t:"Interpret",b:"The argmax class is returned alongside the full probability distribution and token-level breakdown."},
            ].map(({n,t,b})=>(
              <TiltCard key={n} intensity={9}>
                <div className="card" style={{padding:24,height:"100%"}}>
                  <div style={{fontFamily:"'Cormorant Garamond',serif",fontWeight:700,fontSize:32,color:`${T.gold}44`,letterSpacing:"-.02em",marginBottom:12}}>{n}</div>
                  <div style={{fontFamily:"'Cormorant Garamond',serif",fontWeight:600,fontSize:18,color:T.cream,marginBottom:10}}>{t}</div>
                  <div style={{fontSize:13,color:"rgba(245,240,232,.3)",lineHeight:1.75,fontWeight:300}}>{b}</div>
                </div>
              </TiltCard>
            ))}
          </div>
        </div>
      </section>

      {/* FOOTER */}
      <footer style={{borderTop:`1px solid ${T.border}`,padding:"28px 48px",display:"flex",justifyContent:"space-between",alignItems:"center"}}>
        <div style={{display:"flex",alignItems:"center",gap:10}}>
          <div style={{width:20,height:20,border:`1px solid ${T.gold}`,transform:"rotate(45deg)",display:"flex",alignItems:"center",justifyContent:"center"}}>
            <div style={{width:8,height:8,background:T.gold,transform:"rotate(0deg)",clipPath:"polygon(50% 0%,100% 50%,50% 100%,0% 50%)"}}/>
          </div>
          <span style={{fontFamily:"'Cormorant Garamond',serif",fontSize:14,color:`${T.cream}55`}}>Lexara</span>
        </div>
        <div style={{fontFamily:"'JetBrains Mono',monospace",fontSize:10,color:"rgba(201,168,76,.2)",letterSpacing:".06em",textTransform:"uppercase"}}>
          Real-Time · Accurate · Instant Results
        </div>
        <div style={{fontFamily:"'Rajdhani',sans-serif",fontSize:10,color:"rgba(245,240,232,.15)",letterSpacing:".15em",textTransform:"uppercase"}}>Powered by Lexara</div>
      </footer>
    </div>
  );
}
