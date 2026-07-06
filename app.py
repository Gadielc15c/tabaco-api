"""TFM Tobacco leaf classifier - cloud inference server.

Runs the full mobilenet_v2 PyTorch model (no quantization, full accuracy).
The K210/Maixduino sends a camera JPEG; the server returns {class, confidence}.

Endpoints:
  GET  /            -> live dashboard (auto-refreshing)
  GET  /health      -> {"status": "ok"}
  POST /predict     -> body: raw JPEG/PNG bytes  -> {"label","conf","probs"}
  WS   /ws          -> send JPEG bytes per frame -> receive JSON per frame
"""

import io
import json
import time

import torch
import torch.nn as nn
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from torchvision import models, transforms

CLASSES = ["alternaria_alternata", "cercospora_nicotianae", "none_present"]
LABELS_ES = {"alternaria_alternata": "Alternaria",
             "cercospora_nicotianae": "Cercospora",
             "none_present": "Sana"}
CKPT = "model_best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]
_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=_MEAN, std=_STD),
])


def build_model():
    m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.3),
                                 nn.Linear(m.classifier[1].in_features, len(CLASSES)))
    m.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    m.eval().to(DEVICE)
    return m


model = build_model()
app = FastAPI(title="TFM Tobacco Classifier")

# shared last result for the dashboard
last = {"label": "-", "conf": 0, "ts": 0, "n": 0}


@torch.no_grad()
def infer(img_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = _tf(img).unsqueeze(0).to(DEVICE)
    logits = model(x)[0]
    probs = torch.softmax(logits, 0).tolist()
    idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    key = CLASSES[idx]
    res = {
        "label": LABELS_ES.get(key, key),
        "key": key,
        "conf": round(probs[idx] * 100, 1),
        "probs": {LABELS_ES.get(CLASSES[i], CLASSES[i]): round(p * 100, 1)
                  for i, p in enumerate(probs)},
    }
    last.update(label=res["label"], conf=res["conf"], ts=time.time(), n=last["n"] + 1)
    return res


@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


@app.post("/predict")
async def predict(request: Request):
    body = await request.body()
    if not body:
        return JSONResponse({"error": "empty body"}, status_code=400)
    try:
        return infer(body)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    try:
        while True:
            data = await sock.receive_bytes()
            try:
                await sock.send_text(json.dumps(infer(data)))
            except Exception as e:
                await sock.send_text(json.dumps({"error": str(e)}))
    except WebSocketDisconnect:
        pass


@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>TFM Tabaco</title><style>
*{box-sizing:border-box;margin:0}body{font-family:system-ui,sans-serif;background:#0c1220;
color:#eef;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:18px}
h1{font-size:18px;color:#7fe0ff;margin:8px 0 4px;text-align:center}
.sub{font-size:12px;color:#7a88a0;margin-bottom:14px}
.stage{width:100%;max-width:420px}
video,canvas,#preview{width:100%;border-radius:14px;background:#000;aspect-ratio:4/3;object-fit:cover}
.row{display:flex;gap:10px;margin:12px 0}
button{flex:1;padding:14px;border:0;border-radius:12px;font-size:15px;font-weight:700;
background:#00c896;color:#04121a}button.alt{background:#26324a;color:#cfe}
button:active{transform:scale(.98)}
.card{background:#131c30;border-radius:16px;padding:20px;margin-top:14px;box-shadow:0 8px 24px #0008}
.lbl{font-size:30px;font-weight:800}.pct{font-size:38px;font-weight:800;color:#00c896;float:right}
.bar{height:12px;background:#26324a;border-radius:8px;overflow:hidden;margin:8px 0}
.fill{height:100%;transition:width .4s}
.prob{display:flex;justify-content:space-between;font-size:13px;color:#9fb;margin:3px 0}
.hint{font-size:12px;color:#7a88a0;margin-top:8px;text-align:center}
label.up{display:block}input[type=file]{display:none}</style></head><body>
<h1>TFM . Clasificador de Tabaco</h1>
<div class="sub">Cercospora . Alternaria . Sana &nbsp;|&nbsp; modelo en la nube</div>
<div class="stage">
<video id="v" autoplay playsinline></video>
<canvas id="c" style="display:none"></canvas>
<div class="row">
<button id="shot">Capturar y clasificar</button>
<label class="up" style="flex:1"><button class="alt" type="button" onclick="document.getElementById('f').click()">Subir foto</button>
<input type="file" id="f" accept="image/*"></label>
</div>
<div class="card" id="res" style="display:none">
<span class="pct" id="pct">0%</span><div class="lbl" id="lbl">-</div>
<div class="bar"><div class="fill" id="fill" style="width:0%;background:#00c896"></div></div>
<div id="probs"></div></div>
<div class="hint" id="hint">Apunta una hoja de tabaco y toca Capturar</div>
</div>
<script>
const COLORS={Alternaria:'#ff8c00',Cercospora:'#ffd200',Sana:'#00dc78'};
const v=document.getElementById('v'),c=document.getElementById('c');
navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'}})
 .then(s=>v.srcObject=s).catch(e=>{document.getElementById('hint').textContent='Sin camara: usa Subir foto ('+e+')';});
async function send(blob){
 document.getElementById('hint').textContent='Clasificando...';
 try{let r=await fetch('/predict',{method:'POST',body:blob});let d=await r.json();show(d);}
 catch(e){document.getElementById('hint').textContent='Error: '+e;}
}
function show(d){
 if(d.error){document.getElementById('hint').textContent='Error: '+d.error;return;}
 let col=COLORS[d.label]||'#00c896';
 document.getElementById('res').style.display='block';
 document.getElementById('lbl').textContent=d.label;
 document.getElementById('lbl').style.color=col;
 document.getElementById('pct').textContent=d.conf+'%';
 let f=document.getElementById('fill');f.style.width=d.conf+'%';f.style.background=col;
 let h='';for(let k in d.probs){h+='<div class=prob><span>'+k+'</span><span>'+d.probs[k]+'%</span></div>';}
 document.getElementById('probs').innerHTML=h;
 document.getElementById('hint').textContent='Listo. Captura otra cuando quieras.';
}
document.getElementById('shot').onclick=()=>{
 c.width=v.videoWidth||640;c.height=v.videoHeight||480;
 c.getContext('2d').drawImage(v,0,0,c.width,c.height);
 c.toBlob(b=>send(b),'image/jpeg',0.8);
};
document.getElementById('f').onchange=e=>{if(e.target.files[0])send(e.target.files[0]);};
</script></body></html>"""


@app.get("/last")
def get_last():
    return last
