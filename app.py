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
<title>TFM Tabaco - Live</title><style>
*{box-sizing:border-box;margin:0}body{font-family:system-ui,sans-serif;background:#0c1220;
color:#eef;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:24px}
h1{font-size:20px;color:#7fe0ff;margin:12px}.card{background:#131c30;border-radius:16px;
padding:28px;width:100%;max-width:380px;margin-top:16px;box-shadow:0 8px 24px #0008}
.lbl{font-size:32px;font-weight:800;margin:8px 0}.pct{font-size:44px;font-weight:800;color:#00c896}
.bar{height:14px;background:#26324a;border-radius:8px;overflow:hidden;margin:14px 0}
.fill{height:100%;background:linear-gradient(90deg,#00c896,#7fe0ff);transition:width .4s}
.meta{font-size:12px;color:#7a88a0;margin-top:16px}</style></head><body>
<h1>TFM . Clasificador de Tabaco (nube)</h1>
<div class="card"><div class="lbl" id="lbl">esperando...</div>
<div class="bar"><div class="fill" id="fill" style="width:0%"></div></div>
<div class="pct" id="pct">0%</div>
<div class="meta" id="meta">sin datos aun</div></div>
<script>async function up(){try{let r=await fetch('/last');let d=await r.json();
document.getElementById('lbl').textContent=d.label;
document.getElementById('pct').textContent=d.conf+'%';
document.getElementById('fill').style.width=d.conf+'%';
document.getElementById('meta').textContent='muestras: '+d.n+(d.ts?(' . hace '+Math.round(Date.now()/1000-d.ts)+'s'):'');
}catch(e){}}setInterval(up,1000);up();</script></body></html>"""


@app.get("/last")
def get_last():
    return last
