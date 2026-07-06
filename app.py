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

CLASSES = ["alternaria_alternata", "cercospora_nicotianae", "healthy"]
LABELS_ES = {"alternaria_alternata": "Alternaria",
             "cercospora_nicotianae": "Cercospora",
             "healthy": "Sana"}
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


_feat = {}


def build_model():
    m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.3),
                                 nn.Linear(m.classifier[1].in_features, len(CLASSES)))
    m.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    m.eval().to(DEVICE)
    # capture last conv feature maps for CAM (localization -> boxes)
    m.features.register_forward_hook(lambda mod, i, o: _feat.__setitem__("f", o.detach()))
    return m


model = build_model()
# classifier weights for CAM: Linear is classifier[1]
_cam_w = model.classifier[1].weight.detach()   # (num_classes, 1280)
app = FastAPI(title="TFM Tobacco Detector")

# shared last result for the dashboard
last = {"label": "-", "conf": 0, "ts": 0, "n": 0}


def _boxes_from_cam(cam, thresh=0.45, max_boxes=4):
    """cam: 2D tensor HxW in [0,1]. Return normalized [x,y,w,h] boxes via
    connected components of the thresholded activation."""
    import numpy as np
    m = cam.cpu().numpy()
    H, W = m.shape
    mask = m >= thresh
    if not mask.any():
        return []
    # simple flood-fill connected components (4-neighbour)
    lbl = np.zeros((H, W), dtype=np.int32)
    cur = 0
    boxes = []
    stack = []
    for sy in range(H):
        for sx in range(W):
            if mask[sy, sx] and lbl[sy, sx] == 0:
                cur += 1
                lbl[sy, sx] = cur
                stack.append((sy, sx))
                minx = maxx = sx
                miny = maxy = sy
                area = 0
                while stack:
                    y, x = stack.pop()
                    area += 1
                    if x < minx: minx = x
                    if x > maxx: maxx = x
                    if y < miny: miny = y
                    if y > maxy: maxy = y
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and lbl[ny, nx] == 0:
                            lbl[ny, nx] = cur
                            stack.append((ny, nx))
                if area >= 2:  # drop 1-pixel noise
                    boxes.append((area, minx, miny, maxx, maxy))
    boxes.sort(reverse=True)
    out = []
    for _, minx, miny, maxx, maxy in boxes[:max_boxes]:
        x = minx / W
        y = miny / H
        w = (maxx - minx + 1) / W
        h = (maxy - miny + 1) / H
        out.append([round(x, 3), round(y, 3), round(w, 3), round(h, 3)])
    return out


def infer(img_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    x = _tf(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)[0]
        probs = torch.softmax(logits, 0)
        idx = int(torch.argmax(probs).item())
        # CAM for predicted class: weighted sum of feature maps
        fmap = _feat["f"][0]                      # (1280, h, w)
        cam = torch.tensordot(_cam_w[idx], fmap, dims=([0], [0]))  # (h, w)
        cam = torch.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()
    probs = probs.tolist()
    key = CLASSES[idx]
    # only box disease classes (Sana = no lesion)
    boxes = [] if key == "healthy" else _boxes_from_cam(cam)
    res = {
        "label": LABELS_ES.get(key, key),
        "key": key,
        "conf": round(probs[idx] * 100, 1),
        "probs": {LABELS_ES.get(CLASSES[i], CLASSES[i]): round(p * 100, 1)
                  for i, p in enumerate(probs)},
        "boxes": boxes,
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
<title>TFM Tabaco Detector</title><style>
*{box-sizing:border-box;margin:0}body{font-family:system-ui,sans-serif;background:#0c1220;
color:#eef;min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:16px}
h1{font-size:18px;color:#7fe0ff;margin:6px 0 2px;text-align:center}
.sub{font-size:12px;color:#7a88a0;margin-bottom:12px}
.stage{position:relative;width:100%;max-width:440px;aspect-ratio:4/3;border-radius:14px;overflow:hidden;background:#000}
video,#ov{position:absolute;inset:0;width:100%;height:100%}
video{object-fit:cover}
.row{display:flex;gap:10px;margin:12px 0;width:100%;max-width:440px}
button{flex:1;padding:13px;border:0;border-radius:12px;font-size:14px;font-weight:700;background:#26324a;color:#cfe}
button.on{background:#00c896;color:#04121a}button:active{transform:scale(.98)}
.card{width:100%;max-width:440px;background:#131c30;border-radius:16px;padding:16px 20px;box-shadow:0 8px 24px #0008}
.lbl{font-size:26px;font-weight:800}.pct{font-size:32px;font-weight:800;float:right}
.bar{height:10px;background:#26324a;border-radius:8px;overflow:hidden;margin:8px 0}
.fill{height:100%;transition:width .3s}
.prob{display:flex;justify-content:space-between;font-size:12px;color:#9fb;margin:2px 0}
.hint{font-size:12px;color:#7a88a0;margin-top:8px;text-align:center}
input[type=file]{display:none}</style></head><body>
<h1>TFM . Detector de Tabaco</h1>
<div class="sub">Enfermedad + localizacion (cajas) &nbsp;|&nbsp; modelo en la nube</div>
<div class="stage"><video id="v" autoplay playsinline muted></video><canvas id="ov"></canvas></div>
<div class="row">
<button id="live" class="on">Detener</button>
<button onclick="document.getElementById('f').click()">Subir foto</button>
<input type="file" id="f" accept="image/*">
</div>
<div class="card"><span class="pct" id="pct">--</span><div class="lbl" id="lbl">iniciando...</div>
<div class="bar"><div class="fill" id="fill" style="width:0%;background:#00c896"></div></div>
<div id="probs"></div><div class="hint" id="hint">Apunta una hoja de tabaco</div></div>
<canvas id="cap" style="display:none"></canvas>
<script>
const COLORS={Alternaria:'#ff8c00',Cercospora:'#ffd200',Sana:'#00dc78'};
const v=document.getElementById('v'),ov=document.getElementById('ov'),cap=document.getElementById('cap');
let live=true,busy=false,lastBoxes=[],lastCol='#00c896';
navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'}})
 .then(s=>{v.srcObject=s;requestAnimationFrame(draw);loop();})
 .catch(e=>{document.getElementById('hint').textContent='Sin camara: usa Subir foto';});
function draw(){
 ov.width=ov.clientWidth;ov.height=ov.clientHeight;
 const g=ov.getContext('2d');g.clearRect(0,0,ov.width,ov.height);
 g.lineWidth=3;g.strokeStyle=lastCol;g.font='bold 14px system-ui';g.fillStyle=lastCol;
 for(const b of lastBoxes){
  const x=b[0]*ov.width,y=b[1]*ov.height,w=b[2]*ov.width,h=b[3]*ov.height;
  g.strokeRect(x,y,w,h);g.fillText('lesion',x+2,y>14?y-4:y+14);
 }
 requestAnimationFrame(draw);
}
function toBlob(){cap.width=v.videoWidth||640;cap.height=v.videoHeight||480;
 cap.getContext('2d').drawImage(v,0,0,cap.width,cap.height);
 return new Promise(r=>cap.toBlob(r,'image/jpeg',0.75));}
async function classify(blob){
 busy=true;
 try{let r=await fetch('/predict',{method:'POST',body:blob});let d=await r.json();show(d);}
 catch(e){document.getElementById('hint').textContent='Error red';}
 busy=false;
}
function show(d){
 if(d.error){document.getElementById('hint').textContent=d.error;return;}
 let col=COLORS[d.label]||'#00c896';lastCol=col;lastBoxes=d.boxes||[];
 document.getElementById('lbl').textContent=d.label;
 document.getElementById('lbl').style.color=col;
 document.getElementById('pct').textContent=d.conf+'%';
 document.getElementById('pct').style.color=col;
 let f=document.getElementById('fill');f.style.width=d.conf+'%';f.style.background=col;
 let h='';for(let k in d.probs){h+='<div class=prob><span>'+k+'</span><span>'+d.probs[k]+'%</span></div>';}
 document.getElementById('probs').innerHTML=h;
 document.getElementById('hint').textContent=lastBoxes.length+' lesion(es) detectada(s)';
}
async function loop(){while(true){if(live&&!busy&&v.videoWidth){await classify(await toBlob());}
 await new Promise(r=>setTimeout(r,1200));}}
document.getElementById('live').onclick=e=>{live=!live;e.target.textContent=live?'Detener':'Reanudar';e.target.className=live?'on':'';};
document.getElementById('f').onchange=e=>{if(e.target.files[0]){live=false;document.getElementById('live').textContent='Reanudar';document.getElementById('live').className='';classify(e.target.files[0]);}};
</script></body></html>"""


@app.get("/last")
def get_last():
    return last
