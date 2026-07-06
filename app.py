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

CLASSES = ["alternaria_alternata", "cercospora_nicotianae", "healthy", "no_hoja"]
LABELS_ES = {"alternaria_alternata": "Alternaria",
             "cercospora_nicotianae": "Cercospora",
             "healthy": "Sana",
             "no_hoja": "No es hoja"}
NO_BOX = {"healthy", "no_hoja"}   # clases sin lesiones que localizar
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
    boxes = [] if key in NO_BOX else _boxes_from_cam(cam)
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
#view{width:100%;max-width:460px;aspect-ratio:4/3;border-radius:14px;background:#000;display:block}
.row{display:flex;gap:10px;margin:12px 0;width:100%;max-width:460px}
button{flex:1;padding:14px;border:0;border-radius:12px;font-size:14px;font-weight:700;background:#00c896;color:#04121a}
button.alt{background:#26324a;color:#cfe}button:active{transform:scale(.98)}
.card{width:100%;max-width:460px;background:#131c30;border-radius:16px;padding:16px 20px;box-shadow:0 8px 24px #0008}
.lbl{font-size:26px;font-weight:800}.pct{font-size:32px;font-weight:800;float:right}
.bar{height:10px;background:#26324a;border-radius:8px;overflow:hidden;margin:8px 0}
.fill{height:100%;transition:width .3s}
.prob{display:flex;justify-content:space-between;font-size:12px;color:#9fb;margin:2px 0}
.hint{font-size:12px;color:#7a88a0;margin-top:8px;text-align:center}
input[type=file]{display:none}</style></head><body>
<h1>TFM . Detector de Tabaco</h1>
<div class="sub">Enfermedad + localizacion (cajas) &nbsp;|&nbsp; modelo en la nube</div>
<canvas id="view"></canvas>
<div class="row">
<button onclick="document.getElementById('f').click()">Subir foto de hoja</button>
<button id="cambtn" class="alt" type="button">Camara</button>
<input type="file" id="f" accept="image/*">
</div>
<div class="card"><span class="pct" id="pct">--</span><div class="lbl" id="lbl">Sube una foto de hoja</div>
<div class="bar"><div class="fill" id="fill" style="width:0%;background:#00c896"></div></div>
<div id="probs"></div><div class="hint" id="hint">Sube una foto o abre la camara</div></div>
<video id="v" autoplay playsinline muted style="display:none"></video>
<canvas id="cap" style="display:none"></canvas>
<script>
const COLORS={Alternaria:'#ff8c00',Cercospora:'#ffd200',Sana:'#00dc78'};
const view=document.getElementById('view'),g=view.getContext('2d');
const v=document.getElementById('v'),cap=document.getElementById('cap');
let boxes=[],col='#00c896',live=false,busy=false,srcImg=null;

function fit(){view.width=460;view.height=345;}
fit();
function render(source){
 // draw source stretched to canvas; boxes are normalized to full image so they align
 g.fillStyle='#000';g.fillRect(0,0,view.width,view.height);
 if(source) g.drawImage(source,0,0,view.width,view.height);
 g.lineWidth=3;g.strokeStyle=col;g.font='bold 15px system-ui';g.fillStyle=col;
 for(const b of boxes){
  const x=b[0]*view.width,y=b[1]*view.height,w=b[2]*view.width,h=b[3]*view.height;
  g.strokeRect(x,y,w,h);
  g.fillStyle=col;g.fillRect(x,Math.max(0,y-16),54,16);
  g.fillStyle='#000';g.fillText('lesion',x+3,Math.max(12,y-4));g.fillStyle=col;
 }
}
function toBlob(source,w,h){cap.width=w;cap.height=h;cap.getContext('2d').drawImage(source,0,0,w,h);
 return new Promise(r=>cap.toBlob(r,'image/jpeg',0.85));}
async function classify(blob){
 busy=true;document.getElementById('hint').textContent='Clasificando...';
 try{let r=await fetch('/predict',{method:'POST',body:blob});let d=await r.json();show(d);}
 catch(e){document.getElementById('hint').textContent='Error de red';}
 busy=false;
}
function show(d){
 if(d.error){document.getElementById('hint').textContent=d.error;return;}
 col=COLORS[d.label]||'#00c896';boxes=d.boxes||[];
 document.getElementById('lbl').textContent=d.label;
 document.getElementById('lbl').style.color=col;
 document.getElementById('pct').textContent=d.conf+'%';
 document.getElementById('pct').style.color=col;
 let f=document.getElementById('fill');f.style.width=d.conf+'%';f.style.background=col;
 let h='';for(let k in d.probs){h+='<div class=prob><span>'+k+'</span><span>'+d.probs[k]+'%</span></div>';}
 document.getElementById('probs').innerHTML=h;
 document.getElementById('hint').textContent=(boxes.length?boxes.length+' lesion(es)':'sin lesiones')+' . toca otra foto o camara';
 render(srcImg||v);
}
// UPLOAD: show the image AND boxes
document.getElementById('f').onchange=e=>{
 const file=e.target.files[0];if(!file)return;
 live=false;document.getElementById('cambtn').textContent='Camara';
 const img=new Image();
 img.onload=async()=>{srcImg=img;boxes=[];render(img);
  classify(await toBlob(img,img.naturalWidth,img.naturalHeight));};
 img.src=URL.createObjectURL(file);
};
// CAMERA: live loop drawing frames + boxes
document.getElementById('cambtn').onclick=async e=>{
 if(!live){
  try{const s=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'}}});v.srcObject=s;}
  catch(err){document.getElementById('hint').textContent='Sin camara, usa Subir foto';return;}
  live=true;srcImg=null;e.target.textContent='Detener camara';loop();
 } else {live=false;e.target.textContent='Camara';}
};
async function loop(){while(live){if(v.videoWidth){render(v);if(!busy)await classify(await toBlob(v,v.videoWidth,v.videoHeight));}
 await new Promise(r=>setTimeout(r,1000));}}
</script></body></html>"""


@app.get("/last")
def get_last():
    return last
