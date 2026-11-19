# Generic label OCR on zone sheets: returns (sheet, label, cx, cy, xmin, h, score) for texts matching a regex
import sys, re, json
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
Image.MAX_IMAGE_PIXELS = None
D, OUT, sheets, rx = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3].split(',')], re.compile(sys.argv[4])
F = float(sys.argv[5]) if len(sys.argv) > 5 else 0.6
TILE, OV = 2400, 260
ocr = RapidOCR(); out = []
for n in sheets:
    im = Image.open(f'{D}/1130070-{n:03d}.tif').convert('L'); W, H = im.size
    x0, x1, y0, y1 = int(W*0.025), int(W*0.905), int(H*0.045), int(H*0.73)
    found = []
    for ty in range(y0, y1, TILE - OV):
        for tx in range(x0, x1, TILE - OV):
            bx = (tx, ty, min(tx + TILE, x1), min(ty + TILE, y1))
            t = im.crop(bx); t = t.resize((int(t.width*F), int(t.height*F)), Image.LANCZOS)
            r, _ = ocr(np.asarray(t.convert('RGB')))
            for box, txt, sc in r or []:
                b = np.array(box, float) / F
                for m in rx.finditer(txt.replace(' ', '').replace('O', '0')):
                    L = max(len(txt.replace(' ', '')), 1)
                    xa = b[:, 0].min() + (b[:, 0].max() - b[:, 0].min()) * m.start() / L
                    xb = b[:, 0].min() + (b[:, 0].max() - b[:, 0].min()) * m.end() / L
                    found.append((m.group(), (xa + xb) / 2 + bx[0], b[:, 1].mean() + bx[1], xa + bx[0], b[:, 1].max() - b[:, 1].min(), float(sc)))
    keep = []
    for p in sorted(found, key=lambda q: (q[1], q[2])):
        if any(p[0] == q[0] and abs(p[1]-q[1]) < 80 and abs(p[2]-q[2]) < 80 for q in keep[-60:]): continue
        keep.append(p)
    print(n, len(keep), flush=True)
    out += [(n,) + k for k in keep]
json.dump(out, open(OUT, 'w'))
