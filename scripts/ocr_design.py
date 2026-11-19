# OCR design labels on L-07 zone sheets: contour numbers (big single digits) and WS/WB/EL/+x.x spot labels
import sys, re, json
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
Image.MAX_IMAGE_PIXELS = None
D, OUT = sys.argv[1:3]
sheets = [int(s) for s in sys.argv[3].split(',')]
F = 0.5; TILE = 2400; OV = 260
ocr = RapidOCR()
lab = re.compile(r'(WS|WB|EL|TF)?\s*\+\s*(\d{1,2}(?:\.\d{1,2})?)')
out = []
for n in sheets:
    im = Image.open(f'{D}/1130070-{n:03d}.tif').convert('L')
    W, H = im.size
    x0, x1, y0, y1 = int(W*0.025), int(W*0.905), int(H*0.045), int(H*0.73)
    found = []
    for ty in range(y0, y1, TILE - OV):
        for tx in range(x0, x1, TILE - OV):
            bx = (tx, ty, min(tx + TILE, x1), min(ty + TILE, y1))
            t = im.crop(bx); t = t.resize((int(t.width*F), int(t.height*F)), Image.LANCZOS)
            r, _ = ocr(np.asarray(t.convert('RGB')))
            for box, txt, sc in r or []:
                box = np.array(box, float) / F
                hgt = box[:, 1].max() - box[:, 1].min(); wid = box[:, 0].max() - box[:, 0].min()
                cx, cy = box[:, 0].mean() + bx[0], box[:, 1].mean() + bx[1]
                t2 = txt.replace(' ', '').replace('＋', '+')
                if re.fullmatch(r'[1-9]', t2) and hgt > 100 and wid < 1.2 * hgt:
                    found.append(('C', float(t2), cx, cy, float(sc), hgt))
                    continue
                m = lab.search(t2)
                if m:
                    # the ⊕ mark sits just left of the label text
                    lx = box[:, 0].min() + bx[0] - 40
                    found.append((m.group(1) or 'SP', float(m.group(2)), lx, cy, float(sc), hgt))
    keep = []
    for p in sorted(found, key=lambda q: (q[2], q[3])):
        if any(p[0] == q[0] and p[1] == q[1] and abs(p[2]-q[2]) < 80 and abs(p[3]-q[3]) < 80 for q in keep[-50:]): continue
        keep.append(p)
    print(n, len(keep), sorted({k[0] for k in keep}), flush=True)
    out += [(n,) + k for k in keep]
json.dump(out, open(OUT, 'w'))
