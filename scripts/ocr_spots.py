# OCR the spot heights on L-03 zone sheets -> TWD97 points
import sys, re, json
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
Image.MAX_IMAGE_PIXELS = None
D, SCR, OUT = sys.argv[1:4]
sheets = [int(s) for s in sys.argv[4].split(',')]
F = 0.75; TILE = 2000; OV = 220          # tile size in full-res px
ocr = RapidOCR()
pat = re.compile(r'(?<![\d.])\d{1,2}\.\d{2}(?!\d)')
res_all = []
for n in sheets:
    im = Image.open(f'{D}/1130070-{n:03d}.tif').convert('L')
    W, H = im.size
    x0, x1, y0, y1 = int(W*0.025), int(W*0.905), int(H*0.045), int(H*0.73)
    found = []
    for ty in range(y0, y1, TILE - OV):
        for tx in range(x0, x1, TILE - OV):
            bx = (tx, ty, min(tx + TILE, x1), min(ty + TILE, y1))
            t = im.crop(bx); tw, th = t.size
            t = t.resize((int(tw*F), int(th*F)), Image.LANCZOS).rotate(90, expand=True)   # CCW: text becomes horizontal
            r, _ = ocr(np.asarray(t.convert('RGB')))
            for box, txt, sc in r or []:
                ms = list(pat.finditer(txt.replace(' ', '')))
                if not ms: continue
                box = np.array(box, float)
                # rotated(CCW 90) coords (u,v) -> scaled tile coords: x = th_s - 1 - v ... rotate back
                ths = int(th*F)
                L = len(txt.replace(' ', ''))
                for m in ms:
                    # fraction along the text where the number starts / ends
                    f0, f1 = m.start() / max(L, 1), m.end() / max(L, 1)
                    xl, xr = box[:, 0].min(), box[:, 0].max(); yc = box[:, 1].mean()
                    us = xl + (xr - xl) * f0
                    # anchor: start of the number (the survey mark sits just before it)
                    u, v = us, yc
                    # inverse of PIL rotate(90, expand): (u,v) in rotated <- original scaled (x,y): u = y, v = (tws-1) - x
                    tws = int(tw*F)
                    x = (tws - 1) - v; y = u
                    found.append((bx[0] + x / F, bx[1] + y / F, float(m.group()), float(sc)))
    # dedupe overlaps
    found.sort()
    keep = []
    for p in found:
        if any(abs(p[0]-q[0]) < 60 and abs(p[1]-q[1]) < 60 and p[2] == q[2] for q in keep[-60:]): continue
        keep.append(p)
    print(n, 'numbers', len(keep), flush=True)
    res_all += [(n,) + p for p in keep]
json.dump(res_all, open(OUT, 'w'))
