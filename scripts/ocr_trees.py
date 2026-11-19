# OCR the existing-tree index table (L-08-1, sheet 43) -> {id: name, dbh, girth, crown, height, action}
import sys, re, json
import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR
Image.MAX_IMAGE_PIXELS = None
src, out = sys.argv[1:3]
im = Image.open(src).convert('L'); W, H = im.size; k = W / 1600
F = 0.6; ocr = RapidOCR()
boxes = []
TILE = 2600
for ty in range(int(60*k), int(985*k), TILE - 200):
    for tx in range(int(40*k), int(1450*k), TILE - 200):
        bx = (tx, ty, min(tx + TILE, int(1450*k)), min(ty + TILE, int(985*k)))
        t = im.crop(bx); t = t.resize((int(t.width*F), int(t.height*F)), Image.LANCZOS)
        r, _ = ocr(np.asarray(t.convert('RGB')))
        for box, txt, sc in r or []:
            b = np.array(box, float) / F
            boxes.append((b[:, 0].min() + bx[0], b[:, 0].max() + bx[0], b[:, 1].mean() + bx[1], txt.strip(), sc))
# dedupe
uniq = []
for b in sorted(boxes, key=lambda q: (q[2], q[0])):
    if any(abs(b[0]-u[0]) < 25 and abs(b[2]-u[2]) < 25 and b[3] == u[3] for u in uniq[-80:]): continue
    uniq.append(b)
blocks = [(45, 392), (397, 744), (748, 1100), (1103, 1447)]
rowh = 13 * k
trees = {}
for x0, x1 in blocks:
    X0, X1 = x0 * k, x1 * k
    inb = [b for b in uniq if X0 <= (b[0] + b[1]) / 2 <= X1]
    for b in inb:
        m = re.fullmatch(r'([TH])0?(\d{1,3})', b[3].replace('O', '0').replace('o', '0'))
        if not m: continue
        tid = f'{m.group(1)}{int(m.group(2)):02d}' if m.group(1) == 'H' or int(m.group(2)) < 100 else f'T{int(m.group(2))}'
        if m.group(1) == 'T' and int(m.group(2)) < 100: tid = f'T{int(m.group(2)):02d}'
        row = sorted([c for c in inb if abs(c[2] - b[2]) < rowh * 0.45 and c[0] > b[1] - 5], key=lambda c: c[0])
        name = next((c[3] for c in row if re.search(r'[一-鿿]', c[3]) and not re.search(r'修剪|移植|範圍|死亡|移除|需', c[3])), '')
        nums = []
        for c in row:
            for v in re.findall(r'\d+(?:\.\d+)?', c[3]):
                nums.append(float(v))
        action = ' '.join(c[3] for c in row if re.search(r'修剪|移植|範圍|死亡|移除|保留|需', c[3]))
        trees[tid] = dict(name=name, nums=nums[:4], action=action)
print(len(trees))
json.dump(trees, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
