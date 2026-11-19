import sys, json, csv
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
D, SCR, CSV, n, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
T = json.load(open(SCR + '/T27.json')); a, b, c, d = T['a'], T['b'], T['c'], T['d']
Z = json.load(open(SCR + '/zoneT.json'))[str(n)]
A = np.array(Z); Ainv = np.linalg.inv(A)
rows = list(csv.reader(open(CSV, encoding='utf-8-sig', errors='replace')))[1:]
EN = np.array([(float(r[3]), float(r[2])) for r in rows])
Mi = np.linalg.inv(np.array([[a, b], [b, -a]]))
ip = (Mi @ (EN - [c, d]).T).T
zp = (Ainv @ np.c_[ip, np.ones(len(ip))].T).T[:, :2]
im = Image.open(f'{D}/1130070-{n:03d}.tif').convert('RGB')
f = 2400 / im.width
im = im.resize((2400, int(im.height * f)))
dr = ImageDraw.Draw(im)
for x, y in zp * f: dr.ellipse([x-2, y-2, x+2, y+2], fill=(255, 0, 0))
crop = [float(v) for v in sys.argv[6].split(',')] if len(sys.argv) > 6 else None
if crop: im = im.crop(tuple(int(v * 2400) for v in crop))
im.save(out)
