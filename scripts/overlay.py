import csv, sys, json
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
# fit again (copy of fit.py essentials) and save transform
rows = list(csv.reader(open(sys.argv[1], encoding='utf-8-sig', errors='replace')))[1:]
db = np.array([(float(r[3]), float(r[2])) for r in rows if r[1].startswith('db')])
db = db[np.argsort(db[:,1])]
groups = [[db[0]]]
for p in db[1:]:
    if p[1] - groups[-1][-1][1] > 9: groups.append([p])
    else: groups[-1].append(p)
S = np.array([np.mean(g, axis=0) for g in groups])
k = 8.2775
plan = [(120,302),(348,312),(592,290),(811,236),(1101,197),(1362,144),(1612,77),(1880,41)]
P = np.array([((150 + x*0.175)*k, (310 + y*0.175)*k) for x, y in plan])
# general affine-free similarity with reflection: E = a*x + b*y + c ; N = b*x - a*y + d  (reflected similarity)
A = []; B = []
for (x, y), (e, n) in zip(P, S):
    A.append([x, y, 1, 0]); B.append(e)
    A.append([-y, x, 0, 1]); B.append(n)
# reflected: E = a x + b y + c ; N = b x - a y + d -> rows: [x, y,1,0]=E ; [-y, x,0,1]=N with params (a,b,c,d) -> N = -a y + b x + d
sol, *_ = np.linalg.lstsq(np.array(A), np.array(B), rcond=None)
a, b, c, d = sol
T = dict(a=a, b=b, c=c, d=d, note='E=a*x+b*y+c, N=b*x-a*y+d, x,y = full-res px of sheet 27')
json.dump(T, open(sys.argv[3], 'w'), indent=1)
print(T, 'scale', np.hypot(a, b))
# inverse: solve [a b; b -a][x y] = [E-c, N-d]
M = np.array([[a, b], [b, -a]]); Mi = np.linalg.inv(M)
im = Image.open(sys.argv[2]).convert('RGB')
f = 4000 / im.width
im = im.resize((4000, int(im.height * f)), Image.LANCZOS)
dr = ImageDraw.Draw(im)
for r in rows:
    e, n = float(r[3]), float(r[2])
    x, y = Mi @ np.array([e - c, n - d])
    x *= f; y *= f
    col = (255,0,0) if r[1].startswith('db') else (0,90,255)
    dr.ellipse([x-2, y-2, x+2, y+2], fill=col)
im.save(sys.argv[4])
