# Chamfer-refine each L-06 zone sheet against survey points (small search), propagate to siblings.
import sys, json, csv, math
import numpy as np, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
D, SCR, CSV = sys.argv[1:4]
T = json.load(open(SCR + '/T27.json')); a, b, c, d = T['a'], T['b'], T['c'], T['d']
Z = {k: np.array(v) for k, v in json.load(open(SCR + '/zoneT.json')).items()}
rows = list(csv.reader(open(CSV, encoding='utf-8-sig', errors='replace')))[1:]
# only path points (RD*) - they are drawn as path edges on the paving plans
EN = np.array([(float(r[3]), float(r[2])) for r in rows if 'RD' in r[1]])
Mi = np.linalg.inv(np.array([[a, b], [b, -a]]))
P = np.c_[(Mi @ (EN - [c, d]).T).T, np.ones(len(EN))]
for i, zn in enumerate('ABCDEFG'):
    n = 28 + i
    im = Image.open(f'{D}/1130070-{n:03d}.tif').convert('L')
    zf = np.asarray(im.resize((im.width // 4, im.height // 4), Image.BILINEAR))
    dt = cv2.distanceTransform((zf >= 128).astype(np.uint8), cv2.DIST_L2, 3)
    inv = np.linalg.inv(Z[str(n)])
    def R_(p):
        dx, dy, th, sc = p
        return np.array([[math.cos(th)*sc, -math.sin(th)*sc, dx], [math.sin(th)*sc, math.cos(th)*sc, dy], [0, 0, 1]])
    def cost(p):
        q = (R_(p) @ inv @ P.T).T[:, :2] / 4
        ok = (q[:, 0] > dt.shape[1]*0.03) & (q[:, 1] > dt.shape[0]*0.05) & (q[:, 0] < dt.shape[1]*0.9) & (q[:, 1] < dt.shape[0]*0.7)
        if ok.sum() < 30: return 99.0, int(ok.sum())
        return float(np.mean(np.minimum(dt[q[ok, 1].astype(int), q[ok, 0].astype(int)], 10))), int(ok.sum())
    best = (0, 0, 0, 1); bc, npt = cost(best); c0 = bc
    if npt < 30: print(zn, 'too few points', npt); continue
    for gx in range(-320, 321, 20):
        for gy in range(-320, 321, 20):
            cc, _ = cost((gx, gy, 0, 1))
            if cc < bc: bc, best = cc, (gx, gy, 0, 1)
    for sxy, sth, ssc in [(10, .002, .002), (4, .0008, .0008), (1.5, .0003, .0003)]:
        imp = True
        while imp:
            imp = False
            for k, s in enumerate([sxy, sxy, sth, ssc]):
                for sg in (-1, 1):
                    p = list(best); p[k] += sg * s; cc, _ = cost(p)
                    if cc < bc - 1e-4: bc, best, imp = cc, tuple(p), True
    new = np.linalg.inv(R_(best) @ inv); old = Z[str(n)]
    corr = new @ np.linalg.inv(old)
    for first in (7, 36, 47, 56, 64, 72):
        Z[str(first + i)] = corr @ Z[str(first + i)]
    Z[str(n)] = new
    print(zn, 'pts', npt, 'cost', round(c0, 2), '->', round(bc, 2), 'shift(full px)', best[:2], 'rot', round(math.degrees(best[2]), 3), 'scale', round(best[3], 4), flush=True)
json.dump({k: v.tolist() for k, v in Z.items()}, open(SCR + '/zoneT.json', 'w'))
