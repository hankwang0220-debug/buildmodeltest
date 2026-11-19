# Stage 1: L-06 zone sheets -> index (SIFT on zone A, zone offsets for others, chamfer refine on survey points)
# Stage 2: every other zone sheet -> its L-06 sibling (ECC euclidean on blurred images)
import sys, json, csv, math
import numpy as np, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
D, SCR, CSV = sys.argv[1:4]
K = 8.2775
T = json.load(open(SCR + '/T27.json')); a, b, c, d = T['a'], T['b'], T['c'], T['d']
def load(n, f):
    im = Image.open(f'{D}/1130070-{n:03d}.tif').convert('L')
    return np.asarray(im.resize((im.width // f, im.height // f), Image.BILINEAR))
rows = list(csv.reader(open(CSV, encoding='utf-8-sig', errors='replace')))[1:]
EN = np.array([(float(r[3]), float(r[2])) for r in rows])
Mi = np.linalg.inv(np.array([[a, b], [b, -a]]))
idxpx = (Mi @ (EN - [c, d]).T).T
P = np.c_[idxpx, np.ones(len(idxpx))]
zones = 'ABCDEFG'
rect = dict(A=(78,322), B=(410,110), C=(410,322), D=(745,110), E=(745,322), F=(1077,110), G=(1077,322))
series = {'L03': 7, 'L06': 28, 'L07': 36, 'L08': 47, 'L09': 56, 'L10': 64, 'L11': 72}
out = {}
# --- base: SIFT for zone A L-06 (sheet 28)
idx = load(27, 2); z = load(28, 8)
sift = cv2.SIFT_create(nfeatures=40000)
mi = np.zeros_like(idx); mi[int(50*K/2):int(560*K/2), int(40*K/2):int(1460*K/2)] = 255
mz = np.zeros_like(z); mz[int(z.shape[0]*0.04):int(z.shape[0]*0.70), int(z.shape[1]*0.02):int(z.shape[1]*0.905)] = 255
ki, di = sift.detectAndCompute(255 - idx, mi); kz, dz = sift.detectAndCompute(255 - z, mz)
m = cv2.BFMatcher().knnMatch(dz, di, k=2); good = [p for p, q in m if p.distance < 0.8 * q.distance]
A, inl = cv2.estimateAffinePartial2D(np.float32([kz[g.queryIdx].pt for g in good]), np.float32([ki[g.trainIdx].pt for g in good]),
                                     method=cv2.RANSAC, ransacReprojThreshold=4, maxIters=20000)
base = np.diag([2, 2, 1]) @ np.vstack([A, [0, 0, 1]]) @ np.diag([1/8, 1/8, 1])
def chamfer(n, A0, steps):
    zf = load(n, 4); edges = (zf < 128).astype(np.uint8)
    dt = cv2.distanceTransform(1 - edges, cv2.DIST_L2, 3)
    inv = np.linalg.inv(A0)
    def R_(p):
        dx, dy, th, sc = p
        return np.array([[math.cos(th)*sc, -math.sin(th)*sc, dx], [math.sin(th)*sc, math.cos(th)*sc, dy], [0, 0, 1]])
    def cost(p):
        q = (R_(p) @ inv @ P.T).T[:, :2] / 4
        ok = (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < dt.shape[1]-1) & (q[:, 1] < dt.shape[0]*0.72)
        if ok.sum() < 40: return 99.0, int(ok.sum())
        return float(np.mean(np.minimum(dt[q[ok, 1].astype(int), q[ok, 0].astype(int)], 12))), int(ok.sum())
    best = (0, 0, 0, 1); bc, _ = cost(best)
    # coarse translation grid
    for gx in range(-steps, steps + 1, 60):
        for gy in range(-steps, steps + 1, 60):
            cc, _ = cost((gx, gy, 0, 1))
            if cc < bc: bc, best = cc, (gx, gy, 0, 1)
    for sxy, sth, ssc in [(30, .003, .003), (10, .0015, .0015), (4, .0006, .0006), (1.5, .0002, .0002)]:
        imp = True
        while imp:
            imp = False
            for k, s in enumerate([sxy, sxy, sth, ssc]):
                for sg in (-1, 1):
                    p = list(best); p[k] += sg * s; cc, _ = cost(p)
                    if cc < bc - 1e-4: bc, best, imp = cc, tuple(p), True
    _, npt = cost(best)
    return np.linalg.inv(R_(best) @ inv), bc, npt
for i, zn in enumerate(zones):
    n = 28 + i
    off = (np.array(rect[zn]) - np.array(rect['A'])) * K
    A0 = base.copy(); A0[0, 2] += off[0]; A0[1, 2] += off[1]
    Af, bc, npt = chamfer(n, A0, 600)
    out[str(n)] = Af.tolist()
    print('L06', zn, n, 'chamfer', round(bc, 2), 'pts', npt, 'scale', round(math.hypot(Af[0, 0], Af[1, 0]), 4), flush=True)
# --- stage 2: ECC to sibling
def prep(n):
    im = load(n, 8).astype(np.float32) / 255.0
    im[int(im.shape[0]*0.72):, :] = 1.0; im[:, int(im.shape[1]*0.905):] = 1.0
    return cv2.GaussianBlur(1 - im, (0, 0), 3)
for s, first in series.items():
    if s == 'L06': continue
    for i, zn in enumerate(zones):
        n = first + i; ref = 28 + i
        a_ = prep(ref); b_ = prep(n)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            cc, warp = cv2.findTransformECC(a_, b_, warp, cv2.MOTION_EUCLIDEAN,
                                            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6), None, 5)
        except cv2.error as e:
            print(s, zn, n, 'ECC failed'); continue
        W = np.vstack([warp, [0, 0, 1]]).astype(float)   # ref(/8) -> n(/8)
        S = np.diag([8, 8, 1.0])
        n2ref = S @ np.linalg.inv(W) @ np.linalg.inv(S)   # n full -> ref full
        Af = np.array(out[str(ref)]) @ n2ref
        out[str(n)] = Af.tolist()
        print(s, zn, n, 'ecc', round(cc, 3), 'shift', np.round(W[:2, 2] * 8, 1), 'rot', round(math.degrees(math.atan2(W[1, 0], W[0, 0])), 3), flush=True)
json.dump(out, open(SCR + '/zoneT.json', 'w'))
