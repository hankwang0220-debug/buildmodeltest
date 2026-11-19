# Refine L-06 zone sheets against the index plan by ECC (affine) at index/2 resolution,
# then re-derive all sibling sheets with ECC (euclidean) against their L-06 sheet.
import sys, json, math
import numpy as np, cv2
from PIL import Image, ImageFilter
Image.MAX_IMAGE_PIXELS = None
D, SCR = sys.argv[1:3]
K = 8.2775
Z = json.load(open(SCR + '/zoneT.json'))
def load(n, f):
    im = Image.open(f'{D}/1130070-{n:03d}.tif').convert('L')
    return im.resize((im.width // f, im.height // f), Image.BILINEAR)
idx = load(27, 2)
thick = idx.filter(ImageFilter.MaxFilter(9))                     # remove the big zone letters
mask = thick.point(lambda v: 255 if v < 128 else 0).filter(ImageFilter.MaxFilter(13))
idx = Image.composite(Image.new('L', idx.size, 255), idx, mask)
I = cv2.GaussianBlur(1 - np.asarray(idx, np.float32) / 255, (0, 0), 2.5)
I[:int(50*K/2)] = 0; I[int(560*K/2):] = 0
zones = 'ABCDEFG'
for i, zn in enumerate(zones):
    n = 28 + i
    z = np.asarray(load(n, 8), np.float32) / 255
    zz = 1 - z; h, w = zz.shape
    zz[int(h*0.70):] = 0; zz[:, int(w*0.905):] = 0; zz[:int(h*0.045)] = 0
    if zn in 'BD': zz[:int(h*0.32), :int(w*0.68)] = 0            # legend tables on B/D sheets
    A0 = np.array(Z[str(n)])                                      # zone full -> index full
    M = np.diag([.5, .5, 1]) @ A0 @ np.diag([8, 8, 1])            # zone/8 -> index/2
    Wz = cv2.warpAffine(zz, M[:2], (I.shape[1], I.shape[0]))
    Wz = cv2.GaussianBlur(Wz, (0, 0), 2.5)
    valid = (cv2.warpAffine(np.ones_like(zz), M[:2], (I.shape[1], I.shape[0])) > 0.5).astype(np.uint8)
    valid = cv2.erode(valid, np.ones((15, 15), np.uint8))
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(I, Wz, warp, cv2.MOTION_AFFINE,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 400, 1e-7), valid, 1)
    except cv2.error:
        print(zn, 'ECC failed'); continue
    W3 = np.vstack([warp, [0, 0, 1]]).astype(float)               # index/2 -> warped
    # zone full -> index full:  index = (M^-1 W)^-1 ... compose: zone/8 = M^-1 W (index/2)
    idx2zone8 = np.linalg.inv(M) @ W3
    A1 = np.diag([2, 2, 1]) @ np.linalg.inv(idx2zone8) @ np.diag([1/8, 1/8, 1])
    Z[str(n)] = A1.tolist()
    print(zn, n, 'ecc', round(cc, 3), 'shift(index/2 px)', np.round(warp[:, 2], 1), 'lin', np.round(warp[:, :2].ravel(), 4), flush=True)
# siblings
def prep(n):
    im = np.asarray(load(n, 8), np.float32) / 255.0
    im = im.copy(); im[int(im.shape[0]*0.72):, :] = 1.0; im[:, int(im.shape[1]*0.905):] = 1.0
    return cv2.GaussianBlur(1 - im, (0, 0), 3)
series = {'L03': 7, 'L07': 36, 'L08': 47, 'L09': 56, 'L10': 64, 'L11': 72}
for s, first in series.items():
    for i, zn in enumerate(zones):
        n = first + i; ref = 28 + i
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            cc, warp = cv2.findTransformECC(prep(ref), prep(n), warp, cv2.MOTION_EUCLIDEAN,
                                            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6), None, 5)
        except cv2.error:
            print(s, zn, 'ECC failed'); continue
        Wm = np.vstack([warp, [0, 0, 1]]).astype(float); S = np.diag([8, 8, 1.0])
        Z[str(n)] = (np.array(Z[str(ref)]) @ S @ np.linalg.inv(Wm) @ np.linalg.inv(S)).tolist()
json.dump(Z, open(SCR + '/zoneT.json', 'w'))
print('saved')
