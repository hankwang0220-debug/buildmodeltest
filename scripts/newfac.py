# New-construction facilities: point items read off the L-06 zone sheets (preview px of each sheet, 1600 px wide),
# plus paving / revetment highlight overlay from the L-06 mosaic.
import sys, json, math
import numpy as np, cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
DD, SCR, OUT = sys.argv[1:4]
T = json.load(open(SCR + '/T27.json')); a, b = T['a'], T['b']
ZT = {int(k): np.array(v) for k, v in json.load(open(SCR + '/zoneT.json')).items()}
K = 8.2775; MPP = K * math.hypot(a, b); X0, Y0, X1, Y1 = 60.0, 90.0, 1450.0, 560.0
widths = {}
def zuv(n, px, py):
    if n not in widths: widths[n] = Image.open(f'{DD}/1130070-{n:03d}.tif').size[0]
    k = widths[n] / 1600.0
    q = ZT[n] @ np.array([px * k, py * k, 1.0])
    return [round((q[0] / K - X0) * MPP, 2), round((q[1] / K - Y0) * MPP, 2)]
P = [  # (sheet, kind, x, y, note)
 (28, 'wash', 357.5, 693, 'A區'), (30, 'wash', 390.5, 711, 'C區'), (30, 'wash', 1160, 751, 'C區'), (34, 'wash', 97.6, 571, 'G區'),
 (28, 'drop', 1106, 253, 'A區 WS+4.7→4.5'), (30, 'drop', 596, 95.5, 'C區 WS+3.7→3.5'), (30, 'drop', 1313.5, 98, 'C區 WS+3.2→3.0'),
 (31, 'rock', 343, 751, 'D區 生態池上游端'), (31, 'rock', 700, 936, 'D區 生態池下游端'),
 (31, 'perch', 206.7, 826.7, 'D區'), (31, 'perch', 527, 813.5, 'D區'), (31, 'perch', 684, 783, 'D區'), (31, 'perch', 434, 931, 'D區'),
 (32, 'perch', 392.4, 150.4, 'E區'),
 (31, 'log', 763.6, 815.6, 'D區'), (32, 'log', 619.8, 78.4, 'E區'),
]
benches = [(789.2, 438.0, -68), (820.7, 480.3, -77), (839.1, 457.2, -82), (839.9, 514.9, -77), (806.4, 532.1, -65),
           (833.7, 532.1, -76), (828.3, 568.6, -65), (798.8, 599.4, -48), (733.4, 632.0, -40)]
names = dict(wash='新設洗手台', drop='跌水工', rock='疊石', perch='鳥踏區', log='水中枯木微棲地', bench='休憩座椅')
detail = dict(wash='D-08', drop='D-03', rock='D-06', perch='D-07', log='D-07', bench='D-05')
fac = []
for n, kd, x, y, note in P:
    fac.append(dict(kind=kd, name=names[kd], uv=zuv(n, x, y), note=note, ref=detail[kd]))
for x, y, ang in benches:
    p0 = zuv(32, x, y); p1 = zuv(32, x + math.cos(math.radians(ang)), y + math.sin(math.radians(ang)))
    fac.append(dict(kind='bench', name=names['bench'], uv=p0, rot=round(math.atan2(p1[1] - p0[1], p1[0] - p0[0]), 3),
                    len=4.4, note='E區 圓形廣場', ref='D-05'))
# ---------- overlay: paving (hatched band) + revetment (thick lines) ----------
L06 = np.load(OUT + '/L06.npy')
TH, TW = L06.shape; F = TW / (X1 - X0)
M = json.load(open(OUT + '/model2.json', encoding='utf-8'))
S = F / MPP                                   # mosaic px per metre
mp = lambda u, v: (int(round(u * S)), int(round(v * S)))
# new paving = as-built path edges from the 2025.08 survey (RD / 2RD / 3RD strings), each edge buffered ~1.2 m
import re
seqs = {}
for c in M['cloud']:
    mm = re.match(r'^(\d*RD)(\d+)$', c[3])
    if mm: seqs.setdefault(mm.group(1), []).append((int(mm.group(2)), c[0], c[1]))
pave = np.zeros((TH, TW), np.uint8)
for k_, lst in seqs.items():
    lst.sort()
    for (i1, u1, v1), (i2, u2, v2) in zip(lst[:-1], lst[1:]):
        if i2 == i1 + 1 and math.dist((u1, v1), (u2, v2)) < 7:
            cv2.line(pave, mp(u1, v1), mp(u2, v2), 1, int(2.4 * S))
# revetment = thick lines (dash-dot 卵塊石 / solid 植生袋) near the eco-pond, plus the channel banks (WS chains)
dark = (L06 < 110).astype(np.uint8)
thick = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
nl, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(thick, np.ones((9, 9), np.uint8)))
keep = np.zeros(nl, bool)
for i in range(1, nl):
    w, h, ar = st[i, 2], st[i, 3], st[i, 4]
    if max(w, h) / S > 12 and ar / max(w * h, 1) < 0.35: keep[i] = True     # long, not a filled island
near = np.zeros((TH, TW), np.uint8)
lag = np.array([[(x - X0) * MPP * S, (y - Y0) * MPP * S] for x, y in M['lagoon_px']], np.int32) if 'lagoon_px' in M else None
if lag is None:
    lag = np.array([mp(u, v) for u, v in M['lagoon']], np.int32)
cv2.fillPoly(near, [lag], 1); near = cv2.dilate(near, np.ones((int(30 * S), int(30 * S)), np.uint8))
rev = ((thick > 0) & keep[lab] & (near > 0)).astype(np.uint8)
rev = cv2.dilate(rev, np.ones((5, 5), np.uint8))
for ch in M.get('channels', []):
    pts = np.array([mp(p[0], p[1]) for p in ch], np.int32)
    cv2.polylines(rev, [pts], False, 1, int(4.0 * S))
rgba = np.zeros((TH, TW, 4), np.uint8)
rgba[pave > 0] = (255, 150, 30, 255)
rgba[rev > 0] = (230, 40, 90, 255)
im = Image.fromarray(rgba).resize((TW // 2, TH // 2), Image.BILINEAR)
im.save(OUT + '/overlay_new.png', optimize=True)
prev = Image.fromarray(np.stack([L06] * 3, -1)).convert('RGB'); prev.paste(Image.fromarray(rgba), (0, 0), Image.fromarray(rgba))
prev.resize((TW // 3, TH // 3)).save(OUT + '/overlay_check.png')
M['newFacilities'] = fac
M['flow'] = dict(jm=[[round((x - X0) * MPP, 1), round((y - Y0) * MPP, 1)] for x, y in
                     [(60,305),(150,325),(200,334),(280,328),(350,316),(400,301),(450,280),(500,262),(540,240),(600,195),(640,160),(670,120),(700,100),(760,100),(840,100),(900,110),(950,105),(1100,100),(1300,95),(1450,92)]],
                 xd=[[round((x - X0) * MPP, 1), round((y - Y0) * MPP, 1)] for x, y in
                     [(760,120),(820,150),(870,200),(940,250),(1020,235),(1069,229),(1128,216),(1208,196),(1289,175),(1369,152),(1450,135)]])
json.dump(M, open(OUT + '/model2.json', 'w', encoding='utf-8'), ensure_ascii=False)
print('facilities', len(fac), 'pave px', int(pave.sum()), 'rev px', int(rev.sum()), 'overlay', im.size)
