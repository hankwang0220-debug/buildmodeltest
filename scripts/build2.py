# v2 model: existing terrain from L-03 spot heights, design terrain from L-07 labels, trees (L-08), lights (L-11),
# textures mosaicked from the zone sheets (L-06 paving, L-09 planting).
import csv, json, sys, base64, math, re, os
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFilter
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
Image.MAX_IMAGE_PIXELS = None
CSV, DD, SCR, OUTDIR = sys.argv[1:5]
sys.stdout.reconfigure(encoding='utf-8')
exec(open(SCR + '/features.py', encoding='utf-8').read())          # digitised constants shared with v1
T = json.load(open(SCR + '/T27.json')); a, b, c, d = T['a'], T['b'], T['c'], T['d']
ZT = {int(k): np.array(v) for k, v in json.load(open(SCR + '/zoneT.json')).items()}
K = 8.2775; S = math.hypot(a, b); MPP = K * S
X0, Y0, X1, Y1 = 60.0, 90.0, 1450.0, 560.0
Mi = np.linalg.inv(np.array([[a, b], [b, -a]]))
U1, V1 = (X1 - X0) * MPP, (Y1 - Y0) * MPP
def px2uv(p): return ((p[0] - X0) * MPP, (p[1] - Y0) * MPP)
def idx2uv(fx, fy): return ((fx / K - X0) * MPP, (fy / K - Y0) * MPP)
def zone2uv(n, x, y):
    q = ZT[n] @ np.array([x, y, 1.0]); return idx2uv(q[0], q[1])
def en2px(e, n):
    fx, fy = Mi @ np.array([e - c, n - d]); return (fx / K, fy / K)
def uv2en(u, v):
    fx, fy = (X0 + u / MPP) * K, (Y0 + v / MPP) * K
    return a*fx + b*fy + c, b*fx - a*fy + d
inside = lambda u, v: 0 <= u <= U1 and 0 <= v <= V1
wl = np.array([px2uv(p) for p in wall])
def wall_v(u): return np.interp(u, wl[:, 0], wl[:, 1])

# ---------- survey ----------
rows = list(csv.reader(open(CSV, encoding='utf-8-sig', errors='replace')))[1:]
survey = [(r[1].strip(), px2uv(en2px(float(r[3]), float(r[2]))), float(r[4])) for r in rows]

# ---------- L-03 spot heights ----------
raw = json.load(open(SCR + '/spots_raw.json'))
spots = []
for n, x, y, v, sc in raw:
    if sc < 0.6 or not (0.0 <= v <= 16.0): continue
    u, w = zone2uv(n, x, y)
    if inside(u, w): spots.append((u, w, v))
spots = np.array(spots)
# outlier removal against neighbours (8–20 m)
tr = cKDTree(spots[:, :2]); keep = np.ones(len(spots), bool)
for i, p in enumerate(spots):
    nb = [j for j in tr.query_ball_point(p[:2], 14) if j != i]
    if len(nb) >= 3:
        med = np.median(spots[nb, 2])
        if abs(p[2] - med) > 2.2: keep[i] = False
print('spots', len(spots), 'kept', keep.sum())
spots = spots[keep]
# merge duplicates (same point read from two overlapping sheets)
tr = cKDTree(spots[:, :2]); used = np.zeros(len(spots), bool); sp2 = []
for i in range(len(spots)):
    if used[i]: continue
    g = tr.query_ball_point(spots[i, :2], 1.2); used[g] = True; sp2.append(spots[g].mean(0))
spots = np.array(sp2); print('spots merged', len(spots))

# ---------- L-07 design labels ----------
dl = json.load(open(SCR + '/design_raw.json'))
design = []
for n, kind, v, x, y, sc, h in dl:
    if sc < 0.6: continue
    if kind == 'C' and not (1 <= v <= 9): continue
    if kind in ('WS', 'WB', 'EL', 'TF', 'SP') and not (0.3 <= v <= 12): continue
    u, w = zone2uv(n, x, y)
    if inside(u, w): design.append((u, w, v, kind))
print('design labels', len(design), {k: sum(1 for q in design if q[3] == k) for k in 'C WS WB EL TF SP'.split()})

# ---------- surfaces ----------
STEP = 1.5
nu, nv = int(U1 / STEP) + 1, int(V1 / STEP) + 1
gu, gv = np.meshgrid(np.arange(nu) * STEP, np.arange(nv) * STEP)
def dens(line, z, step=6):
    out = []
    for p, q in zip(line[:-1], line[1:]):
        L = math.dist(p, q); k = max(1, int(L / step))
        for i in range(k):
            t = i / k; out.append((px2uv((p[0]+(q[0]-p[0])*t, p[1]+(q[1]-p[1])*t)), z))
    return out
river = dens(jm_bed, 0.3) + dens(xd_bed, 0.3) + dens(jm_toe, 1.0) + dens(xd_toe, 1.0) + dens(jm_far, 2.0)
st = cKDTree(spots[:, :2])
river = [(p, z) for p, z in river if st.query(p)[0] > 20]       # only where L-03 has no data
# the 2025.08 survey is the reference: drop L-03 spot heights within 4 m of any survey point
svp = np.array([p for nm, p, z in survey if not nm.startswith('db') and inside(*p)])
near_sv = cKDTree(svp).query(spots[:, :2])[0] < 4.0
print('L-03 spots replaced by survey', int(near_sv.sum()))
base_pts = [((u, w), z) for (u, w, z), ns in zip(spots, near_sv) if not ns] + river
base_pts += [(p, z) for nm, p, z in survey if not nm.startswith('db') and inside(*p)]
def surface(pts):
    P = np.array([p for p, z in pts]); Z = np.array([z for p, z in pts])
    lin = LinearNDInterpolator(P, Z); near = NearestNDInterpolator(P, Z)
    H = lin(gu, gv); m = np.isnan(H); H[m] = near(gu[m], gv[m])
    H = gaussian_filter(H, 0.8)
    wv = wall_v(gu[0])[None, :]
    H[gv >= wv] = 9.0
    return H
H_ex = surface(base_pts)
# design: drop existing points that disagree with nearby design labels, then add the labels
def hat(H, u, w): return float(H[min(int(round(w / STEP)), nv - 1), min(int(round(u / STEP)), nu - 1)])
# drop groyne-crest labels (modelled as solids) and OCR noise such as mileage text read as "+10" (|change| > 3.5 m)
design = [q for q in design if q[3] != 'EL' and abs(q[2] - hat(H_ex, q[0], q[1])) <= 3.5]
print('design labels after filter', len(design))
dpts = np.array([(u, w, v) for u, w, v, k in design if k != 'WS'])
dt = cKDTree(dpts[:, :2])
des_pts = []
for p, z in base_pts:
    dd, j = dt.query(p)
    if dd < 9 and abs(dpts[j, 2] - z) > 0.4: continue
    des_pts.append((p, z))
des_pts += [((u, w), v) for u, w, v in dpts]
H_de = surface(des_pts)
print('z existing', H_ex.min().round(2), H_ex.max().round(2), 'design', H_de.min().round(2), H_de.max().round(2))
diff = H_de - H_ex
print('cut/fill m3 (grid, park side):', round(float(diff[diff > 0].sum() * STEP * STEP)), round(float(diff[diff < 0].sum() * STEP * STEP)))

# ---------- channel (引水道) water: chain WS points ----------
sys.path.insert(0, SCR)
from channels import make_chains          # WS labels → channel lines (monotonic, ≤45 m links, no jump into the eco-pond)
channels = make_chains(design)

# ---------- trees ----------
table = json.load(open(SCR + '/trees_table.json', encoding='utf-8'))
tp = []
for f in ('treepos_raw.json', 'treepos_raw2.json'):
    if os.path.exists(SCR + '/' + f): tp += json.load(open(SCR + '/' + f))
trees = {}
for n, lab, cx, cy, xa, h, sc in tp:
    m = re.fullmatch(r'([TH])(\d{2,3})', lab)
    if not m or sc < 0.7: continue
    num = int(m.group(2)); tid = f'{m.group(1)}{num:02d}' if num < 100 else f'{m.group(1)}{num}'
    if m.group(1) == 'H' and num > 42 or m.group(1) == 'T' and num > 158: continue
    u, w = zone2uv(n, cx, cy)
    if not inside(u, w): continue
    trees.setdefault(tid, []).append((u, w))
tree_objs = []
for tid, ps in trees.items():
    u, w = np.mean(ps, 0)
    info = table.get(tid, {}); nums = info.get('nums', [])
    crown = nums[2] if len(nums) >= 3 and 0.5 <= nums[2] <= 30 else 6.0
    hgt = nums[3] if len(nums) >= 4 and 2 <= nums[3] <= 20 else 7.0
    act = info.get('action', '')
    status = 'removed' if re.search(r'死亡|移除', act) else 'move' if '移植' in act else 'keep'
    if status == 'removed': continue
    tree_objs.append(dict(id=tid, name=info.get('name', ''), uv=[round(u, 2), round(w, 2)], crown=crown, h=hgt, status=status,
                          palm=bool(re.search(r'葵|椰', info.get('name', '')))))
print('trees placed', len(tree_objs))

# ---------- lights (L-11) ----------
lr = json.load(open(SCR + '/lights_raw.json')) if os.path.exists(SCR + '/lights_raw.json') else []
lights = {}
for n, lab, cx, cy, xa, h, sc in lr:
    if sc < 0.7 or not re.fullmatch(r'G\d{2}', lab): continue
    u, w = zone2uv(n, xa - 100, cy)              # the symbol sits just left of the label
    if inside(u, w): lights.setdefault(lab, []).append((u, w))
light_objs = [dict(name=k, uv=[round(x, 2) for x in np.mean(v, 0)]) for k, v in sorted(lights.items())]
print('lights', len(light_objs), sorted({k[0] for k in lights}))

# ---------- textures: mosaic zone sheets ----------
F = 3.0
TW, TH = int((X1 - X0) * F), int((Y1 - Y0) * F)
rect = dict(A=(78,322), B=(410,110), C=(410,322), D=(745,110), E=(745,322), F=(1077,110), G=(1077,322))
def mosaic(first):
    ty, tx = np.mgrid[0:TH, 0:TW].astype(np.float32)
    ipx = (X0 + (tx + 0.5) / F) * K; ipy = (Y0 + (ty + 0.5) / F) * K    # index full-res px
    # fall back to the full-site plan (sheet 27) wherever no zone sheet covers the pixel
    idx27 = Image.open(f'{DD}/1130070-027.tif').convert('L')
    out = np.asarray(idx27.crop((int(X0*K), int(Y0*K), int(X1*K), int(Y1*K))).resize((TW, TH), Image.LANCZOS)).copy()
    big = Image.fromarray(out).filter(ImageFilter.MaxFilter(9))
    zl = (np.asarray(big.point(lambda v: 255 if v < 110 else 0).filter(ImageFilter.MaxFilter(13))) > 0)
    out[zl] = 255                                          # drop the big zone letters of the index sheet
    for i, zn in enumerate('ABCDEFG'):
        n = first + i
        rx, ry = rect[zn]
        sel = (ipx / K >= rx - 2) & (ipx / K <= rx + 337) & (ipy / K >= ry - 2) & (ipy / K <= ry + 214)
        if not sel.any(): continue
        inv = np.linalg.inv(ZT[n])
        zx = inv[0, 0] * ipx + inv[0, 1] * ipy + inv[0, 2]; zy = inv[1, 0] * ipx + inv[1, 1] * ipy + inv[1, 2]
        im = Image.open(f'{DD}/1130070-{n:03d}.tif').convert('L')
        f = 4; small = np.asarray(im.resize((im.width // f, im.height // f), Image.BOX))
        W, H = im.size
        ok = sel & (zx > W*0.02) & (zx < W*0.905) & (zy > H*0.04) & (zy < H*0.72)
        if zn in 'BDF':
            ok = sel & (zx > W*0.02) & (zx < W*0.905) & (zy > H*0.27) & (zy < H*0.86)   # tables on top; map runs lower
        smp = cv2.remap(small, (zx / f).astype(np.float32), (zy / f).astype(np.float32), cv2.INTER_AREA if False else cv2.INTER_LINEAR, borderValue=255)
        out[ok] = np.minimum(out[ok], smp[ok]) if False else smp[ok]
    return out
def colorize(lines, H):
    Hup = np.array(Image.fromarray(H.astype(np.float32)).resize((TW, TH), Image.BILINEAR))
    land = np.array([150, 185, 112], np.float32); sand = np.array([196, 184, 150], np.float32); city = np.array([200, 200, 196], np.float32)
    t = np.clip((Hup - 1.0) / 2.5, 0, 1)[..., None]
    base = sand * (1 - t) + land * t
    tu = (np.arange(TW) + 0.5) / F * MPP; tv = (np.arange(TH)[:, None] + 0.5) / F * MPP
    behind = np.broadcast_to(tv >= wall_v(tu)[None, :], (TH, TW))
    base[behind] = city
    img = Image.fromarray(base.astype(np.uint8)); dr = ImageDraw.Draw(img)
    tpx = lambda p: ((p[0] - X0) * F, (p[1] - Y0) * F)
    cols = dict(basket=(190,110,80), tennis=(70,130,120), baseball=(176,150,100), court=(110,140,170), field=(120,170,95), skate=(165,165,175))
    for fl in fields: dr.polygon([tpx(p) for p in fl['poly']], fill=cols[fl['kind']])
    dr.polygon([tpx(p) for p in lagoon], fill=(95, 140, 170)); dr.polygon([tpx(p) for p in lagoon_island], fill=(140, 175, 105))
    for ch in channels:
        dr.line([(q[0] / MPP * F, q[1] / MPP * F) for q in ch], fill=(95, 140, 170), width=int(3.5 / MPP * F))
    seqs = {}
    for nm, p, z in survey:
        mm = re.match(r'^(\d*RD)(\d+)$', nm)
        if mm: seqs.setdefault(mm.group(1), []).append((int(mm.group(2)), p))
    pw = max(2, int(2.2 * F / MPP))
    for k_, lst in seqs.items():
        lst.sort()
        for (i1, p1), (i2, p2) in zip(lst[:-1], lst[1:]):
            if i2 == i1 + 1 and math.dist(p1, p2) < 7:
                dr.line([(p1[0] / MPP * F, p1[1] / MPP * F), (p2[0] / MPP * F, p2[1] / MPP * F)], fill=(222, 205, 165), width=pw)
    L = Image.fromarray(lines)
    thick = L.filter(ImageFilter.MaxFilter(9)); mk = thick.point(lambda v: 255 if v < 110 else 0).filter(ImageFilter.MaxFilter(13))
    L = Image.composite(Image.new('L', L.size, 255), L, mk)
    dk = 1 - np.asarray(L, np.float32) / 255
    # privacy: no base-map linework outside the project
    priv = Image.new('L', (TW, TH), 0); pd = ImageDraw.Draw(priv)
    pd.polygon([tpx(p) for p in [(X0, Y0)] + jm_far + [(619, Y0)]], fill=255)
    offs = [(x, y + 2) for x, y in wall]
    pd.polygon([tpx(p) for p in offs + [(X1, offs[-1][1]), (X1, Y1), (X0, Y1)]], fill=255)
    pm = np.asarray(priv) > 0; dk[pm] = 0
    arr = np.asarray(img, np.float32); arr[pm & ~behind] = land
    out = arr * (1 - 0.62 * dk[..., None])
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
L06 = mosaic(28); L09 = mosaic(56)
np.save(OUTDIR + '/L06.npy', L06)
colorize(L06, H_de).save(OUTDIR + '/tex_paving.jpg', quality=80)
colorize(L09, H_de).save(OUTDIR + '/tex_planting.jpg', quality=80)
Image.open(OUTDIR + '/tex_paving.jpg').resize((TW // 3, TH // 3)).save(OUTDIR + '/tex_paving_small.png')
Image.open(OUTDIR + '/tex_planting.jpg').resize((TW // 3, TH // 3)).save(OUTDIR + '/tex_planting_small.png')

# ---------- groynes (as v1) ----------
db = sorted([(p, z, nm) for nm, p, z in survey if nm.startswith('db')], key=lambda t: -t[0][0])
v1 = json.load(open(OUTDIR + '/model.json', encoding='utf-8'))
def enc(H): return base64.b64encode(np.round(H * 100).astype('<i2').tobytes()).decode()
model = dict(v1)
# ---------- survey point attribution: what the drawings show at each measured point ----------
def pip(pt, poly):
    x, y = pt; ins = False; n_ = len(poly)
    for i_ in range(n_):
        x1, y1 = poly[i_]; x2, y2 = poly[(i_ + 1) % n_]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1: ins = not ins
    return ins
def seg_dist(p, a_, b_):
    p, a_, b_ = map(np.asarray, (p, a_, b_)); ab = b_ - a_; t_ = np.clip(np.dot(p - a_, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
    return float(np.linalg.norm(p - (a_ + t_ * ab)))
fpolys = [(fl['name'], [px2uv(q) for q in fl['poly']]) for fl in fields]
lag_uv = [px2uv(q) for q in lagoon]; isl_uv = [px2uv(q) for q in lagoon_island]
gro = v1['groynes'] if 'v1' in dir() else json.load(open(OUTDIR + '/model.json', encoding='utf-8'))['groynes']
lit = [(l['name'], l['uv']) for l in light_objs]
dark = (L06 < 128).astype(np.float32)
integ = cv2.integral(dark)
rpx = int(1.5 / MPP * F)
def pave_density(u, w):
    x, y = int(u / MPP * F), int(w / MPP * F)
    x0_, x1_, y0_, y1_ = max(x - rpx, 0), min(x + rpx, TW - 1), max(y - rpx, 0), min(y + rpx, TH - 1)
    if x1_ <= x0_ or y1_ <= y0_: return 0.0
    return float((integ[y1_, x1_] - integ[y0_, x1_] - integ[y1_, x0_] + integ[y0_, x0_]) / ((x1_ - x0_) * (y1_ - y0_)))
# paving = hatched areas on the L-06 mosaic (dense short strokes); distance (px) from every pixel to the nearest paving
# at mosaic resolution the paving hatch reads as light-grey bands between the edge lines
bl_ = cv2.blur(L06.astype(np.float32), (9, 9))
pmask = ((bl_ < 244) & (bl_ > 150)).astype(np.uint8)
pmask = cv2.morphologyEx(pmask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
pdist = cv2.distanceTransform(1 - pmask, cv2.DIST_L2, 3)
Image.fromarray(pmask * 255).resize((TW // 3, TH // 3)).save(OUTDIR + '/paving_mask.png')
codes = {'RD': '步道', '2RD': '步道', '3RD': '步道', 'db': '丁壩', 'RI': '河岸', 'wb': '水邊', 'ty': '其他', 'sos': '其他',
         'GB': '其他', 'bb': '基準點', 'base': '基準點'}
H_l03 = surface([((u, w), z) for u, w, z in spots] + river)   # pre-construction surface from L-03 only
sv_out = []
for nm, p, z in survey:
    if not inside(*p): continue
    u, w = p; code = re.sub(r'\d+$', '', nm)
    obj = ''
    for g in gro:
        if seg_dist(p, g['root'], g['tip']) < 4.0: obj = '丁壩 ' + g['name']; break
    if not obj:
        for nm_, poly in fpolys:
            if pip(p, poly): obj = nm_; break
    if not obj and pip(p, lag_uv): obj = '生態池島' if pip(p, isl_uv) else '生態池'
    if not obj:
        for ln, luv in lit:
            if math.dist(p, luv) < 2.0: obj = '高燈 ' + ln; break
    if not obj and w >= wall_v(u) - 1: obj = '堤防／快速道路'
    off = 0.0
    if not obj:
        x_, y_ = min(int(u / MPP * F), TW - 1), min(int(w / MPP * F), TH - 1)
        off = float(pdist[y_, x_]) * MPP / F
        obj = '設計鋪面' if off < 1.0 else '設計綠地'
    dz_ = hat(H_de, u, w); l03 = hat(H_l03, u, w)
    sv_out.append([round(u, 2), round(w, 2), round(z, 3), nm, code, codes.get(code, '其他'), obj, round(dz_, 2), round(l03, 2), round(off, 2)])
from collections import Counter
print('survey attribution', Counter(s[6].split(' ')[0] for s in sv_out).most_common())
print('RD on designed paving', sum(1 for s in sv_out if s[4].endswith('RD') and s[6] == '設計鋪面'), 'of', sum(1 for s in sv_out if s[4].endswith('RD')))
offs = np.array([s[9] for s in sv_out if s[4].endswith('RD')])
print('RD offset from paving (m) percentiles 50/75/90/95/max', np.percentile(offs, [50, 75, 90, 95, 100]).round(2))
# map: RD points coloured by offset on the paving mask
mp = Image.fromarray(np.stack([L06] * 3, -1)).convert('RGB'); dm = ImageDraw.Draw(mp)
for s in sv_out:
    if not s[4].endswith('RD'): continue
    x_, y_ = s[0] / MPP * F, s[1] / MPP * F; o = s[9]
    col = (0, 160, 0) if o < 0.75 else (255, 160, 0) if o < 2 else (220, 0, 0)
    dm.ellipse([x_ - 4, y_ - 4, x_ + 4, y_ + 4], fill=col)
mp.save(OUTDIR + '/rd_offset.png')
dd_ = np.array([s[2] - s[7] for s in sv_out if s[4].endswith('RD')])
print('RD measured - design: mean', dd_.mean().round(2), 'abs median', np.median(np.abs(dd_)).round(2))
model.update(heights=enc(H_ex), heightsDesign=enc(H_de), spots=[[round(u, 2), round(w, 2), round(z, 2)] for u, w, z in spots],
             designPts=[[round(u, 2), round(w, 2), v, k] for u, w, v, k in design], channels=channels, trees=tree_objs,
             cloud=sv_out)
if len(light_objs) >= 20:
    hz = lambda u, v: float(H_de[min(int(round(v / STEP)), nv - 1), min(int(round(u / STEP)), nu - 1)])
    model['lights'] = [dict(name=l['name'], uv=l['uv'], z=hz(*l['uv'])) for l in light_objs]
model['meta'] = dict(v1['meta'], version=2)
json.dump(model, open(OUTDIR + '/model2.json', 'w', encoding='utf-8'), ensure_ascii=False)
np.save(OUTDIR + '/diff.npy', diff)
print('done')
