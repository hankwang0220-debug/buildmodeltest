# Build terrain + objects + draped texture for the 3D model.
# All hand-digitised geometry is in "preview px" of sheet 27 (L-06-1, 1600 px wide preview).
import csv, json, sys, base64, io, math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import gaussian_filter
Image.MAX_IMAGE_PIXELS = None

CSV, TIF27, TJSON, OUTDIR = sys.argv[1:5]
T = json.load(open(TJSON))
a, b, c, d = T['a'], T['b'], T['c'], T['d']
K = 8.2775                      # full-res px per preview px
S = math.hypot(a, b)            # metres per full-res px
MPP = K * S                     # metres per preview px
X0, Y0, X1, Y1 = 60.0, 90.0, 1450.0, 560.0   # model extent (preview px)
Mi = np.linalg.inv(np.array([[a, b], [b, -a]]))

def px2uv(p):   # preview px -> local metres (u along plan x, v along plan y/down)
    return ((p[0] - X0) * MPP, (p[1] - Y0) * MPP)
def px2en(p):
    fx, fy = p[0] * K, p[1] * K
    return (a*fx + b*fy + c, b*fx - a*fy + d)
def en2px(e, n):
    fx, fy = Mi @ np.array([e - c, n - d])
    return (fx / K, fy / K)
U1, V1 = px2uv((X1, Y1))

# ---------------- survey ----------------
rows = list(csv.reader(open(CSV, encoding='utf-8-sig', errors='replace')))[1:]
survey = []
for r in rows:
    name = r[1].strip(); e, n, z = float(r[3]), float(r[2]), float(r[4])
    px = en2px(e, n)
    survey.append((name, px, z, e, n))

# ---------------- digitised features (preview px) ----------------
wall = [(60,497),(150,497),(250,498),(350,500),(450,502),(550,505),(650,507),(750,508),(850,508),(940,505),
        (1000,493),(1060,478),(1120,458),(1180,436),(1240,410),(1300,383),(1360,355),(1420,328),(1450,314)]
WALL_W = 13.0  # m, levee + Shuiyuan expressway band (L-12 sections)
jm_far = [(60,270),(100,272),(140,283),(160,290),(200,300),(240,305),(280,307),(320,303),(360,298),(385,292),
          (398,280),(425,262),(450,250),(480,238),(531,196),(580,158),(606,138),(619,90)]
jm_toe = [(60,338),(110,345),(150,362),(175,370),(200,369),(225,362),(260,353),(300,345),(340,337),(380,330),
          (420,320),(460,310),(500,300),(560,272),(614,240),(660,205),(689,180),(700,140),(707,113)]
jm_bed = [(60,305),(150,325),(200,334),(280,328),(350,316),(400,301),(450,280),(500,262),(540,240),(600,195),
          (640,160),(670,120),(700,100),(760,100),(840,100),(900,110),(950,105)]
xd_toe = [(707,113),(740,140),(770,170),(800,200),(830,230),(853,261),(900,262),(940,291),(1020,275),(1069,269),
          (1128,256),(1208,236),(1289,215),(1369,192),(1450,175)]
xd_bed = [(760,120),(820,150),(870,200),(940,250),(1020,235),(1069,229),(1128,216),(1208,196),(1289,175),
          (1369,152),(1450,135),(1450,90),(1200,90),(1000,90),(800,90)]
sandbar_hills = [(710.7,225.6),(742,199),(827,240),(669.5,259.5),(635.6,279)]
lagoon = [(742,326),(756.5,307.9),(773.5,292.2),(790.5,287.3),(814.7,283.7),(839,280),(863.2,278.9),(887.4,278.4),
          (911.6,281.3),(921.3,291),(922.5,300.7),(911.6,308),(902,312.8),(894.7,322.4),(887.4,332.1),(882.6,341.8),
          (868,349.1),(839,351.5),(814.7,350.3),(790.5,344.2),(766.3,337),(742,332)]
lagoon_island = [(771,311.5),(790,306),(830,307),(840,313),(830,322),(790,323),(772,318)]
channel_A = [(136,496),(139,470),(142,440),(145,410),(147,395),(152,384),(162,374)]
fields = [
 dict(name='籃球場', kind='basket', poly=[(154.4,420.6),(205.8,420.6),(205.8,452),(154.4,452)]),
 dict(name='籃球場', kind='basket', poly=[(211,413),(262,413),(262,452),(211,452)]),
 dict(name='網球場', kind='tennis', poly=[(279,382.8),(334.6,382.8),(334.6,412.8),(279,412.8)]),
 dict(name='網球場', kind='tennis', poly=[(279,412.8),(334.6,412.8),(334.6,443.6),(279,443.6)]),
 dict(name='既有棒球場', kind='baseball', poly=[(480,347),(707,368),(700,484),(472,463)]),
 dict(name='球場', kind='court', poly=[(1017.9,390.1),(1051.4,377.8),(1075.6,440.3),(1043.4,453.7)]),
 dict(name='球場', kind='court', poly=[(968.2,455.1),(997.7,443),(1007.1,467.2),(977.6,479.3)]),
 dict(name='球場', kind='court', poly=[(1001.2,441.7),(1029.9,429.6),(1039.9,452.4),(1010.3,464.5)]),
 dict(name='運動場', kind='field', poly=[(1093.6,355.8),(1199.9,355.8),(1200.9,381.3),(1093.6,388.5)]),
 dict(name='花式溜冰場', kind='skate', poly=[(1208.5,333),(1274.2,295.4),(1297,335.6),(1231.3,370.5)]),
]
lights = {'#1':(170.9,417.9),'#2':(189.3,418.4),'#3':(170.4,454.2),'#4':(190.3,454),'#5':(224.5,410.4),
 '#6':(248.6,410.4),'#8':(224.2,451.5),'#10':(249.6,451.5),'#11':(297.8,380.8),'#12':(319,380.8),'#13':(298.5,412.8),
 '#14':(319.6,412.8),'#15':(297.3,447.4),'#16':(297.3,450),'#17':(318.6,447.9),'#18':(476,403),'#19':(472,447.2),
 '#20':(499.5,473.8),'#21':(546,482),'#22':(1020.5,398.7),'#23':(1044.7,455),'#24':(1059.5,389.3),'#25':(1068.9,412.1),
 '#26':(1082.3,441.7),'#27':(1121.5,353.1),'#28':(1176.2,353.1),'#29':(1125.2,390.7),'#30':(1177.6,390.7),
 '#31':(1110.5,432.3),'#32':(1144,419.7),'☆1(遷移)':(1207.9,331.6),'☆2(遷移)':(1274.2,295.4)}
# groyne spec table (L-06-1): name, L, H, theta
spec_jm = [('景1',5,20),('景2',6,18),('景3',6.5,15),('景4',6.5,15),('景5',7.5,15),('景6',8,15),('景7',7,15),('景8',6,15)]
spec_xd = [('新1',6,15),('新2',7,15),('新3',8,15),('新4',7,15),('新5',6,15)]
xd_roots = [(1070,274),(1124.5,260),(1170,248.5),(1214,236.5),(1259,224)]

# ---------------- terrain control points ----------------
ctrl = []
for name, px, z, e, n in survey:
    if name.startswith('db'): continue          # groyne body -> modelled as solids
    ctrl.append((px, z))
def dens(line, z, step=6):
    out = []
    for p, q in zip(line[:-1], line[1:]):
        L = math.dist(p, q); k = max(1, int(L / step))
        for i in range(k):
            t = i / k; out.append(((p[0]+(q[0]-p[0])*t, p[1]+(q[1]-p[1])*t), z))
    out.append((line[-1], z)); return out
ctrl += dens(jm_toe, 1.0) + dens(jm_bed, 0.3) + dens(xd_toe, 1.0) + dens(xd_bed, 0.3)
ctrl += dens(jm_far, 2.0)
ctrl += dens([(x, y - 25) for x, y in jm_far[:10]], 5.0)
ctrl += [(p, 4.0) for p in sandbar_hills]
ctrl += dens(lagoon, 2.6) + [(p, 1.6) for p in lagoon_island] and dens(lagoon, 2.6)
lc = np.mean(np.array(lagoon_island), axis=0)
ctrl += [((x*0.6+lc[0]*0.4, y*0.6+lc[1]*0.4), 3.6) for x, y in lagoon_island]
# lagoon bed: ring between outer polygon and island
lag = np.array(lagoon); li = np.array(lagoon_island)
for i in range(len(lag)):
    p = lag[i]; j = np.argmin(((li - p)**2).sum(1)); q = li[j]
    ctrl.append((tuple(p*0.55 + q*0.45), 1.6))
wall_in = []  # bike path along levee, ~7.6 m (L-12 sections)
for (x, y) in wall: wall_in.append(((x, y - 4), 7.6))
ctrl += wall_in

P = np.array([px2uv(p) for p, z in ctrl]); Z = np.array([z for p, z in ctrl])
STEP = 1.5
nu, nv = int(U1 / STEP) + 1, int(V1 / STEP) + 1
gu, gv = np.meshgrid(np.arange(nu) * STEP, np.arange(nv) * STEP)
lin = LinearNDInterpolator(P, Z); near = NearestNDInterpolator(P, Z)
H = lin(gu, gv); m = np.isnan(H); H[m] = near(gu[m], gv[m])
H = gaussian_filter(H, 1.2)
# behind the levee: city ground
wl = np.array([px2uv(p) for p in wall])
wall_v = np.interp(gu[0], wl[:, 0], wl[:, 1])
behind = gv >= (wall_v[None, :] + WALL_W)
H[behind] = 9.0
H[(gv >= wall_v[None, :]) & ~behind] = 9.0
print('grid', nu, nv, 'z range', H.min(), H.max())

# ---------------- groynes ----------------
db = [(px, z, e, n) for name, px, z, e, n in survey if name.startswith('db')]
db.sort(key=lambda t: t[3])
groups = [[db[0]]]
for t in db[1:]:
    if t[3] - groups[-1][-1][3] > 9: groups.append([t])
    else: groups[-1].append(t)
groynes = []
for (nm, L, th), g in zip(spec_jm, groups):
    pts = np.array([px2uv(t[0]) for t in g])
    root = pts[np.argmax(pts[:, 1])]; tip = pts[np.argmin(pts[:, 1])]
    groynes.append(dict(name=nm, root=root.tolist(), tip=tip.tolist(), L=L, theta=th, source='竣工測量點'))
xt = np.array([px2uv(p) for p in xd_toe])
for (nm, L, th), r in zip(spec_xd, xd_roots):
    ru = np.array(px2uv(r))
    i = int(np.argmin(((xt - ru)**2).sum(1))); i = min(max(i, 1), len(xt) - 2)
    tang = xt[i+1] - xt[i-1]; tang /= np.linalg.norm(tang)    # downstream->upstream? xd: x increases upstream
    nrm = np.array([tang[1], -tang[0]])
    if nrm[1] > 0: nrm = -nrm                                    # towards river (smaller v)
    ang = math.radians(th); ax = nrm * math.cos(ang) + tang * math.sin(ang)
    tip = ru + ax * (L + 2.0)
    root = ru - ax * 2.0
    groynes.append(dict(name=nm, root=root.tolist(), tip=tip.tolist(), L=L, theta=th, source='設計圖'))

# ---------------- texture ----------------
F = 3.0
TW, TH = int((X1 - X0) * F), int((Y1 - Y0) * F)
def tpx(p): return ((p[0] - X0) * F, (p[1] - Y0) * F)
Hup = np.array(Image.fromarray(H.astype(np.float32)).resize((TW, TH), Image.BILINEAR))
base = np.zeros((TH, TW, 3), np.float32)
land = np.array([150, 185, 112], np.float32); sand = np.array([196, 184, 150], np.float32)
city = np.array([200, 200, 196], np.float32)
t = np.clip((Hup - 1.0) / 2.5, 0, 1)[..., None]
base[:] = sand * (1 - t) + land * t
tv = (np.arange(TH)[:, None] + 0.5) / F * MPP
tu = (np.arange(TW)[None, :] + 0.5) / F * MPP
wv = np.interp(tu[0], wl[:, 0], wl[:, 1])[None, :]
base[np.broadcast_to(tv >= wv, (TH, TW))] = city
img = Image.fromarray(base.astype(np.uint8))
dr = ImageDraw.Draw(img)
colors = dict(basket=(190,110,80), tennis=(70,130,120), baseball=(176,150,100), court=(110,140,170),
              field=(120,170,95), skate=(165,165,175))
for f in fields:
    dr.polygon([tpx(p) for p in f['poly']], fill=colors[f['kind']])
dr.polygon([tpx(p) for p in lagoon], fill=(95,140,170))
dr.polygon([tpx(p) for p in lagoon_island], fill=(140,175,105))
dr.line([tpx(p) for p in channel_A], fill=(95,140,170), width=int(3.0 / MPP * F))
# as-built paths from survey (RD / 2RD / 3RD point strings)
import re
seqs = {}
for name, px, z, e, n in survey:
    mm = re.match(r'^(\d*RD)(\d+)$', name)
    if mm: seqs.setdefault(mm.group(1), []).append((int(mm.group(2)), px))
pw = max(2, int(2.2 / MPP * F))
for k, lst in seqs.items():
    lst.sort()
    for (i1, p1), (i2, p2) in zip(lst[:-1], lst[1:]):
        if i2 == i1 + 1 and math.dist(p1, p2) * MPP < 7:
            dr.line([tpx(p1), tpx(p2)], fill=(222, 205, 165), width=pw)
# plan linework (multiply), with the big zone letters removed
src = Image.open(TIF27).convert('L')
cr = src.crop((int(X0*K), int(Y0*K), int(X1*K), int(Y1*K))).resize((TW, TH), Image.LANCZOS)
thick = cr.filter(ImageFilter.MaxFilter(11))           # only very thick strokes stay dark
mask = thick.point(lambda v: 255 if v < 128 else 0).filter(ImageFilter.MaxFilter(15))
cr = Image.composite(Image.new('L', cr.size, 255), cr, mask)
dk = 1.0 - np.asarray(cr, np.float32) / 255.0
# privacy: drop base-map linework outside the project (houses behind the levee, parcels across 景美溪)
priv = Image.new('L', (TW, TH), 0); pd = ImageDraw.Draw(priv)
pd.polygon([tpx(p) for p in [(X0, Y0)] + jm_far + [(619, Y0)]], fill=255)
offs = [(x, y + 2) for x, y in wall]
pd.polygon([tpx(p) for p in offs + [(X1, offs[-1][1]), (X1, Y1), (X0, Y1)]], fill=255)
pm = np.asarray(priv) > 0
dk[pm] = 0.0
img_arr = np.asarray(img).copy(); img_arr[pm & ~np.broadcast_to(tv >= wv, (TH, TW))] = land.astype(np.uint8)
img = Image.fromarray(img_arr)
out = np.asarray(img, np.float32) * (1.0 - 0.62 * dk[..., None])
tex = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
tex.save(OUTDIR + '/texture.jpg', quality=82)
tex.resize((TW // 3, TH // 3)).save(OUTDIR + '/texture_small.png')

# ---------------- objects ----------------
def uvl(lst): return [list(px2uv(p)) for p in lst]
def zat(u, v):
    i = min(max(int(round(v / STEP)), 0), nv - 1); j = min(max(int(round(u / STEP)), 0), nu - 1)
    return float(H[i, j])
light_objs = [dict(name=k, uv=list(px2uv(p)), z=zat(*px2uv(p))) for k, p in lights.items()]
field_objs = [dict(name=f['name'], kind=f['kind'], poly=uvl(f['poly'])) for f in fields]
labels = [dict(t=t_, uv=list(px2uv(p))) for t_, p in [('景美溪',(300,325)),('新店溪',(1200,150)),('A',(250,470)),('B',(600,230)),
          ('C',(590,420)),('D',(900,220)),('E',(900,440)),('F',(1300,250)),('G',(1350,420)),('水源快速道路',(700,515))]]
H16 = np.round(H * 100).astype('<i2')
model = dict(
    meta=dict(title='新店溪及景美溪匯流口護岸及景觀改善工程', step=STEP, nu=nu, nv=nv, U=U1, V=V1, mpp=MPP,
              twd97=dict(a=a/ S * MPP / K * K, note='see en()'), X0=X0, Y0=Y0, K=K, T=[a, b, c, d]),
    heights=base64.b64encode(H16.tobytes()).decode(),
    wall=uvl(wall), wallW=WALL_W, lagoon=uvl(lagoon), lagoonIsland=uvl(lagoon_island),
    groynes=groynes, lights=light_objs, fields=field_objs, labels=labels,
    survey=[[round(px2uv(px)[0], 2), round(px2uv(px)[1], 2), round(z, 3), re.sub(r'\d+$', '', name)]
            for name, px, z, e, n in survey if X0 <= px[0] <= X1 and Y0 <= px[1] <= Y1])
json.dump(model, open(OUTDIR + '/model.json', 'w', encoding='utf-8'), ensure_ascii=False)
print('texture', TW, TH, 'groynes', len(groynes), 'lights', len(light_objs))
for g in groynes: print(g['name'], np.round(g['root'],1), np.round(g['tip'],1), round(math.dist(g['root'], g['tip']),1))
