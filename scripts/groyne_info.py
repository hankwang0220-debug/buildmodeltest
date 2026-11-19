# Per-groyne metrics vs. the design rules of thumb in 水利署電子報〈淺談丁壩設計〉(110.10.29)
import sys, json, math
import numpy as np
SCR, OUT = sys.argv[1:3]
sys.stdout.reconfigure(encoding='utf-8')
M = json.load(open(OUT + '/model2.json', encoding='utf-8'))
exec(open(SCR + '/features.py', encoding='utf-8').read())
MPP = M['meta']['mpp']; X0, Y0 = M['meta']['X0'], M['meta']['Y0']
uv = lambda p: np.array(((p[0] - X0) * MPP, (p[1] - Y0) * MPP))
def poly(line): return [uv(p) for p in line]
def seg_near(p, pts):
    best = (1e9, None, None)
    for a_, b_ in zip(pts[:-1], pts[1:]):
        ab = b_ - a_; t = np.clip(np.dot(p - a_, ab) / np.dot(ab, ab), 0, 1); q = a_ + ab * t; d = np.linalg.norm(p - q)
        if d < best[0]: best = (d, q, ab / np.linalg.norm(ab))
    return best
flows = [np.array(M['flow']['jm']), np.array(M['flow']['xd'])]
far = {'景': poly(jm_far), '新': None}
G = M['groynes']
# --- re-derive 景 groyne axes from the surveyed outline points (PCA), tip = end toward the river (smaller v)
db = np.array([[c[0], c[1]] for c in M['cloud'] if c[4] == 'db'])
db = db[np.argsort(db[:, 0])]
cl = [[db[0]]]
for p in db[1:]:
    (cl[-1].append(p) if p[0] - cl[-1][-1][0] < 9 else cl.append([p]))
cl = [np.array(c) for c in cl if len(c) >= 4]
print('db clusters', len(cl))
jm = [g for g in G if g['name'].startswith('景')]
# survey outlines include the riprap toe and are stretched along the flow, so the survey gives the position (centroid)
# and the drawing gives the direction: θ from the L-06-1 table, tip leaning upstream as drawn (向上型)
for g, c in zip(sorted(jm, key=lambda g: g['root'][0]), cl):
    m = c.mean(0)
    fd = min((seg_near(m, list(f)) for f in flows), key=lambda x: x[0])[2]
    nrm = np.array([fd[1], -fd[0]])
    if nrm[1] > 0: nrm = -nrm
    th = math.radians(g['theta']); ax = nrm * math.cos(th) - fd * math.sin(th)
    half = g['L'] / 2 + 1.5
    g['root'] = (m - ax * half).round(2).tolist(); g['tip'] = (m + ax * half).round(2).tolist()
    g['source'] = '位置：竣工測量點；方向：設計圖（向上型）'
# --- 新 groynes (no survey): design θ from the L-06-1 table, leaning upstream as drawn (向上型)
for g in G:
    if not g['name'].startswith('新'): continue
    r = np.array(g['root']); t = np.array(g['tip']); Ln = np.linalg.norm(t - r)
    fd = min((seg_near(r, list(f)) for f in flows), key=lambda x: x[0])[2]
    nrm = np.array([fd[1], -fd[0]])
    if nrm[1] > 0: nrm = -nrm
    th = math.radians(g['theta']); ax = nrm * math.cos(th) - fd * math.sin(th)
    g['tip'] = (r + ax * Ln).round(2).tolist(); g['source'] = '設計圖（L-06-1 θ，向上型）'
info = []
for i, g in enumerate(G):
    r, t = np.array(g['root']), np.array(g['tip']); ax = (t - r) / np.linalg.norm(t - r)
    fd = min((seg_near(r, list(f)) for f in flows), key=lambda x: x[0])[2]
    along = float(np.dot(ax, fd))                      # >0: tip points downstream
    ang = math.degrees(math.acos(min(1, abs(float(np.dot(ax, np.array([-fd[1], fd[0]])))))))
    kind = '直角型' if ang < 8 else ('向下型' if along > 0 else '向上型')
    # river width: from the groyne root straight across to the far bank (景美溪 only; 新店溪 far bank is outside the model)
    width = None
    if g['name'].startswith('景'):
        for k in range(1, 400):
            p = r + ax * k * 0.5
            d, _, _ = seg_near(p, far['景'])
            if d < 0.6: width = k * 0.5; break
    same = [h for h in G if h['name'][0] == g['name'][0]]
    j = same.index(g)
    nxt = same[j + 1] if j + 1 < len(same) else None
    spacing = float(np.linalg.norm(np.array(nxt['root']) - r)) if nxt else None
    info.append(dict(name=g['name'], L=g['L'], theta=g['theta'], type=kind, angle=round(ang, 1),
                     width=round(width, 1) if width else None, LW=round(g['L'] / width * 100, 1) if width else None,
                     spacing=round(spacing, 1) if spacing else None, SL=round(spacing / g['L'], 1) if spacing else None,
                     h=2.0))   # crest EL2.5 – bed ≈ EL0.5 (D-03-2)
    print(info[-1])
M['groyneInfo'] = info
json.dump(M, open(OUT + '/model2.json', 'w', encoding='utf-8'), ensure_ascii=False)
