import csv, sys, json
import numpy as np
rows = list(csv.reader(open(sys.argv[1], encoding='utf-8-sig', errors='replace')))[1:]
db = np.array([(float(r[3]), float(r[2])) for r in rows if r[1].startswith('db')])
# cluster by N gap
order = np.argsort(db[:,1]); db = db[order]
groups = [[db[0]]]
for p in db[1:]:
    if p[1] - groups[-1][-1][1] > 9: groups.append([p])
    else: groups[-1].append(p)
cents = [np.mean(g, axis=0) for g in groups]
for g, c in zip(groups, cents):
    g = np.array(g)
    print(len(g), 'cent E,N', c.round(2), 'extentE', g[:,0].min().round(1), g[:,0].max().round(1), 'extentN', np.ptp(g[:,1]).round(1))
# plan centroids in g1 crop coords -> full-res px
k = 8.2775
plan = [(120,302),(348,312),(592,290),(811,236),(1101,197),(1362,144),(1612,77),(1880,41)]
P = np.array([((150 + x*0.175)*k, (310 + y*0.175)*k) for x, y in plan])
# survey groups ordered by N ascending -> plan left->right? check orientation: plan x increases with N
S = np.array(cents)
# similarity fit: [E,N] = s*R*[x,y] + t  (allow reflection)
def fit(P, S):
    best=None
    for refl in (1,-1):
        Q = P.copy(); Q[:,1]*=refl
        mp, ms = Q.mean(0), S.mean(0)
        A, B = Q-mp, S-ms
        U, sig, Vt = np.linalg.svd(A.T@B)
        R = (U@Vt).T
        if np.linalg.det(R) < 0: continue
        s = sig.sum()/ (A**2).sum()
        res = S - (s*(R@A.T).T + ms)
        rms = np.sqrt((res**2).sum(1).mean())
        if best is None or rms < best[0]: best=(rms, refl, s, R, mp, ms, res)
    return best
for order_name, SS in (('asc', S), ('desc', S[::-1])):
    rms, refl, s, R, mp, ms, res = fit(P, SS)
    print(order_name, 'rms', rms, 'refl', refl, 'm/px', s, 'rot deg', np.degrees(np.arctan2(R[1,0],R[0,0])))
    print(res.round(2))
