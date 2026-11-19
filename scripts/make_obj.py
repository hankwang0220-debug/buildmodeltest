# Export OBJ/MTL (Z-up, local TWD97 offset) from model.json
import json, sys, base64, math, os
import numpy as np
model, outdir = sys.argv[1], sys.argv[2]
M = json.load(open(model, encoding='utf-8')); meta = M['meta']
a, b, c, d = meta['T']; K = meta['K']; MPP = meta['mpp']; X0, Y0 = meta['X0'], meta['Y0']
OE, ON = 303800.0, 2765600.0
def en(u, v):
    fx = (X0 + u / MPP) * K; fy = (Y0 + v / MPP) * K
    return a*fx + b*fy + c - OE, b*fx - a*fy + d - ON
nu, nv, st = meta['nu'], meta['nv'], meta['step']
HEX = np.frombuffer(base64.b64decode(M['heights']), '<i2').reshape(nv, nu) / 100.0
HDE = np.frombuffer(base64.b64decode(M.get('heightsDesign', M['heights'])), '<i2').reshape(nv, nu) / 100.0
H = HDE
L = []; vi = 1; vt = 1
def obj(name, mtl): L.append(f'o {name}\nusemtl {mtl}')
for tname, HH in (('terrain_design', HDE), ('terrain_existing', HEX)):
    obj(tname, 'plan')
    for i in range(nv):
        for j in range(nu):
            E, N = en(j*st, i*st); L.append(f'v {E:.3f} {N:.3f} {HH[i,j]:.3f}')
    if vt == 1:
        for i in range(nv):
            for j in range(nu):
                L.append(f'vt {j*st/meta["U"]:.5f} {1 - i*st/meta["V"]:.5f}')
    for i in range(nv - 1):
        for j in range(nu - 1):
            p = vi + i*nu + j; q = 1 + i*nu + j
            A, B, C, D = p, p+1, p+nu, p+nu+1; a_, b_, c_, d_ = q, q+1, q+nu, q+nu+1
            L.append(f'f {A}/{a_} {B}/{b_} {C}/{c_}'); L.append(f'f {B}/{b_} {D}/{d_} {C}/{c_}')
    vi += nu*nv; vt = 2
# water plane EL1.0
obj('water_EL1.0', 'water')
for u, v in [(0,0),(meta['U'],0),(meta['U'],meta['V']),(0,meta['V'])]:
    E, N = en(u, v); L.append(f'v {E:.3f} {N:.3f} 1.000')
L.append(f'f {vi} {vi+1} {vi+2} {vi+3}'); vi += 4
# groynes
for g in M['groynes']:
    r = np.array(g['root']); t = np.array(g['tip']); ax = t - r; D = np.linalg.norm(ax); ax /= D
    n = np.array([-ax[1], ax[0]]); bed = 0.3
    zc = lambda s: max(2.5 - s/4, 1.3)
    def sec(s):
        cc = r + ax*s; z = zc(s); hw = 0.5 + (z - bed)
        return [(cc + n*hw, bed), (cc + n*0.5, z), (cc - n*0.5, z), (cc - n*hw, bed)]
    pts = sec(0) + sec(D) + [(t + ax*(zc(D) - bed), bed)]
    obj('groyne_' + g['name'], 'stone')
    for p, z in pts:
        E, N = en(*p); L.append(f'v {E:.3f} {N:.3f} {z:.3f}')
    for f in [(0,4,1),(1,4,5),(1,5,2),(2,5,6),(2,6,3),(3,6,7),(0,1,2),(0,2,3),(4,8,5),(5,8,6),(6,8,7)]:
        L.append('f ' + ' '.join(str(vi + k) for k in f))
    vi += 9
# levee / expressway band
w = np.array(M['wall']); W = M['wallW']
obj('levee_expressway', 'concrete')
for i, p in enumerate(w):
    a0 = w[max(i-1, 0)]; b0 = w[min(i+1, len(w)-1)]; dd = b0 - a0; dd /= np.linalg.norm(dd); o = np.array([-dd[1], dd[0]])
    for q, z in [(p, 5.0), (p, 15.0), (p + o*W, 15.0), (p + o*W, 5.0)]:
        E, N = en(*q); L.append(f'v {E:.3f} {N:.3f} {z:.3f}')
for i in range(len(w) - 1):
    A = vi + i*4; B = A + 4
    for k in range(3): L.append(f'f {A+k} {B+k} {B+k+1} {A+k+1}')
vi += len(w)*4
# lights as simple poles (quad prisms 12 m)
obj('lights', 'steel')
def hz(u, v):
    i = min(max(int(round(v/st)), 0), nv-1); j = min(max(int(round(u/st)), 0), nu-1); return H[i, j]
for l in M['lights']:
    u, v = l['uv']; z0 = hz(u, v); E, N = en(u, v)
    for z in (z0, z0 + 12):
        for dx, dy in [(-.15,-.15),(.15,-.15),(.15,.15),(-.15,.15)]:
            L.append(f'v {E+dx:.3f} {N+dy:.3f} {z:.3f}')
    for k in range(4):
        k2 = (k+1) % 4; L.append(f'f {vi+k} {vi+k2} {vi+4+k2} {vi+4+k}')
    vi += 8
# trees: trunk prism + octahedral crown sized from the L-08-1 table
for t in M.get('trees', []):
    u, v = t['uv']; z0 = hz(u, v); E, N = en(u, v); h = t['h']; r = max(t['crown'] / 2, 1.2)
    th = h if t.get('palm') else h * 0.45
    obj('tree_' + t['id'], 'trunk')
    for z in (z0, z0 + th):
        for dx, dy in [(-.2,-.2),(.2,-.2),(.2,.2),(-.2,.2)]: L.append(f'v {E+dx:.3f} {N+dy:.3f} {z:.3f}')
    for k in range(4):
        k2 = (k+1) % 4; L.append(f'f {vi+k} {vi+k2} {vi+4+k2} {vi+4+k}')
    vi += 8
    obj('crown_' + t['id'], 'crown')
    cz = z0 + th + (h - th) * 0.5 if not t.get('palm') else z0 + h; rz = (h - th) * 0.6 if not t.get('palm') else r * 0.5
    for dx, dy, dz in [(r,0,0),(-r,0,0),(0,r,0),(0,-r,0),(0,0,rz),(0,0,-rz)]: L.append(f'v {E+dx:.3f} {N+dy:.3f} {cz+dz:.3f}')
    for f in [(0,2,4),(2,1,4),(1,3,4),(3,0,4),(2,0,5),(1,2,5),(3,1,5),(0,3,5)]:
        L.append('f ' + ' '.join(str(vi + k) for k in f))
    vi += 6
hdr = ['# 新店溪及景美溪匯流口護岸及景觀改善工程 3D model',
       '# objects: terrain_design (L-07), terrain_existing (L-03), water_EL1.0, groyne_*, levee_expressway, lights (L-11), tree_*/crown_* (L-08)', f'# Z-up, metres. X = TWD97 E - {OE:.0f}, Y = TWD97 N - {ON:.0f}, Z = elevation',
       'mtllib model.mtl']
open(os.path.join(outdir, 'model.obj'), 'w', encoding='utf-8').write('\n'.join(hdr + L) + '\n')
open(os.path.join(outdir, 'model.mtl'), 'w', encoding='utf-8').write(
"""newmtl plan
Kd 1 1 1
map_Kd texture.jpg
newmtl water
Kd 0.31 0.53 0.66
d 0.6
newmtl stone
Kd 0.55 0.54 0.51
newmtl concrete
Kd 0.72 0.71 0.68
newmtl steel
Kd 0.42 0.44 0.45
newmtl trunk
Kd 0.43 0.32 0.21
newmtl crown
Kd 0.31 0.49 0.23
""")
print('vertices', vi - 1)
