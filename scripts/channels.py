# Chain the L-07 water-surface labels (WS) into channel centre lines.
# Rules: start from the highest WS; next point = nearest one that is not higher than the current one and
# lies within MAXLINK m; a WS label sitting on top of an equal WB label is an OCR misread of that WB and is dropped.
# Usable as a module (make_chains) or as a script that patches model2.json.
import math, sys, json
MAXLINK = 45.0
def make_chains(design, verbose=False):
    ws = [(u, w, v) for u, w, v, k in design if k == 'WS']
    wb = [(u, w, v) for u, w, v, k in design if k == 'WB']
    ws = [p for p in ws if not any(abs(p[2] - q[2]) < 0.05 and math.dist(p[:2], q[:2]) < 4 for q in wb)]
    rem = sorted(ws, key=lambda p: -p[2]); chains = []
    while rem:
        cur = [rem.pop(0)]
        while True:
            # drops at a 跌水工 are ~0.2 m; a bigger drop to a very close label is usually a misread WB, so penalise it
            cand = [(math.dist(cur[-1][:2], q[:2]) + 100 * max(0.0, cur[-1][2] - q[2] - 0.3), math.dist(cur[-1][:2], q[:2]), i)
                    for i, q in enumerate(rem) if q[2] <= cur[-1][2] + 0.05]
            cand = [c for c in cand if c[1] <= MAXLINK]
            if not cand: break
            cur.append(rem.pop(min(cand)[2]))
        chains.append(cur)
    out = [[[round(u, 2), round(w, 2), round(v - 0.2, 2)] for u, w, v in ch] for ch in chains if len(ch) >= 2]
    if verbose:
        for ch in chains:
            segs = [round(math.dist(a[:2], b[:2]), 1) for a, b in zip(ch[:-1], ch[1:])]
            print(len(ch), 'WS', [p[2] for p in ch], 'u', [round(p[0]) for p in ch], 'seg', segs)
    return out
if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    path = sys.argv[1]
    M = json.load(open(path, encoding='utf-8'))
    M['channels'] = make_chains([tuple(d) for d in M['designPts']], verbose=True)
    json.dump(M, open(path, 'w', encoding='utf-8'), ensure_ascii=False)
    print('channels', len(M['channels']))
