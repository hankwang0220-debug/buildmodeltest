import sys, json, numpy as np
from PIL import Image, ImageDraw
sys.stdout.reconfigure(encoding='utf-8')
O = sys.argv[1]
d = np.load(O + '/diff.npy')
r = np.clip(d / 2.5, -1, 1)
img = np.zeros(d.shape + (3,), np.uint8) + 235
img[..., 0] = np.where(r > 0, 235, 235 + r * 200).astype(np.uint8)
img[..., 2] = np.where(r < 0, 235, 235 - r * 200).astype(np.uint8)
img[..., 1] = (235 - np.abs(r) * 200).astype(np.uint8)
im = Image.fromarray(img).resize((d.shape[1] * 2, d.shape[0] * 2), Image.NEAREST)
M = json.load(open(O + '/model2.json', encoding='utf-8'))
dr = ImageDraw.Draw(im); st = M['meta']['step']
for u, v, val, k in M['designPts']:
    x, y = u / st * 2, v / st * 2
    dr.ellipse([x-3, y-3, x+3, y+3], outline=(0, 0, 0)); dr.text((x+4, y-6), f'{k}{val}', fill=(0, 0, 0))
im.save(O + '/diff.png')
print(sorted(set(l['name'][0] for l in M['lights'])), len(M['lights']))
