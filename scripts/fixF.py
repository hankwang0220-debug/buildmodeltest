import json, sys
import numpy as np
SCR = sys.argv[1]
Z = {k: np.array(v) for k, v in json.load(open(SCR + '/zoneT.json')).items()}
V1 = {k: np.array(v) for k, v in json.load(open(SCR + '/zoneT_v1.json')).items()}
bad = Z['33']; good = V1['33']
for n in ['12', '41', '52', '61', '69', '77']:
    Z[n] = good @ np.linalg.inv(bad) @ Z[n]
Z['33'] = good
json.dump({k: v.tolist() for k, v in Z.items()}, open(SCR + '/zoneT.json', 'w'))
print('fixed F')
