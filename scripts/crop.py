import sys
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
# usage: crop.py tif out x0 y0 x1 y1 (in 1600-wide preview coords) outwidth
tif, out = sys.argv[1], sys.argv[2]
x0, y0, x1, y1, ow = map(float, sys.argv[3:8])
im = Image.open(tif)
k = im.width / 1600.0
c = im.crop((int(x0*k), int(y0*k), int(x1*k), int(y1*k))).convert('L')
r = ow / c.width
c = c.resize((int(ow), int(c.height*r)), Image.LANCZOS)
c.save(out)
print(c.size, 'full-res px per preview px', k)
