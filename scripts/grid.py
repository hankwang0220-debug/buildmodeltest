import sys
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
# grid.py tif out x0 y0 x1 y1 outw step   (preview coords)
tif, out = sys.argv[1], sys.argv[2]
x0, y0, x1, y1, ow, st = map(float, sys.argv[3:9])
im = Image.open(tif)
k = im.width / 1600.0
c = im.crop((int(x0*k), int(y0*k), int(x1*k), int(y1*k))).convert('RGB')
r = ow / c.width
c = c.resize((int(ow), int(c.height*r)), Image.LANCZOS)
d = ImageDraw.Draw(c)
f = ow / (x1 - x0)
import math
gx = math.ceil(x0/st)*st
while gx < x1:
    X = (gx-x0)*f; d.line([(X,0),(X,c.height)], fill=(255,0,0) if gx%(st*5)==0 else (255,170,170), width=1)
    if gx % (st*5) == 0: d.text((X+2, 2), str(int(gx)), fill=(255,0,0))
    gx += st
gy = math.ceil(y0/st)*st
while gy < y1:
    Y = (gy-y0)*f; d.line([(0,Y),(c.width,Y)], fill=(0,0,255) if gy%(st*5)==0 else (170,170,255), width=1)
    if gy % (st*5) == 0: d.text((2, Y+2), str(int(gy)), fill=(0,0,255))
    gy += st
c.save(out)
print(c.size)
