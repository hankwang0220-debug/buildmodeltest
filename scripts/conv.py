import sys, os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
src, dst, w = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.makedirs(dst, exist_ok=True)
for f in sorted(os.listdir(src)):
    if not f.lower().endswith('.tif'): continue
    im = Image.open(os.path.join(src, f))
    if f.endswith('001.tif'): print(im.size, im.mode, im.info.get('dpi'))
    im = im.convert('L')
    r = w / im.width
    im = im.resize((w, int(im.height * r)), Image.LANCZOS)
    im.save(os.path.join(dst, f[:-4] + '.png'), optimize=True)
print('done')
