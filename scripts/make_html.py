import sys, base64
tpl, model, tex, tex2, out, standalone = sys.argv[1:7]
ovl = sys.argv[7] if len(sys.argv) > 7 else None
h = open(tpl, encoding='utf-8').read()
m = open(model, encoding='utf-8').read()
enc = lambda p, mt='image/jpeg': f'data:{mt};base64,' + base64.b64encode(open(p, 'rb').read()).decode()
h = h.replace('__MODEL__', m).replace('__TEX3__', enc(ovl, 'image/png') if ovl else '').replace('__TEX2__', enc(tex2)).replace('__TEX__', enc(tex))
open(out, 'w', encoding='utf-8').write(h)
open(standalone, 'w', encoding='utf-8').write('<!doctype html>\n<html lang="zh-Hant"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"></head><body style="margin:0">\n' + h + '\n</body></html>')
print(len(h))
