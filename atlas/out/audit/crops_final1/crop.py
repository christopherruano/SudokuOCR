import sys
from PIL import Image
# usage: crop.py img out x0 y0 x1 y1 [scale] (fractions or pixels)
p,o=sys.argv[1],sys.argv[2]; b=[float(v) for v in sys.argv[3:7]]
s=float(sys.argv[7]) if len(sys.argv)>7 else 1
im=Image.open(p).convert('L'); W,H=im.size
if all(v<=1 for v in b): b=[b[0]*W,b[1]*H,b[2]*W,b[3]*H]
c=im.crop(tuple(int(v) for v in b))
if s!=1: c=c.resize((int(c.width*s),int(c.height*s)),Image.LANCZOS)
c.save(o)
