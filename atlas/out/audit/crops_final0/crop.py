import sys
from PIL import Image
src,out,x0,y0,x1,y1=sys.argv[1:7]
im=Image.open(src)
W,H=im.size
box=tuple(int(float(v)*(W if i%2==0 else H)) if float(v)<=1 else int(v) for i,v in enumerate([x0,y0,x1,y1]))
c=im.crop(box)
s=float(sys.argv[7]) if len(sys.argv)>7 else 1
if s!=1: c=c.resize((int(c.width*s),int(c.height*s)),Image.LANCZOS)
c.save('/Users/chrisruano/Research/SudokuOCR/atlas/out/audit/crops_final0/'+out)
print(box,c.size)
