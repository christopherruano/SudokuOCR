#!/usr/bin/env python3
"""Merge a left and right page scan into ONE side-by-side composite image so a reader sees both halves
in a single view. Left page (row labels + early columns) on the left, right page (continuation columns)
on the right, height-normalized, thin divider between. Usage: make_composite.py <left_ih> <right_ih> <out>"""
import sys
from PIL import Image, ImageDraw
def scan(ih):
    return f"census_site/img/{ih}.jpg"
def main(lih, rih, out):
    L=Image.open(scan(lih)).convert("RGB"); R=Image.open(scan(rih)).convert("RGB")
    h=max(L.height,R.height)
    def fit(im):
        if im.height!=h:
            w=int(im.width*h/im.height); im=im.resize((w,h))
        return im
    L,R=fit(L),fit(R)
    gap=8
    comp=Image.new("RGB",(L.width+gap+R.width,h),"white")
    comp.paste(L,(0,0)); comp.paste(R,(L.width+gap,0))
    d=ImageDraw.Draw(comp); d.rectangle([L.width+2,0,L.width+gap-2,h],fill=(200,40,40))
    # cap width for model input (<=2600px wide keeps text legible without huge payload)
    if comp.width>2600:
        s=2600/comp.width; comp=comp.resize((2600,int(h*s)))
    comp.save(out,"JPEG",quality=88)
    print(f"composite {comp.width}x{comp.height} -> {out} (LEFT {L.width}px | divider | RIGHT)")
if __name__=="__main__":
    main(sys.argv[1],sys.argv[2],sys.argv[3])
