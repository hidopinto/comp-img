"""
we need to do the following with dcraw:
1) white balancing using the camera's profile: -w achieves this
2) demosaicing using high-quality interpolation: -q 3
3) use sRGB as the output colour space: -o 1
4) use linear 16-bit images -4
5) write Tiff files -T

for windows command prompt, run from the downloads folder of Zach's drive (has the dcraw program in it already)

for /l %i in (1,1,16) do dcraw -w -T -q 3 -o 1 -4 C:\Users\zacharyl\Downloads\nef_files\exposure%i.nef
"""