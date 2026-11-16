"""Prep GRaMC gabbro IPF images for build_new_source.py.

 * finds the translucent grey scale-bar box (bottom-left, ~250x140 px at native size) and paints a
   generous rectangle over it white, so it is not segmented as a grain
 * downsamples to <= 4,000,000 px with the same formula segment_grains.py uses (so the segmenter
   does not resize again and the exclusion box stays in the right coordinates)
 * writes <dst>.exclude.json with the blanked box; build_new_source.py drops any grain touching it
   (a grain partly hidden by the box would get a wrong aspect ratio)

usage: python3 -I clean_gabbro.py <src_png> <dst_png>
"""
import json, sys
import numpy as np
from PIL import Image
from skimage import measure

Image.MAX_IMAGE_PIXELS = None
MAX_PIXELS = 4_000_000
BOX_H, BOX_C0, BOX_C1, MARGIN = 140, 20, 305, 8

src, dst = sys.argv[1], sys.argv[2]
img = Image.open(src).convert("RGB")
a = np.array(img)
H, W = a.shape[:2]
r, g, b = [a[..., i].astype(int) for i in range(3)]
grey = (abs(r - g) < 6) & (abs(g - b) < 6) & (r > 80) & (r < 135)
reg = np.zeros_like(grey); reg[int(H * 0.8):, : int(W * 0.15)] = True
lab = measure.label(grey & reg, connectivity=2)
if lab.max() == 0:
    print(src, "NO scale bar box found"); sys.exit(1)
r0 = min(p.bbox[0] for p in measure.regionprops(lab) if p.area > 500)
box = [max(0, r0 - MARGIN), BOX_C0, min(H, r0 + BOX_H + MARGIN), BOX_C1]

scale = 1.0
if H * W > MAX_PIXELS:
    scale = (MAX_PIXELS / (H * W)) ** 0.5
    img = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
a = np.array(img)
sbox = [int(box[0] * scale), int(box[1] * scale), int(np.ceil(box[2] * scale)), int(np.ceil(box[3] * scale))]
a[sbox[0]:sbox[2], sbox[1]:sbox[3]] = 255
Image.fromarray(a).save(dst)
json.dump({"boxes": [sbox], "note": "r0,c0,r1,c1 in the saved image's pixels"}, open(dst + ".exclude.json", "w"))
print(f"{src.split('/')[-1]}: {W}x{H} -> {a.shape[1]}x{a.shape[0]}  blanked rows {sbox[0]}-{sbox[2]} cols {sbox[1]}-{sbox[3]}")
