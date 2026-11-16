"""Remove non-grain objects the segmenter picks up from figure furniture, from a v2 dataset.

Rules (applied only to sources named on the command line, default = all rows):
  * thin / hollow bounding box  : area_px / (crop_w*crop_h) < 0.15      (image-frame line fragments)
  * grey frame/scale-bar pieces : colour saturation < 12 and mean R < 150 and fill < 0.30
  * giant region                : area_px > 200,000 px (~5% of a 4 MP map) -- almost certainly several
                                  grains merged where boundary lines are missing, so its AR is meaningless
Removed rows are written to <dataset>/removed_artifacts.csv with the reason, and their crops deleted.

usage: python3 -I filter_artifacts.py <dataset_dir> [source_image ...]
"""
import csv, sys
from pathlib import Path

ds = Path(sys.argv[1]); only = set(sys.argv[2:])
rows = list(csv.DictReader((ds / "all_grains.csv").open()))
fields = list(rows[0].keys())
for r in rows:
    for k in r:
        if k not in fields: fields.append(k)


def reason(r):
    try:
        w, h, a = int(r["crop_w_px"]), int(r["crop_h_px"]), float(r["area_px"])
        fill = a / (w * h)
        c = [float(r["mean_color_r"]), float(r["mean_color_g"]), float(r["mean_color_b"])]
    except (KeyError, ValueError, ZeroDivisionError):
        return None
    if fill < 0.15: return "thin_or_hollow_bbox"
    if max(c) - min(c) < 12 and c[0] < 150 and fill < 0.30: return "grey_frame_or_scalebar_piece"
    if a > 200_000: return "giant_merged_region"
    return None


keep, removed = [], []
for r in rows:
    why = reason(r) if (not only or r["source_image"] in only) else None
    (removed if why else keep).append((r, why))

for r, why in removed:
    pref = r["source_image"].rsplit("_IPF", 1)[0]
    f = ds / "grain_crops" / r["morphology_label"] / f"{pref}_grain_{int(float(r['grain_id'])):04d}_ar{float(r['aspect_ratio']):.2f}.png"
    if f.exists(): f.unlink()

with (ds / "all_grains.csv").open("w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=fields, restval="", extrasaction="ignore")
    wr.writeheader(); wr.writerows(r for r, _ in keep)
with (ds / "removed_artifacts.csv").open("w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=["reason"] + fields, restval="", extrasaction="ignore")
    wr.writeheader(); wr.writerows({"reason": why, **r} for r, why in removed)
print(f"kept {len(keep)}, removed {len(removed)}")
for r, why in removed:
    print(f"  {r['source_image'][:14]:14s} grain {r['grain_id']:>4s} {r['crop_w_px']}x{r['crop_h_px']} area {r['area_px']} -> {why}")
