"""Add one or more new IPF map images to a v2 dataset.

Runs your existing segment_grains.py pipeline unchanged (same blur, quantization, size filter,
brightness gate, quality gate, AR>=2.0 label), but writes crops in the v2 format:
  * grain isolated by its mask (neighbors black), padded to a SQUARE at native size
  * all_grains.csv rows get ambiguous / too_small / crop_w_px / crop_h_px
  * rows are appended; re-running the same image replaces its old rows (idempotent)
  * an overlay image per source is saved to <dataset>/visualizations/ -- LOOK AT IT before trusting
    the new source (the quality gate does not catch every bad segmentation)

usage:
  python3 -I build_new_source.py --repo <repo_root> --dataset <Dataset_v2> \
        [--tier unverified] <image1> [<image2> ...]
"""
import argparse, csv, importlib.util, os, sys
from pathlib import Path
import numpy as np
from PIL import Image

AMBIG_LO, AMBIG_HI = 1.7, 2.3
MIN_SIDE_PX = 8
NEW_FIELDS = ["ambiguous", "too_small", "crop_w_px", "crop_h_px", "touches_border", "group"]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def pad_square(arr):
    h, w = arr.shape[:2]
    s = max(h, w)
    out = np.zeros((s, s, 3), dtype=arr.dtype)
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = arr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tier", default="unverified")
    ap.add_argument("--group", default="", help="specimen/build group name written to the `group` column "
                    "(all images of the same specimen share one group so splits never separate them)")
    ap.add_argument("images", nargs="+")
    a = ap.parse_args()

    repo, ds = Path(a.repo), Path(a.dataset)
    seg = load(repo / "Dataset" / "segment_grains.py", "segment_grains")
    common = load(repo / "morphology_cnn" / "common.py", "common")

    csv_path = ds / "all_grains.csv"
    existing = list(csv.DictReader(csv_path.open())) if csv_path.exists() else []
    fields = list(existing[0].keys()) if existing else None
    (ds / "visualizations").mkdir(parents=True, exist_ok=True)
    for c in ("columnar", "equiaxed"):
        (ds / "grain_crops" / c).mkdir(parents=True, exist_ok=True)

    for img_path in a.images:
        src_name = os.path.basename(img_path)
        prefix = common.sanitize_source_prefix(src_name)
        # replace any prior rows/crops for this source
        existing = [r for r in existing if r["source_image"] != src_name]
        for c in ("columnar", "equiaxed"):
            for old in (ds / "grain_crops" / c).glob(f"{prefix}_grain_*.png"):
                old.unlink()

        rows = seg.process_image(img_path, str(ds / "visualizations" / prefix), save_crops_dir=None, label=prefix)
        if rows is None:
            print(f"  -> {src_name} failed the boundary quality gate and was NOT added. "
                  f"Needs a .ctf/.ang route or a different segmentation method.")
            continue

        # re-derive the (possibly downsampled) array exactly as process_image did, then write v2 crops
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)
        if arr.shape[0] * arr.shape[1] > seg.MAX_PIXELS:
            sc = (seg.MAX_PIXELS / (arr.shape[0] * arr.shape[1])) ** 0.5
            img = img.resize((int(img.width * sc), int(img.height * sc)), Image.LANCZOS)
            arr = np.array(img)
        labels = seg.segment_grains(arr, verbose=False)

        # optional <image>.exclude.json: boxes (r0,c0,r1,c1) hiding part of the map; drop grains touching them
        import json
        side = Path(str(img_path) + ".exclude.json")
        boxes = json.loads(side.read_text())["boxes"] if side.exists() else []
        if boxes:
            def hits(r):
                return any(r["bbox_minr"] < b[2] + 4 and r["bbox_maxr"] > b[0] - 4 and
                           r["bbox_minc"] < b[3] + 4 and r["bbox_maxc"] > b[1] - 4 for b in boxes)
            n0 = len(rows)
            rows = [r for r in rows if not hits(r)]
            print(f"  excluded {n0 - len(rows)} grains touching masked box(es)")
        Hh, Ww = arr.shape[:2]

        for r in rows:
            gid = r["grain_id"]
            r0, c0, r1, c1 = r["bbox_minr"], r["bbox_minc"], r["bbox_maxr"], r["bbox_maxc"]
            mask = labels[r0:r1, c0:c1] == gid
            crop = arr[r0:r1, c0:c1].copy()
            crop[~mask] = 0
            h, w = crop.shape[:2]
            fname = f"{prefix}_grain_{gid:04d}_ar{r['aspect_ratio']:.2f}.png"
            Image.fromarray(pad_square(crop)).save(ds / "grain_crops" / r["morphology_label"] / fname)
            ar = r["aspect_ratio"]
            r["confidence_tier"] = a.tier
            r["ambiguous"] = str(AMBIG_LO <= ar <= AMBIG_HI)
            r["too_small"] = str(max(h, w) < MIN_SIDE_PX)
            r["crop_w_px"], r["crop_h_px"] = w, h
            if "touches_border" not in r:      # newer segment_grains.py already provides it (0/1)
                r["touches_border"] = int(r0 == 0 or c0 == 0 or r1 >= Hh or c1 >= Ww)
            r["group"] = a.group or r.get("group", "")
        existing.extend(rows)
        n_amb = sum(r["ambiguous"] == "True" for r in rows)
        print(f"  -> added {len(rows)} grains from {src_name} (prefix {prefix}); {n_amb} in ambiguous band")

    if existing:
        fields = list(fields or existing[-1].keys())
        for r in existing:            # union of columns, so older rows keep working (blank for new columns)
            for k in r.keys():
                if k not in fields:
                    fields.append(k)
        with csv_path.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields, restval="", extrasaction="ignore")
            wr.writeheader(); wr.writerows(existing)
    print("dataset rows now:", len(existing))


if __name__ == "__main__":
    main()
