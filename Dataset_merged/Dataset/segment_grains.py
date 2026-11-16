"""
segment_grains.py

Segments individual grains from an EBSD/IPF color map image and measures
each one: size, aspect ratio, orientation. Classifies each grain as
columnar or equiaxed based on its actual measured aspect ratio (a more
rigorous criterion than labeling a whole cropped patch by eye).

METHOD
------
IPF maps encode each grain as a near-flat color, usually separated by
thin dark boundary lines. This pipeline:
  1. Median-blurs the image slightly to suppress within-grain color
     gradient/texture noise without blurring across real grain-to-grain
     color jumps.
  2. Quantizes color into coarse Lab buckets (absorbs compression noise
     while grouping perceptually similar colors).
  3. Labels connected components of constant quantized color -- each
     component is a candidate grain.
  4. Filters out components that are too small (noise/fragments,
     threshold chosen adaptively per image) or that are actually
     near-white background/border or near-black boundary-line network
     rather than real grain interior.
  5. Measures each surviving component with skimage.regionprops and
     classifies it columnar/equiaxed by aspect ratio.

QUALITY GATE
------------
This method assumes grains are reasonably large, mostly-flat-colored
regions. It works well on images with visible dark boundary lines
between grains (confirmed visually: Dr. Lu's reference image, AM Bench
montages). It does NOT work reliably on images with no boundary lines
and continuous internal color gradients (confirmed: LEROY serial-section
IPF renders) -- on those, grains fragment into many small pieces and
the aspect-ratio signal becomes unreliable.

check_boundary_quality() flags this automatically by measuring the
fraction of dark (boundary-like) pixels in the image. Images that fail
the check are EXCLUDED from the per-grain dataset rather than included
with unreliable measurements -- see the README in the output package
for exactly which images passed/failed and why.
"""

from PIL import Image
import numpy as np
from skimage import measure
from skimage.segmentation import relabel_sequential
import cv2
import csv
import os

ASPECT_RATIO_THRESHOLD = 2.0      # columnar if aspect_ratio >= this (matches team's deck, page 3)
COLOR_QUANT_STEP = 12             # bucket size for color quantization
ADAPTIVE_MIN_SIZE_PERCENTILE = 40 # components below this percentile of real-grain-scale sizes = noise
ADAPTIVE_MIN_SIZE_FLOOR = 30      # absolute floor, drops single/few-pixel speckle regardless
WHITE_BG_CUTOFF = 240             # mean RGB above this = white border/background, not a grain
BLACK_BOUNDARY_CUTOFF = 35        # mean RGB below this = boundary-line network, not a grain
MAX_ASPECT_RATIO = 25             # sanity ceiling; above this = segmentation artifact, drop it
MIN_DARK_BOUNDARY_FRACTION = 0.03 # quality gate: image needs at least this fraction of dark
                                    # (boundary-like) pixels for the method to be reliable
MAX_PIXELS = 4_000_000             # images larger than this get downsampled before
                                     # segmentation -- grain SHAPE survives downsampling
                                     # fine, but full-res connected-component labeling on
                                     # very large montages (tens of millions of px) is
                                     # too slow to be practical here
AMBIGUOUS_MIN_AR = 1.7
AMBIGUOUS_MAX_AR = 2.3


def check_boundary_quality(rgb_array):
    """Return (passes: bool, dark_fraction: float). Images with too few
    dark/boundary pixels can't be reliably segmented by this method."""
    brightness = rgb_array.mean(axis=-1)
    dark_fraction = (brightness < 60).mean()
    return dark_fraction >= MIN_DARK_BOUNDARY_FRACTION, dark_fraction


def segment_grains(rgb_array, min_grain_size=None, verbose=True, stats=None):
    """Return an integer label array where each grain has a unique ID."""
    smoothed = cv2.medianBlur(rgb_array, 5)
    lab = cv2.cvtColor(smoothed, cv2.COLOR_RGB2LAB).astype(np.int32)
    lab_step = np.array([COLOR_QUANT_STEP, COLOR_QUANT_STEP, COLOR_QUANT_STEP])
    quant = (lab // lab_step) * lab_step
    color_id = quant[:, :, 0] * 1_000_000 + quant[:, :, 1] * 1_000 + quant[:, :, 2]
    labels = measure.label(color_id, connectivity=1)

    sizes = np.bincount(labels.ravel())
    component_sizes = sizes[1:]

    if min_grain_size is None:
        candidate_sizes = component_sizes[component_sizes >= ADAPTIVE_MIN_SIZE_FLOOR]
        if len(candidate_sizes) > 0:
            min_grain_size = max(ADAPTIVE_MIN_SIZE_FLOOR,
                                  np.percentile(candidate_sizes, ADAPTIVE_MIN_SIZE_PERCENTILE))
        else:
            min_grain_size = ADAPTIVE_MIN_SIZE_FLOOR
        if verbose:
            print(f"  (adaptive min grain size: {min_grain_size:.0f} px)")

    small_mask = sizes[labels] < min_grain_size
    if stats is not None:
        stats["size_filter_pixels"] = int(np.sum(small_mask))
        stats["size_filter_components"] = int(np.sum((component_sizes > 0) & (component_sizes < min_grain_size)))
    labels[small_mask] = 0

    mean_brightness = rgb_array.mean(axis=-1)
    n_labels = labels.max()
    if n_labels > 0:
        sum_brightness = np.bincount(labels.ravel(), weights=mean_brightness.ravel(),
                                      minlength=n_labels + 1)
        count = np.bincount(labels.ravel(), minlength=n_labels + 1)
        count[count == 0] = 1
        avg_brightness_per_label = sum_brightness / count
        bg_label_ids = np.where(
            (avg_brightness_per_label > WHITE_BG_CUTOFF) |
            (avg_brightness_per_label < BLACK_BOUNDARY_CUTOFF)
        )[0]
        bg_label_ids = bg_label_ids[bg_label_ids != 0]
        bg_pixel_mask = np.isin(labels, bg_label_ids)
        if stats is not None:
            stats["brightness_gate_pixels"] = int(np.sum(bg_pixel_mask))
            stats["brightness_gate_components"] = int(len(bg_label_ids))
        labels[bg_pixel_mask] = 0

    labels, _, _ = relabel_sequential(labels)
    return labels


def characterize_grains(
    labels,
    rgb_array,
    source_name="",
    confidence_tier="",
    group="",
    stats=None,
):
    """Run regionprops on the label map and classify each grain."""
    props = measure.regionprops(labels, intensity_image=rgb_array.mean(axis=-1))
    rows = []
    for p in props:
        major = p.axis_major_length
        minor = p.axis_minor_length if p.axis_minor_length > 0 else 1e-6
        aspect_ratio = major / minor

        if aspect_ratio > MAX_ASPECT_RATIO:
            if stats is not None:
                stats["aspect_ratio_filter_components"] = stats.get("aspect_ratio_filter_components", 0) + 1
            continue

        morphology_label = "columnar" if aspect_ratio >= ASPECT_RATIO_THRESHOLD else "equiaxed"
        minr, minc, maxr, maxc = p.bbox
        mean_color = rgb_array[p.coords[:, 0], p.coords[:, 1]].mean(axis=0)
        bbox_height = maxc - minc
        bbox_width = maxr - minr
        feret_aspect_ratio = max(bbox_height, bbox_width) / max(1, min(bbox_height, bbox_width))
        touches_border = minr == 0 or minc == 0 or maxr == labels.shape[0] or maxc == labels.shape[1]
        ambiguous = AMBIGUOUS_MIN_AR <= aspect_ratio <= AMBIGUOUS_MAX_AR

        rows.append({
            "source_image": source_name,
            "group": group or source_name,
            "grain_id": p.label,
            "area_px": p.area,
            "equivalent_diameter_px": round(p.equivalent_diameter_area, 2),
            "major_axis_px": round(major, 2),
            "minor_axis_px": round(minor, 2),
            "aspect_ratio": round(aspect_ratio, 3),
            "feret_aspect_ratio": round(feret_aspect_ratio, 3),
            "touches_border": int(touches_border),
            "ambiguous_weight": 0.5 if ambiguous else 1.0,
            "confidence_tier": confidence_tier,
            "orientation_deg": round(np.degrees(p.orientation), 2),
            "centroid_row": round(p.centroid[0], 1),
            "centroid_col": round(p.centroid[1], 1),
            "bbox_minr": minr, "bbox_minc": minc, "bbox_maxr": maxr, "bbox_maxc": maxc,
            "mean_color_r": round(mean_color[0], 1),
            "mean_color_g": round(mean_color[1], 1),
            "mean_color_b": round(mean_color[2], 1),
            "morphology_label": morphology_label,
        })
    return rows


def save_csv(rows, out_path, mode="w"):
    fieldnames = ["source_image", "grain_id", "area_px", "equivalent_diameter_px",
                  "major_axis_px", "minor_axis_px", "aspect_ratio", "orientation_deg",
                  "centroid_row", "centroid_col", "bbox_minr", "bbox_minc",
                  "bbox_maxr", "bbox_maxc", "mean_color_r", "mean_color_g",
                  "mean_color_b", "morphology_label", "confidence_tier", "group",
                  "feret_aspect_ratio", "touches_border", "ambiguous_weight"]
    write_header = mode == "w" or not os.path.exists(out_path)
    with open(out_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def save_visualization(labels, rows, out_path):
    label_to_class = {r["grain_id"]: r["morphology_label"] for r in rows}
    vis = np.zeros((*labels.shape, 3), dtype=np.uint8)
    columnar_color = np.array([220, 60, 60])
    equiaxed_color = np.array([60, 120, 220])
    for grain_id, cls in label_to_class.items():
        mask = labels == grain_id
        vis[mask] = columnar_color if cls == "columnar" else equiaxed_color
    Image.fromarray(vis).save(out_path, quality=90)


def save_grain_crops(labels, rows, rgb_array, out_dir, prefix=""):
    os.makedirs(os.path.join(out_dir, "columnar"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "equiaxed"), exist_ok=True)
    for r in rows:
        gid = r["grain_id"]
        minr, minc, maxr, maxc = r["bbox_minr"], r["bbox_minc"], r["bbox_maxr"], r["bbox_maxc"]
        mask = labels[minr:maxr, minc:maxc] == gid
        crop = rgb_array[minr:maxr, minc:maxc].copy()
        crop[~mask] = 0
        fname = f"{prefix}grain_{gid:04d}_ar{r['aspect_ratio']:.2f}.png"
        Image.fromarray(crop).save(os.path.join(out_dir, r["morphology_label"], fname))


def process_image(
    img_path,
    out_prefix,
    save_crops_dir=None,
    label=None,
    confidence_tier="",
    group="",
):
    """Full pipeline for one image: quality-gate, segment, characterize,
    save CSV + visualization (+ optional per-grain crops). Returns the
    rows list, or None if the image failed the quality gate.

    out_prefix: path prefix for the visualization output file (can include
        directories, e.g. 'grain_dataset/visualizations/DrLu').
    label: short identifier used as a filename prefix for per-grain crops
        (no directories). Defaults to the basename of out_prefix."""
    if label is None:
        label = os.path.basename(out_prefix)

    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    source_name = os.path.basename(img_path)

    n_pixels = arr.shape[0] * arr.shape[1]
    if n_pixels > MAX_PIXELS:
        scale = (MAX_PIXELS / n_pixels) ** 0.5
        new_size = (int(img.width * scale), int(img.height * scale))
        print(f"  Downsampling from {img.size} to {new_size} for tractable segmentation")
        img = img.resize(new_size, Image.LANCZOS)
        arr = np.array(img)

    passes, dark_frac = check_boundary_quality(arr)
    print(f"\n{source_name} ({arr.shape[1]}x{arr.shape[0]}) -- dark boundary fraction: {dark_frac:.4f}")
    if not passes:
        print(f"  SKIPPED: insufficient boundary contrast for reliable segmentation "
              f"(need >= {MIN_DARK_BOUNDARY_FRACTION}, got {dark_frac:.4f})")
        return None

    stats = {}
    labels = segment_grains(arr, stats=stats)
    n_grains = labels.max()
    print(f"  Found {n_grains} grains after filtering.")

    rows = characterize_grains(
        labels,
        arr,
        source_name=source_name,
        confidence_tier=confidence_tier,
        group=group or source_name,
        stats=stats,
    )
    n_columnar = sum(1 for r in rows if r["morphology_label"] == "columnar")
    n_equiaxed = sum(1 for r in rows if r["morphology_label"] == "equiaxed")
    print(f"  columnar: {n_columnar}, equiaxed: {n_equiaxed}")
    print(
        "  dropped: "
        f"size={stats.get('size_filter_components', 0)}, "
        f"brightness={stats.get('brightness_gate_components', 0)}, "
        f"aspect_ratio>{MAX_ASPECT_RATIO}={stats.get('aspect_ratio_filter_components', 0)}"
    )

    save_visualization(labels, rows, f"{out_prefix}_labeled_viz.jpg")
    if save_crops_dir:
        save_grain_crops(labels, rows, arr, save_crops_dir, prefix=f"{label}_")

    return rows


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "/mnt/user-data/uploads/Dr.Lu_Grain_Structure_Image"
    out_prefix = sys.argv[2] if len(sys.argv) > 2 else "grain_analysis"
    rows = process_image(src, out_prefix, save_crops_dir=f"{out_prefix}_crops")
    if rows:
        save_csv(rows, f"{out_prefix}_grains.csv")
        print(f"Saved {out_prefix}_grains.csv")
