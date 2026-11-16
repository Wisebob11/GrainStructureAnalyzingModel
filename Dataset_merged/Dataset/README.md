# Per-Grain Segmentation Dataset (Columnar/Equiaxed Characterization)

Generated: June 2026
Total grains measured: 2,105 (across 2 source images)

This v2 dataset is a per-grain dataset rather than a 256x256 patch
classification dataset. Instead of labeling a whole square crop as
"columnar" or "equiaxed," this pipeline finds EVERY INDIVIDUAL GRAIN in
an image, measures its actual geometry, and classifies each grain on its
own measured aspect ratio.

## Folder structure

    Dataset/
        all_grains.csv              per-grain measurements (the core dataset)
        visualizations/              one labeled-by-class image per source
        grain_crops/
            columnar/                 1,412 individual grain images (masked, cropped)
            equiaxed/                 693 individual grain images (masked, cropped)
        sources.csv                  source/specimen metadata and physical scale
        segment_grains.py            the pipeline script (re-run on new images)

## all_grains.csv columns

| column | meaning |
|---|---|
| source_image | which source file this grain came from |
| grain_id | unique ID within that source image |
| area_px | grain area in pixels |
| equivalent_diameter_px | diameter of a circle with the same area (grain size proxy) |
| major_axis_px / minor_axis_px | best-fit ellipse axis lengths |
| aspect_ratio | major/minor -- the core shape metric |
| feret_aspect_ratio | bounding-rectangle aspect ratio, a non-convex-shape check |
| orientation_deg | best-fit ellipse angle, -90 to 90 |
| centroid_row / centroid_col | grain center, pixel coordinates |
| bbox_minr/minc/maxr/maxc | bounding box, pixel coordinates |
| mean_color_r/g/b | average RGB of the grain (IPF orientation proxy -- see caveat below) |
| morphology_label | "columnar" if aspect_ratio >= 2.0, else "equiaxed" |
| confidence_tier | "high_confidence" or "moderate_confidence" -- see below |
| touches_border | 1 when the segmented grain is truncated by the image edge |
| ambiguous_weight | 0.5 for AR 1.7–2.3, otherwise 1.0 |

## IMPORTANT: confidence tiers and the quality gate

Not every image can be reliably segmented by this method. The pipeline
finds grains by detecting flat-colored regions separated by dark
boundary lines. Some real EBSD renders (e.g. the LEROY serial-section
images, and one of the Area_1 sub-crops) have NO drawn boundary lines --
grains blend into each other through continuous color gradients instead.
On those images this method fragments grains into hundreds of small,
unreliable pieces and was EXCLUDED automatically by a quality gate
(measuring the fraction of dark/boundary-like pixels in the image;
images below 3% were dropped rather than included with bad data).

**high_confidence** (Dr. Lu's reference image, 602 grains): clear dark
boundary lines, manageable resolution, segmentation visually verified
against the original image region-by-region. Aspect ratio and size
numbers here are trustworthy at the individual-grain level.

**moderate_confidence** (Area_1_1 AM Bench sub-crop, 1,503 grains): real
EBSD data with boundary lines present, but segmentation shows more
fragmentation than the high-confidence source (visually verified -- some
grains split into 2-3 pieces, a minority of pixels misclassified at
region edges). Individual-grain numbers are noisier here; AGGREGATE
statistics (mean aspect ratio, % columnar, size distribution) are still
directionally meaningful, but don't treat any single moderate-confidence
grain's measurement as precise.

**Excluded** (LEROY IPFX/Y/Z, Area_1 full montage at native or
downsampled resolution, Area_1_1crop001IPF): either failed the boundary
quality gate (no dark lines to segment on) or produced visually
unreliable fragmentation when checked. Not included in all_grains.csv.
If you need per-grain data from these specific images, they would
require a different segmentation method (e.g. watershed transform on a
gradient/edge map, or k-means clustering in color space) that wasn't
built out here -- flagging this as a known gap rather than guessing.

## Aggregate statistics (for context)

| Tier | n grains | % columnar | mean aspect ratio | mean equiv. diameter (px) |
|---|---|---|---|---|
| high_confidence | 602 | 62.0% | 3.66 | 33.6 |
| moderate_confidence | 1,503 | 69.1% | 3.06 | 11.5 |

Note the diameter values are in PIXELS, not microns -- converting to a
real physical unit requires the scale bar / pixel size for each source
image (e.g. Dr. Lu's image and the AM Bench montages have a scale bar
baked into the image; you'd need to measure that bar's pixel length and
divide by its labeled physical length to get a µm/pixel conversion
factor before comparing grain sizes ACROSS different source images, since
they are almost certainly captured at different magnifications).

## On inferring material properties from this data

This is the part worth being precise about, since it's easy to overstate.

**Reasonably supportable from this data:**
- Relative solidification/processing conditions: a high fraction of
  columnar grains suggests directional cooling (e.g. near a build wall
  or mold surface during solidification); a transition to equiaxed
  suggests either a change in cooling direction/rate or post-process
  recrystallization.
- Relative grain-size-based strength comparisons BETWEEN regions of the
  SAME image at the SAME scale (Hall-Petch relationship: smaller grains
  generally correlate with higher yield strength) -- this is a
  comparative statement, not an absolute number.
- Texture/anisotropy hints from mean_color_rgb, since IPF color does
  encode crystallographic orientation -- BUT see the caveat directly
  below before using this for anything quantitative.

**NOT supportable from this data alone:**
- Absolute mechanical properties (yield strength in MPa, hardness,
  ductility, fatigue life) -- these need actual mechanical test data
  calibrated to this specific alloy, not just grain geometry.
- Precise texture/anisotropy analysis -- mean_color_rgb is a rough proxy
  computed by averaging this pipeline's own pixel values, NOT a
  substitute for real Euler-angle orientation data from the EBSD scan
  itself. Two different crystal orientations can map to similar-looking
  colors depending on the IPF color key used, and JPEG/BMP compression
  shifts pixel values slightly. If real quantitative texture analysis is
  needed, that has to come from the underlying .ctf/.ang orientation
  data (via MTEX or similar), not from this pipeline's color averaging.
- Alloy composition, corrosion resistance, creep behavior -- not
  inferable from grain morphology at all.

If Dr. Lu is expecting quantitative property predictions (e.g. "this
grain structure implies X MPa yield strength"), that needs a labeled
training set pairing grain statistics with KNOWN mechanical test results
for calibration -- worth raising directly, since it changes what's
buildable in the time available.

## Regenerating or extending this dataset

Run: `python3 segment_grains.py <image_path> <output_prefix>`

The split builder also requires an explicit `--test-source`; use `--val-source`
for a second held-out source. Add the same `group` to A1/A2/A3 or IPF-X/Y/Z
rows when they represent the same specimen/build. Never split duplicate views
of one group across train, validation, and test.

When calling `process_image`, pass `confidence_tier` and `group` explicitly;
the CSV writer preserves both values for downstream filtering and leakage-safe
splitting.

This prints the boundary-quality check result and either processes the
image or explains why it was skipped. Always check the printed dark
boundary fraction and the saved `_labeled_viz.jpg` before trusting new
output -- silent fragmentation on a boundary-free image is the most
likely failure mode (see "Excluded" above).

---

## Update: padded-square crops and a third source (gabbro)

What changed (nothing from the earlier dataset was deleted):
- `grain_crops/` now holds crops padded (centered, black) to a SQUARE at native size, so a resize no longer stretches
  thin grains. Names use the exact prefix `sanitize_source_prefix()` produces (`Area_1_1_...`, `DrLu_...`).
- `grain_crops_original/` is the previous, unpadded crop set (old `Area1_1_...` names), kept untouched.
- `all_grains.csv` keeps every earlier column and adds `ambiguous`, `too_small`, `crop_w_px`, `crop_h_px`.
  Earlier rows have blank `feret_aspect_ratio` and `touches_border` (those two columns were added to the segmenter after
  they were made).
- New source: 4 maps from GRaMC (gabbroic nodules, Gigoldur, Iceland), `group` = `gabbro_gigoldur`, tier `unverified`.
  Only the `05_AreaFiltered` stage of each sample is used; stages 01-03 are the same grains and would duplicate them.
  The grey scale-bar box was painted out first and grains touching it were dropped; `removed_artifacts.csv` lists
  the non-grain objects removed (frame-line fragments and one ~500,000 px merged region in 01a2).
- Overlays for the four new maps are in `visualizations/` next to the two existing ones.
- Segmentation caveat: the gabbro maps were segmented with the current (Lab-quantized) `segment_grains.py`; Area_1_1 and
  Dr. Lu were segmented earlier with the RGB-quantized version and were not re-run (their source images are not in the repo).
