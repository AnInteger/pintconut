"""Validate 'find bead edge from a center click' against user ground truth.

Trust model: this script does NOT grade itself. It reads YOUR hand-labeled truth
and reports objective numbers. Run it yourself to verify.

Reads only:
  - the photo (default NYQC4978.png)
  - tests/validation/gt_NYQC4978.txt   (your truth: "cx cy r" per bead)

PRIMARY (the production algorithm — graded against truth):
  find_bead_radius  : single-source-of-truth, imported from src.bead_label_service.
               grad_outer (OUTERMOST radial-gradient ring above 0.6*peak; a
               highlight is an INNER ring, the true edge is the OUTERMOST, so it
               skips highlights) + radius clamp to [0.8, 1.2]*median(prior_radii)
               once >=3 priors exist (kills ballooning). Prior = the OTHER beads'
               true radii (best case: rest of board already labeled).

Baselines in main's table (comparison only, NOT used in production):
  gradient   : strongest mean-Sobel-gradient ring. Fails on highlights
               (highlight<->red ring is strongest, at ~half the true radius).
  grad_outer : same grad_outer step, but UN-clamped (no prior) — shows what the
               clamp rescues on pathological beads like #11.
  coverage   : outermost radius whose circle is >50% edge-like pixels (85th-pct
               threshold); distinguishes the true full-circle edge from a
               neighbor's partial-arc edge.

Deterministic (random.seed(42)). Usage:
    python tests/validation/eval_edge_finder.py [image] [gt.txt]
"""
import sys
import os
import random
import statistics

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.bead_label_service import gradient_magnitude, find_bead_radius, ring_profile

IMG_DEFAULT = "training/test_images/training/1/NYQC4978.png"
GT_DEFAULT = "tests/validation/gt_NYQC4978.txt"
OVERLAY_OUT = "tests/validation/overlay_algo_vs_gt_NYQC4978.png"
PRIMARY = "find_bead_radius"


def load_gt(path):
    out = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 3:
                out.append(tuple(float(v) for v in p))
    return out


def find_radius_gradient(gmag, cx, cy, r_min=3, r_max=120):
    """Strongest gradient ring (baseline). Fails on highlights."""
    prof, rs = ring_profile(gmag, cx, cy, r_min, r_max)
    return rs[int(np.argmax(prof))] if prof.max() > 0 else r_min


def find_radius_coverage(gmag, cx, cy, r_min=3, r_max=120, n_ang=120, cov_thr=0.5):
    """Outer radius where MOST of the circle is edge-like (a full-circle edge).

    Distinguishes the true bead edge from neighboring beads: a neighbor's edge
    only crosses PART of the circle (coverage ~0.1), while the true edge covers
    ~the whole circle. So we keep the largest radius whose circle has >cov_thr
    fraction of 'edge-like' pixels. An inner highlight ring is at a smaller
    radius, so the largest high-coverage radius is the true outer edge.
    'edge-like' threshold = 85th percentile of gradient magnitude (adaptive)."""
    H, W = gmag.shape
    Tg = float(np.percentile(gmag, 85))
    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    cos, sin = np.cos(ang), np.sin(ang)
    edge = r_min
    seen = False
    for r in range(r_min, r_max + 1):
        xs = np.round(cx + r * cos).astype(int)
        ys = np.round(cy + r * sin).astype(int)
        ok = (xs >= 0) & (ys >= 0) & (xs < W) & (ys < H)
        if ok.sum() < n_ang * 0.6:
            continue
        cov = float((gmag[ys[ok], xs[ok]] > Tg).mean())
        if cov > cov_thr:
            edge = r
            seen = True
    return edge if seen else r_min


def find_radius_saturation(img, cx, cy, r_max=120):
    """Outer boundary of high-saturation region (Otsu T on local crop)."""
    S = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
    H, W = S.shape
    x0, y0 = max(0, int(cx - r_max)), max(0, int(cy - r_max))
    x1, y1 = min(W, int(cx + r_max)), min(H, int(cy + r_max))
    crop = S[y0:y1, x0:x1]
    if crop.size == 0:
        return r_max
    T, _ = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ang = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    cos, sin = np.cos(ang), np.sin(ang)
    seen_body = False
    for r in range(3, r_max + 1):
        xs = np.round(cx + r * cos).astype(int)
        ys = np.round(cy + r * sin).astype(int)
        ok = (xs >= 0) & (ys >= 0) & (xs < W) & (ys < H)
        if ok.sum() < 120 * 0.6:
            continue
        frac = float((S[ys[ok], xs[ok]] > T).mean())
        if not seen_body:
            if frac > 0.7:
                seen_body = True
            continue
        if frac < 0.5:
            return r
    return r_max


def box(cx, cy, r):
    return (cx - r, cy - r, cx + r, cy + r)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = aa + ab - inter
    return inter / u if u > 0 else 0.0


def med(xs):
    return statistics.median(xs)


def run_method(name, img, gmag, gt):
    radii, r_errs, ious = [], [], []
    for (cx, cy, r_gt) in gt:
        if name == "gradient":
            r_algo = find_radius_gradient(gmag, cx, cy)
        elif name == "grad_outer":
            # no prior_radii -> clamp skipped -> raw (un-clamped) grad_outer baseline
            r_algo = find_bead_radius(gmag, cx, cy)[0]
        elif name == "coverage":
            r_algo = find_radius_coverage(gmag, cx, cy)
        else:
            r_algo = find_radius_saturation(img, cx, cy)
        radii.append(r_algo)
        r_errs.append(abs(r_algo - r_gt))
        ious.append(iou(box(cx, cy, r_algo), box(cx, cy, r_gt)))
    return radii, r_errs, ious


def run_primary(gmag, gt):
    """Evaluate PRIMARY (production find_bead_radius) with best-case prior.

    Prior for each bead = the OTHER beads' true radii, simulating 'rest of the
    board is already labeled'. find_bead_radius then clamps to
    [0.8, 1.2]*median(prior) when >=3 priors exist."""
    truths = [r for _, _, r in gt]
    radii, r_errs, ious = [], [], []
    for i, (cx, cy, r_gt) in enumerate(gt):
        prior = truths[:i] + truths[i + 1:]   # other beads' true radii as prior
        r_algo, _warn = find_bead_radius(gmag, cx, cy, prior_radii=prior)
        radii.append(r_algo)
        r_errs.append(abs(r_algo - r_gt))
        ious.append(iou(box(cx, cy, r_algo), box(cx, cy, r_gt)))
    return radii, r_errs, ious


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else IMG_DEFAULT
    gt_path = sys.argv[2] if len(sys.argv) > 2 else GT_DEFAULT
    img = cv2.imread(img_path)
    if img is None:
        sys.exit(f"cannot read {img_path}")
    gmag = gradient_magnitude(img)
    gt = load_gt(gt_path)
    random.seed(42)
    print(f"image: {img.shape[1]}x{img.shape[0]}  |  {len(gt)} ground-truth beads\n")

    methods = ["gradient", "grad_outer", "coverage"]
    labels = {"gradient": "grad", "grad_outer": "gOut", "coverage": "cov"}
    results = {m: run_method(m, img, gmag, gt) for m in methods}

    print(f"  {'bead':>4} {'r_gt':>4} |" + "".join(f" {labels[m]:>4} {'dR':>3}" for m in methods) + "  catastrophic?")
    for i, (cx, cy, r_gt) in enumerate(gt):
        row = f"  {i+1:>4} {r_gt:>4.0f} |"
        bad = []
        for m in methods:
            ra = results[m][0][i]
            re = results[m][1][i]
            row += f" {ra:>4} {re:>3.0f}"
            if re > 10:
                bad.append(labels[m])
        row += "  " + (",".join(bad) if bad else "ok")
        print(row)

    print("\n  -- summary (exact center) --")
    for m in methods:
        re, io = results[m][1], results[m][2]
        nf = sum(1 for e in re if e > 10)
        print(f"  {m:>10}: |dR| median={med(re):.2f} max={max(re):.0f} | "
              f"IoU median={med(io):.2f} min={min(io):.2f} | catastrophic={nf}/{len(gt)}")

    # jitter on the raw grad_outer baseline (no center refinement, no clamp)
    print(f"\n  -- grad_outer baseline, click jitter (no center refinement) --")
    for d in (2, 4):
        j_io, j_re = [], []
        for (cx, cy, r_gt) in gt:
            for _ in range(5):
                a = random.uniform(0, 2 * np.pi)
                jx, jy = cx + d * np.cos(a), cy + d * np.sin(a)
                r_algo = find_bead_radius(gmag, jx, jy)[0]
                j_io.append(iou(box(jx, jy, r_algo), box(cx, cy, r_gt)))
                j_re.append(abs(r_algo - r_gt))
        print(f"    +-{d}px -> IoU median={med(j_io):.2f} mean={statistics.mean(j_io):.2f} | |dR| median={med(j_re):.2f}")

    print(f"\n  -- PRIMARY: {PRIMARY} (production: grad_outer + clamp) --")
    pr, pe, pi = run_primary(gmag, gt)
    pnf = sum(1 for e in pe if e > 10)
    print(f"  {PRIMARY}: |dR| median={med(pe):.2f} max={max(pe):.0f} | "
          f"IoU median={med(pi):.2f} min={min(pi):.2f} | off-by>10px={pnf}/{len(gt)}")
    for i, (cx, cy, r_gt) in enumerate(gt):
        flag = "  <-- off" if pe[i] > 10 else ""
        print(f"    bead {i+1}: r_gt={r_gt:.0f} r_algo={pr[i]} |dR|={pe[i]:.0f} IoU={pi[i]:.2f}{flag}")

    # overlay: green=truth, magenta=PRIMARY
    disp = img.copy()
    for (cx, cy, r_gt), r_algo in zip(gt, pr):
        cv2.circle(disp, (int(cx), int(cy)), int(r_gt), (0, 255, 0), 2)
        cv2.circle(disp, (int(cx), int(cy)), int(r_algo), (255, 0, 255), 1)
    cv2.imwrite(OVERLAY_OUT, disp)
    print(f"\noverlay -> {OVERLAY_OUT}  (green=truth, magenta={PRIMARY})")


if __name__ == "__main__":
    main()
