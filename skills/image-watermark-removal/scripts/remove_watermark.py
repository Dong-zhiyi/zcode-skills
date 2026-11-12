#!/usr/bin/env python3
"""Remove a watermark from an image.

The watermark is treated as a mark whose brightness differs from the
surrounding background.  The script builds a mask inside a bounding box,
inpaints the marked pixels, then (for flat backgrounds) transfers the
background texture from a neighbouring clean patch so the repair is invisible.

Typical use
-----------
  # 1. look at the picture, read off the watermark's pixel box
  # 2. let the script pick the threshold itself
  python remove_watermark.py  in.png  out.png  --bbox 1575 1870 1990 1990

  # force the mark to be the bright side, or pass an explicit threshold
  python remove_watermark.py  in.png  out.png  --bbox 100 100 400 200 --brightness above
  python remove_watermark.py  in.png  out.png  --bbox 100 100 400 200 --threshold 180

  # not sure where the mark is? ask for candidate regions, then pick one
  python remove_watermark.py  in.png  out.png  --detect

Only NumPy + OpenCV + Pillow are required.
"""

import argparse
import sys

import numpy as np


def _dependencies():
    try:
        import cv2
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        sys.exit(
            "Missing dependency: %s\n"
            "Install with:  pip install numpy opencv-python pillow" % exc
        )
    return cv2, Image


# --------------------------------------------------------------------------- #
# mask building
# --------------------------------------------------------------------------- #

def build_mask(img, bbox, brightness, threshold, dilate, margin=0):
    """Return (mask, info).  mask is uint8 0/255 over the whole image."""
    cv2, _ = _dependencies()
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(w, x1 + margin)
    y1 = min(h, y1 + margin)

    roi = img[y0:y1, x0:x1]
    if roi.size == 0:
        raise SystemExit("empty bounding box")

    lum = roi.mean(axis=2)

    # background reference: a frame on the ROI border, assumed to be clean
    t = max(6, min(roi.shape) // 10)
    ring = np.concatenate(
        [lum[:t, :].ravel(), lum[-t:, :].ravel(),
         lum[:, :t].ravel(), lum[:, -t:].ravel()]
    )
    ring_mean = float(ring.mean())
    ring_std = float(ring.std())

    if threshold is None:
        thr = float(cv2.threshold(lum.astype(np.uint8), 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    else:
        thr = float(threshold)

    if brightness == "auto":
        above = lum > thr
        below = lum < thr
        d_above = abs(float(lum[above].mean()) - ring_mean) if above.any() else -1.0
        d_below = abs(float(lum[below].mean()) - ring_mean) if below.any() else -1.0
        brightness = "above" if d_above >= d_below else "below"

    marked = (lum > thr) if brightness == "above" else (lum < thr)

    mask = np.zeros((h, w), np.uint8)
    mask[y0:y1, x0:x1] = np.where(marked, 255, 0).astype(np.uint8)

    # close pinholes inside strokes, then grow to swallow the anti-aliased halo
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8))

    info = dict(
        bbox=(x0, y0, x1, y1),
        threshold=thr,
        brightness=brightness,
        ring_mean=round(ring_mean, 1),
        ring_std=round(ring_std, 1),
        mask_pixels=int((mask > 0).sum()),
        mask_fraction=round(float((mask > 0).sum()) / (h * w), 5),
    )
    return mask, info


def detect_candidates(img, top=10, min_area=800):
    """Heuristically propose watermark regions.

    Finds pixels that deviate from a large median (i.e. anything that is not
    background), dilates so that thin strokes merge into blobs, then ranks the
    blobs by how close they sit to the image edge -- watermarks live in corners
    and margins, the subject usually does not.  Returns a list of dicts.

    Note: one watermark can come back as several adjacent blobs (an icon and
    the text beside it, say).  Merge overlapping/nearby boxes before use.
    """
    cv2, _ = _dependencies()
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # deviation from a large median = anything that is not background
    med = cv2.medianBlur(gray, 51)
    diff = cv2.absdiff(gray, med)
    thr = max(25.0, float(np.percentile(diff, 95.0)))
    m = np.where(diff > thr, 255, 0).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # merge strokes of one mark into a single blob
    m = cv2.dilate(m, np.ones((25, 25), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    reach = 0.1 * min(h, w)
    cands = []
    for i in range(1, n):
        x, y, ww, hh, area = stats[i]
        if area < min_area:
            continue
        # distance from the component to the nearest image edge
        dist = min(x, y, w - (x + ww), h - (y + hh))
        prox = float(np.exp(-max(0, dist) / reach))
        score = float(area) * prox
        patch = gray[y:y + hh, x:x + ww]
        cands.append(dict(
            score=score, area=int(area), bbox=(int(x), int(y), int(x + ww), int(y + hh)),
            edge_dist=int(max(0, dist)),
            mean_lum=round(float(patch.mean()), 1),
            mark="bright" if float(patch.mean()) > float(med[y:y + hh, x:x + ww].mean()) else "dark",
        ))
    cands.sort(key=lambda c: -c["score"])
    return cands[:top]


# --------------------------------------------------------------------------- #
# removal
# --------------------------------------------------------------------------- #

def remove(img, mask, inpaint_radius, texture, texture_sigma, feather):
    """Inpaint the mask, then optionally restore background texture."""
    cv2, _ = _dependencies()
    h, w = img.shape[:2]
    result = cv2.inpaint(img, mask, inpaint_radius, cv2.INPAINT_NS)

    if texture == "off":
        return result, "inpaint only"

    # Ring statistics decide whether the background is flat enough that a
    # shifted patch is a fair stand-in for it.  The ring is measured on the
    # repaired result, which is clean by now.
    inner = cv2.erode(mask, np.ones((25, 25), np.uint8)) > 0
    if not inner.any():
        return result, "inpaint only (mask too small for texture transfer)"

    # High-pass texture from clean neighbours, shifted into place.
    cols = (mask > 0).any(axis=0).nonzero()[0]
    rows = (mask > 0).any(axis=1).nonzero()[0]
    dx = max(1, int(cols.max() - cols.min()) + 40)
    dy = max(1, int(rows.max() - rows.min()) + 40)

    def shifted(sx, sy):
        """out[y, x] = img[y + sy, x + sx]; negative pulls from the left/above."""
        out = np.zeros_like(img)
        x0, x1 = max(0, -sx), min(w, w - sx)
        y0, y1 = max(0, -sy), min(h, h - sy)
        if x1 > x0 and y1 > y0:
            out[y0:y1, x0:x1] = img[y0 + sy:y1 + sy, x0 + sx:x1 + sx]
        return out

    # Only directions with enough clean source pixels are usable, so a mark in
    # any corner still finds a neighbour to borrow texture from.
    patches = []
    if cols.min() >= dx:
        patches.append(("left", shifted(-dx, 0)))
    if cols.max() + dx <= w:
        patches.append(("right", shifted(dx, 0)))
    if rows.min() >= dy:
        patches.append(("above", shifted(0, -dy)))
    if rows.max() + dy <= h:
        patches.append(("below", shifted(0, dy)))
    if not patches:
        return result, "inpaint only (image too small for texture transfer)"

    # A shifted patch is only a fair stand-in for the target when its low
    # frequency matches; otherwise the background carries real structure and
    # shifting it in would be wrong.
    low_target = cv2.GaussianBlur(result.mean(axis=2), (0, 0), 25)
    scored = []
    for name, p in patches:
        lp = cv2.GaussianBlur(p.mean(axis=2), (0, 0), 25)
        mismatch = float(np.abs(low_target - lp)[inner].mean())
        scored.append((mismatch, name, p))
    scored.sort()
    best_mismatch, best_name, _ = scored[0]
    if texture == "auto" and best_mismatch > 3.0:
        return result, "inpaint only (neighbour patch off by %.1f levels)" % best_mismatch

    tex = np.zeros_like(img, dtype=np.float64)
    for _, _, p in scored:
        tex += p.astype(np.float64) - cv2.GaussianBlur(
            p.astype(np.float64), (0, 0), texture_sigma)
    tex /= len(scored)

    alpha = cv2.GaussianBlur(mask.astype(np.float64), (0, 0), feather)
    alpha = alpha / max(alpha.max(), 1e-6)
    result = result.astype(np.float64) + tex * alpha[:, :, None]
    return np.clip(result, 0, 255).astype(np.uint8), (
        "inpaint + texture transfer (%s, low-freq match %.1f)" % (best_name, best_mismatch))


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #

def main(argv=None):
    cv2, Image = _dependencies()

    p = argparse.ArgumentParser(
        description="Remove a watermark from an image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("input", help="source image")
    p.add_argument("output", help="where to write the cleaned image")
    p.add_argument("--bbox", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"),
                   help="watermark bounding box in pixels")
    p.add_argument("--detect", action="store_true",
                   help="print candidate regions and exit (no output written)")
    p.add_argument("--brightness", choices=["auto", "above", "below"],
                   default="auto",
                   help="is the mark brighter or darker than the background?")
    p.add_argument("--threshold", type=float, default=None,
                   help="luminance threshold (default: Otsu inside the box)")
    p.add_argument("--dilate", type=int, default=7,
                   help="mask dilation radius in px (default 7)")
    p.add_argument("--margin", type=int, default=0,
                   help="grow the bounding box by this many px (default 0)")
    p.add_argument("--inpaint-radius", type=int, default=15,
                   help="OpenCV inpaint radius (default 15)")
    p.add_argument("--texture", choices=["auto", "on", "off"], default="auto",
                   help="restore background texture (default auto)")
    p.add_argument("--texture-sigma", type=float, default=8.0,
                   help="texture band-pass sigma (default 8)")
    p.add_argument("--feather", type=float, default=4.0,
                   help="blend feather sigma (default 4)")
    p.add_argument("--mask-out", default=None,
                   help="also write the mask as a PNG for inspection")
    args = p.parse_args(argv)

    img = np.array(Image.open(args.input).convert("RGB"))
    print("input: %s  %dx%d" % (args.input, img.shape[1], img.shape[0]))

    if args.detect:
        cands = detect_candidates(img)
        if not cands:
            print("no candidate regions found")
            return 0
        print("candidate regions, most watermark-like first:")
        print("  score  area  edge_dist  mark   bbox(x0 y0 x1 y1)")
        for c in cands:
            x0, y0, x1, y1 = c["bbox"]
            print("  %6.0f %6d %7d  %6s   %d %d %d %d" % (
                c["score"], c["area"], c["edge_dist"], c["mark"], x0, y0, x1, y1))
        print("\npick one and re-run with --bbox X0 Y0 X1 Y1")
        print("(one watermark is often split into several nearby boxes, e.g. an "
              "icon plus the text beside it -- merge them into a single bbox)")
        return 0

    if args.bbox is None:
        p.error("--bbox is required unless --detect is used")

    mask, info = build_mask(img, args.bbox, args.brightness, args.threshold,
                            args.dilate, args.margin)
    print("mask: %s" % info)

    result, note = remove(img, mask, args.inpaint_radius, args.texture,
                          args.texture_sigma, args.feather)
    print("repair: %s" % note)

    Image.fromarray(result).save(args.output)
    print("written: %s" % args.output)

    if args.mask_out:
        cv2.imwrite(args.mask_out, mask)
        print("mask: %s" % args.mask_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
