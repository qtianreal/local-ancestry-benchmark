"""Render the submission TIFFs from the current figure PDFs.

Figure numbers follow the order the floats appear in the manuscript, which is
the order LaTeX assigns. The mapping is written out explicitly here rather
than inferred from filenames, because the working filenames (fig0..fig5) date
from an earlier arrangement and no longer match the printed numbers.

Run after make_figures.py. Requires Pillow and macOS `sips` for rasterising.
"""

import subprocess
import sys
from pathlib import Path

from PIL import Image

FIG = Path("figures")
OUT = FIG / "submission"
DPI = 600            # Genome Research asks >=300 for halftone, more for line art
WIDTH_IN = 7.2       # widest figure in the manuscript

# printed number -> working file
MAIN = {
    1: "fig0_overview",
    2: "fig1_accuracy_vs_fst",
    3: "fig3_mechanism",
    4: "fig4_real_data",
    5: "fig5_haplotype",
}
SUPP = {"S1": "fig2_transfer_matrix"}


def render(src: Path, dst: Path):
    png = dst.with_suffix(".png")
    subprocess.run(["sips", "-s", "format", "png", "--resampleWidth",
                    str(int(WIDTH_IN * DPI)), str(src), "--out", str(png)],
                   check=True, capture_output=True)
    with Image.open(png) as im:
        im.convert("RGB").save(dst, format="TIFF", compression="tiff_lzw",
                               dpi=(DPI, DPI))
    png.unlink()
    return dst.stat().st_size


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Clear anything from a previous numbering so stale files cannot be
    # uploaded by mistake.
    for old in list(OUT.glob("*.tif")) + list(OUT.glob("*.eps")):
        old.unlink()
    stale = FIG / "submission_numbered"
    if stale.exists():
        for f in stale.iterdir():
            f.unlink()
        stale.rmdir()
        print("removed figures/submission_numbered (superseded numbering)")

    missing = []
    for num, stem in list(MAIN.items()) + list(SUPP.items()):
        src = FIG / f"{stem}.pdf"
        if not src.exists():
            missing.append(str(src))
            continue
        dst = OUT / f"Fig{num}.tif"
        size = render(src, dst)
        print(f"  Fig{num:<3} <- {stem:<22} {size / 1e6:5.1f} MB")
    if missing:
        sys.exit(f"missing source figures: {missing}")
    print(f"\nwrote {len(MAIN) + len(SUPP)} TIFFs to {OUT} at {DPI} dpi")


if __name__ == "__main__":
    main()
