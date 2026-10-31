# Editable manuscript figures

The training-design diagram and corpus montage use T1-encoded Latin Modern Roman,
matching the manuscript's `lmodern` text family. Text and lines remain vector objects.
The five numerical plots are unchanged from the reviewed release.

From the repository root, rebuild only these two figures with Python 3 and pdfLaTeX:

```sh
python Manuscript/Figure_Source/build_figures.py --output Manuscript/Source/figures
```

Two LaTeX passes are required for page-relative TikZ placement. The helper checks
second-pass logs for errors and overflow. It writes temporary build files outside
the source directory; `--build-dir` can retain them for inspection.

In `Overleaf_Source.zip`, this directory is included as `figure_source/`. The main
manuscript consumes prebuilt PDFs in `figures/`, so compiling `main.tex` does not
require Python, TikZ figure compilation or shell escape. For local regeneration
from an extracted ZIP, run `python figure_source/build_figures.py --output figures`.

The 24 PNGs retain the exact embedded pixels from the previous dataset figure.
No mouth image is synthesized, retouched, sharpened or resampled. The original
figure embeds 132 x 132 RGB display images of the documented 88 x 88 grayscale
input crops. `Frame_Preservation.json` records their order and pixel hashes;
original data-level provenance remains in `Data/Dataset_Figure_Provenance.json`
and `Data/Dataset_Figure_Frames.csv` at repository level. Dataset/media terms continue
to apply. This editorial update does not add training or evaluation evidence.
