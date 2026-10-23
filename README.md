# CO3057 Project 2 draft

This bundle contains the Vietnamese LaTeX report, the experiment script, user-supplied inputs, and measured outputs. The cover, member page, and contents styling follow the supplied Project 1 Overleaf source; all Project 2 technical content and experiment assets are separate.

## Reproduce the current experiments

Requirements: Python 3.12 or newer, NumPy, SciPy, Pillow, and Matplotlib.

Put the uploaded images in `assets/` before running the script. Use these names for the gradient-domain experiment:

- `source.png` and `background.png`, plus `mask.png` or a transparent source image.
- The source and mask must have the same dimensions; the source must fit inside the background.

Put at least two same-sized panorama images in the same directory, named `view_1.jpg`, `view_2.jpg`, and so on. Other names are accepted when these gradient-domain images are not present.

```bash
python -m pip install numpy scipy pillow matplotlib
python pipeline.py
```

The script reads all inputs from `assets/`, runs a discrete Poisson solve and two custom feature pipelines, then saves only derived output figures and `results/metrics.json` under `results/`.

Build the report with XeLaTeX, run twice so the contents and references settle:

```bash
xelatex -interaction=nonstopmode CV_Project_2_Draft.tex
xelatex -interaction=nonstopmode CV_Project_2_Draft.tex
```

## Scope and limitations

- The stitching methods are educational implementations: Harris corners with normalized intensity patches and an ORB-style FAST plus oriented BRIEF descriptor. They are not the standard OpenCV SIFT or ORB implementations.
- The panorama homographies are estimated from feature matches; no synthetic ground-truth transform is used.
- The `logo.png` is included only for the report cover style.
