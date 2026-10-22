# CO3057 Project 2 draft

This bundle contains the Vietnamese LaTeX report, the experiment script, generated inputs and outputs, and measured metrics. The cover, member page, and contents styling follow the supplied Project 1 Overleaf source; all Project 2 technical content and experiment assets are separate.

## Reproduce the current experiments

Requirements: Python 3.12 or newer, NumPy, SciPy, Pillow, and Matplotlib.

```bash
python -m pip install numpy scipy pillow matplotlib
python draft_pipeline.py
```

The script creates the synthetic balloon compositing pair and three projectively transformed views of one synthetic panorama scene. It runs a discrete Poisson solve and two custom feature pipelines, then saves the output figures and `results/metrics.json`.

Build the report with XeLaTeX, run twice so the contents and references settle:

```bash
xelatex -interaction=nonstopmode CV_Project_2_Draft.tex
xelatex -interaction=nonstopmode CV_Project_2_Draft.tex
```

## Scope and limitations

- All test scenes are procedural illustrations, not photographs. The current panorama results cover one generated scene, so they are a reproducible pipeline check rather than evidence across real image sets.
- The stitching methods are educational implementations: Harris corners with normalized intensity patches and an ORB-style FAST plus oriented BRIEF descriptor. They are not the standard OpenCV SIFT or ORB implementations.
- Before submission, replace or supplement the generated views with one or more real 3–4 image sequences captured from approximately the same camera location, rerun the comparisons on different scenes, and add the team's public repository URL. The supplied report marks these as outstanding requirements.
- The `logo.png` is included only for the report cover style.
