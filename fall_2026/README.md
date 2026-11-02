# MATH 4512: Fall 2026

This folder contains the starter materials for **Project Investigation 1**:
discrete Laplacians, boundary conditions, and spectral convergence. The
mathematical tasks and submission requirements are in the project handout on
Carmen.

## Start here

If you are new to Python, begin with `python_primer.ipynb`. Then open
`investigation01.ipynb`. The accompanying files `common.py` and
`investigation01.py` are part of the starter; keep them in this folder.

The starter gives you a complete D-D computation, including mode overlays,
eigenvalue comparisons, and a refinement study. Your computational work is to
derive and implement the N-N, D-N, and N-D cell-centered boundary rows, then
reuse the supplied plotting helpers for those cases.

## Google Colab (recommended)

Open either notebook on GitHub and choose **Open in Colab**, or replace
`github.com` with [githubtocolab.com](https://githubtocolab.com) in that
notebook's URL. Sign in with a Google account before running or saving work.
Run the setup cell first, then run cells in order. The notebook works on a
copy in your Google Drive and does not access the instructor's private course
repository.

After completing your work, use the browser print command to save a PDF that
shows the relevant code and output, as directed in the handout.

## Local use

Local installation is optional. With Python 3.10 or newer, install the listed
dependencies and run the starter from this directory:

```sh
python3 -m pip install -r requirements.txt
python3 investigation01.py
```

The starter provides infrastructure, not a completed submission. In
particular, derive the required finite-difference boundary rows and explain
your mathematical and computational results in your own words.
