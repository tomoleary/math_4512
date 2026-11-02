# MATH 4512: Project Starters, Autumn 2026

## Start here

If you are new to Python, begin with `python_primer.ipynb`. It introduces the
small set of Python and NumPy patterns used in Investigation 1: variables,
arrays, indexing, loops, functions, matrix-vector multiplication, and plots.
It is intentionally written in a direct style; you do not need to understand
advanced Python features to begin the project.

The recommended route is Google Colab.
You need to sign in with a Google account to execute and save a notebook in
Colab. A local Conda installation or Docker setup can also be used, but these
are self-managed alternatives and are not the supported course path.

Each starter runs one small baseline. The mathematical questions, further
experiments, and grading requirements are in the project handout on Carmen.
An unchanged run is not a completed investigation.

Complete Investigation 1 and two electives. At least one elective must be
computational (3, 4, or an approved computational 5).
Investigation 1 requires analytical derivations and the computational comparisons in Part E.

## Files

| Investigation | Notebook | Script |
| --- | --- | --- |
| 1. Discrete Laplacian and spectrum | `investigation01.ipynb` | `investigation01.py` |
| 2. Heat and wave modes | `investigation02.ipynb` | `investigation02.py` |
| 3. Spectra and geometry | `investigation03.ipynb` | `investigation03.py` |
| 4. Nonlinear heat and Newton | `investigation04.ipynb` | `investigation04.py` |
| 5. Approved application | `investigation05.ipynb` | `investigation05.py` |

`common.py` supplies matrix assembly, eigensolvers, and output helpers.
`nonlinear.py` supplies small-grid damped Newton and nonlinear heat solvers.
Read the relevant helper alongside your mathematical derivation.

Investigation 1 uses a nodal grid for D-D and a separate cell-centered grid
for N-N and mixed boundaries. Its required cell_centered_laplacian is left
unfinished so students can implement their own derived boundary rows.

Investigation 2 intentionally leaves `unforced_solution` and
`switched_modal_response` for students to derive and implement. Its baseline
uses a numerical ODE reference rather than printing the modal solution formulas.
Investigation 4 leaves `manufactured_source` for students to derive and implement.
All baselines run without calling these unfinished functions.
Investigation 5 is a model-adaptation scaffold; the example is not itself an
approved custom project. Follow the handout's six-item proposal checklist and
100-point custom-project rubric; organize the report under its eight categories.

## Browser Route: Google Colab

The starter bundle is available in the [MATH 4512 project repository](https://github.com/tomoleary/math_4512). Open your chosen notebook on GitHub and use **Open in Colab**, or open Colab and select the GitHub notebook URL.
The notebook's setup cell downloads only that public repository if the helper
files are not already available. It does not access the instructor's private
course repository. A Google account may be needed to run Colab.

The notebooks include editable Python, not just calls to a hidden solution.
Run the setup, modify parameters and experiment cells, and save your own copy.
Output figures and CSV files are placed in `outputs/investigationXX/`.
Download the notebook and any required outputs before the runtime ends.

The notebooks' setup assumes this exact public folder name. To work locally,
run the provided folder directly.

## Local Route

Use a recent Python 3 installation (3.10 or newer recommended). Keep all files
in this folder together. From this folder, install dependencies and run, e.g.:

```sh
python3 -m pip install -r requirements.txt
python3 investigation04.py
python3 -m unittest -v test_starters
```

A virtual environment is recommended if you manage Python locally. The code
uses NumPy, SciPy, and Matplotlib. Jupyter is optional for the notebook route.
Scripts save figures without requiring a graphical desktop. An alternative
output location can be set with the `MATH4512_OUTPUT_DIR` environment variable.

The starter defaults are intentionally small. In particular, the dense
finite-difference Jacobian in Investigations 4 and 5 is educational, not an
efficient large-PDE solver. Larger grids may need sparse Jacobians and solvers.

## Verification and Interpretation

Always record parameters, units/scaling, boundary conditions, and numerical
tolerances. An eigenpair or nonlinear residual measures the algebraic solve;
it is not the same as error relative to a continuous PDE. Use the independent
checks requested in the handout. Public implementation tests check the supplied
infrastructure; they do not carry out the assessed studies or grade your work.

Useful primary references:

- [NumPy linear solves](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html)
- [SciPy symmetric sparse eigenproblems](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html)
- [SciPy numerical ODE reference solver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
- [NIST FiPy nonlinear diffusion example](https://pages.nist.gov/fipy/en/latest/generated/examples.diffusion.newtonCoupled.html)

## Submission and Privacy

AI tools may help write, debug, and understand code. Mathematical derivations,
analysis, and interpretation must be your own work. Briefly acknowledge AI
use, verify results, and be prepared to explain your submission.
Follow the handout for final requirements. Do not put student names, grades, submissions,
or private course records into public commits or issues. Submit your assessed
work through the course's designated channel, not as a public pull request.
