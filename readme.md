
# Estimate Jones Polynomials for Ising anyons on IBM Quantum devices

This is code to reproduce the results of ...
Combine this repo with the experimental data in zenodo ... to explore other data fittings or plot the results.

The notebook `example-knots-circuits.ipynb` shows a walkthrough of how to generate the Hadamard test circuits and the stretched zero-noise extrapolation copies.

Once the experimental data is downloaded from zenodo and the 'analysis' folder is placed inside this repo the `view...` notebooks can be used to plot the results. The download contains the ZNE fits, but `generate_fits.py` can be re-run or extended to compute new zero-noise extrapolation fits. This script `generate_fits.py` uses the pipeline tool `luigi`, see [the luigi docs](https://luigi.readthedocs.io/en/stable/) for how to run it.

Have any questions? Contact me here: [christopher.self@quantinuum.com](mailto:christopher.self@quantinuum.com).

## Requirements

The main requirements are:

```python
pandas ~= 1.4
qiskit ~= 0.25
jupyterlab
luigi
```

Everything will definitely work with these versions. Newer versions of `pandas` may not be able to load the data files. Newer versions of `qiskit` should be compatible, maybe with small fixes. `luigi` is only used by the script `generate_fits.py`.

## Files and folders

- `example-knots-circuits.ipynb` walkthrough for how to generate the Hadamard test circuits
- `generate_fits.py` script to perform ZNE fits to the raw data
- `knots_circuits.py` functions to generate Hadamard test circuits
- `view-hadamard-test-results.ipynb` plot the results for the complex partition function
- `view-jones-poly-estimates.ipynb` plot the final results for the Jones polynomial