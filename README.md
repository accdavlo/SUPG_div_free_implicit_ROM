# GFSUPG
This repository implements the global-flux (GF) streamline-upwind Petrov-Galerkin (SUPG) method for Finite Element methods for the solution of the linear acoustic equations in 2D, as described in the paper [Structure preserving nodal continuous Finite Elements via Global Flux quadrature](https://arxiv.org/abs/2407.10579). 
It allows to preserve truly multi-dimensional moving equilibria as vortices. 
The code is written in python and uses scipy for the assembly of the matrices.

The code will be used in the context of the coding week [SunHype2026](https://elenagaburro.it/sunhype2026.html) to extend it in the implicit setting and for Model Order Reduction (MOR) applications.

## How to install
The code is in python and all dependencies can be found in `pyproject.toml`. The code can be installed using pip, better after having created a virtual environment.

If you use conda ([how to install miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview))

```bash
conda create -n gfsupg python=3.11
conda activate gfsupg
```
And then install the code, either using pip:

```bash
pip install -e .
```

Then, if you want to use jupyter notebooks, you need to run also 

```bash
python -m ipykernel install --user --name=gfsupg --display-name "Python (gfsupg)"
```

The solver is in [gfsupg/solver.py](gfsupg/solver.py) and the problem definition is in [gfsupg/problem.py](gfsupg/problem.py) including all the test cases.
Some auxiliary functions are in [gfsupg/plotting.py](gfsupg/plotting.py) and some quadrature utilities are in [gfsupg/quadr.py](gfsupg/quadr.py).

To start, the main script to run is [apps/test_SUPG.py](apps/test_SUPG.py) where you can select the test case to run and the SUPG and the GF-SUPG codes and compare them.

There are several other scripts in the `apps` folder to test various aspects of the method: convergence tests, perturbation tests, testing the kernels of the methods, plot scripts, testing the operators on analytical div-free solutions, etc.

## Code notation vs paper notation
Symbols and definitions in 1D, omitting the cell $\mathcal{C}_{ij}$ indexes, using $k$ for the row index and $\tilde{k}$ for column index and $x$ as geometrical variable or nodes rescaled in $[0,1]$ (similarly $y$ can be used)


| Symbol                           | Definition                                                 | Code name        |
| -------------------------------- | ---------------------------------------------------------- | ---------------- |
|$M_{k\tilde{k}}$                   | $\int \psi_k \psi_{\tilde{k}}$              | mass|
|$L_{k\tilde{k}}$                   | $\delta_{k \tilde{k}}\int \psi_k$          | lump\_mass|
|$(E_x)^{k}_{\tilde{k}}$            | $\psi_{\tilde{k}}(x^k)$                     | evol\_mass|
|$(I_x)^k_{\tilde{k}}$   | $\int_{0}^{s^k}\psi_{\tilde{k}}$            | int\_mass|
|$(D^x)_{k\tilde{k}}$ | $\int \partial_x \psi_k \psi_{\tilde{k}}$ | deriv\_i|
|$(D_x)_{k\tilde{k}}$ | $\int \psi_k \partial_x \psi_{\tilde{k}}$ | deriv\_j|
|$(D_x^x)_{k\tilde{k}}$ | $\int \partial_x \psi_k \partial_x \psi_{\tilde{k}}$ | deriv\_ij|
|$(\tilde{D}_x^x)_{k\tilde{k}}$ | ${D}_x^x E_x = D^x_x$ | deriv\_ij\_tilde|
|$(\tilde{D}^x)_{k\tilde{k}}$ | ${D}_x^x I_x$ | deriv\_i\_tilde|
|$(\bar{D}_x)_{k\tilde{k}}$ | ${D}_x E_x  = D_x$ | deriv\_j\_bar|
|$\tilde{M}_{k\tilde{k}}$  | ${D}_x I_x$ | der\_int\_tilde|
|$\bar{M}_{k\tilde{k}}$  | $M_x E_x = M_x$ | mass\_bar|
|$(\bar{I}_x)_{k\tilde{k}}$  | $M_x I_x$ | mass\_int| 


|Definition | Code name|
|-----------|----------|
|$M \otimes M$  | mass|
|$L \otimes L$  | lump\_mass|
|$\tilde{M} \otimes \tilde{M}$  | mass\_tilde\_tilde|
|$D_x \otimes M$  | IDx|
|$M \otimes D_y$  | IDy|
|$D^x \otimes M$  | DxI|
|$M \otimes D^y$  | DyI|
|$D_x \otimes D_y$  | IDxy|
|$D_x^x \otimes M$  | DxDx|
|$M \otimes D_y^y$  | DyDy|
|$D^x \otimes D_y$  | DxDy|
|$D_x \otimes D^y$  | DyDx|
|$D_x^x \otimes D_y$  | DxDxy|
|$D_x \otimes D^y_y$  | DyDxy|
|$M \otimes I_y$  | int\_y|
|$I_x \otimes M$  | int\_x|
|$D^x \otimes I_y$  | Dx\_int|
|$I_x \otimes D^y$  | Dy\_int|
|$L^{-1}\otimes L^{-1}$  | inv\_lump|  
|$\tilde{D}_x^x \otimes \tilde{M}$  | DxDx\_tilde |
|$\tilde{M} \otimes \tilde{D}_y^y$  | DyDy\_tilde |
|$\tilde{D}^x \otimes \bar{D}_y$  | DxDy\_tilde |
|$\bar{D}_x \otimes \tilde{D}^y$  | DyDx\_tilde |
|$\bar{D}_x \otimes \tilde{M}$  | IDx\_tilde |
|$\tilde{M} \otimes \bar{D}_y$  | IDy\_tilde |
|$\tilde{M} \otimes \bar{M}$  | mass\_tilde\_x |
|$\bar{M} \otimes \tilde{M}$  | mass\_tilde\_y |
|$\tilde{D}^x \otimes \bar{M}$  | DxI\_tilde |
|$\bar{M} \otimes \tilde{D}^y$  | DyI\_tilde | 
|$\tilde{M} \otimes \bar{I}_y$  | int\_y\_tilde | 
|$\bar{I}_x \otimes \tilde{M}$  | int\_x\_tilde | 
|$\tilde{D}^x \otimes \tilde{M}$  | DxM\_tilde | 
|$\tilde{M} \otimes  \tilde{D}^y$  | DyM\_tilde |
|$\tilde{M} \otimes  \tilde{M}$  | mass\_tilde | 
|$\tilde{D}^x  \otimes \bar{I}_y$  | Dx\_int\_tilde | 
|$D^xM^{-1}_xD_x \otimes  M^y$  | DxDx3 | 
|$M^x\otimes  D^yM^{-1}_yD_y$  | DyDy3 | 
|$D^xM^{-1}_xD_x \otimes  D_yI_y$  | DxDx3\_tilde | 
|$D_xI_x\otimes  D^yM^{-1}_yD_y$  | DyDy3\_tilde | 
|$(D^x_x-D^xM^{-1}_xD_x) \otimes  M_y$  | ZxMy | 
|$M_x\otimes  (D^y_y -D^yM^{-1}_yD_y )$  | ZyMx | 
|$(D^x_x-D^xM^{-1}_xD_x) \otimes  D_yI_y$  | ZxMy\_tilde | 
|$D_xI_x\otimes ( D^y_y -D^yM^{-1}_yD_y)$  | ZyMx\_tilde| 
|$(D^x_x-D^xM^{-1}_xD_x)I_x \otimes  D_y$   | DyZx\_int | 
|$D_x\otimes ( D^y_y -D^yM^{-1}_yD_y)I_y$  | DxZy\_int| 
|$(D^x_x-D^xM^{-1}_xD_x)I_x \otimes  D_yI_y$   | My\_tilde\_Zx\_int | 
|$D_xI_x\otimes ( D^y_y -D^yM^{-1}_yD_y)I_y$  | Mx\_tilde\_Zy\_int| 
|$(D^x_x-D^xM^{-1}_xD_x)I_x\otimes M_y$  | Zx\_int\_My| 
|$M_x\otimes ( D^y_y -D^yM^{-1}_yD_y)I_y$  | Zy\_int\_Mx| 
