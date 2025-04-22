# Quark Propagator via Dyson–Schwinger Equation

**Solution of the quark Dyson-Schwinger equation (DSE)** done as a project at **ECT\* Doctoral Training Program, May 2–18, 2022**, *Hadron Physics with Functional Methods*.

Among the tasks, we found the quark mass \( M(p^2) \) for different flavours (see Fig. below).  
For further detail, see the PDF file in the repo.

![Quark mass function for different flavors](./Quark_mass.png)

---

## Description

This project numerically solves the Dyson–Schwinger equation (DSE) for the quark propagator in QCD using non-perturbative methods. The DSE is treated in the Landau gauge under the rainbow-ladder truncation, and includes an effective Maris-Tandy interaction.

Key features:
- Real and complex \( p^2 \) support
- 3D visualization of dressing functions
- Numerical integration with Gauss–Legendre and Gauss–Chebyshev quadratures

---

## Theoretical Background

The quark propagator is written as:
\[
S^{-1}(p) = i \!\not\!p \, A(p^2) + B(p^2) \equiv A(p^2)(i\!\not\!p + M(p^2))
\]
where \( A(p^2) \) and \( M(p^2) = B(p^2)/A(p^2) \) are the dressing functions.  
These are extracted from the Dyson–Schwinger equation with a specified model for the effective interaction.

See the accompanying PDF file for full derivation and explanation.

---

## Project Structure

| File | Description |
|------|-------------|
| `dressed_func_builder.py` | Main solver for the DSE in real domain |
| `Dressed_Function_Builder_3D.py` | Extension to complex \( p^2 \), builds 3D plots of \( \sigma_v \), \( \sigma_s \) |
| `module_integration.py` | Gauss–Legendre/Chebyshev integration routines, Legendre polynomial roots and weights |
| `module_spline.py` | Cubic spline interpolation used for dressing functions |
| `Result_*.txt` | Output files from simulations |
| `Quark_mass.png` | Plot of \( M(p^2=0) \) for various quark flavours |

---

## Installation & Requirements

```bash
git clone https://github.com/enricotrotti/quark_propagator.git
cd quark_propagator
pip install -r requirements.txt
