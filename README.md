# torch-pbm

A lightweight PyTorch implementation of the **Parallel Multi-Stepsize Proximal Bundle Method (PBM)** for nonsmooth convex optimization.

This repository provides a GPU-compatible, batched implementation of PBM and its variants, designed for experimentation, research, and benchmarking.

---

## Features

- Parallel multi-stepsize PBM (Alg2-style)
- Two-cut proximal bundle model
- Batched PyTorch implementation
- GPU acceleration support
- Modular oracle interface for custom objectives
- Example implementations:
  - Quadratic objective
  - Quadratic + L1 (nonsmooth)

---

## Installation

Clone the repository and install locally:

```bash
git clone https://github.com/nicoleli04/torch-pbm.git
cd torch-pbm
pip install -e .
```
---

## Quick Start 

import torch
from torch_pbm import ParallelPBM, QuadraticOracle

device = "cuda" if torch.cuda.is_available() else "cpu"

# Problem setup
d = 1000
Q = torch.randn(d, d, device=device)
Q = Q.T @ Q  # make PSD
x0 = torch.randn(d, device=device)

oracle = QuadraticOracle(Q)

# Solver
solver = ParallelPBM(
    rho_bar=15.0,
    num_instances=5,
    beta=0.75,
    m=0.0,
)

result = solver.solve(x0, oracle, max_iter=1000)

print("Final objective:", result.best_values[-1])

---

## Method Overview

We solve problems of the form:

[
\min_x f(x)
]

where ( f ) is convex and possibly nonsmooth.

PBM builds a piecewise linear model of ( f ) using subgradients and solves:

[
\min_y \max_i { f_i + \langle v_i, y - z_i \rangle } + \frac{\rho}{2} |y - x_k|^2
]

This implementation uses a two-cut approximation, enabling an efficient closed-form solution.


---

## Parallel PBM

We run multiple instances with different proximal parameters:

[
\rho_j = \rho_{\text{bar}} \cdot 2^j
]

At each iteration:
	•	All instances propose candidates
	•	The best candidate is selected
	•	Models are updated (descent / null step)

---

## Example Objectives

Quadratic

[
f(x) = \frac{1}{2} x^T Q x
]

Quadratic + L1

[
f(x) = \frac{1}{2} x^T Q x + \lambda |x|_1
]

Where ( Q = A^T A ), ensuring convexity.

---

## Creating Objectives

To use your own function, define an oracle:

```
class MyOracle:
    def f_batch(self, X):
        # X: (d, J)
        return ...

    def g_batch(self, X):
        # X: (d, J)
        return ...
```
Then call:
```
solver.solve(x0, oracle)
```

---

## Running Examples

```python examples/quadratic_demo.py```
This will generate:
	•	Objective convergence plots
	•	Step-type statistics
	•	Multi-rho selection behavior






