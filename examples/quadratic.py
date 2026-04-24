import os
import torch
import matplotlib.pyplot as plt

from torch_pbm import ParallelPBM, QuadraticOracle, QuadraticL1Oracle
from torch_pbm.utils import make_psd_matrix, time_gpu


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    os.makedirs("plots", exist_ok=True)

    d = 2000
    max_iter = 10000
    seed = 0

    Q = make_psd_matrix(d, device=device, seed=seed)

    # Choose one:
    oracle = QuadraticOracle(Q)
    # oracle = QuadraticL1Oracle(Q, lam=1e-2)

    torch.manual_seed(seed)
    x0 = torch.randn(d, device=device, dtype=torch.float32)

    solver = ParallelPBM(
        rho_bar=15.0,
        num_instances=5,
        beta=0.75,
        m=0.0,
    )

    time_ms, result = time_gpu(
        lambda: solver.solve(x0=x0, oracle=oracle, max_iter=max_iter)
    )

    print(f"Runtime: {time_ms:.1f} ms")
    print(f"Final best value: {result.best_values[-1]:.6g}")
    print("Rho grid:", result.rho_grid.tolist())
    print("First 20 selected rho:", result.selected_rhos[:20])

    counts = {}
    for j in result.selected_indices:
        counts[j] = counts.get(j, 0) + 1

    print("\nSelection frequency:")
    for j, rho in enumerate(result.rho_grid):
        print(f"j={j}, rho={rho.item():g}: {counts.get(j, 0)} times")

    # For QuadraticOracle and QuadraticL1Oracle with Q = A^T A,
    # the optimal value is f* = 0 at x* = 0.
    f_star = 0.0
    eps_gap = 1e-12

    xs = range(1, max_iter + 1)

    plt.figure(figsize=(9, 6))

    best_gap = [max(v - f_star, eps_gap) for v in result.best_values]
    plt.semilogy(xs, best_gap, linewidth=2, label="best gap")

    for j in range(solver.num_instances):
        curve_gap = [
            max(result.all_values[k][j].item() - f_star, eps_gap)
            for k in range(max_iter)
        ]
        plt.semilogy(
            xs,
            curve_gap,
            linestyle="--",
            alpha=0.8,
            label=f"rho={result.rho_grid[j].item():g}",
        )

    plt.xlabel("Iteration")
    plt.ylabel("Cost gap f(x_k) - f*")
    plt.title("Parallel Proximal Bundle Method")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/cost_gap.png", dpi=250)
    print("Saved plots/cost_gap.png")

    plt.figure(figsize=(9, 5))
    plt.plot(xs, result.descent_fractions, label="descent fraction")
    plt.plot(xs, result.null_fractions, label="null fraction")
    plt.xlabel("Iteration")
    plt.ylabel("Fraction")
    plt.title("Step Type Fractions")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/step_fractions.png", dpi=250)
    print("Saved plots/step_fractions.png")

    plt.figure(figsize=(9, 4))
    plt.semilogy(xs, result.selected_rhos)
    plt.xlabel("Iteration")
    plt.ylabel("Selected rho")
    plt.title("Selected rho over time")
    plt.tight_layout()
    plt.savefig("plots/selected_rho.png", dpi=250)
    print("Saved plots/selected_rho.png")


if __name__ == "__main__":
    main()