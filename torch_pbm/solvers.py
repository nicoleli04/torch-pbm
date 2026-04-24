from dataclasses import dataclass
import torch


@dataclass
class PBMResult:
    x_final: torch.Tensor
    best_values: list
    all_values: list
    descent_fractions: list
    null_fractions: list
    selected_indices: list
    selected_rhos: list
    rho_grid: torch.Tensor


class ParallelPBM:
    """
    This implementation runs multiple PBM instances in parallel with different
    constant rho values.

    Uses a two cut model:
        m(y) = max{
            f1 + <v1, y - z_base>,
            f2 + <v2, y - z_base>
        }.
    """

    def __init__(self, rho_bar=15.0, num_instances=5, beta=0.75, m=0.0):
        self.rho_bar = rho_bar
        self.num_instances = num_instances
        self.beta = beta
        self.m = m

    @staticmethod
    @torch.no_grad()
    def analytic_sol_batch(Xk, f1, f2, v1, v2, rho):
        diff = v1 - v2
        denom = torch.sum(diff * diff, dim=0) + 1e-12

        raw = rho * (f2 - f1) / denom
        theta = torch.minimum(torch.ones_like(raw), raw)

        s = (1.0 - theta) * v1 + theta * v2
        z = Xk - s / rho

        return z, s

    @staticmethod
    def model_eval(z_base, f1, f2, v1, v2, Y):
        dY = Y - z_base
        return torch.maximum(
            f1 + torch.sum(v1 * dY, dim=0),
            f2 + torch.sum(v2 * dY, dim=0),
        )

    @torch.no_grad()
    def _rho_grid(self, device, dtype):
        exponents = torch.arange(
            self.num_instances,
            device=device,
            dtype=dtype,
        )
        return self.rho_bar * (2.0 ** exponents)

    @torch.no_grad()
    def _init_state(self, x0, rho, oracle):
        Xk = x0[:, None].repeat(1, self.num_instances)

        g0 = oracle.g_batch(Xk)
        f0 = oracle.f_batch(Xk)

        Z0 = Xk - g0 / rho
        f_lin_at_Z0 = f0 + torch.sum(g0 * (Z0 - Xk), dim=0)

        z_base = Z0.clone()
        f1 = f_lin_at_Z0.clone()
        f2 = f_lin_at_Z0.clone()
        v1 = g0.clone()
        v2 = g0.clone()

        return Xk, z_base, f1, f2, v1, v2

    @torch.no_grad()
    def _step(self, Xk, z_base, f1, f2, v1, v2, rho, oracle):
        beta_t = torch.as_tensor(self.beta, device=Xk.device, dtype=Xk.dtype)
        m_t = torch.as_tensor(self.m, device=Xk.device, dtype=Xk.dtype)

        fxk = oracle.f_batch(Xk)

        j_star = torch.argmin(fxk)
        x_star = Xk[:, j_star:j_star + 1]
        fx_star = fxk[j_star]

        Z, S = self.analytic_sol_batch(Xk, f1, f2, v1, v2, rho)
        fkZ = self.model_eval(z_base, f1, f2, v1, v2, Z)

        fz = oracle.f_batch(Z)
        phiZ = fz + 0.5 * m_t * torch.sum((Z - Xk) ** 2, dim=0)

        threshold = fxk - fkZ
        truegap = fxk - phiZ

        descent = truegap >= beta_t * threshold
        nullstep = ~descent

        good_descent = descent & (fz <= fx_star)
        reset_to_best = descent & (fx_star < fz)

        Xnext = Xk.clone()

        if bool(good_descent.any()):
            Xnext[:, good_descent] = Z[:, good_descent]

        if bool(reset_to_best.any()):
            n_reset = int(reset_to_best.sum().item())
            Xnext[:, reset_to_best] = x_star.expand(-1, n_reset)

        # Null step: keep center, refine two-cut model.
        if bool(nullstep.any()):
            gz = oracle.g_batch(Z)
            new_v = gz + m_t * (Z - Xk)

            z_base[:, nullstep] = Z[:, nullstep]
            f1[nullstep] = fkZ[nullstep]
            f2[nullstep] = phiZ[nullstep]
            v1[:, nullstep] = S[:, nullstep]
            v2[:, nullstep] = new_v[:, nullstep]

        # Good descent: move center, restart with fresh one-cut model.
        if bool(good_descent.any()):
            Xgd = Xnext[:, good_descent]
            gx = oracle.g_batch(Xgd)
            rho_gd = rho[good_descent]

            Zgd = Xgd - gx / rho_gd
            fx = oracle.f_batch(Xgd)
            f_lin_at_Zgd = fx + torch.sum(gx * (Zgd - Xgd), dim=0)

            z_base[:, good_descent] = Zgd
            f1[good_descent] = f_lin_at_Zgd
            f2[good_descent] = f_lin_at_Zgd
            v1[:, good_descent] = gx
            v2[:, good_descent] = gx

        # Communication/reset: if another instance already has a better center,
        # reset this instance to that center and restart its model.
        if bool(reset_to_best.any()):
            Xreset = Xnext[:, reset_to_best]
            gxreset = oracle.g_batch(Xreset)
            rho_reset = rho[reset_to_best]

            Zreset = Xreset - gxreset / rho_reset
            fxreset = oracle.f_batch(Xreset)
            f_lin_at_Zreset = fxreset + torch.sum(
                gxreset * (Zreset - Xreset),
                dim=0,
            )

            z_base[:, reset_to_best] = Zreset
            f1[reset_to_best] = f_lin_at_Zreset
            f2[reset_to_best] = f_lin_at_Zreset
            v1[:, reset_to_best] = gxreset
            v2[:, reset_to_best] = gxreset

        return Xnext, z_base, f1, f2, v1, v2, descent, nullstep, j_star

    @torch.no_grad()
    def solve(self, x0, oracle, max_iter=10000):
        """
        Run the parallel PBM solver.

        Parameters
        ----------
        x0:
            Initial point, shape (d,).
        oracle:
            Objective oracle with f_batch and g_batch methods.
        max_iter:
            Number of outer iterations.

        Returns
        -------
        PBMResult
        """
        if x0.ndim != 1:
            raise ValueError("x0 must have shape (d,).")

        rho = self._rho_grid(device=x0.device, dtype=x0.dtype)
        Xk, z_base, f1, f2, v1, v2 = self._init_state(x0, rho, oracle)

        best_values = []
        all_values = []
        descent_fractions = []
        null_fractions = []
        selected_indices = []
        selected_rhos = []

        for _ in range(max_iter):
            Xk, z_base, f1, f2, v1, v2, descent, nullstep, j_star = self._step(
                Xk, z_base, f1, f2, v1, v2, rho, oracle
            )

            fxk = oracle.f_batch(Xk)

            best_values.append(float(fxk.min().item()))
            all_values.append(fxk.detach().cpu())
            descent_fractions.append(float(descent.float().mean().item()))
            null_fractions.append(float(nullstep.float().mean().item()))
            selected_indices.append(int(j_star.item()))
            selected_rhos.append(float(rho[j_star].item()))

        return PBMResult(
            x_final=Xk,
            best_values=best_values,
            all_values=all_values,
            descent_fractions=descent_fractions,
            null_fractions=null_fractions,
            selected_indices=selected_indices,
            selected_rhos=selected_rhos,
            rho_grid=rho.detach().cpu(),
        )