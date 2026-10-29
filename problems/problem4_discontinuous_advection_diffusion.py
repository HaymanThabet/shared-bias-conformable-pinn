"""
Discontinuous-coefficient advection--diffusion benchmark used in Section 4.5.

The script compares a standard PINN with the shared-bias formulation under
the settings reported in the manuscript.  The exact initial condition is
embedded in both trial solutions, and the coefficient jumps are excluded
from the independent residual check.  Figures are written relative to the
script directory.
"""

import os
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    torch.set_num_threads(1)
except Exception:
    pass

# Figure quality
plt.rcParams["figure.dpi"] = 180
# Rasterized artists embedded in PDF are limited to 300 dpi.
# Text, axes, labels, legends, and line plots remain vector graphics.
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["pdf.compression"] = 9
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["lines.linewidth"] = 2.2

COLOR_STD = "#1f3b73"
COLOR_SB = "#d95f02"
COLOR_EX = "#2e7d32"

# Reproducibility
device = "cpu"
dtype = torch.float64
torch.set_default_dtype(dtype)

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

# ============================================================
# Discontinuous-coefficient time-conformable
# advection--diffusion problem
#
# T_q u + a(x) u_x - nu u_xx = f(x,t)
# u_ex(x,t) = exp(-t) sin(omega x)
# ============================================================
q_frac = 0.7

# Frequency used for the manuscript experiment.
omega = 5.0

if omega not in {5.0, 10.0, 20.0, 30.0}:
    raise ValueError("omega must be 5.0, 10.0, 20.0, or 30.0.")

omega_tag = str(int(omega))
nu = 0.01 / math.pi

x_left = 0.0
x_right = 2.0 * math.pi
T_final = 1.0

# Output folder
script_dir = Path(__file__).resolve().parent
output_dir = (
    script_dir
    / "Figs"
    / "Ex4_Discontinuous_Advection_Diffusion"
    / f"omega_{omega_tag}"
)
output_dir.mkdir(parents=True, exist_ok=True)

# Exact solution and forcing
def exact_solution(x, t):
    return torch.exp(-t) * torch.sin(omega * x)


# Periodic discontinuous advection coefficient.
# The endpoint values agree, so the coefficient is compatible
# with the periodic spatial domain.
a_outer = 0.5
a_middle = 2.0
x_jump_1 = 0.5 * math.pi
x_jump_2 = 1.5 * math.pi


def advection_coefficient(x):
    outer_region = (x < x_jump_1) | (x > x_jump_2)
    return torch.where(
        outer_region,
        torch.full_like(x, a_outer),
        torch.full_like(x, a_middle),
    )


def forcing_term(x, t, q):
    e_t = torch.exp(-t)
    s = torch.sin(omega * x)
    c = torch.cos(omega * x)
    a = advection_coefficient(x)

    return (
        -t ** (1.0 - q) * e_t * s
        + a * omega * e_t * c
        + nu * omega**2 * e_t * s
    )


def u0(x):
    return torch.sin(omega * x)


# Feature maps
def input_features_std(x, t):
    return torch.cat([x, t], dim=1)


def periodic_features(x, t):
    return torch.cat(
        [torch.sin(omega * x), torch.cos(omega * x), t],
        dim=1,
    )


# Standard PINN
class StandardPINN(nn.Module):
    def __init__(self, input_dim=2, hid_dim=80, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, X):
        return self.fc2(torch.tanh(self.fc1(X)))


# Shared-bias network
class SharedBiasNet(nn.Module):
    def __init__(self, input_dim=3, hid_dim=80, output_dim=1):
        super().__init__()
        self.W1 = nn.Parameter(
            torch.randn(input_dim, hid_dim) * math.sqrt(1.0 / input_dim)
        )
        self.W2 = nn.Parameter(
            torch.randn(hid_dim, output_dim) * math.sqrt(1.0 / hid_dim)
        )
        self.b1 = nn.Parameter(
            torch.randn(1, 1) * math.sqrt(1.0 / input_dim)
        )
        self.b2 = nn.Parameter(
            torch.randn(1, output_dim) * math.sqrt(1.0 / output_dim)
        )

    def forward(self, X):
        Z = X @ self.W1 + self.b1
        H = torch.tanh(Z)
        return H @ self.W2 + self.b2


# Trial solutions
def trial_solution_std(model, x, t):
    X = input_features_std(x, t)
    return u0(x) + t * model(X)


def trial_solution_sb(model, x, t):
    X = periodic_features(x, t)
    return u0(x) + t * model(X)


# Automatic differentiation
def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


# Residuals
def advection_diffusion_residual_std(model, x, t, q):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    u = trial_solution_std(model, x, t)
    u_t = grad(u, t)
    u_x = grad(u, x)
    u_xx = grad(u_x, x)

    Tq_u = t ** (1.0 - q) * u_t
    a = advection_coefficient(x)
    f = forcing_term(x, t, q)

    return Tq_u + a * u_x - nu * u_xx - f


def advection_diffusion_residual_sb(model, x, t, q):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    u = trial_solution_sb(model, x, t)
    u_t = grad(u, t)
    u_x = grad(u, x)
    u_xx = grad(u_x, x)

    Tq_u = t ** (1.0 - q) * u_t
    a = advection_coefficient(x)
    f = forcing_term(x, t, q)

    return Tq_u + a * u_x - nu * u_xx - f


# Initial losses
def initial_loss_std(model, x0):
    t0 = torch.zeros_like(x0)
    return torch.mean(
        (trial_solution_std(model, x0, t0) - u0(x0)) ** 2
    )


def initial_loss_sb(model, x0):
    t0 = torch.zeros_like(x0)
    return torch.mean(
        (trial_solution_sb(model, x0, t0) - u0(x0)) ** 2
    )


# Periodic boundary losses
def periodic_bc_loss_std(model, t_b):
    xL = torch.full_like(t_b, x_left, requires_grad=True)
    xR = torch.full_like(t_b, x_right, requires_grad=True)

    tL = t_b.clone().detach().requires_grad_(True)
    tR = t_b.clone().detach().requires_grad_(True)

    uL = trial_solution_std(model, xL, tL)
    uR = trial_solution_std(model, xR, tR)

    uL_x = grad(uL, xL)
    uR_x = grad(uR, xR)

    return torch.mean((uL - uR) ** 2) + torch.mean((uL_x - uR_x) ** 2)


def periodic_bc_loss_sb(model, t_b):
    xL = torch.full_like(t_b, x_left, requires_grad=True)
    xR = torch.full_like(t_b, x_right, requires_grad=True)

    tL = t_b.clone().detach().requires_grad_(True)
    tR = t_b.clone().detach().requires_grad_(True)

    uL = trial_solution_sb(model, xL, tL)
    uR = trial_solution_sb(model, xR, tR)

    uL_x = grad(uL, xL)
    uR_x = grad(uR, xR)

    return torch.mean((uL - uR) ** 2) + torch.mean((uL_x - uR_x) ** 2)


# Total losses
def total_loss_std(model, x_f, t_f, x0, t_b, q):
    res = advection_diffusion_residual_std(model, x_f, t_f, q)
    loss_pde = torch.mean(res**2)
    loss_ic = initial_loss_std(model, x0)
    loss_bc = periodic_bc_loss_std(model, t_b)
    loss = loss_pde + loss_ic + loss_bc
    return loss, loss_pde, loss_ic, loss_bc


def total_loss_sb(model, x_f, t_f, x0, t_b, q):
    res = advection_diffusion_residual_sb(model, x_f, t_f, q)
    loss_pde = torch.mean(res**2)
    loss_ic = initial_loss_sb(model, x0)
    loss_bc = periodic_bc_loss_sb(model, t_b)
    loss = loss_pde + loss_ic + loss_bc
    return loss, loss_pde, loss_ic, loss_bc


# Conformable-gradient schedule and update
def lr_alpha(i):
    return 1.5 - 2.0 * (
        math.atan(i) + math.pi / 2.0
    ) / math.pi / 2.0


def LR(i, LR0=1e-2):
    a = max(lr_alpha(i), 1e-8)
    return LR0 * a


@torch.no_grad()
def Conformable_fractional_variable_gradient_PINN(
    model, i, LR0=1e-2
):
    alpha_opt = max(lr_alpha(i), 1.0e-14)
    Lr = LR(i, LR0=LR0)

    # Blockwise conformable-gradient scaling: each trainable block
    # is scaled by its own Euclidean norm, consistent with the
    # formulation stated in the manuscript.
    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        parameter_norm = torch.norm(parameter).clamp_min(1.0e-14)
        scaled_gradient = (
            parameter_norm ** alpha_opt
        ) * parameter.grad / alpha_opt

        parameter -= Lr * scaled_gradient


# Training settings
hid_dim = 80
num_f = 200
num_x0 = 100
num_tb = 100
num_iters = 10000
LR0 = 1e-2
clip_value = 10.0

adam_iters = num_iters
adam_lr = 1e-3
eps_t = 1e-4

# Shared-bias initialization and sampling.
torch.manual_seed(SEED)
np.random.seed(SEED)

shared_model = SharedBiasNet(
    input_dim=3,
    hid_dim=hid_dim,
    output_dim=1,
).to(device)

t_f = eps_t + (
    T_final - eps_t
) * torch.rand(num_f, 1, device=device)

x_f = x_left + (
    x_right - x_left
) * torch.rand(num_f, 1, device=device)

x0 = torch.linspace(
    x_left,
    x_right,
    num_x0,
    device=device,
).reshape(-1, 1)

t_b = torch.linspace(
    eps_t,
    T_final,
    num_tb,
    device=device,
).reshape(-1, 1)

# Standard-PINN initialization.
torch.manual_seed(SEED)
standard_model = StandardPINN(
    input_dim=2,
    hid_dim=hid_dim,
    output_dim=1,
).to(device)


# ============================================================
# Manufactured-problem check (away from the coefficient jumps)
# ============================================================
def verify_manufactured_problem():
    x_verify = torch.tensor(
        [
            0.23,
            0.91,
            1.40,
            1.80,
            2.80,
            4.40,
            5.10,
            5.95,
        ],
        dtype=dtype,
        device=device,
    ).reshape(-1, 1)
    t_verify = torch.tensor(
        [
            0.05,
            0.14,
            0.26,
            0.39,
            0.53,
            0.67,
            0.82,
            0.96,
        ],
        dtype=dtype,
        device=device,
    ).reshape(-1, 1)

    x = x_verify.clone().detach().requires_grad_(True)
    t = t_verify.clone().detach().requires_grad_(True)

    u = exact_solution(x, t)
    u_t = grad(u, t)
    u_x = grad(u, x)
    u_xx = grad(u_x, x)

    residual = (
        t ** (1.0 - q_frac) * u_t
        + advection_coefficient(x) * u_x
        - nu * u_xx
        - forcing_term(x, t, q_frac)
    )

    x_ic = torch.linspace(
        x_left,
        x_right,
        101,
        device=device,
    ).reshape(-1, 1)
    t_zero = torch.zeros_like(x_ic)
    ic_error = torch.max(
        torch.abs(
            exact_solution(x_ic, t_zero) - u0(x_ic)
        )
    )

    t_bc_v = torch.linspace(
        0.0,
        T_final,
        51,
        device=device,
    ).reshape(-1, 1)

    xL = torch.full_like(
        t_bc_v,
        x_left,
        requires_grad=True,
    )
    xR = torch.full_like(
        t_bc_v,
        x_right,
        requires_grad=True,
    )
    tL = t_bc_v.clone().detach().requires_grad_(True)
    tR = t_bc_v.clone().detach().requires_grad_(True)

    uL = exact_solution(xL, tL)
    uR = exact_solution(xR, tR)
    uL_x = grad(uL, xL)
    uR_x = grad(uR, xR)

    bc_u_error = torch.max(torch.abs(uL - uR))
    bc_ux_error = torch.max(torch.abs(uL_x - uR_x))

    a0 = advection_coefficient(
        torch.tensor([[0.0]], dtype=dtype, device=device)
    ).item()
    a1m = advection_coefficient(
        torch.tensor([[x_jump_1 - 1.0e-8]], dtype=dtype, device=device)
    ).item()
    a1p = advection_coefficient(
        torch.tensor([[x_jump_1 + 1.0e-8]], dtype=dtype, device=device)
    ).item()
    a2m = advection_coefficient(
        torch.tensor([[x_jump_2 - 1.0e-8]], dtype=dtype, device=device)
    ).item()
    a2p = advection_coefficient(
        torch.tensor([[x_jump_2 + 1.0e-8]], dtype=dtype, device=device)
    ).item()
    a2pi = advection_coefficient(
        torch.tensor([[2.0 * math.pi]], dtype=dtype, device=device)
    ).item()

    print(
        "\nManufactured-problem verification"
        f"\n  max PDE residual = {torch.max(torch.abs(residual)).item():.3e}"
        f"\n  max IC error     = {ic_error.item():.3e}"
        f"\n  periodic u error = {bc_u_error.item():.3e}"
        f"\n  periodic ux error= {bc_ux_error.item():.3e}"
        f"\n  a(0) / a(2pi)    = {a0:.1f} / {a2pi:.1f}"
        f"\n  jump at pi/2     = {a1m:.1f} -> {a1p:.1f}"
        f"\n  jump at 3pi/2    = {a2m:.1f} -> {a2p:.1f}"
    )


verify_manufactured_problem()


# Train standard PINN
optimizer_std = torch.optim.Adam(
    standard_model.parameters(),
    lr=adam_lr,
)

Loss_std = []
Loss_pde_std = []
Loss_ic_std = []
Loss_bc_std = []

start_std = time.time()

for i in range(adam_iters + 1):
    optimizer_std.zero_grad(set_to_none=True)

    loss, loss_pde, loss_ic, loss_bc = total_loss_std(
        standard_model,
        x_f,
        t_f,
        x0,
        t_b,
        q_frac,
    )

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Standard PINN stopped at iteration {i}: invalid loss."
        )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        standard_model.parameters(),
        max_norm=clip_value,
    )
    optimizer_std.step()

    Loss_std.append(loss.item())
    Loss_pde_std.append(loss_pde.item())
    Loss_ic_std.append(loss_ic.item())
    Loss_bc_std.append(loss_bc.item())

    if i % 200 == 0:
        print(
            f"[Standard PINN] Iteration: {i:5d} | "
            f"Loss: {loss.item():.6e} | "
            f"PDE: {loss_pde.item():.6e} | "
            f"IC: {loss_ic.item():.6e} | "
            f"BC: {loss_bc.item():.6e}"
        )

elapsed_std = time.time() - start_std


# Train shared-bias PINN
Loss_sb = []
Loss_pde_sb = []
Loss_ic_sb = []
Loss_bc_sb = []

start_sb = time.time()

for i in range(num_iters + 1):
    for p in shared_model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    loss, loss_pde, loss_ic, loss_bc = total_loss_sb(
        shared_model,
        x_f,
        t_f,
        x0,
        t_b,
        q_frac,
    )

    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Shared-bias PINN stopped at iteration {i}: invalid loss."
        )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        shared_model.parameters(),
        max_norm=clip_value,
    )
    Conformable_fractional_variable_gradient_PINN(
        shared_model,
        i,
        LR0=LR0,
    )

    Loss_sb.append(loss.item())
    Loss_pde_sb.append(loss_pde.item())
    Loss_ic_sb.append(loss_ic.item())
    Loss_bc_sb.append(loss_bc.item())

    if i % 200 == 0:
        print(
            f"[Shared-bias PINN] Iteration: {i:5d} | "
            f"Loss: {loss.item():.6e} | "
            f"PDE: {loss_pde.item():.6e} | "
            f"IC: {loss_ic.item():.6e} | "
            f"BC: {loss_bc.item():.6e} | "
            f"alpha_opt: {lr_alpha(i):.6e} | "
            f"LR: {LR(i, LR0):.6e}"
        )

elapsed_sb = time.time() - start_sb


# Evaluation
@torch.no_grad()
def evaluate_model(model, trial_function, nx=256, nt=201):
    xg = torch.linspace(x_left, x_right, nx, device=device)
    tg = torch.linspace(0.0, T_final, nt, device=device)

    X, T = torch.meshgrid(xg, tg, indexing="ij")
    x_flat = X.reshape(-1, 1)
    t_flat = T.reshape(-1, 1)

    U_pred = trial_function(
        model,
        x_flat,
        t_flat,
    ).reshape(nx, nt)

    U_exact = exact_solution(
        x_flat,
        t_flat,
    ).reshape(nx, nt)

    error = U_pred - U_exact

    E_inf = torch.max(torch.abs(error))
    E_2 = torch.sqrt(torch.mean(error**2))
    E_2_rel = torch.norm(error) / torch.norm(U_exact)

    return (
        xg.cpu().numpy(),
        tg.cpu().numpy(),
        U_pred.cpu().numpy(),
        U_exact.cpu().numpy(),
        E_inf.item(),
        E_2.item(),
        E_2_rel.item(),
    )


def recompute_final_loss_std():
    loss, _, _, _ = total_loss_std(
        standard_model,
        x_f,
        t_f,
        x0,
        t_b,
        q_frac,
    )
    return loss.detach().item()


def recompute_final_loss_sb():
    loss, _, _, _ = total_loss_sb(
        shared_model,
        x_f,
        t_f,
        x0,
        t_b,
        q_frac,
    )
    return loss.detach().item()


(
    xg,
    tg,
    U_std,
    U_exact,
    E_inf_std,
    E_2_std,
    E_2_rel_std,
) = evaluate_model(
    standard_model,
    trial_solution_std,
    nx=256,
    nt=201,
)

(
    _,
    _,
    U_sb,
    _,
    E_inf_sb,
    E_2_sb,
    E_2_rel_sb,
) = evaluate_model(
    shared_model,
    trial_solution_sb,
    nx=256,
    nt=201,
)

(
    xg_s,
    tg_s,
    U_std_s,
    U_exact_s,
    _,
    _,
    _,
) = evaluate_model(
    standard_model,
    trial_solution_std,
    nx=500,
    nt=260,
)

(
    _,
    _,
    U_sb_s,
    _,
    _,
    _,
    _,
) = evaluate_model(
    shared_model,
    trial_solution_sb,
    nx=500,
    nt=260,
)

final_loss_std = recompute_final_loss_std()
final_loss_sb = recompute_final_loss_sb()


# ============================================================
# Independent residual and convergence diagnostics
# These are post-training diagnostics and do not affect either optimizer.
# ============================================================
def independent_residual_rms(
    model,
    residual_function,
    nx=257,
    nt=201,
):
    xg_r = torch.linspace(
        x_left,
        x_right,
        nx,
        device=device,
    )
    tg_r = torch.linspace(
        eps_t,
        T_final,
        nt,
        device=device,
    )

    # Exclude the two coefficient-jump locations from the
    # deterministic residual grid.
    delta = 1.0e-10
    mask = (
        (torch.abs(xg_r - x_jump_1) > delta)
        & (torch.abs(xg_r - x_jump_2) > delta)
    )
    xg_r = xg_r[mask]

    Xr, Tr = torch.meshgrid(
        xg_r,
        tg_r,
        indexing="ij",
    )
    xr = Xr.reshape(-1, 1)
    tr = Tr.reshape(-1, 1)

    residual = residual_function(
        model,
        xr,
        tr,
        q_frac,
    )

    return torch.sqrt(
        torch.mean(residual**2)
    ).detach().item()


def first_iteration_below(loss_history, tolerance=1.0e-3):
    for iteration, value in enumerate(loss_history):
        if value <= tolerance:
            return iteration
    return None


E_r_std = independent_residual_rms(
    standard_model,
    advection_diffusion_residual_std,
)
E_r_sb = independent_residual_rms(
    shared_model,
    advection_diffusion_residual_sb,
)

N_tol_std = first_iteration_below(
    Loss_std,
    tolerance=1.0e-3,
)
N_tol_sb = first_iteration_below(
    Loss_sb,
    tolerance=1.0e-3,
)


# Results tables
results = [
    (
        "Standard PINN + Adam",
        E_inf_std,
        E_2_std,
        E_2_rel_std,
        E_r_std,
        final_loss_std,
        N_tol_std,
        elapsed_std,
    ),
    (
        "Shared-bias PINN + conformable GD",
        E_inf_sb,
        E_2_sb,
        E_2_rel_sb,
        E_r_sb,
        final_loss_sb,
        N_tol_sb,
        elapsed_sb,
    ),
]

print(
    f"\nDiscontinuous-coefficient advection--diffusion "
    f"comparison for omega = {omega_tag}:\n"
)

header = (
    f"{'Method':<38}"
    f"{'E_inf':>15}"
    f"{'E_2':>15}"
    f"{'E_2_rel':>15}"
    f"{'E_r':>15}"
    f"{'Final loss':>15}"
    f"{'N_tol':>10}"
    f"{'CPU time (s)':>15}"
)
print(header)
print("-" * len(header))

for (
    method,
    e_inf,
    e_2,
    e_2_rel,
    e_r,
    final_loss,
    n_tol,
    cpu,
) in results:
    n_tol_text = str(n_tol) if n_tol is not None else ">10000"
    print(
        f"{method:<38}"
        f"{e_inf:15.6e}"
        f"{e_2:15.6e}"
        f"{e_2_rel:15.6e}"
        f"{e_r:15.6e}"
        f"{final_loss:15.6e}"
        f"{n_tol_text:>10}"
        f"{cpu:15.3f}"
    )

print("\nLaTeX table:\n")
print(r"\begin{table}[htbp]")
print(r"\centering")
print(
    r"\caption{Accuracy, independent residual, convergence, "
    r"and CPU-time comparison for the discontinuous-coefficient "
    rf"time-conformable advection--diffusion problem with $\omega={omega_tag}$."
    r"}"
)
print(
    rf"\label{{tab:discontinuous_advection_diffusion_omega{omega_tag}}}"
)
print(r"\scriptsize")
print(r"\begin{tabular}{lccccccc}")
print(r"\hline")
print(
    r"Method & $E_{\infty}$ & $E_2$ & $E_{2,\mathrm{rel}}$ & "
    r"$E_r$ & Final loss & $N_{\mathrm{tol}}$ & CPU time (s) \\"
)
print(r"\hline")

for (
    method,
    e_inf,
    e_2,
    e_2_rel,
    e_r,
    final_loss,
    n_tol,
    cpu,
) in results:
    n_tol_text = str(n_tol) if n_tol is not None else r"$>10{,}000$"
    print(
        f"{method} & {e_inf:.6e} & {e_2:.6e} & "
        f"{e_2_rel:.6e} & {e_r:.6e} & {final_loss:.6e} & "
        f"{n_tol_text} & {cpu:.3f} \\\\"
    )

print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")


# PDF figures
plt.close("all")


def save_pdf(fig, stem):
    """Save a compact journal-quality PDF.

    Rasterized artists are stored at 300 dpi, while text, axes,
    legends, and ordinary line plots remain vector based.
    """
    fig.savefig(
        output_dir / f"{stem}.pdf",
        format="pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
        metadata={
            "Creator": "Matplotlib",
            "Producer": "Matplotlib PDF backend",
        },
    )


# Combined loss histories
fig1, ax1 = plt.subplots(figsize=(7.6, 4.4), dpi=240)
ax1.semilogy(
    Loss_std,
    color=COLOR_STD,
    label="Standard PINN",
)
ax1.semilogy(
    Loss_sb,
    "--",
    color=COLOR_SB,
    label="Shared-bias PINN",
)
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Training loss")
ax1.set_title(rf"Training-loss comparison, $\omega={omega_tag}$")
ax1.set_xlim(left=0)
ax1.margins(x=0)
ax1.legend()
fig1.tight_layout()
save_pdf(fig1, f"discontinuous_advection_diffusion_loss_comparison_omega{omega_tag}")
plt.close(fig1)

# Exact, standard-PINN, and shared-bias surfaces
X_plot_s, T_plot_s = np.meshgrid(
    xg_s,
    tg_s,
    indexing="ij",
)

fig2 = plt.figure(
    figsize=(20.0, 6.0),
    dpi=240,
    constrained_layout=True,
)

surface_data = [
    ("Exact solution", U_exact_s),
    ("Standard PINN", U_std_s),
    ("Shared-bias PINN", U_sb_s),
]

for index, (title, values) in enumerate(surface_data, start=1):
    ax = fig2.add_subplot(1, 3, index, projection="3d")
    surf = ax.plot_surface(
        X_plot_s,
        T_plot_s,
        values,
        cmap="viridis",
        rcount=220,
        ccount=180,
        linewidth=0,
        edgecolor="none",
        antialiased=True,
        rasterized=True,
    )
    ax.set_xlabel("x", labelpad=10)
    ax.set_ylabel("t", labelpad=10)
    ax.set_zlabel("u(x,t)", labelpad=10)
    ax.set_title(rf"{title}, $\omega={omega_tag}$", pad=12)
    ax.view_init(elev=28, azim=-135)
    fig2.colorbar(surf, ax=ax, shrink=0.73, pad=0.08)

save_pdf(fig2, f"discontinuous_advection_diffusion_solution_surfaces_omega{omega_tag}")
plt.close(fig2)

# Three-dimensional absolute errors
fig3 = plt.figure(
    figsize=(14.0, 6.0),
    dpi=240,
    constrained_layout=True,
)

error_surface_data = [
    ("Standard PINN error", np.abs(U_std_s - U_exact_s)),
    ("Shared-bias PINN error", np.abs(U_sb_s - U_exact_s)),
]

for index, (title, values) in enumerate(
    error_surface_data,
    start=1,
):
    ax = fig3.add_subplot(1, 2, index, projection="3d")
    surf = ax.plot_surface(
        X_plot_s,
        T_plot_s,
        values,
        cmap="magma",
        rcount=220,
        ccount=180,
        linewidth=0,
        edgecolor="none",
        antialiased=True,
        rasterized=True,
    )
    ax.set_xlabel("x", labelpad=10)
    ax.set_ylabel("t", labelpad=10)
    ax.set_zlabel("|error|", labelpad=10)
    ax.set_title(rf"{title}, $\omega={omega_tag}$", pad=12)
    ax.view_init(elev=28, azim=-135)
    fig3.colorbar(surf, ax=ax, shrink=0.75, pad=0.08)

save_pdf(fig3, f"discontinuous_advection_diffusion_error_surfaces_omega{omega_tag}")
plt.close(fig3)

# Final-time comparison
fig4, ax4 = plt.subplots(figsize=(7.6, 4.4), dpi=240)
ax4.plot(
    xg,
    U_exact[:, -1],
    color=COLOR_EX,
    label="Exact",
)
ax4.plot(
    xg,
    U_std[:, -1],
    "--",
    color=COLOR_STD,
    label="Standard PINN",
)
ax4.plot(
    xg,
    U_sb[:, -1],
    "-.",
    color=COLOR_SB,
    label="Shared-bias PINN",
)
ax4.set_xlabel("x")
ax4.set_ylabel("u(x,1)")
ax4.set_title(rf"Final-time comparison at $t=1$, $\omega={omega_tag}$")
ax4.legend()
fig4.tight_layout()
save_pdf(fig4, f"discontinuous_advection_diffusion_final_time_omega{omega_tag}")
plt.close(fig4)

# Absolute-error heatmaps
fig5, axes = plt.subplots(
    1,
    2,
    figsize=(11.5, 4.8),
    dpi=240,
    constrained_layout=True,
)

heatmap_data = [
    ("Standard PINN error", np.abs(U_std - U_exact)),
    ("Shared-bias PINN error", np.abs(U_sb - U_exact)),
]

for ax, (title, values) in zip(axes, heatmap_data):
    im = ax.imshow(
        values.T,
        extent=[x_left, x_right, 0.0, T_final],
        origin="lower",
        aspect="auto",
        cmap="magma",
        rasterized=True,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(rf"{title}, $\omega={omega_tag}$")
    fig5.colorbar(im, ax=ax, shrink=0.9)

save_pdf(fig5, f"discontinuous_advection_diffusion_error_heatmaps_omega{omega_tag}")
plt.close(fig5)

print(f"\nPDF figures saved in:\n{output_dir}")
