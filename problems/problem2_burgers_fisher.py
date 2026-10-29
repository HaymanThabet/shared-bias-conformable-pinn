"""Forward comparison for the manufactured Burgers--Fisher problem.

The script trains a standard PINN with Adam and a shared-bias PINN with
conformable-gradient updates. It reports the global errors, final loss, and
CPU time, and saves the figures used in the manuscript as PDF files.
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

# Figure output
script_dir = Path(__file__).resolve().parent
output_dir = script_dir / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# Plot settings
plt.rcParams["figure.dpi"] = 220
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

# Reproducibility and precision
device = "cpu"
dtype = torch.float64
torch.set_default_dtype(dtype)
torch.manual_seed(0)
np.random.seed(0)

# Problem parameters
q_frac = 0.7
nu = 0.01
lam = 1.0

x_left = -math.pi
x_right = math.pi
T_final = 1.0

# Training settings
clip_value = 10.0
hid_dim = 100
num_f = 200
num_x0 = 100
num_tb = 100
num_iters = 20000
LR0 = 5e-2

# The baseline uses the same width and iteration budget.
hidden_dim = hid_dim
adam_iters = num_iters
adam_lr = 51e-3

eps_t = 1e-4

# Manufactured solution and source term
def exact_solution(x, t):
    return 0.5 + 0.25 * torch.exp(-t) * (torch.sin(2.0 * x) + 0.5 * torch.sin(4.0 * x))

def forcing_term(x, t, q):
    e_t = torch.exp(-t)
    A = torch.sin(2.0 * x) + 0.5 * torch.sin(4.0 * x)

    u = 0.5 + 0.25 * e_t * A
    u_t = -0.25 * e_t * A
    u_x = 0.25 * e_t * (2.0 * torch.cos(2.0 * x) + 2.0 * torch.cos(4.0 * x))
    u_xx = 0.25 * e_t * (-4.0 * torch.sin(2.0 * x) - 8.0 * torch.sin(4.0 * x))

    return t**(1.0 - q) * u_t + u * u_x - nu * u_xx - lam * u * (1.0 - u)

def u0(x):
    return 0.5 + 0.25 * (torch.sin(2.0 * x) + 0.5 * torch.sin(4.0 * x))

# Feature map and differentiation helper
def periodic_features(x, t):
    return torch.cat(
        [
            torch.sin(2.0 * x),
            torch.cos(2.0 * x),
            torch.sin(4.0 * x),
            torch.cos(4.0 * x),
            t,
        ],
        dim=1,
    )

def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True
    )[0]


def save_pdf(fig, stem):
    """Save one manuscript figure as a compact PDF."""
    fig.savefig(
        output_dir / f"{stem}.pdf",
        format="pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )

# Standard PINN
class StandardPINN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=100, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, X):
        return self.fc2(self.act(self.fc1(X)))

def standard_features(x, t):
    return torch.cat([x, t], dim=1)

def trial_solution_std(model, x, t):
    X = standard_features(x, t)
    return u0(x) + t * model(X)

def standard_residual(model, x, t, q):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    u = trial_solution_std(model, x, t)
    u_t = grad(u, t)
    u_x = grad(u, x)
    u_xx = grad(u_x, x)

    Tq_u = t**(1.0 - q) * u_t
    f = forcing_term(x, t, q)

    return Tq_u + u * u_x - nu * u_xx - lam * u * (1.0 - u) - f

def standard_initial_loss(model, x0):
    t0 = torch.zeros_like(x0)
    u_pred = trial_solution_std(model, x0, t0)
    return torch.mean((u_pred - u0(x0))**2)

def standard_boundary_loss(model, t_b):
    xL = torch.full_like(t_b, x_left, requires_grad=True)
    xR = torch.full_like(t_b, x_right, requires_grad=True)
    tL = t_b.clone().detach().requires_grad_(True)
    tR = t_b.clone().detach().requires_grad_(True)

    uL = trial_solution_std(model, xL, tL)
    uR = trial_solution_std(model, xR, tR)

    uL_x = grad(uL, xL)
    uR_x = grad(uR, xR)

    loss_u = torch.mean((uL - uR)**2)
    loss_ux = torch.mean((uL_x - uR_x)**2)
    return loss_u + loss_ux

def total_loss_std(model, x_f, t_f, x0, t_b, q, w_pde=1.0, w_ic=1.0, w_bc=1.0):
    res = standard_residual(model, x_f, t_f, q)
    loss_pde = torch.mean(res**2)
    loss_ic = standard_initial_loss(model, x0)
    loss_bc = standard_boundary_loss(model, t_b)

    loss = w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc
    return loss, loss_pde, loss_ic, loss_bc

@torch.no_grad()
def evaluate_standard(model, nx=256, nt=201):
    xg = torch.linspace(x_left, x_right, nx, device=device)
    tg = torch.linspace(0.0, T_final, nt, device=device)

    X, T = torch.meshgrid(xg, tg, indexing="ij")
    x_flat = X.reshape(-1, 1)
    t_flat = T.reshape(-1, 1)

    U_pred = trial_solution_std(model, x_flat, t_flat).reshape(nx, nt)
    U_exact = exact_solution(x_flat, t_flat).reshape(nx, nt)

    error = U_pred - U_exact
    e_inf = torch.max(torch.abs(error))
    e_2 = torch.sqrt(torch.mean(error**2))
    e_2_rel = torch.norm(error) / torch.norm(U_exact)

    return (
        xg.cpu().numpy(),
        tg.cpu().numpy(),
        U_pred.cpu().numpy(),
        U_exact.cpu().numpy(),
        e_inf.item(),
        e_2.item(),
        e_2_rel.item()
    )

# Shared-bias PINN
class SharedBiasPINN(nn.Module):
    def __init__(self, input_dim=5, hid_dim=100, output_dim=1):
        super().__init__()
        self.hid_dim = hid_dim

        self.W1 = nn.Parameter(torch.empty(input_dim, hid_dim, dtype=dtype))
        self.W2 = nn.Parameter(torch.empty(hid_dim, output_dim, dtype=dtype))
        self.b1 = nn.Parameter(torch.zeros(1, 1, dtype=dtype))
        self.b2 = nn.Parameter(torch.zeros(1, output_dim, dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.W1)
        nn.init.xavier_normal_(self.W2)
        with torch.no_grad():
            self.b1.zero_()
            self.b2.zero_()

    def forward(self, X):
        Z = X @ self.W1 + self.b1
        H = torch.tanh(Z)
        Y = H @ self.W2 + self.b2
        return Y

def trial_solution_sb(model, x, t):
    X = periodic_features(x, t)
    return u0(x) + t * model(X)

def shared_residual(model, x, t, q):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    u = trial_solution_sb(model, x, t)
    u_t = grad(u, t)
    u_x = grad(u, x)
    u_xx = grad(u_x, x)

    Tq_u = t**(1.0 - q) * u_t
    f = forcing_term(x, t, q)

    return Tq_u + u * u_x - nu * u_xx - lam * u * (1.0 - u) - f

def shared_boundary_loss(model, t_b):
    xL = torch.full_like(t_b, x_left, requires_grad=True)
    xR = torch.full_like(t_b, x_right, requires_grad=True)
    tL = t_b.clone().detach().requires_grad_(True)
    tR = t_b.clone().detach().requires_grad_(True)

    uL = trial_solution_sb(model, xL, tL)
    uR = trial_solution_sb(model, xR, tR)

    uL_x = grad(uL, xL)
    uR_x = grad(uR, xR)

    loss_u = torch.mean((uL - uR) ** 2)
    loss_ux = torch.mean((uL_x - uR_x) ** 2)
    return loss_u + loss_ux

def total_loss_sb(model, x_f, t_f, t_b, q, w_pde=1.0, w_bc=1.0):
    res = shared_residual(model, x_f, t_f, q)
    loss_pde = torch.mean(res**2)
    loss_bc = shared_boundary_loss(model, t_b)
    loss = w_pde * loss_pde + w_bc * loss_bc
    return loss, loss_pde, loss_bc

def lr_alpha(i):
    """Scheduled conformable order used by the optimizer."""
    return 1.0 - math.atan(i) / math.pi

def learning_rate(i, LR0=3e-2):
    a = max(lr_alpha(i), 1.0e-8)
    return LR0 * a

@torch.no_grad()
def apply_conformable_gradient(model, i, LR0=7e-3):
    alpha_opt = max(lr_alpha(i), 1.0e-14)
    step_size = learning_rate(i, LR0=LR0)

    for p in model.parameters():
        if p.grad is None:
            continue
        p_norm = torch.norm(p).clamp_min(1.0e-14)
        scaled_grad = (p_norm ** alpha_opt) * p.grad / alpha_opt
        p -= step_size * scaled_grad

def sample_interior(num_f, eps_t):
    t_f = eps_t + (T_final - eps_t) * torch.rand(num_f, 1, device=device)
    x_f = x_left + (x_right - x_left) * torch.rand(num_f, 1, device=device)
    return x_f, t_f

def sample_boundary_times(num_tb, eps_t):
    return torch.linspace(eps_t, T_final, num_tb, device=device).reshape(-1, 1)

@torch.no_grad()
def evaluate_shared(model, nx=256, nt=201):
    xg = torch.linspace(x_left, x_right, nx, device=device)
    tg = torch.linspace(0.0, T_final, nt, device=device)

    X, T = torch.meshgrid(xg, tg, indexing="ij")
    x_flat = X.reshape(-1, 1)
    t_flat = T.reshape(-1, 1)

    U_pred = trial_solution_sb(model, x_flat, t_flat).reshape(nx, nt)
    U_exact = exact_solution(x_flat, t_flat).reshape(nx, nt)

    error = U_pred - U_exact
    e_inf = torch.max(torch.abs(error))
    e_2 = torch.sqrt(torch.mean(error**2))
    e_2_rel = torch.norm(error) / torch.norm(U_exact)

    return (
        xg.cpu().numpy(),
        tg.cpu().numpy(),
        U_pred.cpu().numpy(),
        U_exact.cpu().numpy(),
        e_inf.item(),
        e_2.item(),
        e_2_rel.item()
    )

# Reporting helpers
def print_results(global_rows):
    print("\nGlobal comparison:\n")
    header = (
        f"{'Method':<24}"
        f"{'E_inf':>15}"
        f"{'E_2':>15}"
        f"{'E_2_rel':>15}"
        f"{'Final loss':>15}"
        f"{'CPU time (s)':>15}"
    )
    print(header)
    print("-" * len(header))

    for method, e_inf, e_2, e_2_rel, final_loss, cpu_time in global_rows:
        print(
            f"{method:<24}"
            f"{e_inf:15.6e}"
            f"{e_2:15.6e}"
            f"{e_2_rel:15.6e}"
            f"{final_loss:15.6e}"
            f"{cpu_time:15.3f}"
        )


def make_latex_table(global_rows, caption, label):
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\hline")
    lines.append(
        "Method & $E_{\\infty}$ & $E_2$ & "
        "$E_{2,\\mathrm{rel}}$ & Final loss & CPU time (s) \\\\"
    )
    lines.append("\\hline")

    for method, e_inf, e_2, e_2_rel, final_loss, cpu_time in global_rows:
        lines.append(
            f"{method} & {e_inf:.6e} & {e_2:.6e} & "
            f"{e_2_rel:.6e} & {final_loss:.6e} & "
            f"{cpu_time:.3f} \\\\"
        )

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)

# Train both models
global_error_rows = []

# Standard PINN with Adam
std_model = StandardPINN(input_dim=2, hidden_dim=hidden_dim, output_dim=1).to(device)

t_f_std = eps_t + (T_final - eps_t) * torch.rand(num_f, 1, device=device)
x_f_std = x_left + (x_right - x_left) * torch.rand(num_f, 1, device=device)

x0_std = torch.linspace(x_left, x_right, num_x0, device=device).reshape(-1, 1)
t_b_std = torch.linspace(eps_t, T_final, num_tb, device=device).reshape(-1, 1)

optimizer_std = torch.optim.Adam(std_model.parameters(), lr=adam_lr)

loss_history_std = []
t0_wall_std = time.time()

for i in range(adam_iters + 1):
    optimizer_std.zero_grad()

    loss_std, loss_pde_std, loss_ic_std, loss_bc_std = total_loss_std(
        std_model, x_f_std, t_f_std, x0_std, t_b_std, q_frac
    )

    if torch.isnan(loss_std) or torch.isinf(loss_std):
        print(f"Training stopped: invalid loss at iteration {i}")
        break

    loss_std.backward()
    torch.nn.utils.clip_grad_norm_(std_model.parameters(), max_norm=clip_value)
    optimizer_std.step()

    loss_history_std.append(loss_std.item())

    if i % 200 == 0:
        print(
            f"Iter {i:5d} | "
            f"Loss = {loss_std.item():.6e} | "
            f"PDE = {loss_pde_std.item():.6e} | "
            f"IC = {loss_ic_std.item():.6e} | "
            f"BC = {loss_bc_std.item():.6e}"
        )

elapsed_std = time.time() - t0_wall_std
final_loss_std = loss_history_std[-1] if loss_history_std else float("nan")

(
    xg_std,
    tg_std,
    U_pred_std,
    U_exact_std,
    e_inf_std,
    e_2_std,
    e_2_rel_std
) = evaluate_standard(std_model, nx=256, nt=201)

(
    xg_std_s,
    tg_std_s,
    U_pred_std_s,
    U_exact_std_s,
    _,
    _,
    _
) = evaluate_standard(std_model, nx=500, nt=260)

global_error_rows.append(
    (
        "Standard PINN",
        e_inf_std,
        e_2_std,
        e_2_rel_std,
        final_loss_std,
        elapsed_std
    )
)

# Shared-bias PINN with conformable-gradient training
sb_model = SharedBiasPINN(input_dim=5, hid_dim=hid_dim, output_dim=1).to(device)

x_f_sb, t_f_sb = sample_interior(num_f=num_f, eps_t=eps_t)
t_b_sb = sample_boundary_times(num_tb=num_tb, eps_t=eps_t)

loss_history_sb = []
t0_wall_sb = time.time()

for i in range(num_iters + 1):
    if i % 200 == 0:
        x_f_sb, t_f_sb = sample_interior(num_f=num_f, eps_t=eps_t)

    sb_model.zero_grad(set_to_none=True)

    loss_sb, loss_pde_sb, loss_bc_sb = total_loss_sb(
        sb_model, x_f_sb, t_f_sb, t_b_sb, q_frac,
        w_pde=1.0, w_bc=1.0
    )

    if torch.isnan(loss_sb) or torch.isinf(loss_sb):
        print(f"Shared-bias PINN stopped: invalid loss at iteration {i}")
        break

    loss_sb.backward()
    torch.nn.utils.clip_grad_norm_(sb_model.parameters(), max_norm=clip_value)
    apply_conformable_gradient(sb_model, i, LR0=LR0)

    loss_history_sb.append(loss_sb.item())

elapsed_sb = time.time() - t0_wall_sb
final_loss_sb = loss_history_sb[-1] if loss_history_sb else float("nan")

(
    xg_sb,
    tg_sb,
    U_pred_sb,
    U_exact_sb,
    e_inf_sb,
    e_2_sb,
    e_2_rel_sb
) = evaluate_shared(sb_model, nx=256, nt=201)

(
    xg_sb_s,
    tg_sb_s,
    U_pred_sb_s,
    U_exact_sb_s,
    _,
    _,
    _
) = evaluate_shared(sb_model, nx=500, nt=260)

global_error_rows.append(
    (
        "Shared-bias PINN",
        e_inf_sb,
        e_2_sb,
        e_2_rel_sb,
        final_loss_sb,
        elapsed_sb
    )
)

# Figures

# Training-loss comparison
fig1, ax1 = plt.subplots(figsize=(7.6, 4.4), dpi=240)
ax1.semilogy(loss_history_std, color=COLOR_STD, label="Standard PINN")
ax1.semilogy(loss_history_sb, color=COLOR_SB, label="Shared-bias PINN")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Loss")
ax1.set_title("Loss history comparison")
ax1.legend()
fig1.tight_layout()
save_pdf(fig1, "compare_loss_history_prob2")
plt.close(fig1)

# Final-time comparison
fig2, ax2 = plt.subplots(figsize=(7.6, 4.4), dpi=240)
ax2.plot(xg_std, U_exact_std[:, -1], color=COLOR_EX, label="Exact")
ax2.plot(xg_std, U_pred_std[:, -1], "--", color=COLOR_STD, label="Standard PINN")
ax2.plot(xg_sb, U_pred_sb[:, -1], "-.", color=COLOR_SB, label="Shared-bias PINN")
ax2.set_xlabel("x")
ax2.set_ylabel("u(x,1)")
ax2.set_title("Final-time comparison at $t=1$")
ax2.legend()
fig2.tight_layout()
save_pdf(fig2, "compare_final_time_prob2")
plt.close(fig2)

# Solution heatmaps
fig3, axes = plt.subplots(1, 3, figsize=(15.8, 4.8), dpi=240, constrained_layout=True)

im0 = axes[0].imshow(
    U_exact_std.T,
    extent=[x_left, x_right, 0.0, T_final],
    origin="lower",
    aspect="auto",
    cmap="viridis",
    rasterized=True,
)
axes[0].set_title("Exact")
axes[0].set_xlabel("x")
axes[0].set_ylabel("t")
fig3.colorbar(im0, ax=axes[0], shrink=0.88)

im1 = axes[1].imshow(
    U_pred_std.T,
    extent=[x_left, x_right, 0.0, T_final],
    origin="lower",
    aspect="auto",
    cmap="viridis",
    rasterized=True,
)
axes[1].set_title("Standard PINN")
axes[1].set_xlabel("x")
axes[1].set_ylabel("t")
fig3.colorbar(im1, ax=axes[1], shrink=0.88)

im2 = axes[2].imshow(
    U_pred_sb.T,
    extent=[x_left, x_right, 0.0, T_final],
    origin="lower",
    aspect="auto",
    cmap="viridis",
    rasterized=True,
)
axes[2].set_title("Shared-bias PINN")
axes[2].set_xlabel("x")
axes[2].set_ylabel("t")
fig3.colorbar(im2, ax=axes[2], shrink=0.88)

save_pdf(fig3, "compare_solution_heatmaps_prob2")
plt.close(fig3)

# Error heatmaps
fig4, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), dpi=240, constrained_layout=True)

im3 = axes[0].imshow(
    np.abs(U_pred_std - U_exact_std).T,
    extent=[x_left, x_right, 0.0, T_final],
    origin="lower",
    aspect="auto",
    cmap="magma",
    rasterized=True,
)
axes[0].set_title("Standard PINN error")
axes[0].set_xlabel("x")
axes[0].set_ylabel("t")
fig4.colorbar(im3, ax=axes[0], shrink=0.88)

im4 = axes[1].imshow(
    np.abs(U_pred_sb - U_exact_sb).T,
    extent=[x_left, x_right, 0.0, T_final],
    origin="lower",
    aspect="auto",
    cmap="magma",
    rasterized=True,
)
axes[1].set_title("Shared-bias PINN error")
axes[1].set_xlabel("x")
axes[1].set_ylabel("t")
fig4.colorbar(im4, ax=axes[1], shrink=0.88)

save_pdf(fig4, "compare_error_heatmaps_prob2")
plt.close(fig4)

# Three-dimensional solution comparison
X_plot_std_s, T_plot_std_s = np.meshgrid(xg_std_s, tg_std_s, indexing="ij")
X_plot_sb_s, T_plot_sb_s = np.meshgrid(xg_sb_s, tg_sb_s, indexing="ij")

fig5 = plt.figure(figsize=(22.0, 6.4), dpi=240, constrained_layout=True)

ax5 = fig5.add_subplot(1, 3, 1, projection="3d")
surf0 = ax5.plot_surface(
    X_plot_std_s, T_plot_std_s, U_exact_std_s,
    cmap="viridis", rcount=240, ccount=200,
    linewidth=0, edgecolor="none", antialiased=True,
    rasterized=True,
)
ax5.set_xlabel("x", labelpad=10)
ax5.set_ylabel("t", labelpad=10)
ax5.set_zlabel("u(x,t)", labelpad=10)
ax5.set_title("Exact solution", pad=12)
ax5.view_init(elev=28, azim=-135)
fig5.colorbar(surf0, ax=ax5, shrink=0.72, pad=0.08)

ax6 = fig5.add_subplot(1, 3, 2, projection="3d")
surf1 = ax6.plot_surface(
    X_plot_std_s, T_plot_std_s, U_pred_std_s,
    cmap="viridis", rcount=240, ccount=200,
    linewidth=0, edgecolor="none", antialiased=True,
    rasterized=True,
)
ax6.set_xlabel("x", labelpad=10)
ax6.set_ylabel("t", labelpad=10)
ax6.set_zlabel("u(x,t)", labelpad=10)
ax6.set_title("Standard PINN", pad=12)
ax6.view_init(elev=28, azim=-135)
fig5.colorbar(surf1, ax=ax6, shrink=0.72, pad=0.08)

ax7 = fig5.add_subplot(1, 3, 3, projection="3d")
surf2 = ax7.plot_surface(
    X_plot_sb_s, T_plot_sb_s, U_pred_sb_s,
    cmap="viridis", rcount=240, ccount=200,
    linewidth=0, edgecolor="none", antialiased=True,
    rasterized=True,
)
ax7.set_xlabel("x", labelpad=10)
ax7.set_ylabel("t", labelpad=10)
ax7.set_zlabel("u(x,t)", labelpad=10)
ax7.set_title("Shared-bias PINN", pad=12)
ax7.view_init(elev=28, azim=-135)
fig5.colorbar(surf2, ax=ax7, shrink=0.72, pad=0.08)

save_pdf(fig5, "compare_3d_solutions_prob2")
plt.close(fig5)

# Three-dimensional error comparison
fig6 = plt.figure(figsize=(16.0, 6.2), dpi=240, constrained_layout=True)

ax8 = fig6.add_subplot(1, 2, 1, projection="3d")
surf3 = ax8.plot_surface(
    X_plot_std_s, T_plot_std_s, np.abs(U_pred_std_s - U_exact_std_s),
    cmap="magma", rcount=240, ccount=200,
    linewidth=0, edgecolor="none", antialiased=True,
    rasterized=True,
)
ax8.set_xlabel("x", labelpad=10)
ax8.set_ylabel("t", labelpad=10)
ax8.set_zlabel("|error|", labelpad=10)
ax8.set_title("Standard PINN error", pad=12)
ax8.view_init(elev=28, azim=-135)
fig6.colorbar(surf3, ax=ax8, shrink=0.75, pad=0.08)

ax9 = fig6.add_subplot(1, 2, 2, projection="3d")
surf4 = ax9.plot_surface(
    X_plot_sb_s, T_plot_sb_s, np.abs(U_pred_sb_s - U_exact_sb_s),
    cmap="magma", rcount=240, ccount=200,
    linewidth=0, edgecolor="none", antialiased=True,
    rasterized=True,
)
ax9.set_xlabel("x", labelpad=10)
ax9.set_ylabel("t", labelpad=10)
ax9.set_zlabel("|error|", labelpad=10)
ax9.set_title("Shared-bias PINN error", pad=12)
ax9.view_init(elev=28, azim=-135)
fig6.colorbar(surf4, ax=ax9, shrink=0.75, pad=0.08)

save_pdf(fig6, "compare_3d_errors_prob2")
plt.close(fig6)

# Final report
print_results(global_error_rows)

latex_global = make_latex_table(
    global_error_rows,
    caption="Global errors, final training loss, and CPU time for the manufactured time-conformable Burgers--Fisher problem.",
    label="tab:burgers_fisher_global_comparison"
)

print("\nLaTeX table:\n")
print(latex_global)

print(f"\nPDF figures saved in:\n{output_dir}")