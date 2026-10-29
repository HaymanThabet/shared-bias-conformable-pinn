"""Forward solver for the manufactured time-conformable Burgers problem.

The script compares a standard PINN trained with Adam against a
shared-bias PINN trained with the conformable-gradient update.
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

# Plot settings
plt.rcParams["figure.dpi"] = 180
# Dense surfaces are rasterized in the PDF; labels and curves stay vector.
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


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


set_seed(SEED)

# Manufactured problem:
# T_q u + u u_x - nu u_xx = f(x,t),
# u_exact(x,t) = exp(-t) sin(omega x).
q_order = 0.7

# Frequency used in the current run.
omega = 5.0

if omega not in {5.0, 10.0, 20.0, 30.0}:
    raise ValueError("omega must be 5.0, 10.0, 20.0, or 30.0.")

omega_tag = str(int(omega))
nu = 0.01 / math.pi

x_left = 0.0
x_right = 2.0 * math.pi
t_final = 1.0

# Output folder
script_dir = Path(__file__).resolve().parent
output_dir = script_dir / "outputs" / f"omega_{omega_tag}"
output_dir.mkdir(parents=True, exist_ok=True)

# Exact solution and forcing
def exact_solution(x, t):
    return torch.exp(-t) * torch.sin(omega * x)


def forcing_term(x, t, q):
    e_t = torch.exp(-t)
    s = torch.sin(omega * x)
    c = torch.cos(omega * x)

    return (
        -t ** (1.0 - q) * e_t * s
        + omega * e_t**2 * s * c
        + nu * omega**2 * e_t * s
    )


def u0(x):
    return torch.sin(omega * x)


# Feature maps
def standard_features(x, t):
    return torch.cat([x, t], dim=1)


def periodic_features(x, t):
    return torch.cat(
        [torch.sin(omega * x), torch.cos(omega * x), t],
        dim=1,
    )


# Standard PINN
class StandardPINN(nn.Module):
    def __init__(self, input_dim=2, hidden_width=80, output_dim=1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_width)
        self.fc2 = nn.Linear(hidden_width, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, features):
        hidden = torch.tanh(self.fc1(features))
        return self.fc2(hidden)


# Shared-bias PINN
class SharedBiasPINN(nn.Module):
    def __init__(self, input_dim=3, hidden_width=80, output_dim=1):
        super().__init__()
        self.W1 = nn.Parameter(
            torch.randn(input_dim, hidden_width) * math.sqrt(1.0 / input_dim)
        )
        self.W2 = nn.Parameter(
            torch.randn(hidden_width, output_dim) * math.sqrt(1.0 / hidden_width)
        )
        self.b1 = nn.Parameter(
            torch.randn(1, 1) * math.sqrt(1.0 / input_dim)
        )
        self.b2 = nn.Parameter(
            torch.randn(1, output_dim) * math.sqrt(1.0 / output_dim)
        )

    def forward(self, features):
        hidden = torch.tanh(features @ self.W1 + self.b1)
        return hidden @ self.W2 + self.b2


# Trial solutions
def trial_solution_std(model, x, t):
    features = standard_features(x, t)
    return u0(x) + t * model(features)


def trial_solution_sb(model, x, t):
    features = periodic_features(x, t)
    return u0(x) + t * model(features)


# Automatic differentiation
def differentiate(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


# Residuals
def standard_residual(model, x, t, q):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    u = trial_solution_std(model, x, t)
    u_t = differentiate(u, t)
    u_x = differentiate(u, x)
    u_xx = differentiate(u_x, x)

    tq_u = t ** (1.0 - q) * u_t
    f = forcing_term(x, t, q)

    return tq_u + u * u_x - nu * u_xx - f


def shared_residual(model, x, t, q):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    u = trial_solution_sb(model, x, t)
    u_t = differentiate(u, t)
    u_x = differentiate(u, x)
    u_xx = differentiate(u_x, x)

    tq_u = t ** (1.0 - q) * u_t
    f = forcing_term(x, t, q)

    return tq_u + u * u_x - nu * u_xx - f


# Initial losses
def standard_initial_loss(model, x0):
    t0 = torch.zeros_like(x0)
    return torch.mean(
        (trial_solution_std(model, x0, t0) - u0(x0)) ** 2
    )


def shared_initial_loss(model, x0):
    t0 = torch.zeros_like(x0)
    return torch.mean(
        (trial_solution_sb(model, x0, t0) - u0(x0)) ** 2
    )


# Periodic boundary losses
def standard_boundary_loss(model, t_b):
    xL = torch.full_like(t_b, x_left, requires_grad=True)
    xR = torch.full_like(t_b, x_right, requires_grad=True)

    tL = t_b.clone().detach().requires_grad_(True)
    tR = t_b.clone().detach().requires_grad_(True)

    uL = trial_solution_std(model, xL, tL)
    uR = trial_solution_std(model, xR, tR)

    uL_x = differentiate(uL, xL)
    uR_x = differentiate(uR, xR)

    return torch.mean((uL - uR) ** 2) + torch.mean((uL_x - uR_x) ** 2)


def shared_boundary_loss(model, t_b):
    xL = torch.full_like(t_b, x_left, requires_grad=True)
    xR = torch.full_like(t_b, x_right, requires_grad=True)

    tL = t_b.clone().detach().requires_grad_(True)
    tR = t_b.clone().detach().requires_grad_(True)

    uL = trial_solution_sb(model, xL, tL)
    uR = trial_solution_sb(model, xR, tR)

    uL_x = differentiate(uL, xL)
    uR_x = differentiate(uR, xR)

    return torch.mean((uL - uR) ** 2) + torch.mean((uL_x - uR_x) ** 2)


# Total losses
def standard_loss(model, x_f, t_f, x0, t_b, q):
    res = standard_residual(model, x_f, t_f, q)
    loss_pde = torch.mean(res**2)
    loss_ic = standard_initial_loss(model, x0)
    loss_bc = standard_boundary_loss(model, t_b)
    loss = loss_pde + loss_ic + loss_bc
    return loss, loss_pde, loss_ic, loss_bc


def shared_loss(model, x_f, t_f, x0, t_b, q):
    res = shared_residual(model, x_f, t_f, q)
    loss_pde = torch.mean(res**2)
    loss_ic = shared_initial_loss(model, x0)
    loss_bc = shared_boundary_loss(model, t_b)
    loss = loss_pde + loss_ic + loss_bc
    return loss, loss_pde, loss_ic, loss_bc


# Conformable-gradient update
def conformable_order(i):
    return 1.5 - 2.0 * (
        math.atan(i) + math.pi / 2.0
    ) / math.pi / 2.0


def scheduled_learning_rate(i, base_learning_rate=1e-2):
    a = max(conformable_order(i), 1e-8)
    return base_learning_rate * a


@torch.no_grad()
def conformable_gradient_step(
    model, i, base_learning_rate=1e-2
):
    alpha_opt = max(conformable_order(i), 1.0e-14)
    step_size = scheduled_learning_rate(
        i,
        base_learning_rate=base_learning_rate,
    )

    # Scale each parameter block by its own Euclidean norm.
    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        parameter_norm = torch.norm(parameter).clamp_min(1.0e-14)
        scaled_gradient = (
            parameter_norm ** alpha_opt
        ) * parameter.grad / alpha_opt

        parameter -= step_size * scaled_gradient


# Training settings
hidden_width = 80
n_residual = 200
n_initial = 100
n_boundary = 100
num_iters = 10000
base_learning_rate = 1e-2
clip_value = 10.0

adam_iters = num_iters
adam_lr = 1e-3
eps_t = 1e-4

# Initialize the shared-bias model before drawing the training points.
set_seed(SEED)

shared_model = SharedBiasPINN(
    input_dim=3,
    hidden_width=hidden_width,
    output_dim=1,
).to(device)

t_f = eps_t + (
    t_final - eps_t
) * torch.rand(n_residual, 1, device=device)

x_f = x_left + (
    x_right - x_left
) * torch.rand(n_residual, 1, device=device)

x0 = torch.linspace(
    x_left,
    x_right,
    n_initial,
    device=device,
).reshape(-1, 1)

t_b = torch.linspace(
    eps_t,
    t_final,
    n_boundary,
    device=device,
).reshape(-1, 1)

# Reset the seed so the standard model has a reproducible initialization.
torch.manual_seed(SEED)
standard_model = StandardPINN(
    input_dim=2,
    hidden_width=hidden_width,
    output_dim=1,
).to(device)


# Train standard PINN
optimizer_std = torch.optim.Adam(
    standard_model.parameters(),
    lr=adam_lr,
)

standard_loss_history = []
standard_pde_history = []
standard_ic_history = []
standard_bc_history = []

start_std = time.time()

for i in range(adam_iters + 1):
    optimizer_std.zero_grad(set_to_none=True)

    loss, loss_pde, loss_ic, loss_bc = standard_loss(
        standard_model,
        x_f,
        t_f,
        x0,
        t_b,
        q_order,
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

    standard_loss_history.append(loss.item())
    standard_pde_history.append(loss_pde.item())
    standard_ic_history.append(loss_ic.item())
    standard_bc_history.append(loss_bc.item())

    if i % 200 == 0:
        print(
            f"[Standard PINN] Iteration: {i:5d} | "
            f"Loss: {loss.item():.6e} | "
            f"PDE: {loss_pde.item():.6e} | "
            f"IC: {loss_ic.item():.6e} | "
            f"BC: {loss_bc.item():.6e}"
        )

standard_time = time.time() - start_std


# Train shared-bias PINN
shared_loss_history = []
shared_pde_history = []
shared_ic_history = []
shared_bc_history = []

start_sb = time.time()

for i in range(num_iters + 1):
    for p in shared_model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    loss, loss_pde, loss_ic, loss_bc = shared_loss(
        shared_model,
        x_f,
        t_f,
        x0,
        t_b,
        q_order,
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
    conformable_gradient_step(
        shared_model,
        i,
        base_learning_rate=base_learning_rate,
    )

    shared_loss_history.append(loss.item())
    shared_pde_history.append(loss_pde.item())
    shared_ic_history.append(loss_ic.item())
    shared_bc_history.append(loss_bc.item())

    if i % 200 == 0:
        print(
            f"[Shared-bias PINN] Iteration: {i:5d} | "
            f"Loss: {loss.item():.6e} | "
            f"PDE: {loss_pde.item():.6e} | "
            f"IC: {loss_ic.item():.6e} | "
            f"BC: {loss_bc.item():.6e} | "
            f"alpha_opt: {conformable_order(i):.6e} | "
            f"LR: {scheduled_learning_rate(i, base_learning_rate):.6e}"
        )

shared_time = time.time() - start_sb


# Evaluation
@torch.no_grad()
def evaluate_model(model, trial_function, nx=256, nt=201):
    xg = torch.linspace(x_left, x_right, nx, device=device)
    tg = torch.linspace(0.0, t_final, nt, device=device)

    x_mesh, t_mesh = torch.meshgrid(xg, tg, indexing="ij")
    x_flat = x_mesh.reshape(-1, 1)
    t_flat = t_mesh.reshape(-1, 1)

    u_pred = trial_function(
        model,
        x_flat,
        t_flat,
    ).reshape(nx, nt)

    u_exact = exact_solution(
        x_flat,
        t_flat,
    ).reshape(nx, nt)

    error = u_pred - u_exact

    e_inf = torch.max(torch.abs(error))
    e_2 = torch.sqrt(torch.mean(error**2))
    e_2_rel = torch.norm(error) / torch.norm(u_exact)

    return (
        xg.cpu().numpy(),
        tg.cpu().numpy(),
        u_pred.cpu().numpy(),
        u_exact.cpu().numpy(),
        e_inf.item(),
        e_2.item(),
        e_2_rel.item(),
    )


def recompute_final_loss_std():
    loss, _, _, _ = standard_loss(
        standard_model,
        x_f,
        t_f,
        x0,
        t_b,
        q_order,
    )
    return loss.detach().item()


def recompute_final_loss_sb():
    loss, _, _, _ = shared_loss(
        shared_model,
        x_f,
        t_f,
        x0,
        t_b,
        q_order,
    )
    return loss.detach().item()


(
    xg,
    tg,
    u_standard,
    u_exact,
    e_inf_standard,
    e_2_standard,
    e_2_rel_standard,
) = evaluate_model(
    standard_model,
    trial_solution_std,
    nx=256,
    nt=201,
)

(
    _,
    _,
    u_shared,
    _,
    e_inf_shared,
    e_2_shared,
    e_2_rel_shared,
) = evaluate_model(
    shared_model,
    trial_solution_sb,
    nx=256,
    nt=201,
)

(
    xg_s,
    tg_s,
    u_standard_surface,
    u_exact_surface,
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
    u_shared_surface,
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


# Report results
results = [
    (
        "Standard PINN",
        e_inf_standard,
        e_2_standard,
        e_2_rel_standard,
        final_loss_std,
        standard_time,
    ),
    (
        "Shared-bias PINN",
        e_inf_shared,
        e_2_shared,
        e_2_rel_shared,
        final_loss_sb,
        shared_time,
    ),
]

print(f"\nGlobal comparison for omega = {omega_tag}:\n")
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

for method, e_inf, e_2, e_2_rel, final_loss, cpu in results:
    print(
        f"{method:<24}"
        f"{e_inf:15.6e}"
        f"{e_2:15.6e}"
        f"{e_2_rel:15.6e}"
        f"{final_loss:15.6e}"
        f"{cpu:15.3f}"
    )

print("\nLaTeX table:\n")
print(r"\begin{table}[htbp]")
print(r"\centering")
print(
    r"\caption{Global error, final loss, and CPU-time "
    r"comparison for the manufactured time-conformable "
    rf"Burgers problem with $\omega={omega_tag}$."
    r"}"
)
print(rf"\label{{tab:burgers_omega{omega_tag}_global}}")
print(r"\begin{tabular}{lccccc}")
print(r"\hline")
print(
    r"Method & $E_{\infty}$ & $E_2$ & "
    r"$E_{2,\mathrm{rel}}$ & Final loss & CPU time (s) \\"
)
print(r"\hline")

for method, e_inf, e_2, e_2_rel, final_loss, cpu in results:
    print(
        f"{method} & {e_inf:.6e} & {e_2:.6e} & "
        f"{e_2_rel:.6e} & {final_loss:.6e} & "
        f"{cpu:.3f} \\\\"
    )

print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")


# Figures
plt.close("all")


def save_pdf(fig, stem):
    """Save a figure as a compact PDF."""
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
    standard_loss_history,
    color=COLOR_STD,
    label="Standard PINN",
)
ax1.semilogy(
    shared_loss_history,
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
save_pdf(fig1, f"burgers_loss_comparison_omega{omega_tag}")
plt.close(fig1)

# Exact, standard-PINN, and shared-bias surfaces
x_surface, t_surface = np.meshgrid(
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
    ("Exact solution", u_exact_surface),
    ("Standard PINN", u_standard_surface),
    ("Shared-bias PINN", u_shared_surface),
]

for index, (title, values) in enumerate(surface_data, start=1):
    ax = fig2.add_subplot(1, 3, index, projection="3d")
    surf = ax.plot_surface(
        x_surface,
        t_surface,
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

save_pdf(fig2, f"burgers_solution_surfaces_omega{omega_tag}")
plt.close(fig2)

# Three-dimensional absolute errors
fig3 = plt.figure(
    figsize=(14.0, 6.0),
    dpi=240,
    constrained_layout=True,
)

error_surface_data = [
    ("Standard PINN error", np.abs(u_standard_surface - u_exact_surface)),
    ("Shared-bias PINN error", np.abs(u_shared_surface - u_exact_surface)),
]

for index, (title, values) in enumerate(
    error_surface_data,
    start=1,
):
    ax = fig3.add_subplot(1, 2, index, projection="3d")
    surf = ax.plot_surface(
        x_surface,
        t_surface,
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

save_pdf(fig3, f"burgers_error_surfaces_omega{omega_tag}")
plt.close(fig3)

# Final-time comparison
fig4, ax4 = plt.subplots(figsize=(7.6, 4.4), dpi=240)
ax4.plot(
    xg,
    u_exact[:, -1],
    color=COLOR_EX,
    label="Exact",
)
ax4.plot(
    xg,
    u_standard[:, -1],
    "--",
    color=COLOR_STD,
    label="Standard PINN",
)
ax4.plot(
    xg,
    u_shared[:, -1],
    "-.",
    color=COLOR_SB,
    label="Shared-bias PINN",
)
ax4.set_xlabel("x")
ax4.set_ylabel("u(x,1)")
ax4.set_title(rf"Final-time comparison at $t=1$, $\omega={omega_tag}$")
ax4.legend()
fig4.tight_layout()
save_pdf(fig4, f"burgers_final_time_omega{omega_tag}")
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
    ("Standard PINN error", np.abs(u_standard - u_exact)),
    ("Shared-bias PINN error", np.abs(u_shared - u_exact)),
]

for ax, (title, values) in zip(axes, heatmap_data):
    im = ax.imshow(
        values.T,
        extent=[x_left, x_right, 0.0, t_final],
        origin="lower",
        aspect="auto",
        cmap="magma",
        rasterized=True,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(rf"{title}, $\omega={omega_tag}$")
    fig5.colorbar(im, ax=ax, shrink=0.9)

save_pdf(fig5, f"burgers_error_heatmaps_omega{omega_tag}")
plt.close(fig5)

print(f"\nPDF figures saved in:\n{output_dir}")
