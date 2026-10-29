"""
Optimizer-swapped study for the manufactured time-conformable Burgers problem.

The script compares:
    1. A standard PINN trained with conformable-gradient descent.
    2. A shared-bias PINN trained with Adam.

The governing equation is

    T_q u + u u_x - nu u_xx = f(x,t),

with exact solution

    u(x,t) = exp(-t) sin(omega x).

The equation, network architectures, initialization order, collocation
points, iteration counts, learning rates, error measures, table output,
and figure filenames are unchanged.
"""

import math
import os
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

try:
    torch.set_num_threads(1)
except RuntimeError:
    pass


# ---------------------------------------------------------------------
# Plot settings
# ---------------------------------------------------------------------
plt.rcParams.update(
    {
        "figure.dpi": 180,
        "savefig.dpi": 500,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "lines.linewidth": 2.2,
    }
)

COLOR_STANDARD = "#1f3b73"
COLOR_SHARED = "#d95f02"
COLOR_EXACT = "#2e7d32"


# ---------------------------------------------------------------------
# Reproducibility and numerical precision
# ---------------------------------------------------------------------
DEVICE = "cpu"
DTYPE = torch.float64
SEED = 0

torch.set_default_dtype(DTYPE)
torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------
# Problem parameters
# ---------------------------------------------------------------------
Q = 0.7
OMEGA = 5.0

if OMEGA not in {5.0, 10.0, 20.0}:
    raise ValueError("OMEGA must be 5.0, 10.0, or 20.0.")

OMEGA_TAG = str(int(OMEGA))
NU = 0.01 / math.pi

X_LEFT = 0.0
X_RIGHT = 2.0 * math.pi
FINAL_TIME = 1.0
MINIMUM_TIME = 1.0e-4


# ---------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------
HIDDEN_UNITS = 80
NUM_RESIDUAL_POINTS = 200
NUM_INITIAL_POINTS = 100
NUM_BOUNDARY_POINTS = 100
NUM_ITERATIONS = 10000

CONFORMABLE_BASE_RATE = 1.0e-2
ADAM_RATE = 1.0e-3
GRADIENT_CLIP = 10.0


# ---------------------------------------------------------------------
# Output folder
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs" / f"omega_{OMEGA_TAG}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Exact solution, forcing term, and initial condition
# ---------------------------------------------------------------------
def exact_solution(x, t):
    return torch.exp(-t) * torch.sin(OMEGA * x)


def initial_condition(x):
    return torch.sin(OMEGA * x)


def forcing_term(x, t, q):
    decay = torch.exp(-t)
    sine = torch.sin(OMEGA * x)
    cosine = torch.cos(OMEGA * x)

    return (
        -t ** (1.0 - q) * decay * sine
        + OMEGA * decay**2 * sine * cosine
        + NU * OMEGA**2 * decay * sine
    )


# ---------------------------------------------------------------------
# Feature maps
# ---------------------------------------------------------------------
def standard_features(x, t):
    return torch.cat((x, t), dim=1)


def periodic_features(x, t):
    return torch.cat(
        (
            torch.sin(OMEGA * x),
            torch.cos(OMEGA * x),
            t,
        ),
        dim=1,
    )


# ---------------------------------------------------------------------
# Neural-network models
# ---------------------------------------------------------------------
class StandardPINN(nn.Module):
    """One-hidden-layer PINN with ordinary hidden biases."""

    def __init__(self, input_dim=2, hidden_dim=80, output_dim=1):
        super().__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.hidden_layer.weight)
        nn.init.zeros_(self.hidden_layer.bias)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, inputs):
        hidden_output = torch.tanh(self.hidden_layer(inputs))
        return self.output_layer(hidden_output)


class SharedBiasPINN(nn.Module):
    """One-hidden-layer PINN with one common hidden-layer bias."""

    def __init__(self, input_dim=3, hidden_dim=80, output_dim=1):
        super().__init__()

        # The parameter creation order is preserved.
        self.W1 = nn.Parameter(
            torch.randn(input_dim, hidden_dim)
            * math.sqrt(1.0 / input_dim)
        )
        self.W2 = nn.Parameter(
            torch.randn(hidden_dim, output_dim)
            * math.sqrt(1.0 / hidden_dim)
        )
        self.b1 = nn.Parameter(
            torch.randn(1, 1) * math.sqrt(1.0 / input_dim)
        )
        self.b2 = nn.Parameter(
            torch.randn(1, output_dim)
            * math.sqrt(1.0 / output_dim)
        )

    def forward(self, inputs):
        hidden_argument = inputs @ self.W1 + self.b1
        hidden_output = torch.tanh(hidden_argument)
        return hidden_output @ self.W2 + self.b2


# ---------------------------------------------------------------------
# Trial solutions
# ---------------------------------------------------------------------
def standard_trial_solution(model, x, t):
    network_output = model(standard_features(x, t))
    return initial_condition(x) + t * network_output


def shared_trial_solution(model, x, t):
    network_output = model(periodic_features(x, t))
    return initial_condition(x) + t * network_output


# ---------------------------------------------------------------------
# Automatic differentiation
# ---------------------------------------------------------------------
def derivative(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


# ---------------------------------------------------------------------
# Differential-equation residuals
# ---------------------------------------------------------------------
def standard_residual(model, x, t):
    x = x.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)

    u = standard_trial_solution(model, x, t)
    u_t = derivative(u, t)
    u_x = derivative(u, x)
    u_xx = derivative(u_x, x)

    return (
        t ** (1.0 - Q) * u_t
        + u * u_x
        - NU * u_xx
        - forcing_term(x, t, Q)
    )


def shared_residual(model, x, t):
    x = x.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)

    u = shared_trial_solution(model, x, t)
    u_t = derivative(u, t)
    u_x = derivative(u, x)
    u_xx = derivative(u_x, x)

    return (
        t ** (1.0 - Q) * u_t
        + u * u_x
        - NU * u_xx
        - forcing_term(x, t, Q)
    )


# ---------------------------------------------------------------------
# Initial and periodic boundary losses
# ---------------------------------------------------------------------
def standard_initial_loss(model, initial_x):
    initial_t = torch.zeros_like(initial_x)
    prediction = standard_trial_solution(
        model, initial_x, initial_t
    )
    return torch.mean(
        (prediction - initial_condition(initial_x)) ** 2
    )


def shared_initial_loss(model, initial_x):
    initial_t = torch.zeros_like(initial_x)
    prediction = shared_trial_solution(
        model, initial_x, initial_t
    )
    return torch.mean(
        (prediction - initial_condition(initial_x)) ** 2
    )


def standard_boundary_loss(model, boundary_t):
    left_x = torch.full_like(
        boundary_t, X_LEFT, requires_grad=True
    )
    right_x = torch.full_like(
        boundary_t, X_RIGHT, requires_grad=True
    )

    left_t = boundary_t.detach().clone().requires_grad_(True)
    right_t = boundary_t.detach().clone().requires_grad_(True)

    left_u = standard_trial_solution(model, left_x, left_t)
    right_u = standard_trial_solution(model, right_x, right_t)

    left_u_x = derivative(left_u, left_x)
    right_u_x = derivative(right_u, right_x)

    value_loss = torch.mean((left_u - right_u) ** 2)
    derivative_loss = torch.mean((left_u_x - right_u_x) ** 2)

    return value_loss + derivative_loss


def shared_boundary_loss(model, boundary_t):
    left_x = torch.full_like(
        boundary_t, X_LEFT, requires_grad=True
    )
    right_x = torch.full_like(
        boundary_t, X_RIGHT, requires_grad=True
    )

    left_t = boundary_t.detach().clone().requires_grad_(True)
    right_t = boundary_t.detach().clone().requires_grad_(True)

    left_u = shared_trial_solution(model, left_x, left_t)
    right_u = shared_trial_solution(model, right_x, right_t)

    left_u_x = derivative(left_u, left_x)
    right_u_x = derivative(right_u, right_x)

    value_loss = torch.mean((left_u - right_u) ** 2)
    derivative_loss = torch.mean((left_u_x - right_u_x) ** 2)

    return value_loss + derivative_loss


# ---------------------------------------------------------------------
# Total training losses
# ---------------------------------------------------------------------
def standard_total_loss(
    model,
    residual_x,
    residual_t,
    initial_x,
    boundary_t,
):
    residual = standard_residual(
        model, residual_x, residual_t
    )

    residual_loss = torch.mean(residual**2)
    initial_loss = standard_initial_loss(model, initial_x)
    boundary_loss = standard_boundary_loss(model, boundary_t)

    total_loss = residual_loss + initial_loss + boundary_loss
    return total_loss, residual_loss, initial_loss, boundary_loss


def shared_total_loss(
    model,
    residual_x,
    residual_t,
    initial_x,
    boundary_t,
):
    residual = shared_residual(
        model, residual_x, residual_t
    )

    residual_loss = torch.mean(residual**2)
    initial_loss = shared_initial_loss(model, initial_x)
    boundary_loss = shared_boundary_loss(model, boundary_t)

    total_loss = residual_loss + initial_loss + boundary_loss
    return total_loss, residual_loss, initial_loss, boundary_loss


# ---------------------------------------------------------------------
# Conformable-gradient update
# ---------------------------------------------------------------------
def conformable_order(iteration):
    return (
        1.5
        - 2.0
        * (math.atan(iteration) + math.pi / 2.0)
        / math.pi
        / 2.0
    )


def effective_learning_rate(iteration):
    alpha = max(conformable_order(iteration), 1.0e-8)
    return CONFORMABLE_BASE_RATE * alpha


@torch.no_grad()
def apply_conformable_gradient(model, iteration):
    alpha = max(conformable_order(iteration), 1.0e-14)
    learning_rate = effective_learning_rate(iteration)

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        parameter_norm = torch.norm(parameter).clamp_min(1.0e-14)
        scaled_gradient = (
            parameter_norm**alpha
        ) * parameter.grad / alpha

        parameter -= learning_rate * scaled_gradient


# ---------------------------------------------------------------------
# Model initialization and collocation points
# ---------------------------------------------------------------------
# The original random-number sequence is preserved:
# shared model, time points, space points, then standard model.
torch.manual_seed(SEED)
np.random.seed(SEED)

shared_model = SharedBiasPINN(
    input_dim=3,
    hidden_dim=HIDDEN_UNITS,
    output_dim=1,
).to(DEVICE)

residual_t = MINIMUM_TIME + (
    FINAL_TIME - MINIMUM_TIME
) * torch.rand(NUM_RESIDUAL_POINTS, 1, device=DEVICE)

residual_x = X_LEFT + (
    X_RIGHT - X_LEFT
) * torch.rand(NUM_RESIDUAL_POINTS, 1, device=DEVICE)

initial_x = torch.linspace(
    X_LEFT,
    X_RIGHT,
    NUM_INITIAL_POINTS,
    device=DEVICE,
).reshape(-1, 1)

boundary_t = torch.linspace(
    MINIMUM_TIME,
    FINAL_TIME,
    NUM_BOUNDARY_POINTS,
    device=DEVICE,
).reshape(-1, 1)

torch.manual_seed(SEED)

standard_model = StandardPINN(
    input_dim=2,
    hidden_dim=HIDDEN_UNITS,
    output_dim=1,
).to(DEVICE)


# ---------------------------------------------------------------------
# Train the standard PINN with conformable-gradient descent
# ---------------------------------------------------------------------
standard_loss_history = []
standard_residual_history = []
standard_initial_history = []
standard_boundary_history = []

standard_start = time.time()

for iteration in range(NUM_ITERATIONS + 1):
    standard_model.zero_grad(set_to_none=True)

    (
        loss,
        residual_loss,
        initial_loss,
        boundary_loss,
    ) = standard_total_loss(
        standard_model,
        residual_x,
        residual_t,
        initial_x,
        boundary_t,
    )

    if not torch.isfinite(loss):
        raise RuntimeError(
            "Standard PINN + conformable GD stopped at iteration "
            f"{iteration}: invalid loss."
        )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        standard_model.parameters(),
        max_norm=GRADIENT_CLIP,
    )
    apply_conformable_gradient(standard_model, iteration)

    standard_loss_history.append(loss.item())
    standard_residual_history.append(residual_loss.item())
    standard_initial_history.append(initial_loss.item())
    standard_boundary_history.append(boundary_loss.item())

    if iteration % 200 == 0:
        print(
            "[Standard PINN + conformable GD] "
            f"Iteration: {iteration:5d} | "
            f"Loss: {loss.item():.6e} | "
            f"PDE: {residual_loss.item():.6e} | "
            f"IC: {initial_loss.item():.6e} | "
            f"BC: {boundary_loss.item():.6e} | "
            f"alpha_opt: {conformable_order(iteration):.6e} | "
            f"LR: {effective_learning_rate(iteration):.6e}"
        )

standard_time = time.time() - standard_start


# ---------------------------------------------------------------------
# Train the shared-bias PINN with Adam
# ---------------------------------------------------------------------
shared_optimizer = torch.optim.Adam(
    shared_model.parameters(),
    lr=ADAM_RATE,
)

shared_loss_history = []
shared_residual_history = []
shared_initial_history = []
shared_boundary_history = []

shared_start = time.time()

for iteration in range(NUM_ITERATIONS + 1):
    shared_optimizer.zero_grad(set_to_none=True)

    (
        loss,
        residual_loss,
        initial_loss,
        boundary_loss,
    ) = shared_total_loss(
        shared_model,
        residual_x,
        residual_t,
        initial_x,
        boundary_t,
    )

    if not torch.isfinite(loss):
        raise RuntimeError(
            "Shared-bias PINN + Adam stopped at iteration "
            f"{iteration}: invalid loss."
        )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        shared_model.parameters(),
        max_norm=GRADIENT_CLIP,
    )
    shared_optimizer.step()

    shared_loss_history.append(loss.item())
    shared_residual_history.append(residual_loss.item())
    shared_initial_history.append(initial_loss.item())
    shared_boundary_history.append(boundary_loss.item())

    if iteration % 200 == 0:
        print(
            "[Shared-bias PINN + Adam] "
            f"Iteration: {iteration:5d} | "
            f"Loss: {loss.item():.6e} | "
            f"PDE: {residual_loss.item():.6e} | "
            f"IC: {initial_loss.item():.6e} | "
            f"BC: {boundary_loss.item():.6e}"
        )

shared_time = time.time() - shared_start


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(model, trial_solution, nx=256, nt=201):
    x_grid = torch.linspace(
        X_LEFT, X_RIGHT, nx, device=DEVICE
    )
    t_grid = torch.linspace(
        0.0, FINAL_TIME, nt, device=DEVICE
    )

    x_mesh, t_mesh = torch.meshgrid(
        x_grid, t_grid, indexing="ij"
    )
    x_flat = x_mesh.reshape(-1, 1)
    t_flat = t_mesh.reshape(-1, 1)

    prediction = trial_solution(
        model, x_flat, t_flat
    ).reshape(nx, nt)

    reference = exact_solution(
        x_flat, t_flat
    ).reshape(nx, nt)

    error = prediction - reference

    return (
        x_grid.cpu().numpy(),
        t_grid.cpu().numpy(),
        prediction.cpu().numpy(),
        reference.cpu().numpy(),
        torch.max(torch.abs(error)).item(),
        torch.sqrt(torch.mean(error**2)).item(),
        (torch.norm(error) / torch.norm(reference)).item(),
    )


def recompute_standard_final_loss():
    loss, _, _, _ = standard_total_loss(
        standard_model,
        residual_x,
        residual_t,
        initial_x,
        boundary_t,
    )
    return loss.detach().item()


def recompute_shared_final_loss():
    loss, _, _, _ = shared_total_loss(
        shared_model,
        residual_x,
        residual_t,
        initial_x,
        boundary_t,
    )
    return loss.detach().item()


(
    x_grid,
    t_grid,
    standard_solution,
    exact_grid,
    standard_e_inf,
    standard_e_2,
    standard_e_2_rel,
) = evaluate_model(
    standard_model,
    standard_trial_solution,
    nx=256,
    nt=201,
)

(
    _,
    _,
    shared_solution,
    _,
    shared_e_inf,
    shared_e_2,
    shared_e_2_rel,
) = evaluate_model(
    shared_model,
    shared_trial_solution,
    nx=256,
    nt=201,
)

(
    x_surface,
    t_surface,
    standard_surface,
    exact_surface,
    _,
    _,
    _,
) = evaluate_model(
    standard_model,
    standard_trial_solution,
    nx=500,
    nt=260,
)

(
    _,
    _,
    shared_surface,
    _,
    _,
    _,
    _,
) = evaluate_model(
    shared_model,
    shared_trial_solution,
    nx=500,
    nt=260,
)

standard_final_loss = recompute_standard_final_loss()
shared_final_loss = recompute_shared_final_loss()


# ---------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------
results = [
    (
        "Standard PINN + conformable GD",
        standard_e_inf,
        standard_e_2,
        standard_e_2_rel,
        standard_final_loss,
        standard_time,
    ),
    (
        "Shared-bias PINN + Adam",
        shared_e_inf,
        shared_e_2,
        shared_e_2_rel,
        shared_final_loss,
        shared_time,
    ),
]

print(
    f"\nOptimizer-swapped comparison for omega = {OMEGA_TAG}:\n"
)

header = (
    f"{'Method':<34}"
    f"{'E_inf':>15}"
    f"{'E_2':>15}"
    f"{'E_2_rel':>15}"
    f"{'Final loss':>15}"
    f"{'CPU time (s)':>15}"
)

print(header)
print("-" * len(header))

for method, e_inf, e_2, e_2_rel, final_loss, cpu_time in results:
    print(
        f"{method:<34}"
        f"{e_inf:15.6e}"
        f"{e_2:15.6e}"
        f"{e_2_rel:15.6e}"
        f"{final_loss:15.6e}"
        f"{cpu_time:15.3f}"
    )

print("\nLaTeX table:\n")
print(r"\begin{table}[htbp]")
print(r"\centering")
print(
    r"\caption{Global errors, final training loss, and CPU time "
    r"for the optimizer-swapped manufactured time-conformable "
    rf"Burgers experiment with $\omega={OMEGA_TAG}$."
    r"}"
)
print(
    rf"\label{{tab:burgers_optimizer_swap_omega{OMEGA_TAG}}}"
)
print(r"\begin{tabular}{lccccc}")
print(r"\hline")
print(
    r"Method & $E_{\infty}$ & $E_2$ & "
    r"$E_{2,\mathrm{rel}}$ & Final loss & CPU time (s) \\"
)
print(r"\hline")

for method, e_inf, e_2, e_2_rel, final_loss, cpu_time in results:
    print(
        f"{method} & {e_inf:.6e} & {e_2:.6e} & "
        f"{e_2_rel:.6e} & {final_loss:.6e} & "
        f"{cpu_time:.3f} \\\\"
    )

print(r"\hline")
print(r"\end{tabular}")
print(r"\end{table}")


# ---------------------------------------------------------------------
# Figure output
# ---------------------------------------------------------------------
plt.close("all")


def save_pdf(figure, stem):
    """Save a PDF, using a fallback name when Windows locks the target."""

    target_path = OUTPUT_DIR / f"{stem}.pdf"

    try:
        figure.savefig(
            target_path,
            format="pdf",
            bbox_inches="tight",
        )
        return target_path
    except PermissionError:
        counter = 1

        while True:
            fallback_path = (
                OUTPUT_DIR / f"{stem}_new{counter}.pdf"
            )

            if not fallback_path.exists():
                figure.savefig(
                    fallback_path,
                    format="pdf",
                    bbox_inches="tight",
                )
                print(
                    "\nWarning: the target PDF is locked:\n"
                    f"{target_path}\n"
                    "The figure was saved instead as:\n"
                    f"{fallback_path}\n"
                )
                return fallback_path

            counter += 1


# Combined loss histories
figure_1, axis_1 = plt.subplots(
    figsize=(7.6, 4.4), dpi=240
)

axis_1.semilogy(
    standard_loss_history,
    color=COLOR_STANDARD,
    label="Standard PINN + conformable GD",
)
axis_1.semilogy(
    shared_loss_history,
    "--",
    color=COLOR_SHARED,
    label="Shared-bias PINN + Adam",
)
axis_1.set_xlabel("Iteration")
axis_1.set_ylabel("Training loss")
axis_1.set_title(
    "Optimizer-swapped training-loss comparison, "
    rf"$\omega={OMEGA_TAG}$"
)
axis_1.set_xlim(left=0)
axis_1.margins(x=0)
axis_1.legend()

figure_1.tight_layout()
save_pdf(
    figure_1,
    f"burgers_optimizer_swap_loss_comparison_omega{OMEGA_TAG}",
)
plt.close(figure_1)


# Exact and numerical solution surfaces
x_surface_mesh, t_surface_mesh = np.meshgrid(
    x_surface,
    t_surface,
    indexing="ij",
)

figure_2 = plt.figure(
    figsize=(20.0, 6.0),
    dpi=240,
    constrained_layout=True,
)

solution_surfaces = [
    ("Exact solution", exact_surface),
    (
        "Standard PINN + conformable GD",
        standard_surface,
    ),
    (
        "Shared-bias PINN + Adam",
        shared_surface,
    ),
]

for index, (title, values) in enumerate(
    solution_surfaces, start=1
):
    axis = figure_2.add_subplot(
        1, 3, index, projection="3d"
    )

    surface = axis.plot_surface(
        x_surface_mesh,
        t_surface_mesh,
        values,
        cmap="viridis",
        rcount=220,
        ccount=180,
        linewidth=0,
        edgecolor="none",
        antialiased=True,
    )

    axis.set_xlabel("x", labelpad=10)
    axis.set_ylabel("t", labelpad=10)
    axis.set_zlabel("u(x,t)", labelpad=10)
    axis.set_title(
        rf"{title}, $\omega={OMEGA_TAG}$", pad=12
    )
    axis.view_init(elev=28, azim=-135)

    figure_2.colorbar(
        surface,
        ax=axis,
        shrink=0.73,
        pad=0.08,
    )

save_pdf(
    figure_2,
    f"burgers_optimizer_swap_solution_surfaces_omega{OMEGA_TAG}",
)
plt.close(figure_2)


# Three-dimensional absolute-error surfaces
figure_3 = plt.figure(
    figsize=(14.0, 6.0),
    dpi=240,
    constrained_layout=True,
)

error_surfaces = [
    (
        "Standard PINN + CFG error",
        np.abs(standard_surface - exact_surface),
    ),
    (
        "Shared-bias PINN + Adam error",
        np.abs(shared_surface - exact_surface),
    ),
]

for index, (title, values) in enumerate(
    error_surfaces, start=1
):
    axis = figure_3.add_subplot(
        1, 2, index, projection="3d"
    )

    surface = axis.plot_surface(
        x_surface_mesh,
        t_surface_mesh,
        values,
        cmap="magma",
        rcount=220,
        ccount=180,
        linewidth=0,
        edgecolor="none",
        antialiased=True,
    )

    axis.set_xlabel("x", labelpad=10)
    axis.set_ylabel("t", labelpad=10)
    axis.set_zlabel("|error|", labelpad=10)
    axis.set_title(
        rf"{title}, $\omega={OMEGA_TAG}$", pad=12
    )
    axis.view_init(elev=28, azim=-135)

    figure_3.colorbar(
        surface,
        ax=axis,
        shrink=0.75,
        pad=0.08,
    )

save_pdf(
    figure_3,
    f"burgers_optimizer_swap_error_surfaces_omega{OMEGA_TAG}",
)
plt.close(figure_3)


# Final-time comparison
figure_4, axis_4 = plt.subplots(
    figsize=(7.6, 4.4), dpi=240
)

axis_4.plot(
    x_grid,
    exact_grid[:, -1],
    color=COLOR_EXACT,
    label="Exact",
)
axis_4.plot(
    x_grid,
    standard_solution[:, -1],
    "--",
    color=COLOR_STANDARD,
    label="Standard PINN + conformable GD",
)
axis_4.plot(
    x_grid,
    shared_solution[:, -1],
    "-.",
    color=COLOR_SHARED,
    label="Shared-bias PINN + Adam",
)

axis_4.set_xlabel("x")
axis_4.set_ylabel("u(x,1)")
axis_4.set_title(
    rf"Final-time comparison at $t=1$, $\omega={OMEGA_TAG}$"
)
axis_4.legend()

figure_4.tight_layout()
save_pdf(
    figure_4,
    f"burgers_optimizer_swap_final_time_omega{OMEGA_TAG}",
)
plt.close(figure_4)


# Absolute-error heatmaps
figure_5, axes = plt.subplots(
    1,
    2,
    figsize=(11.5, 4.8),
    dpi=240,
    constrained_layout=True,
)

heatmaps = [
    (
        "Standard PINN + CFG error",
        np.abs(standard_solution - exact_grid),
    ),
    (
        "Shared-bias PINN + Adam error",
        np.abs(shared_solution - exact_grid),
    ),
]

for axis, (title, values) in zip(axes, heatmaps):
    image = axis.imshow(
        values.T,
        extent=[X_LEFT, X_RIGHT, 0.0, FINAL_TIME],
        origin="lower",
        aspect="auto",
        cmap="magma",
    )

    axis.set_xlabel("x")
    axis.set_ylabel("t")
    axis.set_title(
        rf"{title}, $\omega={OMEGA_TAG}$"
    )

    figure_5.colorbar(image, ax=axis, shrink=0.9)

save_pdf(
    figure_5,
    f"burgers_optimizer_swap_error_heatmaps_omega{OMEGA_TAG}",
)
plt.close(figure_5)

print(f"\nPDF figures saved in:\n{OUTPUT_DIR}")
