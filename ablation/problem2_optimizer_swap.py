"""
Optimizer-swapped study for the manufactured time-conformable
Burgers--Fisher problem.

The script compares:
    1. A standard PINN trained with conformable-gradient descent.
    2. A shared-bias PINN trained with Adam.

The governing equation is

    T_q u + u u_x - nu u_xx - lambda u(1-u) = f(x,t).

The equation, network architectures, random-number sequence,
collocation strategy, resampling interval, iteration counts, learning
rates, gradient clipping, error measures, table output, and figure
filenames are unchanged.
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
# Output folder
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Plot settings
# ---------------------------------------------------------------------
plt.rcParams.update(
    {
        "figure.dpi": 220,
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
# Reproducibility and precision
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
NU = 0.01
REACTION_RATE = 1.0

X_LEFT = -math.pi
X_RIGHT = math.pi
FINAL_TIME = 1.0
MINIMUM_TIME = 1.0e-4


# ---------------------------------------------------------------------
# Training settings
# ---------------------------------------------------------------------
HIDDEN_UNITS = 100
NUM_RESIDUAL_POINTS = 200
NUM_INITIAL_POINTS = 100
NUM_BOUNDARY_POINTS = 100
NUM_ITERATIONS = 20000

CONFORMABLE_BASE_RATE = 5.0e-2
ADAM_RATE = 51.0e-3
GRADIENT_CLIP = 10.0
RESAMPLING_INTERVAL = 200


# ---------------------------------------------------------------------
# Exact solution, initial condition, and forcing term
# ---------------------------------------------------------------------
def exact_solution(x, t):
    spatial_mode = (
        torch.sin(2.0 * x) + 0.5 * torch.sin(4.0 * x)
    )
    return 0.5 + 0.25 * torch.exp(-t) * spatial_mode


def initial_condition(x):
    return 0.5 + 0.25 * (
        torch.sin(2.0 * x) + 0.5 * torch.sin(4.0 * x)
    )


def forcing_term(x, t, q):
    decay = torch.exp(-t)
    spatial_mode = (
        torch.sin(2.0 * x) + 0.5 * torch.sin(4.0 * x)
    )

    u = 0.5 + 0.25 * decay * spatial_mode
    u_t = -0.25 * decay * spatial_mode
    u_x = 0.25 * decay * (
        2.0 * torch.cos(2.0 * x)
        + 2.0 * torch.cos(4.0 * x)
    )
    u_xx = 0.25 * decay * (
        -4.0 * torch.sin(2.0 * x)
        - 8.0 * torch.sin(4.0 * x)
    )

    return (
        t ** (1.0 - q) * u_t
        + u * u_x
        - NU * u_xx
        - REACTION_RATE * u * (1.0 - u)
    )


# ---------------------------------------------------------------------
# Feature maps
# ---------------------------------------------------------------------
def standard_features(x, t):
    return torch.cat((x, t), dim=1)


def periodic_features(x, t):
    return torch.cat(
        (
            torch.sin(2.0 * x),
            torch.cos(2.0 * x),
            torch.sin(4.0 * x),
            torch.cos(4.0 * x),
            t,
        ),
        dim=1,
    )


# ---------------------------------------------------------------------
# Neural-network models
# ---------------------------------------------------------------------
class StandardPINN(nn.Module):
    """One-hidden-layer PINN with ordinary hidden biases."""

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

    def forward(self, inputs):
        return self.fc2(self.act(self.fc1(inputs)))


class SharedBiasPINN(nn.Module):
    """One-hidden-layer PINN with one common hidden-layer bias."""

    def __init__(self, input_dim=5, hidden_dim=100, output_dim=1):
        super().__init__()

        # The parameter registration and initialization order is preserved.
        self.W1 = nn.Parameter(
            torch.empty(input_dim, hidden_dim, dtype=DTYPE)
        )
        self.W2 = nn.Parameter(
            torch.empty(hidden_dim, output_dim, dtype=DTYPE)
        )
        self.b1 = nn.Parameter(torch.zeros(1, 1, dtype=DTYPE))
        self.b2 = nn.Parameter(
            torch.zeros(1, output_dim, dtype=DTYPE)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.W1)
        nn.init.xavier_normal_(self.W2)

        with torch.no_grad():
            self.b1.zero_()
            self.b2.zero_()

    def forward(self, inputs):
        hidden_argument = inputs @ self.W1 + self.b1
        hidden_output = torch.tanh(hidden_argument)
        return hidden_output @ self.W2 + self.b2


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
# Trial solutions
# ---------------------------------------------------------------------
def standard_trial_solution(model, x, t):
    network_output = model(standard_features(x, t))
    return initial_condition(x) + t * network_output


def shared_trial_solution(model, x, t):
    network_output = model(periodic_features(x, t))
    return initial_condition(x) + t * network_output


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
        - REACTION_RATE * u * (1.0 - u)
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
        - REACTION_RATE * u * (1.0 - u)
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
    boundary_t,
):
    residual = shared_residual(
        model, residual_x, residual_t
    )

    residual_loss = torch.mean(residual**2)
    boundary_loss = shared_boundary_loss(model, boundary_t)

    total_loss = residual_loss + boundary_loss
    return total_loss, residual_loss, boundary_loss


# ---------------------------------------------------------------------
# Conformable-gradient update
# ---------------------------------------------------------------------
def conformable_order(iteration):
    return 1.0 - math.atan(iteration) / math.pi


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
# Sampling
# ---------------------------------------------------------------------
def sample_residual_points():
    # Time is sampled before space to preserve the original RNG order.
    residual_t = MINIMUM_TIME + (
        FINAL_TIME - MINIMUM_TIME
    ) * torch.rand(NUM_RESIDUAL_POINTS, 1, device=DEVICE)

    residual_x = X_LEFT + (
        X_RIGHT - X_LEFT
    ) * torch.rand(NUM_RESIDUAL_POINTS, 1, device=DEVICE)

    return residual_x, residual_t


def sample_initial_points():
    return torch.linspace(
        X_LEFT,
        X_RIGHT,
        NUM_INITIAL_POINTS,
        device=DEVICE,
    ).reshape(-1, 1)


def sample_boundary_times():
    return torch.linspace(
        MINIMUM_TIME,
        FINAL_TIME,
        NUM_BOUNDARY_POINTS,
        device=DEVICE,
    ).reshape(-1, 1)


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


# ---------------------------------------------------------------------
# Table output
# ---------------------------------------------------------------------
def print_global_table(rows):
    print("\nGlobal comparison table:\n")

    header = (
        f"{'Method':<38}"
        f"{'E_inf':>15}"
        f"{'E_2':>15}"
        f"{'E_2_rel':>15}"
        f"{'Final loss':>15}"
        f"{'CPU time (s)':>15}"
    )

    print(header)
    print("-" * len(header))

    for method, e_inf, e_2, e_2_rel, final_loss, cpu_time in rows:
        print(
            f"{method:<38}"
            f"{e_inf:15.6e}"
            f"{e_2:15.6e}"
            f"{e_2_rel:15.6e}"
            f"{final_loss:15.6e}"
            f"{cpu_time:15.3f}"
        )


def make_latex_table(rows):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{Global errors, final training loss, and CPU time "
            r"for the optimizer-swapped manufactured time-conformable "
            r"Burgers--Fisher experiment.}"
        ),
        r"\label{tab:burgers_fisher_optimizer_swap}",
        r"\begin{tabular}{lccccc}",
        r"\hline",
        (
            r"Method & $E_{\infty}$ & $E_2$ & "
            r"$E_{2,\mathrm{rel}}$ & Final loss & CPU time (s) \\"
        ),
        r"\hline",
    ]

    for method, e_inf, e_2, e_2_rel, final_loss, cpu_time in rows:
        lines.append(
            f"{method} & {e_inf:.6e} & {e_2:.6e} & "
            f"{e_2_rel:.6e} & {final_loss:.6e} & "
            f"{cpu_time:.3f} \\\\"
        )

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Figure output
# ---------------------------------------------------------------------
def save_pdf(figure, stem):
    """Save a PDF and use a fallback name when Windows locks the target."""

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
                    f"\nWarning: the target PDF is locked:\n"
                    f"{target_path}\n"
                    f"The figure was saved instead as:\n"
                    f"{fallback_path}\n"
                )
                return fallback_path

            counter += 1


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------
def main():
    rows = []

    # The original RNG sequence is preserved:
    # standard model, standard points, shared model, shared points.
    standard_model = StandardPINN(
        input_dim=2,
        hidden_dim=HIDDEN_UNITS,
        output_dim=1,
    ).to(DEVICE)

    standard_x, standard_t = sample_residual_points()
    initial_x = sample_initial_points()
    standard_boundary_t = sample_boundary_times()

    standard_history = []
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
            standard_x,
            standard_t,
            initial_x,
            standard_boundary_t,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Standard PINN + conformable GD stopped at "
                f"iteration {iteration}: invalid loss."
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            standard_model.parameters(),
            max_norm=GRADIENT_CLIP,
        )
        apply_conformable_gradient(standard_model, iteration)

        standard_history.append(loss.item())

        if iteration % 200 == 0:
            print(
                "[Standard PINN + conformable GD] "
                f"Iter {iteration:5d} | "
                f"Loss = {loss.item():.6e} | "
                f"PDE = {residual_loss.item():.6e} | "
                f"IC = {initial_loss.item():.6e} | "
                f"BC = {boundary_loss.item():.6e} | "
                f"alpha = {conformable_order(iteration):.6e} | "
                f"LR = {effective_learning_rate(iteration):.6e}"
            )

    standard_time = time.time() - standard_start
    standard_final_loss = (
        standard_history[-1] if standard_history else float("nan")
    )

    (
        standard_x_grid,
        standard_t_grid,
        standard_solution,
        standard_exact,
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
        standard_surface_x,
        standard_surface_t,
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

    rows.append(
        (
            "Standard PINN + conformable GD",
            standard_e_inf,
            standard_e_2,
            standard_e_2_rel,
            standard_final_loss,
            standard_time,
        )
    )

    shared_model = SharedBiasPINN(
        input_dim=5,
        hidden_dim=HIDDEN_UNITS,
        output_dim=1,
    ).to(DEVICE)

    shared_x, shared_t = sample_residual_points()
    shared_boundary_t = sample_boundary_times()

    shared_optimizer = torch.optim.Adam(
        shared_model.parameters(),
        lr=ADAM_RATE,
    )

    shared_history = []
    shared_start = time.time()

    for iteration in range(NUM_ITERATIONS + 1):
        if iteration % RESAMPLING_INTERVAL == 0:
            shared_x, shared_t = sample_residual_points()

        shared_optimizer.zero_grad(set_to_none=True)

        loss, residual_loss, boundary_loss = shared_total_loss(
            shared_model,
            shared_x,
            shared_t,
            shared_boundary_t,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Shared-bias PINN + Adam stopped at "
                f"iteration {iteration}: invalid loss."
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            shared_model.parameters(),
            max_norm=GRADIENT_CLIP,
        )
        shared_optimizer.step()

        shared_history.append(loss.item())

        if iteration % 200 == 0:
            print(
                "[Shared-bias PINN + Adam] "
                f"Iter {iteration:5d} | "
                f"Loss = {loss.item():.6e} | "
                f"PDE = {residual_loss.item():.6e} | "
                f"BC = {boundary_loss.item():.6e}"
            )

    shared_time = time.time() - shared_start
    shared_final_loss = (
        shared_history[-1] if shared_history else float("nan")
    )

    (
        shared_x_grid,
        shared_t_grid,
        shared_solution,
        shared_exact,
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
        shared_surface_x,
        shared_surface_t,
        shared_surface,
        shared_exact_surface,
        _,
        _,
        _,
    ) = evaluate_model(
        shared_model,
        shared_trial_solution,
        nx=500,
        nt=260,
    )

    rows.append(
        (
            "Shared-bias PINN + Adam",
            shared_e_inf,
            shared_e_2,
            shared_e_2_rel,
            shared_final_loss,
            shared_time,
        )
    )

    # Loss history
    figure_1, axis_1 = plt.subplots(
        figsize=(7.6, 4.4), dpi=240
    )
    axis_1.semilogy(
        standard_history,
        color=COLOR_STANDARD,
        label="Standard PINN + conformable GD",
    )
    axis_1.semilogy(
        shared_history,
        color=COLOR_SHARED,
        label="Shared-bias PINN + Adam",
    )
    axis_1.set_xlabel("Iteration")
    axis_1.set_ylabel("Loss")
    axis_1.set_title(
        "Optimizer-swapped loss history comparison"
    )
    axis_1.legend()
    figure_1.tight_layout()
    save_pdf(figure_1, "optimizer_swap_loss_history_prob2")
    plt.close(figure_1)

    # Final-time comparison
    figure_2, axis_2 = plt.subplots(
        figsize=(7.6, 4.4), dpi=240
    )
    axis_2.plot(
        standard_x_grid,
        standard_exact[:, -1],
        color=COLOR_EXACT,
        label="Exact",
    )
    axis_2.plot(
        standard_x_grid,
        standard_solution[:, -1],
        "--",
        color=COLOR_STANDARD,
        label="Standard PINN + conformable GD",
    )
    axis_2.plot(
        shared_x_grid,
        shared_solution[:, -1],
        "-.",
        color=COLOR_SHARED,
        label="Shared-bias PINN + Adam",
    )
    axis_2.set_xlabel("x")
    axis_2.set_ylabel("u(x,1)")
    axis_2.set_title("Final-time comparison at $t=1$")
    axis_2.legend()
    figure_2.tight_layout()
    save_pdf(figure_2, "optimizer_swap_final_time_prob2")
    plt.close(figure_2)

    # Solution heatmaps
    figure_3, axes_3 = plt.subplots(
        1,
        3,
        figsize=(15.8, 4.8),
        dpi=240,
        constrained_layout=True,
    )

    solution_heatmaps = [
        ("Exact", standard_exact),
        ("Standard PINN + conformable GD", standard_solution),
        ("Shared-bias PINN + Adam", shared_solution),
    ]

    for axis, (title, values) in zip(
        axes_3, solution_heatmaps
    ):
        image = axis.imshow(
            values.T,
            extent=[
                X_LEFT,
                X_RIGHT,
                0.0,
                FINAL_TIME,
            ],
            origin="lower",
            aspect="auto",
            cmap="viridis",
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("t")
        figure_3.colorbar(image, ax=axis, shrink=0.88)

    save_pdf(
        figure_3,
        "optimizer_swap_solution_heatmaps_prob2",
    )
    plt.close(figure_3)

    # Error heatmaps
    figure_4, axes_4 = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.8),
        dpi=240,
        constrained_layout=True,
    )

    error_heatmaps = [
        (
            "Standard PINN + conformable GD error",
            np.abs(standard_solution - standard_exact),
        ),
        (
            "Shared-bias PINN + Adam error",
            np.abs(shared_solution - shared_exact),
        ),
    ]

    for axis, (title, values) in zip(
        axes_4, error_heatmaps
    ):
        image = axis.imshow(
            values.T,
            extent=[
                X_LEFT,
                X_RIGHT,
                0.0,
                FINAL_TIME,
            ],
            origin="lower",
            aspect="auto",
            cmap="magma",
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("t")
        figure_4.colorbar(image, ax=axis, shrink=0.88)

    save_pdf(
        figure_4,
        "optimizer_swap_error_heatmaps_prob2",
    )
    plt.close(figure_4)

    # Three-dimensional solution surfaces
    standard_x_mesh, standard_t_mesh = np.meshgrid(
        standard_surface_x,
        standard_surface_t,
        indexing="ij",
    )
    shared_x_mesh, shared_t_mesh = np.meshgrid(
        shared_surface_x,
        shared_surface_t,
        indexing="ij",
    )

    figure_5 = plt.figure(
        figsize=(22.0, 6.4),
        dpi=240,
        constrained_layout=True,
    )

    solution_surfaces = [
        (
            "Exact solution",
            standard_x_mesh,
            standard_t_mesh,
            exact_surface,
        ),
        (
            "Standard PINN + conformable GD",
            standard_x_mesh,
            standard_t_mesh,
            standard_surface,
        ),
        (
            "Shared-bias PINN + Adam",
            shared_x_mesh,
            shared_t_mesh,
            shared_surface,
        ),
    ]

    for index, (title, x_mesh, t_mesh, values) in enumerate(
        solution_surfaces, start=1
    ):
        axis = figure_5.add_subplot(
            1, 3, index, projection="3d"
        )
        surface = axis.plot_surface(
            x_mesh,
            t_mesh,
            values,
            cmap="viridis",
            rcount=240,
            ccount=200,
            linewidth=0,
            edgecolor="none",
            antialiased=True,
        )
        axis.set_xlabel("x", labelpad=10)
        axis.set_ylabel("t", labelpad=10)
        axis.set_zlabel("u(x,t)", labelpad=10)
        axis.set_title(title, pad=12)
        axis.view_init(elev=28, azim=-135)
        figure_5.colorbar(
            surface,
            ax=axis,
            shrink=0.72,
            pad=0.08,
        )

    save_pdf(
        figure_5,
        "optimizer_swap_3d_solutions_prob2",
    )
    plt.close(figure_5)

    # Three-dimensional error surfaces
    figure_6 = plt.figure(
        figsize=(16.0, 6.2),
        dpi=240,
        constrained_layout=True,
    )

    error_surfaces = [
        (
            "Standard PINN + conformable GD error",
            standard_x_mesh,
            standard_t_mesh,
            np.abs(standard_surface - exact_surface),
        ),
        (
            "Shared-bias PINN + Adam error",
            shared_x_mesh,
            shared_t_mesh,
            np.abs(shared_surface - shared_exact_surface),
        ),
    ]

    for index, (title, x_mesh, t_mesh, values) in enumerate(
        error_surfaces, start=1
    ):
        axis = figure_6.add_subplot(
            1, 2, index, projection="3d"
        )
        surface = axis.plot_surface(
            x_mesh,
            t_mesh,
            values,
            cmap="magma",
            rcount=240,
            ccount=200,
            linewidth=0,
            edgecolor="none",
            antialiased=True,
        )
        axis.set_xlabel("x", labelpad=10)
        axis.set_ylabel("t", labelpad=10)
        axis.set_zlabel("|error|", labelpad=10)
        axis.set_title(title, pad=12)
        axis.view_init(elev=28, azim=-135)
        figure_6.colorbar(
            surface,
            ax=axis,
            shrink=0.75,
            pad=0.08,
        )

    save_pdf(
        figure_6,
        "optimizer_swap_3d_errors_prob2",
    )
    plt.close(figure_6)

    print_global_table(rows)

    print("\nLaTeX global error table:\n")
    print(make_latex_table(rows))

    print(f"\nPDF figures saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
