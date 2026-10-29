"""Problem 3: manufactured time-conformable reaction--diffusion equation.

Three frequency pairs are considered:
    (omega_1, omega_2) = (2, 5), (4, 9), and (6, 12).

For each case, the script compares a standard PINN with a shared-bias
PINN trained by the conformable-gradient update. It reports E_inf, E_2,
E_2_rel, the final training loss, and CPU time. All figures are saved as
PDF files.
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
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    torch.set_num_threads(1)
except Exception:
    pass


# -----------------------------------------------------------------------------
# Numerical and plotting settings
# -----------------------------------------------------------------------------
DEVICE = "cpu"
DTYPE = torch.float64
SEED = 0

torch.set_default_dtype(DTYPE)
torch.manual_seed(SEED)
np.random.seed(SEED)

plt.rcParams.update(
    {
        "figure.dpi": 220,
        "savefig.dpi": 300,
        "pdf.compression": 9,
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

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Problem and training parameters
# -----------------------------------------------------------------------------
Q_ORDER = 0.7
NU = 0.01 / math.pi
LAMBDA = 1.0

X_LEFT = 0.0
X_RIGHT = 2.0 * math.pi
T_FINAL = 1.0
TIME_REGULARIZATION = 1.0e-3
MIN_RESIDUAL_TIME = 1.0e-4

HIDDEN_WIDTH = 100
NUM_RESIDUAL_POINTS = 200
NUM_INITIAL_POINTS = 100
NUM_BOUNDARY_POINTS = 100
NUM_ITERATIONS = 20000
GRADIENT_CLIP = 10.0

ADAM_RATE = 5.0e-2
CONFORMABLE_BASE_RATE = 5.0e-2
RESAMPLING_INTERVAL = 200

CASES = (
    ("case1", 2.0, 5.0),
    ("case2", 4.0, 9.0),
    ("case3", 6.0, 12.0),
)


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------
def derivative(outputs, inputs):
    """Differentiate ``outputs`` with respect to ``inputs``."""
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def smooth_curve(values, beta=0.98):
    """Return an exponentially smoothed copy of a loss history."""
    smoothed = []
    running_average = 0.0

    for index, value in enumerate(values, start=1):
        running_average = beta * running_average + (1.0 - beta) * value
        correction = 1.0 - beta**index
        smoothed.append(running_average / correction)

    return smoothed


def save_pdf(fig, filename_stem):
    """Save a figure as PDF, including a fallback for locked files."""
    target = OUTPUT_DIR / f"{filename_stem}.pdf"

    try:
        fig.savefig(target, format="pdf", dpi=300, bbox_inches="tight")
        return target
    except PermissionError:
        suffix = 1
        while True:
            fallback = OUTPUT_DIR / f"{filename_stem}_new{suffix}.pdf"
            if not fallback.exists():
                fig.savefig(
                    fallback,
                    format="pdf",
                    dpi=300,
                    bbox_inches="tight",
                )
                print(
                    f"\nWarning: the target PDF is locked:\n{target}\n"
                    f"The figure was saved instead as:\n{fallback}\n"
                    "Close the original PDF, then replace it with the new file."
                )
                return fallback
            suffix += 1


def sample_interior_points(number_of_points):
    """Sample interior residual points in the space--time domain."""
    time_points = MIN_RESIDUAL_TIME + (
        T_FINAL - MIN_RESIDUAL_TIME
    ) * torch.rand(number_of_points, 1, device=DEVICE)

    space_points = X_LEFT + (X_RIGHT - X_LEFT) * torch.rand(
        number_of_points,
        1,
        device=DEVICE,
    )

    return space_points, time_points


def sample_boundary_times(number_of_points):
    return torch.linspace(
        MIN_RESIDUAL_TIME,
        T_FINAL,
        number_of_points,
        device=DEVICE,
    ).reshape(-1, 1)


# -----------------------------------------------------------------------------
# Neural networks
# -----------------------------------------------------------------------------
class StandardPINN(nn.Module):
    """One-hidden-layer PINN with ordinary neuron-specific biases."""

    def __init__(self, input_dim=2, hidden_dim=100, output_dim=1):
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
        hidden_values = torch.tanh(self.hidden_layer(inputs))
        return self.output_layer(hidden_values)


class SharedBiasPINN(nn.Module):
    """One-hidden-layer network with one common hidden-layer bias."""

    def __init__(self, input_dim=5, hidden_dim=100, output_dim=1):
        super().__init__()
        self.input_weights = nn.Parameter(
            torch.empty(input_dim, hidden_dim, dtype=DTYPE)
        )
        self.output_weights = nn.Parameter(
            torch.empty(hidden_dim, output_dim, dtype=DTYPE)
        )
        self.shared_bias = nn.Parameter(torch.zeros(1, 1, dtype=DTYPE))
        self.output_bias = nn.Parameter(
            torch.zeros(1, output_dim, dtype=DTYPE)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.input_weights)
        nn.init.xavier_normal_(self.output_weights)
        with torch.no_grad():
            self.shared_bias.zero_()
            self.output_bias.zero_()

    def forward(self, inputs):
        hidden_values = torch.tanh(
            inputs @ self.input_weights + self.shared_bias
        )
        return hidden_values @ self.output_weights + self.output_bias


# -----------------------------------------------------------------------------
# Case-dependent exact solution, forcing term, and periodic features
# -----------------------------------------------------------------------------
def build_case_functions(omega_1, omega_2):
    def first_time_factor(time):
        return torch.exp(-torch.sqrt(time + TIME_REGULARIZATION))

    def first_time_derivative(time):
        root = torch.sqrt(time + TIME_REGULARIZATION)
        return -0.5 * torch.exp(-root) / root

    def second_time_factor(time):
        return torch.exp(-time)

    def second_time_derivative(time):
        return -torch.exp(-time)

    def exact_solution(space, time):
        return (
            first_time_factor(time) * torch.cos(omega_1 * space)
            + 0.35
            * second_time_factor(time)
            * torch.sin(omega_2 * space)
        )

    def forcing_term(space, time, order):
        first_factor = first_time_factor(time)
        first_derivative = first_time_derivative(time)
        second_factor = second_time_factor(time)
        second_derivative = second_time_derivative(time)

        solution_time_derivative = (
            first_derivative * torch.cos(omega_1 * space)
            + 0.35
            * second_derivative
            * torch.sin(omega_2 * space)
        )

        solution_second_space_derivative = (
            -(omega_1**2)
            * first_factor
            * torch.cos(omega_1 * space)
            - 0.35
            * (omega_2**2)
            * second_factor
            * torch.sin(omega_2 * space)
        )

        return (
            time ** (1.0 - order) * solution_time_derivative
            - NU * solution_second_space_derivative
            + LAMBDA * exact_solution(space, time)
        )

    def initial_condition(space):
        return exact_solution(space, torch.zeros_like(space))

    def periodic_features(space, time):
        return torch.cat(
            [
                torch.sin(omega_1 * space),
                torch.cos(omega_1 * space),
                torch.sin(omega_2 * space),
                torch.cos(omega_2 * space),
                time,
            ],
            dim=1,
        )

    return exact_solution, forcing_term, initial_condition, periodic_features


# -----------------------------------------------------------------------------
# Standard PINN formulation
# -----------------------------------------------------------------------------
def standard_inputs(space, time):
    return torch.cat([space, time], dim=1)


def standard_trial_solution(model, space, time, initial_condition):
    return initial_condition(space) + time * model(standard_inputs(space, time))


def standard_residual(
    model,
    space,
    time,
    order,
    forcing_term,
    initial_condition,
):
    space = space.clone().detach().requires_grad_(True)
    time = time.clone().detach().requires_grad_(True)

    solution = standard_trial_solution(
        model,
        space,
        time,
        initial_condition,
    )
    solution_time_derivative = derivative(solution, time)
    solution_space_derivative = derivative(solution, space)
    solution_second_space_derivative = derivative(
        solution_space_derivative,
        space,
    )

    conformable_time_derivative = (
        time ** (1.0 - order) * solution_time_derivative
    )

    return (
        conformable_time_derivative
        - NU * solution_second_space_derivative
        + LAMBDA * solution
        - forcing_term(space, time, order)
    )


def standard_initial_loss(model, initial_points, initial_condition):
    initial_times = torch.zeros_like(initial_points)
    prediction = standard_trial_solution(
        model,
        initial_points,
        initial_times,
        initial_condition,
    )
    return torch.mean((prediction - initial_condition(initial_points)) ** 2)


def standard_boundary_loss(model, boundary_times, initial_condition):
    left_points = torch.full_like(
        boundary_times,
        X_LEFT,
        requires_grad=True,
    )
    right_points = torch.full_like(
        boundary_times,
        X_RIGHT,
        requires_grad=True,
    )
    left_times = boundary_times.clone().detach().requires_grad_(True)
    right_times = boundary_times.clone().detach().requires_grad_(True)

    left_solution = standard_trial_solution(
        model,
        left_points,
        left_times,
        initial_condition,
    )
    right_solution = standard_trial_solution(
        model,
        right_points,
        right_times,
        initial_condition,
    )

    left_derivative = derivative(left_solution, left_points)
    right_derivative = derivative(right_solution, right_points)

    value_loss = torch.mean((left_solution - right_solution) ** 2)
    derivative_loss = torch.mean((left_derivative - right_derivative) ** 2)
    return value_loss + derivative_loss


def standard_total_loss(
    model,
    residual_space,
    residual_time,
    initial_points,
    boundary_times,
    order,
    forcing_term,
    initial_condition,
):
    residual = standard_residual(
        model,
        residual_space,
        residual_time,
        order,
        forcing_term,
        initial_condition,
    )
    residual_loss = torch.mean(residual**2)
    initial_loss = standard_initial_loss(
        model,
        initial_points,
        initial_condition,
    )
    boundary_loss = standard_boundary_loss(
        model,
        boundary_times,
        initial_condition,
    )
    total_loss = residual_loss + initial_loss + boundary_loss

    return total_loss, residual_loss, initial_loss, boundary_loss


# -----------------------------------------------------------------------------
# Shared-bias PINN formulation
# -----------------------------------------------------------------------------
def shared_trial_solution(
    model,
    space,
    time,
    initial_condition,
    periodic_features,
):
    features = periodic_features(space, time)
    return initial_condition(space) + time * model(features)


def shared_residual(
    model,
    space,
    time,
    order,
    forcing_term,
    initial_condition,
    periodic_features,
):
    space = space.clone().detach().requires_grad_(True)
    time = time.clone().detach().requires_grad_(True)

    solution = shared_trial_solution(
        model,
        space,
        time,
        initial_condition,
        periodic_features,
    )
    solution_time_derivative = derivative(solution, time)
    solution_space_derivative = derivative(solution, space)
    solution_second_space_derivative = derivative(
        solution_space_derivative,
        space,
    )

    conformable_time_derivative = (
        time ** (1.0 - order) * solution_time_derivative
    )

    return (
        conformable_time_derivative
        - NU * solution_second_space_derivative
        + LAMBDA * solution
        - forcing_term(space, time, order)
    )


def shared_boundary_loss(
    model,
    boundary_times,
    initial_condition,
    periodic_features,
):
    left_points = torch.full_like(
        boundary_times,
        X_LEFT,
        requires_grad=True,
    )
    right_points = torch.full_like(
        boundary_times,
        X_RIGHT,
        requires_grad=True,
    )
    left_times = boundary_times.clone().detach().requires_grad_(True)
    right_times = boundary_times.clone().detach().requires_grad_(True)

    left_solution = shared_trial_solution(
        model,
        left_points,
        left_times,
        initial_condition,
        periodic_features,
    )
    right_solution = shared_trial_solution(
        model,
        right_points,
        right_times,
        initial_condition,
        periodic_features,
    )

    left_derivative = derivative(left_solution, left_points)
    right_derivative = derivative(right_solution, right_points)

    value_loss = torch.mean((left_solution - right_solution) ** 2)
    derivative_loss = torch.mean((left_derivative - right_derivative) ** 2)
    return value_loss + derivative_loss


def shared_total_loss(
    model,
    residual_space,
    residual_time,
    boundary_times,
    order,
    forcing_term,
    initial_condition,
    periodic_features,
):
    residual = shared_residual(
        model,
        residual_space,
        residual_time,
        order,
        forcing_term,
        initial_condition,
        periodic_features,
    )
    residual_loss = torch.mean(residual**2)
    boundary_loss = shared_boundary_loss(
        model,
        boundary_times,
        initial_condition,
        periodic_features,
    )
    total_loss = residual_loss + boundary_loss

    return total_loss, residual_loss, boundary_loss


# -----------------------------------------------------------------------------
# Conformable-gradient update
# -----------------------------------------------------------------------------
def conformable_order(iteration):
    return 1.5 - 2.0 * (
        math.atan(iteration) + math.pi / 2.0
    ) / math.pi / 2.0


def conformable_learning_rate(iteration, base_rate=3.0e-2):
    order = max(conformable_order(iteration), 1.0e-8)
    return base_rate * order


@torch.no_grad()
def apply_conformable_gradient(model, iteration, base_rate=7.0e-3):
    order = max(conformable_order(iteration), 1.0e-14)
    learning_rate = conformable_learning_rate(iteration, base_rate)

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        parameter_norm = torch.norm(parameter).clamp_min(1.0e-14)
        scaled_gradient = (
            parameter_norm**order
        ) * parameter.grad / order
        parameter -= learning_rate * scaled_gradient


# -----------------------------------------------------------------------------
# Training and evaluation
# -----------------------------------------------------------------------------
def train_standard_model(forcing_term, initial_condition):
    model = StandardPINN(
        input_dim=2,
        hidden_dim=HIDDEN_WIDTH,
        output_dim=1,
    ).to(DEVICE)

    residual_time = MIN_RESIDUAL_TIME + (
        T_FINAL - MIN_RESIDUAL_TIME
    ) * torch.rand(NUM_RESIDUAL_POINTS, 1, device=DEVICE)
    residual_space = X_LEFT + (X_RIGHT - X_LEFT) * torch.rand(
        NUM_RESIDUAL_POINTS,
        1,
        device=DEVICE,
    )
    initial_points = torch.linspace(
        X_LEFT,
        X_RIGHT,
        NUM_INITIAL_POINTS,
        device=DEVICE,
    ).reshape(-1, 1)
    boundary_times = sample_boundary_times(NUM_BOUNDARY_POINTS)

    optimizer = torch.optim.Adam(model.parameters(), lr=ADAM_RATE)
    loss_history = []

    start_time = time.time()

    for iteration in range(NUM_ITERATIONS + 1):
        optimizer.zero_grad()

        total_loss, _, _, _ = standard_total_loss(
            model,
            residual_space,
            residual_time,
            initial_points,
            boundary_times,
            Q_ORDER,
            forcing_term,
            initial_condition,
        )

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(
                "Standard PINN stopped: invalid loss at iteration "
                f"{iteration}"
            )
            break

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP,
        )
        optimizer.step()
        loss_history.append(total_loss.item())

    elapsed_time = time.time() - start_time
    final_loss = loss_history[-1] if loss_history else float("nan")

    return model, loss_history, final_loss, elapsed_time


def train_shared_model(forcing_term, initial_condition, periodic_features):
    model = SharedBiasPINN(
        input_dim=5,
        hidden_dim=HIDDEN_WIDTH,
        output_dim=1,
    ).to(DEVICE)

    residual_space, residual_time = sample_interior_points(
        NUM_RESIDUAL_POINTS
    )
    boundary_times = sample_boundary_times(NUM_BOUNDARY_POINTS)
    loss_history = []

    start_time = time.time()

    for iteration in range(NUM_ITERATIONS + 1):
        if iteration % RESAMPLING_INTERVAL == 0:
            residual_space, residual_time = sample_interior_points(
                NUM_RESIDUAL_POINTS
            )

        model.zero_grad(set_to_none=True)

        total_loss, _, _ = shared_total_loss(
            model,
            residual_space,
            residual_time,
            boundary_times,
            Q_ORDER,
            forcing_term,
            initial_condition,
            periodic_features,
        )

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(
                "Shared-bias PINN stopped: invalid loss at iteration "
                f"{iteration}"
            )
            break

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP,
        )
        apply_conformable_gradient(
            model,
            iteration,
            base_rate=CONFORMABLE_BASE_RATE,
        )
        loss_history.append(total_loss.item())

    elapsed_time = time.time() - start_time
    final_loss = loss_history[-1] if loss_history else float("nan")

    return model, loss_history, final_loss, elapsed_time


@torch.no_grad()
def evaluate_model(
    model,
    trial_solution,
    exact_solution,
    initial_condition,
    periodic_features=None,
    nx=256,
    nt=201,
):
    space_grid = torch.linspace(X_LEFT, X_RIGHT, nx, device=DEVICE)
    time_grid = torch.linspace(0.0, T_FINAL, nt, device=DEVICE)

    space_mesh, time_mesh = torch.meshgrid(
        space_grid,
        time_grid,
        indexing="ij",
    )
    flat_space = space_mesh.reshape(-1, 1)
    flat_time = time_mesh.reshape(-1, 1)

    if periodic_features is None:
        prediction = trial_solution(
            model,
            flat_space,
            flat_time,
            initial_condition,
        )
    else:
        prediction = trial_solution(
            model,
            flat_space,
            flat_time,
            initial_condition,
            periodic_features,
        )

    prediction = prediction.reshape(nx, nt)
    exact_values = exact_solution(flat_space, flat_time).reshape(nx, nt)
    error = prediction - exact_values

    e_inf = torch.max(torch.abs(error)).item()
    e_2 = torch.sqrt(torch.mean(error**2)).item()
    e_2_rel = (torch.norm(error) / torch.norm(exact_values)).item()

    return (
        space_grid.cpu().numpy(),
        time_grid.cpu().numpy(),
        prediction.cpu().numpy(),
        exact_values.cpu().numpy(),
        e_inf,
        e_2,
        e_2_rel,
    )


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
def create_case_figures(
    case_name,
    omega_1,
    omega_2,
    standard_loss,
    shared_loss,
    standard_coarse,
    shared_coarse,
    standard_fine,
    shared_fine,
):
    (
        x_standard,
        _,
        prediction_standard,
        exact_standard,
        _,
        _,
        _,
    ) = standard_coarse
    (
        x_shared,
        _,
        prediction_shared,
        exact_shared,
        _,
        _,
        _,
    ) = shared_coarse

    (
        x_standard_fine,
        t_standard_fine,
        prediction_standard_fine,
        exact_standard_fine,
        _,
        _,
        _,
    ) = standard_fine
    (
        x_shared_fine,
        t_shared_fine,
        prediction_shared_fine,
        exact_shared_fine,
        _,
        _,
        _,
    ) = shared_fine

    suffix = (
        f"{case_name}_w1_{int(omega_1)}_w2_{int(omega_2)}"
    )

    # Smoothed training losses
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=240)
    ax.semilogy(
        smooth_curve(standard_loss),
        color=COLOR_STANDARD,
        label="Standard PINN total",
    )
    ax.semilogy(
        smooth_curve(shared_loss),
        color=COLOR_SHARED,
        label="Shared-bias PINN total",
    )
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(
        "Loss history comparison, "
        rf"$\omega_1={int(omega_1)}$, $\omega_2={int(omega_2)}$"
    )
    ax.legend()
    fig.tight_layout()
    save_pdf(fig, f"compare_loss_history_prob3_{suffix}")
    plt.close(fig)

    # Final-time profiles
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=240)
    ax.plot(
        x_standard,
        exact_standard[:, -1],
        color=COLOR_EXACT,
        label="Exact",
    )
    ax.plot(
        x_standard,
        prediction_standard[:, -1],
        "--",
        color=COLOR_STANDARD,
        label="Standard PINN",
    )
    ax.plot(
        x_shared,
        prediction_shared[:, -1],
        "-.",
        color=COLOR_SHARED,
        label="Shared-bias PINN",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("u(x,1)")
    ax.set_title(
        "Final-time comparison at $t=1$, "
        rf"$\omega_1={int(omega_1)}$, $\omega_2={int(omega_2)}$"
    )
    ax.legend()
    fig.tight_layout()
    save_pdf(fig, f"compare_final_time_prob3_{suffix}")
    plt.close(fig)

    # Solution heatmaps
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.8, 4.8),
        dpi=240,
        constrained_layout=True,
    )
    solution_panels = (
        ("Exact", exact_standard),
        ("Standard PINN", prediction_standard),
        ("Shared-bias PINN", prediction_shared),
    )

    for ax, (title, values) in zip(axes, solution_panels):
        image = ax.imshow(
            values.T,
            extent=[X_LEFT, X_RIGHT, 0.0, T_FINAL],
            origin="lower",
            aspect="auto",
            cmap="viridis",
            rasterized=True,
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(image, ax=ax, shrink=0.88)

    save_pdf(fig, f"compare_solution_heatmaps_prob3_{suffix}")
    plt.close(fig)

    # Error heatmaps
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.8),
        dpi=240,
        constrained_layout=True,
    )
    error_panels = (
        (
            "Standard PINN error",
            np.abs(prediction_standard - exact_standard),
        ),
        (
            "Shared-bias PINN error",
            np.abs(prediction_shared - exact_shared),
        ),
    )

    for ax, (title, values) in zip(axes, error_panels):
        image = ax.imshow(
            values.T,
            extent=[X_LEFT, X_RIGHT, 0.0, T_FINAL],
            origin="lower",
            aspect="auto",
            cmap="magma",
            rasterized=True,
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig.colorbar(image, ax=ax, shrink=0.88)

    save_pdf(fig, f"compare_error_heatmaps_prob3_{suffix}")
    plt.close(fig)

    standard_space_mesh, standard_time_mesh = np.meshgrid(
        x_standard_fine,
        t_standard_fine,
        indexing="ij",
    )
    shared_space_mesh, shared_time_mesh = np.meshgrid(
        x_shared_fine,
        t_shared_fine,
        indexing="ij",
    )

    # Three-dimensional solution surfaces
    fig = plt.figure(
        figsize=(22.0, 6.4),
        dpi=240,
        constrained_layout=True,
    )
    surface_panels = (
        (
            standard_space_mesh,
            standard_time_mesh,
            exact_standard_fine,
            "Exact solution",
        ),
        (
            standard_space_mesh,
            standard_time_mesh,
            prediction_standard_fine,
            "Standard PINN",
        ),
        (
            shared_space_mesh,
            shared_time_mesh,
            prediction_shared_fine,
            "Shared-bias PINN",
        ),
    )

    for index, (space_mesh, time_mesh, values, title) in enumerate(
        surface_panels,
        start=1,
    ):
        ax = fig.add_subplot(1, 3, index, projection="3d")
        surface = ax.plot_surface(
            space_mesh,
            time_mesh,
            values,
            cmap="viridis",
            rcount=240,
            ccount=200,
            linewidth=0,
            edgecolor="none",
            antialiased=True,
            rasterized=True,
        )
        ax.set_xlabel("x", labelpad=10)
        ax.set_ylabel("t", labelpad=10)
        ax.set_zlabel("u(x,t)", labelpad=10)
        ax.set_title(title, pad=12)
        ax.view_init(elev=28, azim=-135)
        fig.colorbar(surface, ax=ax, shrink=0.72, pad=0.08)

    save_pdf(fig, f"compare_3d_solutions_prob3_{suffix}")
    plt.close(fig)

    # Three-dimensional error surfaces
    fig = plt.figure(
        figsize=(16.0, 6.2),
        dpi=240,
        constrained_layout=True,
    )
    error_surface_panels = (
        (
            standard_space_mesh,
            standard_time_mesh,
            np.abs(prediction_standard_fine - exact_standard_fine),
            "Standard PINN error",
        ),
        (
            shared_space_mesh,
            shared_time_mesh,
            np.abs(prediction_shared_fine - exact_shared_fine),
            "Shared-bias PINN error",
        ),
    )

    for index, (space_mesh, time_mesh, values, title) in enumerate(
        error_surface_panels,
        start=1,
    ):
        ax = fig.add_subplot(1, 2, index, projection="3d")
        surface = ax.plot_surface(
            space_mesh,
            time_mesh,
            values,
            cmap="magma",
            rcount=240,
            ccount=200,
            linewidth=0,
            edgecolor="none",
            antialiased=True,
            rasterized=True,
        )
        ax.set_xlabel("x", labelpad=10)
        ax.set_ylabel("t", labelpad=10)
        ax.set_zlabel("|error|", labelpad=10)
        ax.set_title(title, pad=12)
        ax.view_init(elev=28, azim=-135)
        fig.colorbar(surface, ax=ax, shrink=0.75, pad=0.08)

    save_pdf(fig, f"compare_3d_errors_prob3_{suffix}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Tables
# -----------------------------------------------------------------------------
def print_results_table(rows):
    print("\nOverall global error table for Problem 3:\n")
    header = (
        f"{'case':>8} {'(omega1,omega2)':>18} {'method':>18} "
        f"{'E_inf':>16} {'E_2':>16} {'E_2_rel':>16} "
        f"{'final_loss':>16} {'time_sec':>14}"
    )
    print(header)
    print("-" * len(header))

    for row in rows:
        (
            case_name,
            omega_1,
            omega_2,
            method,
            e_inf,
            e_2,
            e_2_rel,
            final_loss,
            elapsed_time,
        ) = row
        frequency_pair = f"({int(omega_1)},{int(omega_2)})"
        print(
            f"{case_name:>8} {frequency_pair:>18} {method:>18} "
            f"{e_inf:16.8e} {e_2:16.8e} {e_2_rel:16.8e} "
            f"{final_loss:16.8e} {elapsed_time:14.6f}"
        )


def build_latex_table(rows):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{Global errors, final training loss, and CPU time "
            r"for the three manufactured time-conformable "
            r"reaction--diffusion cases.}"
        ),
        r"\label{tab:reaction_diffusion_global_comparison}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{cclccccc}",
        r"\hline",
        (
            r"Case & $(\omega_1,\omega_2)$ & Method & $E_{\infty}$ "
            r"& $E_2$ & $E_{2,\mathrm{rel}}$ & Final loss & "
            r"CPU time (s) \\"
        ),
        r"\hline",
    ]

    for row in rows:
        (
            case_name,
            omega_1,
            omega_2,
            method,
            e_inf,
            e_2,
            e_2_rel,
            final_loss,
            elapsed_time,
        ) = row
        lines.append(
            f"{case_name} & $({int(omega_1)},{int(omega_2)})$ & "
            f"{method} & {e_inf:.6e} & {e_2:.6e} & "
            f"{e_2_rel:.6e} & {final_loss:.6e} & "
            f"{elapsed_time:.3f} \\\\"
        )

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------
def main():
    all_results = []

    for case_name, omega_1, omega_2 in CASES:
        print("\n" + "=" * 80)
        print(
            f"Running Problem 3 {case_name}: "
            f"omega1 = {omega_1}, omega2 = {omega_2}"
        )
        print("=" * 80)

        (
            exact_solution,
            forcing_term,
            initial_condition,
            periodic_features,
        ) = build_case_functions(omega_1, omega_2)

        standard_model, standard_loss, standard_final_loss, standard_time = (
            train_standard_model(forcing_term, initial_condition)
        )

        standard_coarse = evaluate_model(
            standard_model,
            standard_trial_solution,
            exact_solution,
            initial_condition,
            nx=256,
            nt=201,
        )
        standard_fine = evaluate_model(
            standard_model,
            standard_trial_solution,
            exact_solution,
            initial_condition,
            nx=500,
            nt=260,
        )

        shared_model, shared_loss, shared_final_loss, shared_time = (
            train_shared_model(
                forcing_term,
                initial_condition,
                periodic_features,
            )
        )

        shared_coarse = evaluate_model(
            shared_model,
            shared_trial_solution,
            exact_solution,
            initial_condition,
            periodic_features=periodic_features,
            nx=256,
            nt=201,
        )
        shared_fine = evaluate_model(
            shared_model,
            shared_trial_solution,
            exact_solution,
            initial_condition,
            periodic_features=periodic_features,
            nx=500,
            nt=260,
        )

        _, _, _, _, e_inf_standard, e_2_standard, e_2_rel_standard = (
            standard_coarse
        )
        _, _, _, _, e_inf_shared, e_2_shared, e_2_rel_shared = shared_coarse

        all_results.append(
            (
                case_name,
                omega_1,
                omega_2,
                "Standard PINN",
                e_inf_standard,
                e_2_standard,
                e_2_rel_standard,
                standard_final_loss,
                standard_time,
            )
        )
        all_results.append(
            (
                case_name,
                omega_1,
                omega_2,
                "Shared-bias PINN",
                e_inf_shared,
                e_2_shared,
                e_2_rel_shared,
                shared_final_loss,
                shared_time,
            )
        )

        print(
            f"Completed {case_name}: "
            f"Standard PINN E_2={e_2_standard:.6e}, "
            f"Shared-bias PINN E_2={e_2_shared:.6e}"
        )

        create_case_figures(
            case_name,
            omega_1,
            omega_2,
            standard_loss,
            shared_loss,
            standard_coarse,
            shared_coarse,
            standard_fine,
            shared_fine,
        )

    print_results_table(all_results)
    print("\nLaTeX overall global error table:\n")
    print(build_latex_table(all_results))
    print(f"\nPDF figures saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
