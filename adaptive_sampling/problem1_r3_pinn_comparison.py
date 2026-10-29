"""
Problem 1 comparison with the published R3-sampling PINN baseline.

Representative setting:
    omega = 5

Methods:
    1. Standard PINN + Adam with fixed collocation points.
    2. R3-PINN + Adam with retain--resample--release sampling.
    3. Shared-bias PINN + conformable-gradient training.

R3 source:
    Daw et al., "Mitigating Propagation Failures in Physics-Informed
    Neural Networks using Retain-Resample-Release (R3) Sampling,"
    Proceedings of the 40th International Conference on Machine
    Learning, 2023.

The R3 step follows the supplied implementation:
    - evaluate the absolute residual at the current collocation points;
    - retain points whose absolute residual exceeds its mean;
    - replace all remaining points by uniform samples;
    - update the collocation set after every training iteration.

The governing equation, model initialization, common collocation points,
training order, optimizer settings, R3 rule, evaluation grid, table
output, and figure filenames are unchanged.
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
# Reproducibility and precision
# ---------------------------------------------------------------------
DEVICE = "cpu"
DTYPE = torch.float64
SEED = 0

torch.set_default_dtype(DTYPE)
torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------
# Figure settings
# ---------------------------------------------------------------------
plt.rcParams.update(
    {
        "figure.dpi": 180,
        "savefig.dpi": 500,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
    }
)


# ---------------------------------------------------------------------
# Output folder
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs" / "omega_5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Manufactured time-conformable Burgers problem
# ---------------------------------------------------------------------
Q = 0.7
OMEGA = 5.0
NU = 0.01 / math.pi

X_LEFT = 0.0
X_RIGHT = 2.0 * math.pi
FINAL_TIME = 1.0
MINIMUM_TIME = 1.0e-4

HIDDEN_UNITS = 80
NUM_RESIDUAL_POINTS = 200
NUM_INITIAL_POINTS = 100
NUM_BOUNDARY_POINTS = 100
NUM_ITERATIONS = 10000

ADAM_RATE = 1.0e-3
CONFORMABLE_BASE_RATE = 1.0e-2
GRADIENT_CLIP = 10.0

EVALUATION_NX = 256
EVALUATION_NT = 201


# ---------------------------------------------------------------------
# Exact solution, initial condition, and forcing term
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
def raw_features(x, t):
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

        # The original parameter creation order is preserved.
        self.W1 = nn.Parameter(
            torch.randn(input_dim, hidden_dim)
            * math.sqrt(1.0 / input_dim)
        )
        self.W2 = nn.Parameter(
            torch.randn(hidden_dim, output_dim)
            * math.sqrt(1.0 / hidden_dim)
        )
        self.b1 = nn.Parameter(
            torch.randn(1, 1)
            * math.sqrt(1.0 / input_dim)
        )
        self.b2 = nn.Parameter(
            torch.randn(1, output_dim)
            * math.sqrt(1.0 / output_dim)
        )

    def forward(self, inputs):
        hidden_argument = inputs @ self.W1 + self.b1
        hidden_output = torch.tanh(hidden_argument)
        return hidden_output @ self.W2 + self.b2


def count_trainable_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ---------------------------------------------------------------------
# Trial solutions and automatic differentiation
# ---------------------------------------------------------------------
def standard_trial_solution(model, x, t):
    network_output = model(raw_features(x, t))
    return initial_condition(x) + t * network_output


def shared_trial_solution(model, x, t):
    network_output = model(periodic_features(x, t))
    return initial_condition(x) + t * network_output


def derivative(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


# ---------------------------------------------------------------------
# Residual and loss terms
# ---------------------------------------------------------------------
def burgers_residual(model, trial_solution, x, t):
    local_x = x.detach().clone().requires_grad_(True)
    local_t = t.detach().clone().requires_grad_(True)

    u = trial_solution(model, local_x, local_t)
    u_t = derivative(u, local_t)
    u_x = derivative(u, local_x)
    u_xx = derivative(u_x, local_x)

    return (
        local_t ** (1.0 - Q) * u_t
        + u * u_x
        - NU * u_xx
        - forcing_term(local_x, local_t, Q)
    )


def initial_loss(model, trial_solution, initial_x):
    initial_t = torch.zeros_like(initial_x)
    prediction = trial_solution(
        model,
        initial_x,
        initial_t,
    )

    return torch.mean(
        (prediction - initial_condition(initial_x)) ** 2
    )


def periodic_boundary_loss(
    model,
    trial_solution,
    boundary_t,
):
    left_x = torch.full_like(
        boundary_t,
        X_LEFT,
        requires_grad=True,
    )
    right_x = torch.full_like(
        boundary_t,
        X_RIGHT,
        requires_grad=True,
    )

    left_t = boundary_t.detach().clone().requires_grad_(True)
    right_t = boundary_t.detach().clone().requires_grad_(True)

    left_u = trial_solution(model, left_x, left_t)
    right_u = trial_solution(model, right_x, right_t)

    left_u_x = derivative(left_u, left_x)
    right_u_x = derivative(right_u, right_x)

    value_loss = torch.mean((left_u - right_u) ** 2)
    derivative_loss = torch.mean(
        (left_u_x - right_u_x) ** 2
    )

    return value_loss + derivative_loss


def total_loss(
    model,
    trial_solution,
    residual_x,
    residual_t,
    initial_x,
    boundary_t,
):
    residual = burgers_residual(
        model,
        trial_solution,
        residual_x,
        residual_t,
    )

    residual_loss = torch.mean(residual**2)
    initial_condition_loss = initial_loss(
        model,
        trial_solution,
        initial_x,
    )
    boundary_condition_loss = periodic_boundary_loss(
        model,
        trial_solution,
        boundary_t,
    )

    loss = (
        residual_loss
        + initial_condition_loss
        + boundary_condition_loss
    )

    return (
        loss,
        residual_loss,
        initial_condition_loss,
        boundary_condition_loss,
        residual,
    )


# ---------------------------------------------------------------------
# Retain--resample--release sampler
# ---------------------------------------------------------------------
class R3Sampler:
    """
    Retain points whose absolute residual exceeds the mean residual.

    Every other point is replaced by a new uniform sample.
    """

    def __init__(
        self,
        x_limits,
        t_limits,
        number_of_points,
        initial_x,
        initial_t,
    ):
        self.x_limits = x_limits
        self.t_limits = t_limits
        self.number_of_points = number_of_points

        self.x = initial_x.detach().clone()
        self.t = initial_t.detach().clone()

        expected_shape = (number_of_points, 1)

        if self.x.shape != expected_shape:
            raise ValueError("initial_x has an invalid shape.")
        if self.t.shape != expected_shape:
            raise ValueError("initial_t has an invalid shape.")

    @torch.no_grad()
    def update(self, residual):
        fitness = torch.abs(residual).reshape(-1)

        if fitness.numel() != self.number_of_points:
            raise RuntimeError(
                "Residual count does not match the R3 point count."
            )

        threshold = torch.mean(fitness)
        retain_mask = fitness > threshold

        retained_x = self.x[retain_mask]
        retained_t = self.t[retain_mask]

        number_retained = int(retain_mask.sum().item())
        number_new = self.number_of_points - number_retained

        # Space is sampled before time, as in the supplied implementation.
        new_x = (
            self.x_limits[0]
            + (self.x_limits[1] - self.x_limits[0])
            * torch.rand(
                number_new,
                1,
                device=DEVICE,
                dtype=DTYPE,
            )
        )

        new_t = (
            self.t_limits[0]
            + (self.t_limits[1] - self.t_limits[0])
            * torch.rand(
                number_new,
                1,
                device=DEVICE,
                dtype=DTYPE,
            )
        )

        self.x = torch.cat(
            (retained_x, new_x),
            dim=0,
        ).detach()
        self.t = torch.cat(
            (retained_t, new_t),
            dim=0,
        ).detach()

        expected_shape = (self.number_of_points, 1)

        if self.x.shape != expected_shape:
            raise RuntimeError("R3 x-point count changed.")
        if self.t.shape != expected_shape:
            raise RuntimeError("R3 t-point count changed.")

        return (
            self.x,
            self.t,
            number_retained,
            threshold.item(),
        )


# ---------------------------------------------------------------------
# Conformable-gradient schedule and update
# ---------------------------------------------------------------------
def conformable_order(iteration):
    return (
        1.5
        - (math.atan(iteration) + math.pi / 2.0)
        / math.pi
    )


def effective_learning_rate(iteration, base_rate):
    alpha = max(conformable_order(iteration), 1.0e-8)
    return base_rate * alpha


@torch.no_grad()
def apply_conformable_gradient(
    model,
    iteration,
    base_rate,
):
    alpha = max(conformable_order(iteration), 1.0e-14)
    learning_rate = effective_learning_rate(
        iteration,
        base_rate,
    )

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        parameter_norm = torch.norm(
            parameter
        ).clamp_min(1.0e-14)

        scaled_gradient = (
            parameter_norm**alpha
            * parameter.grad
            / alpha
        )

        parameter -= learning_rate * scaled_gradient


# ---------------------------------------------------------------------
# Common initialization and collocation points
# ---------------------------------------------------------------------
def initialize_experiment():
    """
    Preserve the original random-number sequence.

    The shared-bias model is initialized first. Time points and then
    space points are sampled. The standard and R3 models are then
    initialized independently from the same seed.
    """

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    shared_model = SharedBiasPINN(
        input_dim=3,
        hidden_dim=HIDDEN_UNITS,
        output_dim=1,
    ).to(DEVICE)

    common_t = MINIMUM_TIME + (
        FINAL_TIME - MINIMUM_TIME
    ) * torch.rand(
        NUM_RESIDUAL_POINTS,
        1,
        device=DEVICE,
    )

    common_x = X_LEFT + (
        X_RIGHT - X_LEFT
    ) * torch.rand(
        NUM_RESIDUAL_POINTS,
        1,
        device=DEVICE,
    )

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

    torch.manual_seed(SEED)

    r3_model = StandardPINN(
        input_dim=2,
        hidden_dim=HIDDEN_UNITS,
        output_dim=1,
    ).to(DEVICE)

    r3_sampler = R3Sampler(
        x_limits=(X_LEFT, X_RIGHT),
        t_limits=(MINIMUM_TIME, FINAL_TIME),
        number_of_points=NUM_RESIDUAL_POINTS,
        initial_x=common_x,
        initial_t=common_t,
    )

    return (
        standard_model,
        r3_model,
        shared_model,
        r3_sampler,
        common_x,
        common_t,
        initial_x,
        boundary_t,
    )


# ---------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------
def train_standard(
    model,
    residual_x,
    residual_t,
    initial_x,
    boundary_t,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ADAM_RATE,
    )

    loss_history = []
    start_time = time.perf_counter()

    for iteration in range(NUM_ITERATIONS + 1):
        optimizer.zero_grad(set_to_none=True)

        (
            loss,
            residual_loss,
            initial_condition_loss,
            boundary_condition_loss,
            _,
        ) = total_loss(
            model,
            standard_trial_solution,
            residual_x,
            residual_t,
            initial_x,
            boundary_t,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Standard PINN stopped at iteration "
                f"{iteration}: invalid loss."
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP,
        )
        optimizer.step()

        loss_history.append(loss.item())

        if iteration % 1000 == 0:
            print(
                f"[Standard PINN] Iteration {iteration:5d} | "
                f"Loss {loss.item():.6e} | "
                f"PDE {residual_loss.item():.6e} | "
                f"IC {initial_condition_loss.item():.6e} | "
                f"BC {boundary_condition_loss.item():.6e}"
            )

    cpu_time = time.perf_counter() - start_time

    final_loss, _, _, _, _ = total_loss(
        model,
        standard_trial_solution,
        residual_x,
        residual_t,
        initial_x,
        boundary_t,
    )

    return (
        loss_history,
        final_loss.detach().item(),
        cpu_time,
    )


def train_r3(
    model,
    sampler,
    initial_x,
    boundary_t,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ADAM_RATE,
    )

    loss_history = []
    retained_history = []
    threshold_history = []

    start_time = time.perf_counter()

    for iteration in range(NUM_ITERATIONS + 1):
        optimizer.zero_grad(set_to_none=True)

        (
            loss,
            residual_loss,
            initial_condition_loss,
            boundary_condition_loss,
            residual,
        ) = total_loss(
            model,
            standard_trial_solution,
            sampler.x,
            sampler.t,
            initial_x,
            boundary_t,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "R3-PINN stopped at iteration "
                f"{iteration}: invalid loss."
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP,
        )
        optimizer.step()

        (
            _,
            _,
            number_retained,
            threshold,
        ) = sampler.update(residual.detach())

        loss_history.append(loss.item())
        retained_history.append(number_retained)
        threshold_history.append(threshold)

        if iteration % 1000 == 0:
            print(
                f"[R3-PINN] Iteration {iteration:5d} | "
                f"Loss {loss.item():.6e} | "
                f"PDE {residual_loss.item():.6e} | "
                f"IC {initial_condition_loss.item():.6e} | "
                f"BC {boundary_condition_loss.item():.6e} | "
                f"Retained {number_retained:3d}/"
                f"{NUM_RESIDUAL_POINTS}"
            )

    cpu_time = time.perf_counter() - start_time

    final_loss, _, _, _, _ = total_loss(
        model,
        standard_trial_solution,
        sampler.x,
        sampler.t,
        initial_x,
        boundary_t,
    )

    return (
        loss_history,
        retained_history,
        threshold_history,
        final_loss.detach().item(),
        cpu_time,
    )


def train_shared(
    model,
    residual_x,
    residual_t,
    initial_x,
    boundary_t,
):
    loss_history = []
    start_time = time.perf_counter()

    for iteration in range(NUM_ITERATIONS + 1):
        model.zero_grad(set_to_none=True)

        (
            loss,
            residual_loss,
            initial_condition_loss,
            boundary_condition_loss,
            _,
        ) = total_loss(
            model,
            shared_trial_solution,
            residual_x,
            residual_t,
            initial_x,
            boundary_t,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Shared-bias PINN stopped at iteration "
                f"{iteration}: invalid loss."
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP,
        )
        apply_conformable_gradient(
            model,
            iteration,
            CONFORMABLE_BASE_RATE,
        )

        loss_history.append(loss.item())

        if iteration % 1000 == 0:
            print(
                f"[Shared-bias PINN] Iteration {iteration:5d} | "
                f"Loss {loss.item():.6e} | "
                f"PDE {residual_loss.item():.6e} | "
                f"IC {initial_condition_loss.item():.6e} | "
                f"BC {boundary_condition_loss.item():.6e} | "
                f"alpha {conformable_order(iteration):.6e} | "
                f"LR "
                f"{effective_learning_rate(iteration, CONFORMABLE_BASE_RATE):.6e}"
            )

    cpu_time = time.perf_counter() - start_time

    final_loss, _, _, _, _ = total_loss(
        model,
        shared_trial_solution,
        residual_x,
        residual_t,
        initial_x,
        boundary_t,
    )

    return (
        loss_history,
        final_loss.detach().item(),
        cpu_time,
    )


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(
    model,
    trial_solution,
    nx=EVALUATION_NX,
    nt=EVALUATION_NT,
):
    x_grid = torch.linspace(
        X_LEFT,
        X_RIGHT,
        nx,
        device=DEVICE,
    )
    t_grid = torch.linspace(
        0.0,
        FINAL_TIME,
        nt,
        device=DEVICE,
    )

    x_mesh, t_mesh = torch.meshgrid(
        x_grid,
        t_grid,
        indexing="ij",
    )

    x_flat = x_mesh.reshape(-1, 1)
    t_flat = t_mesh.reshape(-1, 1)

    prediction = trial_solution(
        model,
        x_flat,
        t_flat,
    ).reshape(nx, nt)

    reference = exact_solution(
        x_flat,
        t_flat,
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
# Console and LaTeX tables
# ---------------------------------------------------------------------
def print_results_table(results):
    print("\nR3 comparison for Problem 1 with omega = 5:\n")

    header = (
        f"{'Method':<42}"
        f"{'Parameters':>13}"
        f"{'E_inf':>15}"
        f"{'E_2':>15}"
        f"{'E_2_rel':>15}"
        f"{'Final loss':>15}"
        f"{'CPU time (s)':>15}"
    )

    print(header)
    print("-" * len(header))

    for row in results:
        print(
            f"{row['method']:<42}"
            f"{row['parameters']:13d}"
            f"{row['E_inf']:15.6e}"
            f"{row['E_2']:15.6e}"
            f"{row['E_2_rel']:15.6e}"
            f"{row['final_loss']:15.6e}"
            f"{row['cpu_time']:15.3f}"
        )


def print_latex_table(results):
    print("\nLaTeX table:\n")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(
        r"\caption{Comparison with the published R3-sampling PINN "
        r"for Problem~1 with $\omega=5$.}"
    )
    print(r"\label{tab:problem1_r3_comparison}")
    print(r"\scriptsize")
    print(r"\setlength{\tabcolsep}{3.5pt}")
    print(r"\resizebox{\textwidth}{!}{")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\hline")
    print(
        r"Method & Parameters & $E_{\infty}$ & $E_2$ & "
        r"$E_{2,\mathrm{rel}}$ & Final loss & CPU time (s) \\"
    )
    print(r"\hline")

    for row in results:
        print(
            f"{row['method']} & "
            f"{row['parameters']} & "
            f"{row['E_inf']:.6e} & "
            f"{row['E_2']:.6e} & "
            f"{row['E_2_rel']:.6e} & "
            f"{row['final_loss']:.6e} & "
            f"{row['cpu_time']:.3f} \\\\"
        )

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"}")
    print(r"\end{table}")


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------
def save_figures(
    standard_history,
    r3_history,
    shared_history,
    retained_history,
    r3_sampler,
    x_grid,
    exact_grid,
    standard_prediction,
    r3_prediction,
    shared_prediction,
):
    figure_loss, axis_loss = plt.subplots(
        figsize=(7.8, 4.6),
        dpi=240,
    )

    axis_loss.semilogy(
        standard_history,
        label="Standard PINN + Adam",
    )
    axis_loss.semilogy(
        r3_history,
        linestyle="--",
        label="R3-PINN + Adam",
    )
    axis_loss.semilogy(
        shared_history,
        linestyle="-.",
        label="Shared-bias PINN + conformable GD",
    )

    axis_loss.set_xlabel("Iteration")
    axis_loss.set_ylabel("Training loss")
    axis_loss.set_title(
        r"Training-loss comparison, $\omega=5$"
    )
    axis_loss.set_xlim(left=0)
    axis_loss.margins(x=0)
    axis_loss.legend()
    figure_loss.tight_layout()

    loss_path = (
        OUTPUT_DIR
        / "problem1_r3_loss_comparison_omega5.pdf"
    )
    figure_loss.savefig(
        loss_path,
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(figure_loss)

    figure_final, axis_final = plt.subplots(
        figsize=(7.8, 4.6),
        dpi=240,
    )

    axis_final.plot(
        x_grid,
        exact_grid[:, -1],
        label="Exact",
    )
    axis_final.plot(
        x_grid,
        standard_prediction[:, -1],
        linestyle="--",
        label="Standard PINN + Adam",
    )
    axis_final.plot(
        x_grid,
        r3_prediction[:, -1],
        linestyle=":",
        label="R3-PINN + Adam",
    )
    axis_final.plot(
        x_grid,
        shared_prediction[:, -1],
        linestyle="-.",
        label="Shared-bias PINN + conformable GD",
    )

    axis_final.set_xlabel("x")
    axis_final.set_ylabel("u(x,1)")
    axis_final.set_title(
        r"Final-time comparison at $t=1$, $\omega=5$"
    )
    axis_final.legend()
    figure_final.tight_layout()

    final_path = (
        OUTPUT_DIR
        / "problem1_r3_final_time_comparison_omega5.pdf"
    )
    figure_final.savefig(
        final_path,
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(figure_final)

    figure_sampler, sampler_axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.4),
        dpi=240,
        constrained_layout=True,
    )

    iterations = np.arange(len(retained_history))

    sampler_axes[0].plot(
        iterations,
        retained_history,
    )
    sampler_axes[0].set_xlabel("Iteration")
    sampler_axes[0].set_ylabel("Retained points")
    sampler_axes[0].set_title("R3 retained-point count")
    sampler_axes[0].set_xlim(left=0)
    sampler_axes[0].margins(x=0)

    sampler_axes[1].scatter(
        r3_sampler.x.cpu().numpy().reshape(-1),
        r3_sampler.t.cpu().numpy().reshape(-1),
        s=12,
    )
    sampler_axes[1].set_xlabel("x")
    sampler_axes[1].set_ylabel("t")
    sampler_axes[1].set_title("Final R3 collocation set")
    sampler_axes[1].set_xlim(X_LEFT, X_RIGHT)
    sampler_axes[1].set_ylim(0.0, FINAL_TIME)

    sampler_path = (
        OUTPUT_DIR
        / "problem1_r3_sampling_diagnostics_omega5.pdf"
    )
    figure_sampler.savefig(
        sampler_path,
        format="pdf",
        bbox_inches="tight",
    )
    plt.close(figure_sampler)

    return loss_path, final_path, sampler_path


# ---------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------
def main():
    (
        standard_model,
        r3_model,
        shared_model,
        r3_sampler,
        common_x,
        common_t,
        initial_x,
        boundary_t,
    ) = initialize_experiment()

    (
        standard_history,
        standard_final_loss,
        standard_cpu_time,
    ) = train_standard(
        standard_model,
        common_x,
        common_t,
        initial_x,
        boundary_t,
    )

    (
        r3_history,
        retained_history,
        threshold_history,
        r3_final_loss,
        r3_cpu_time,
    ) = train_r3(
        r3_model,
        r3_sampler,
        initial_x,
        boundary_t,
    )

    (
        shared_history,
        shared_final_loss,
        shared_cpu_time,
    ) = train_shared(
        shared_model,
        common_x,
        common_t,
        initial_x,
        boundary_t,
    )

    (
        x_grid,
        t_grid,
        standard_prediction,
        exact_grid,
        standard_e_inf,
        standard_e_2,
        standard_e_2_rel,
    ) = evaluate_model(
        standard_model,
        standard_trial_solution,
    )

    (
        _,
        _,
        r3_prediction,
        _,
        r3_e_inf,
        r3_e_2,
        r3_e_2_rel,
    ) = evaluate_model(
        r3_model,
        standard_trial_solution,
    )

    (
        _,
        _,
        shared_prediction,
        _,
        shared_e_inf,
        shared_e_2,
        shared_e_2_rel,
    ) = evaluate_model(
        shared_model,
        shared_trial_solution,
    )

    results = [
        {
            "method": "Standard PINN + Adam",
            "parameters": count_trainable_parameters(
                standard_model
            ),
            "E_inf": standard_e_inf,
            "E_2": standard_e_2,
            "E_2_rel": standard_e_2_rel,
            "final_loss": standard_final_loss,
            "cpu_time": standard_cpu_time,
        },
        {
            "method": "R3-PINN + Adam",
            "parameters": count_trainable_parameters(
                r3_model
            ),
            "E_inf": r3_e_inf,
            "E_2": r3_e_2,
            "E_2_rel": r3_e_2_rel,
            "final_loss": r3_final_loss,
            "cpu_time": r3_cpu_time,
        },
        {
            "method": "Shared-bias PINN + conformable GD",
            "parameters": count_trainable_parameters(
                shared_model
            ),
            "E_inf": shared_e_inf,
            "E_2": shared_e_2,
            "E_2_rel": shared_e_2_rel,
            "final_loss": shared_final_loss,
            "cpu_time": shared_cpu_time,
        },
    ]

    print_results_table(results)
    print_latex_table(results)

    loss_path, final_path, sampler_path = save_figures(
        standard_history,
        r3_history,
        shared_history,
        retained_history,
        r3_sampler,
        x_grid,
        exact_grid,
        standard_prediction,
        r3_prediction,
        shared_prediction,
    )

    print(f"\nLoss figure:\n{loss_path}")
    print(f"\nFinal-time figure:\n{final_path}")
    print(f"\nR3 sampling diagnostics:\n{sampler_path}")
    print(f"\nAll files saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
