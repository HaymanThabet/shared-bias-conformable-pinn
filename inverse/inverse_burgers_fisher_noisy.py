"""
Inverse identification of the conformable order for the manufactured
time-conformable Burgers--Fisher problem.

The script compares:
    1. A standard PINN trained with Adam.
    2. A shared-bias PINN trained with conformable-gradient descent.

The unknown conformable order q is recovered from sparse observations
with 0%, 1%, and 5% Gaussian noise. The manufactured forcing is fixed
using the true value q = 0.7. During training, the estimated order is
projected onto [0.10, 0.99].

For each noise level, the script reports:
    - the estimated conformable order;
    - the relative error in q;
    - E_inf, E_2, and E_2_rel;
    - the final training loss;
    - CPU time.

It also saves PDF figures for:
    - q-identification histories;
    - loss histories;
    - final-time profiles;
    - absolute-error heatmaps.

The equation, seeds, sampling order, loss weights, optimizer settings,
resampling interval, evaluation grid, output table, and figure names are
unchanged.
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
# Reproducibility and numerical precision
# ---------------------------------------------------------------------
DEVICE = "cpu"
DTYPE = torch.float64
BASE_SEED = 2026

torch.set_default_dtype(DTYPE)
np.random.seed(BASE_SEED)
torch.manual_seed(BASE_SEED)


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
# Problem parameters
# ---------------------------------------------------------------------
TRUE_Q = 0.7
NU = 0.01
REACTION_RATE = 1.0

X_LEFT = -math.pi
X_RIGHT = math.pi
FINAL_TIME = 1.0
MINIMUM_TIME = 1.0e-4

Q_MIN = 0.10
Q_MAX = 0.99
Q_INITIAL = 0.50


# ---------------------------------------------------------------------
# Inverse-problem settings
# ---------------------------------------------------------------------
NOISE_LEVELS = [0.00, 0.01, 0.05]
NUM_OBSERVATIONS = 100
OBSERVATION_TIME_MIN = 0.05

HIDDEN_UNITS = 100
NUM_RESIDUAL_POINTS = 200
NUM_BOUNDARY_POINTS = 100
NUM_ITERATIONS = 20000
RESAMPLING_INTERVAL = 200
GRADIENT_CLIP = 10.0

PDE_WEIGHT = 1.0
BOUNDARY_WEIGHT = 1.0
DATA_WEIGHT = 10.0

ADAM_RATE = 5.0e-3
CONFORMABLE_BASE_RATE = 5.0e-2

HISTORY_STRIDE = 10


# ---------------------------------------------------------------------
# Optional short diagnostic run
# ---------------------------------------------------------------------
SMOKE_TEST = os.getenv("PINN_SMOKE_TEST", "0") == "1"

if SMOKE_TEST:
    NUM_RESIDUAL_POINTS = 20
    NUM_BOUNDARY_POINTS = 12
    NUM_OBSERVATIONS = 20
    NUM_ITERATIONS = 3
    HISTORY_STRIDE = 1
    NOISE_LEVELS = [0.00]


# ---------------------------------------------------------------------
# Exact solution, initial condition, and fixed forcing term
# ---------------------------------------------------------------------
def exact_solution(x, t):
    spatial_mode = (
        torch.sin(2.0 * x) + 0.5 * torch.sin(4.0 * x)
    )
    return 0.5 + 0.25 * torch.exp(-t) * spatial_mode


def initial_condition(x):
    return exact_solution(x, torch.zeros_like(x))


def fixed_forcing(x, t):
    """Manufactured forcing evaluated once with the true order."""

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
        t ** (1.0 - TRUE_Q) * u_t
        + u * u_x
        - NU * u_xx
        - REACTION_RATE * u * (1.0 - u)
    )


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------
def derivative(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def format_noise_tag(noise_level):
    return str(int(round(100.0 * noise_level)))


def smooth_curve(values, beta=0.98):
    smoothed = []
    running_average = 0.0

    for index, value in enumerate(values, start=1):
        running_average = (
            beta * running_average + (1.0 - beta) * value
        )
        corrected_average = (
            running_average / (1.0 - beta**index)
        )
        smoothed.append(corrected_average)

    return smoothed


def save_pdf(figure, output_dir, stem):
    """Save a PDF, using a fallback name if Windows locks the target."""

    target_path = output_dir / f"{stem}.pdf"

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
                output_dir / f"{stem}_new{counter}.pdf"
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
# Sampling and noisy observations
# ---------------------------------------------------------------------
def sample_residual_points(number, generator):
    # Time is sampled before space to preserve the original RNG order.
    residual_t = MINIMUM_TIME + (
        FINAL_TIME - MINIMUM_TIME
    ) * torch.rand(
        number,
        1,
        generator=generator,
        device=DEVICE,
        dtype=DTYPE,
    )

    residual_x = X_LEFT + (
        X_RIGHT - X_LEFT
    ) * torch.rand(
        number,
        1,
        generator=generator,
        device=DEVICE,
        dtype=DTYPE,
    )

    return residual_x, residual_t


def create_case_data(noise_level, seed):
    """Create common observations and baseline collocation points."""

    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(seed)

    residual_x, residual_t = sample_residual_points(
        NUM_RESIDUAL_POINTS,
        generator,
    )

    boundary_t = torch.linspace(
        MINIMUM_TIME,
        FINAL_TIME,
        NUM_BOUNDARY_POINTS,
        device=DEVICE,
        dtype=DTYPE,
    ).reshape(-1, 1)

    observation_x = X_LEFT + (
        X_RIGHT - X_LEFT
    ) * torch.rand(
        NUM_OBSERVATIONS,
        1,
        generator=generator,
        device=DEVICE,
        dtype=DTYPE,
    )

    observation_t = OBSERVATION_TIME_MIN + (
        FINAL_TIME - OBSERVATION_TIME_MIN
    ) * torch.rand(
        NUM_OBSERVATIONS,
        1,
        generator=generator,
        device=DEVICE,
        dtype=DTYPE,
    )

    clean_observations = exact_solution(
        observation_x,
        observation_t,
    )

    clean_standard_deviation = torch.std(
        clean_observations,
        unbiased=False,
    )
    noise_standard_deviation = (
        noise_level * clean_standard_deviation
    )

    observation_noise = (
        noise_standard_deviation
        * torch.randn(
            clean_observations.shape,
            generator=generator,
            device=DEVICE,
            dtype=DTYPE,
        )
    )
    noisy_observations = (
        clean_observations + observation_noise
    )

    return {
        "residual_x": residual_x,
        "residual_t": residual_t,
        "boundary_t": boundary_t,
        "observation_x": observation_x,
        "observation_t": observation_t,
        "observations": noisy_observations,
        "clean_observations": clean_observations,
        "noise_standard_deviation": float(
            noise_standard_deviation.item()
        ),
    }


# ---------------------------------------------------------------------
# Standard inverse PINN
# ---------------------------------------------------------------------
class StandardInversePINN(nn.Module):
    """Standard one-hidden-layer PINN with trainable conformable order."""

    def __init__(self, hidden_dim=100):
        super().__init__()

        self.fc1 = nn.Linear(2, hidden_dim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, 1)

        self.q_parameter = nn.Parameter(
            torch.tensor(Q_INITIAL, dtype=DTYPE)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def estimated_q(self):
        return self.q_parameter

    def network_output(self, x, t):
        inputs = torch.cat((x, t), dim=1)
        hidden_output = self.activation(self.fc1(inputs))
        return self.fc2(hidden_output)

    def solution(self, x, t):
        return (
            initial_condition(x)
            + t * self.network_output(x, t)
        )


# ---------------------------------------------------------------------
# Shared-bias inverse PINN
# ---------------------------------------------------------------------
class SharedBiasInversePINN(nn.Module):
    """Shared-bias PINN with trainable conformable order."""

    def __init__(self, hidden_dim=100):
        super().__init__()

        # The parameter registration order is preserved.
        self.W1 = nn.Parameter(
            torch.empty(5, hidden_dim, dtype=DTYPE)
        )
        self.W2 = nn.Parameter(
            torch.empty(hidden_dim, 1, dtype=DTYPE)
        )
        self.b1 = nn.Parameter(
            torch.zeros(1, 1, dtype=DTYPE)
        )
        self.b2 = nn.Parameter(
            torch.zeros(1, 1, dtype=DTYPE)
        )
        self.q_parameter = nn.Parameter(
            torch.tensor(Q_INITIAL, dtype=DTYPE)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.W1)
        nn.init.xavier_normal_(self.W2)

        with torch.no_grad():
            self.b1.zero_()
            self.b2.zero_()

    def estimated_q(self):
        return self.q_parameter

    def network_output(self, x, t):
        features = torch.cat(
            (
                torch.sin(2.0 * x),
                torch.cos(2.0 * x),
                torch.sin(4.0 * x),
                torch.cos(4.0 * x),
                t,
            ),
            dim=1,
        )
        hidden_output = torch.tanh(
            features @ self.W1 + self.b1
        )
        return hidden_output @ self.W2 + self.b2

    def solution(self, x, t):
        return (
            initial_condition(x)
            + t * self.network_output(x, t)
        )


# ---------------------------------------------------------------------
# Inverse residual and loss terms
# ---------------------------------------------------------------------
def inverse_residual(model, x, t):
    local_x = x.detach().clone().requires_grad_(True)
    local_t = t.detach().clone().requires_grad_(True)

    u = model.solution(local_x, local_t)
    u_t = derivative(u, local_t)
    u_x = derivative(u, local_x)
    u_xx = derivative(u_x, local_x)

    estimated_q = model.estimated_q()
    conformable_term = (
        local_t ** (1.0 - estimated_q) * u_t
    )

    return (
        conformable_term
        + u * u_x
        - NU * u_xx
        - REACTION_RATE * u * (1.0 - u)
        - fixed_forcing(local_x, local_t)
    )


def periodic_boundary_loss(model, boundary_t):
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

    left_u = model.solution(left_x, left_t)
    right_u = model.solution(right_x, right_t)

    left_u_x = derivative(left_u, left_x)
    right_u_x = derivative(right_u, right_x)

    value_loss = torch.mean((left_u - right_u) ** 2)
    derivative_loss = torch.mean(
        (left_u_x - right_u_x) ** 2
    )

    return value_loss + derivative_loss


def inverse_total_loss(
    model,
    residual_x,
    residual_t,
    boundary_t,
    observation_x,
    observation_t,
    observations,
):
    residual = inverse_residual(
        model,
        residual_x,
        residual_t,
    )

    residual_loss = torch.mean(residual**2)
    boundary_loss = periodic_boundary_loss(
        model,
        boundary_t,
    )

    predicted_observations = model.solution(
        observation_x,
        observation_t,
    )
    data_loss = torch.mean(
        (predicted_observations - observations) ** 2
    )

    total_loss = (
        PDE_WEIGHT * residual_loss
        + BOUNDARY_WEIGHT * boundary_loss
        + DATA_WEIGHT * data_loss
    )

    return (
        total_loss,
        residual_loss,
        boundary_loss,
        data_loss,
    )


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

    model.q_parameter.clamp_(Q_MIN, Q_MAX)


# ---------------------------------------------------------------------
# Standard inverse training
# ---------------------------------------------------------------------
def train_standard_inverse(case_data, model_seed):
    torch.manual_seed(model_seed)

    model = StandardInversePINN(
        hidden_dim=HIDDEN_UNITS
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ADAM_RATE,
    )

    loss_history = []
    q_history = []
    history_iterations = []

    start_time = time.time()

    for iteration in range(NUM_ITERATIONS + 1):
        optimizer.zero_grad(set_to_none=True)

        (
            loss,
            residual_loss,
            boundary_loss,
            data_loss,
        ) = inverse_total_loss(
            model,
            case_data["residual_x"],
            case_data["residual_t"],
            case_data["boundary_t"],
            case_data["observation_x"],
            case_data["observation_t"],
            case_data["observations"],
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Standard inverse PINN stopped at iteration "
                f"{iteration}: invalid loss."
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP,
        )
        optimizer.step()

        with torch.no_grad():
            model.q_parameter.clamp_(Q_MIN, Q_MAX)

        loss_history.append(loss.item())

        if iteration % HISTORY_STRIDE == 0:
            history_iterations.append(iteration)
            q_history.append(
                model.estimated_q().item()
            )

        if iteration % 1000 == 0:
            print(
                "[Standard PINN + Adam] "
                f"Iter {iteration:5d} | "
                f"Loss={loss.item():.6e} | "
                f"PDE={residual_loss.item():.6e} | "
                f"BC={boundary_loss.item():.6e} | "
                f"Data={data_loss.item():.6e} | "
                f"q={model.estimated_q().item():.8f}"
            )

    cpu_time = time.time() - start_time

    return {
        "model": model,
        "loss_history": loss_history,
        "q_history": q_history,
        "history_iterations": history_iterations,
        "final_loss": loss_history[-1],
        "cpu_time": cpu_time,
    }


# ---------------------------------------------------------------------
# Shared-bias inverse training
# ---------------------------------------------------------------------
def train_shared_inverse(
    case_data,
    model_seed,
    sampling_seed,
):
    torch.manual_seed(model_seed)

    model = SharedBiasInversePINN(
        hidden_dim=HIDDEN_UNITS
    ).to(DEVICE)

    sampling_generator = torch.Generator(device=DEVICE)
    sampling_generator.manual_seed(sampling_seed)

    residual_x = case_data["residual_x"].clone()
    residual_t = case_data["residual_t"].clone()

    loss_history = []
    q_history = []
    history_iterations = []

    start_time = time.time()

    for iteration in range(NUM_ITERATIONS + 1):
        if iteration % RESAMPLING_INTERVAL == 0:
            residual_x, residual_t = sample_residual_points(
                NUM_RESIDUAL_POINTS,
                sampling_generator,
            )

        model.zero_grad(set_to_none=True)

        (
            loss,
            residual_loss,
            boundary_loss,
            data_loss,
        ) = inverse_total_loss(
            model,
            residual_x,
            residual_t,
            case_data["boundary_t"],
            case_data["observation_x"],
            case_data["observation_t"],
            case_data["observations"],
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Shared-bias inverse PINN stopped at iteration "
                f"{iteration}: invalid loss."
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP,
        )
        apply_conformable_gradient(model, iteration)

        loss_history.append(loss.item())

        if iteration % HISTORY_STRIDE == 0:
            history_iterations.append(iteration)
            q_history.append(
                model.estimated_q().item()
            )

        if iteration % 1000 == 0:
            print(
                "[Shared-bias PINN + conformable GD] "
                f"Iter {iteration:5d} | "
                f"Loss={loss.item():.6e} | "
                f"PDE={residual_loss.item():.6e} | "
                f"BC={boundary_loss.item():.6e} | "
                f"Data={data_loss.item():.6e} | "
                f"q={model.estimated_q().item():.8f}"
            )

    cpu_time = time.time() - start_time

    return {
        "model": model,
        "loss_history": loss_history,
        "q_history": q_history,
        "history_iterations": history_iterations,
        "final_loss": loss_history[-1],
        "cpu_time": cpu_time,
    }


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate_inverse_model(model, nx=256, nt=201):
    x_grid = torch.linspace(
        X_LEFT,
        X_RIGHT,
        nx,
        device=DEVICE,
        dtype=DTYPE,
    )
    t_grid = torch.linspace(
        0.0,
        FINAL_TIME,
        nt,
        device=DEVICE,
        dtype=DTYPE,
    )

    x_mesh, t_mesh = torch.meshgrid(
        x_grid,
        t_grid,
        indexing="ij",
    )
    x_flat = x_mesh.reshape(-1, 1)
    t_flat = t_mesh.reshape(-1, 1)

    prediction = model.solution(
        x_flat,
        t_flat,
    ).reshape(nx, nt)

    reference = exact_solution(
        x_flat,
        t_flat,
    ).reshape(nx, nt)

    error = prediction - reference
    q_estimate = model.estimated_q().item()

    return {
        "x": x_grid.cpu().numpy(),
        "t": t_grid.cpu().numpy(),
        "prediction": prediction.cpu().numpy(),
        "exact": reference.cpu().numpy(),
        "e_inf": torch.max(torch.abs(error)).item(),
        "e_2": torch.sqrt(torch.mean(error**2)).item(),
        "e_2_rel": (
            torch.norm(error) / torch.norm(reference)
        ).item(),
        "q_estimate": q_estimate,
        "q_relative_error": (
            abs(q_estimate - TRUE_Q) / abs(TRUE_Q)
        ),
    }


# ---------------------------------------------------------------------
# Console and LaTeX tables
# ---------------------------------------------------------------------
def print_plain_table(rows):
    print("\nInverse identification results:\n")

    header = (
        f"{'Noise':<8}"
        f"{'Method':<36}"
        f"{'q_hat':>13}"
        f"{'Rel q err':>14}"
        f"{'E_inf':>14}"
        f"{'E_2':>14}"
        f"{'E_2_rel':>14}"
        f"{'Final loss':>14}"
        f"{'CPU (s)':>12}"
    )

    print(header)
    print("-" * len(header))

    for row in rows:
        print(
            f"{row['noise_label']:<8}"
            f"{row['method']:<36}"
            f"{row['q_estimate']:13.7f}"
            f"{row['q_relative_error']:14.6e}"
            f"{row['e_inf']:14.6e}"
            f"{row['e_2']:14.6e}"
            f"{row['e_2_rel']:14.6e}"
            f"{row['final_loss']:14.6e}"
            f"{row['cpu_time']:12.3f}"
        )


def make_latex_table(rows):
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        (
            r"\caption{Inverse recovery of the conformable order from "
            r"sparse noisy observations for the manufactured "
            r"time-conformable Burgers--Fisher problem.}"
        ),
        r"\label{tab:burgers_fisher_inverse_q}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{clccccccc}",
        r"\hline",
        (
            r"Noise & Method & $\widehat q$ & Relative $q$ error & "
            r"$E_{\infty}$ & $E_2$ & $E_{2,\mathrm{rel}}$ & "
            r"Final loss & CPU time (s) \\"
        ),
        r"\hline",
    ]

    for row in rows:
        lines.append(
            f"{row['noise_label']} & {row['method']} & "
            f"{row['q_estimate']:.7f} & "
            f"{row['q_relative_error']:.6e} & "
            f"{row['e_inf']:.6e} & "
            f"{row['e_2']:.6e} & "
            f"{row['e_2_rel']:.6e} & "
            f"{row['final_loss']:.6e} & "
            f"{row['cpu_time']:.3f} \\\\"
        )

    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------
# Figures for one noise level
# ---------------------------------------------------------------------
def save_case_figures(
    output_dir,
    noise_display,
    noise_file_tag,
    standard_run,
    shared_run,
    standard_evaluation,
    shared_evaluation,
):
    # Estimated-q histories
    figure_q, axis_q = plt.subplots(
        figsize=(7.6, 4.4),
        dpi=240,
    )
    axis_q.plot(
        standard_run["history_iterations"],
        standard_run["q_history"],
        color=COLOR_STANDARD,
        label="Standard PINN + Adam",
    )
    axis_q.plot(
        shared_run["history_iterations"],
        shared_run["q_history"],
        color=COLOR_SHARED,
        label="Shared-bias PINN + conformable GD",
    )
    axis_q.axhline(
        TRUE_Q,
        color=COLOR_EXACT,
        linestyle="--",
        label=rf"True $q={TRUE_Q}$",
    )
    axis_q.set_xlabel("Iteration")
    axis_q.set_ylabel(
        r"Estimated conformable order $\widehat q$"
    )
    axis_q.set_title(
        rf"Inverse identification, noise level {noise_display}"
    )
    axis_q.legend()
    figure_q.tight_layout()

    save_pdf(
        figure_q,
        output_dir,
        f"inverse_q_history_noise{noise_file_tag}",
    )
    plt.close(figure_q)

    # Loss histories
    figure_loss, axis_loss = plt.subplots(
        figsize=(7.6, 4.4),
        dpi=240,
    )
    axis_loss.semilogy(
        smooth_curve(standard_run["loss_history"]),
        color=COLOR_STANDARD,
        label="Standard PINN + Adam",
    )
    axis_loss.semilogy(
        smooth_curve(shared_run["loss_history"]),
        color=COLOR_SHARED,
        label="Shared-bias PINN + conformable GD",
    )
    axis_loss.set_xlabel("Iteration")
    axis_loss.set_ylabel("Total loss")
    axis_loss.set_title(
        rf"Inverse loss histories, noise level {noise_display}"
    )
    axis_loss.legend()
    figure_loss.tight_layout()

    save_pdf(
        figure_loss,
        output_dir,
        f"inverse_loss_history_noise{noise_file_tag}",
    )
    plt.close(figure_loss)

    # Final-time profiles
    figure_final, axis_final = plt.subplots(
        figsize=(7.6, 4.4),
        dpi=240,
    )
    axis_final.plot(
        standard_evaluation["x"],
        standard_evaluation["exact"][:, -1],
        color=COLOR_EXACT,
        label="Exact",
    )
    axis_final.plot(
        standard_evaluation["x"],
        standard_evaluation["prediction"][:, -1],
        "--",
        color=COLOR_STANDARD,
        label="Standard PINN + Adam",
    )
    axis_final.plot(
        shared_evaluation["x"],
        shared_evaluation["prediction"][:, -1],
        "-.",
        color=COLOR_SHARED,
        label="Shared-bias PINN + conformable GD",
    )
    axis_final.set_xlabel("x")
    axis_final.set_ylabel("u(x,1)")
    axis_final.set_title(
        rf"Final-time inverse comparison, noise level {noise_display}"
    )
    axis_final.legend()
    figure_final.tight_layout()

    save_pdf(
        figure_final,
        output_dir,
        f"inverse_final_time_noise{noise_file_tag}",
    )
    plt.close(figure_final)

    # Absolute-error heatmaps
    figure_error, axes = plt.subplots(
        1,
        2,
        figsize=(11.2, 4.8),
        dpi=240,
        constrained_layout=True,
    )

    standard_error = np.abs(
        standard_evaluation["prediction"]
        - standard_evaluation["exact"]
    )
    shared_error = np.abs(
        shared_evaluation["prediction"]
        - shared_evaluation["exact"]
    )

    standard_image = axes[0].imshow(
        standard_error.T,
        extent=[X_LEFT, X_RIGHT, 0.0, FINAL_TIME],
        origin="lower",
        aspect="auto",
        cmap="magma",
    )
    axes[0].set_title("Standard PINN + Adam error")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    figure_error.colorbar(
        standard_image,
        ax=axes[0],
        shrink=0.88,
    )

    shared_image = axes[1].imshow(
        shared_error.T,
        extent=[X_LEFT, X_RIGHT, 0.0, FINAL_TIME],
        origin="lower",
        aspect="auto",
        cmap="magma",
    )
    axes[1].set_title(
        "Shared-bias PINN + conformable GD error"
    )
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    figure_error.colorbar(
        shared_image,
        ax=axes[1],
        shrink=0.88,
    )

    save_pdf(
        figure_error,
        output_dir,
        f"inverse_error_heatmaps_noise{noise_file_tag}",
    )
    plt.close(figure_error)


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------
def main():
    rows = []

    for case_index, noise_level in enumerate(NOISE_LEVELS):
        noise_file_tag = format_noise_tag(noise_level)
        noise_display = f"{noise_file_tag}%"
        noise_label = f"{noise_file_tag}\\%"

        print("\n" + "=" * 90)
        print(
            "Inverse Burgers--Fisher experiment | "
            f"noise level = {noise_file_tag}%"
        )
        print("=" * 90)

        case_output_dir = (
            OUTPUT_DIR / f"noise_{noise_file_tag}"
        )
        case_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        case_data = create_case_data(
            noise_level=noise_level,
            seed=BASE_SEED + 100 * case_index,
        )

        print(
            f"Observation count = {NUM_OBSERVATIONS}, "
            f"noise sigma = "
            f"{case_data['noise_standard_deviation']:.6e}"
        )

        standard_run = train_standard_inverse(
            case_data,
            model_seed=BASE_SEED + 1000 + case_index,
        )
        shared_run = train_shared_inverse(
            case_data,
            model_seed=BASE_SEED + 2000 + case_index,
            sampling_seed=BASE_SEED + 3000 + case_index,
        )

        standard_evaluation = evaluate_inverse_model(
            standard_run["model"]
        )
        shared_evaluation = evaluate_inverse_model(
            shared_run["model"]
        )

        rows.append(
            {
                "noise_label": noise_label,
                "method": "Standard PINN + Adam",
                "q_estimate": standard_evaluation["q_estimate"],
                "q_relative_error": (
                    standard_evaluation["q_relative_error"]
                ),
                "e_inf": standard_evaluation["e_inf"],
                "e_2": standard_evaluation["e_2"],
                "e_2_rel": standard_evaluation["e_2_rel"],
                "final_loss": standard_run["final_loss"],
                "cpu_time": standard_run["cpu_time"],
            }
        )

        rows.append(
            {
                "noise_label": noise_label,
                "method": (
                    "Shared-bias PINN + conformable GD"
                ),
                "q_estimate": shared_evaluation["q_estimate"],
                "q_relative_error": (
                    shared_evaluation["q_relative_error"]
                ),
                "e_inf": shared_evaluation["e_inf"],
                "e_2": shared_evaluation["e_2"],
                "e_2_rel": shared_evaluation["e_2_rel"],
                "final_loss": shared_run["final_loss"],
                "cpu_time": shared_run["cpu_time"],
            }
        )

        save_case_figures(
            case_output_dir,
            noise_display,
            noise_file_tag,
            standard_run,
            shared_run,
            standard_evaluation,
            shared_evaluation,
        )

    print_plain_table(rows)

    print("\nLaTeX inverse-identification table:\n")
    print(make_latex_table(rows))

    print(f"\nPDF figures saved in:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
