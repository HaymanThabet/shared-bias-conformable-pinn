"""
Five-run study for the manufactured time-conformable Burgers--Fisher problem.

The script compares:
    1. A standard one-hidden-layer PINN trained with Adam.
    2. A shared-bias PINN trained with the conformable-gradient update.

For each seed, the script reports:
    E_inf, E_2, E_2_rel, final training loss, and CPU time.

It also saves:
    - individual-run results in CSV format;
    - summary statistics in CSV format;
    - a PDF figure of the mean loss histories with one-standard-deviation bands.

The numerical model, random-seed sequence, collocation strategy, training
settings, and evaluation grid are unchanged.
"""

import csv
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


# ---------------------------------------------------------------------
# Numerical settings
# ---------------------------------------------------------------------
DEVICE = "cpu"
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

SEEDS = [0, 1, 2, 3, 4]

Q = 0.7
NU = 0.01
REACTION_RATE = 1.0

X_LEFT = -math.pi
X_RIGHT = math.pi
FINAL_TIME = 1.0
MINIMUM_TIME = 1.0e-4

HIDDEN_UNITS = 100
NUM_RESIDUAL_POINTS = 200
NUM_INITIAL_POINTS = 100
NUM_BOUNDARY_POINTS = 100
NUM_ITERATIONS = 20000

ADAM_RATE = 51.0e-3
CONFORMABLE_BASE_RATE = 5.0e-2
GRADIENT_CLIP = 10.0
RESAMPLING_INTERVAL = 200

PRINT_EVERY = 2000
EVALUATION_NX = 256
EVALUATION_NT = 201


# ---------------------------------------------------------------------
# Exact solution, initial condition, and forcing term
# ---------------------------------------------------------------------
def exact_solution(x, t):
    spatial_mode = torch.sin(2.0 * x) + 0.5 * torch.sin(4.0 * x)
    return 0.5 + 0.25 * torch.exp(-t) * spatial_mode


def initial_condition(x):
    return 0.5 + 0.25 * (
        torch.sin(2.0 * x) + 0.5 * torch.sin(4.0 * x)
    )


def forcing_term(x, t, q):
    decay = torch.exp(-t)
    spatial_mode = torch.sin(2.0 * x) + 0.5 * torch.sin(4.0 * x)

    u = 0.5 + 0.25 * decay * spatial_mode
    u_t = -0.25 * decay * spatial_mode
    u_x = 0.25 * decay * (
        2.0 * torch.cos(2.0 * x) + 2.0 * torch.cos(4.0 * x)
    )
    u_xx = 0.25 * decay * (
        -4.0 * torch.sin(2.0 * x) - 8.0 * torch.sin(4.0 * x)
    )

    conformable_time_term = t ** (1.0 - q) * u_t
    advection_term = u * u_x
    diffusion_term = -NU * u_xx
    reaction_term = -REACTION_RATE * u * (1.0 - u)

    return (
        conformable_time_term
        + advection_term
        + diffusion_term
        + reaction_term
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
# Neural networks
# ---------------------------------------------------------------------
class StandardPINN(nn.Module):
    """One-hidden-layer PINN with ordinary neuron-dependent biases."""

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
        hidden_output = torch.tanh(self.hidden_layer(inputs))
        return self.output_layer(hidden_output)


class SharedBiasPINN(nn.Module):
    """One-hidden-layer PINN with one trainable hidden-layer bias."""

    def __init__(self, input_dim=5, hidden_dim=100, output_dim=1):
        super().__init__()

        self.input_weights = nn.Parameter(
            torch.empty(input_dim, hidden_dim, dtype=DTYPE)
        )
        self.output_weights = nn.Parameter(
            torch.empty(hidden_dim, output_dim, dtype=DTYPE)
        )
        self.shared_hidden_bias = nn.Parameter(
            torch.zeros(1, 1, dtype=DTYPE)
        )
        self.output_bias = nn.Parameter(
            torch.zeros(1, output_dim, dtype=DTYPE)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.input_weights)
        nn.init.xavier_normal_(self.output_weights)

        with torch.no_grad():
            self.shared_hidden_bias.zero_()
            self.output_bias.zero_()

    def forward(self, inputs):
        hidden_argument = (
            inputs @ self.input_weights + self.shared_hidden_bias
        )
        hidden_output = torch.tanh(hidden_argument)
        return hidden_output @ self.output_weights + self.output_bias


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
def standard_residual(model, x, t, q):
    x = x.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)

    u = standard_trial_solution(model, x, t)
    u_t = derivative(u, t)
    u_x = derivative(u, x)
    u_xx = derivative(u_x, x)

    return (
        t ** (1.0 - q) * u_t
        + u * u_x
        - NU * u_xx
        - REACTION_RATE * u * (1.0 - u)
        - forcing_term(x, t, q)
    )


def shared_residual(model, x, t, q):
    x = x.detach().clone().requires_grad_(True)
    t = t.detach().clone().requires_grad_(True)

    u = shared_trial_solution(model, x, t)
    u_t = derivative(u, t)
    u_x = derivative(u, x)
    u_xx = derivative(u_x, x)

    return (
        t ** (1.0 - q) * u_t
        + u * u_x
        - NU * u_xx
        - REACTION_RATE * u * (1.0 - u)
        - forcing_term(x, t, q)
    )


# ---------------------------------------------------------------------
# Initial and periodic boundary losses
# ---------------------------------------------------------------------
def standard_initial_loss(model, initial_x):
    initial_t = torch.zeros_like(initial_x)
    prediction = standard_trial_solution(model, initial_x, initial_t)
    return torch.mean((prediction - initial_condition(initial_x)) ** 2)


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
    residual = standard_residual(model, residual_x, residual_t, Q)

    residual_loss = torch.mean(residual**2)
    initial_loss = standard_initial_loss(model, initial_x)
    boundary_loss = standard_boundary_loss(model, boundary_t)
    total_loss = residual_loss + initial_loss + boundary_loss

    return total_loss, residual_loss, initial_loss, boundary_loss


def shared_total_loss(model, residual_x, residual_t, boundary_t):
    residual = shared_residual(model, residual_x, residual_t, Q)

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
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def sample_residual_points():
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
def evaluate_model(model, trial_solution):
    x_grid = torch.linspace(
        X_LEFT, X_RIGHT, EVALUATION_NX, device=DEVICE
    )
    t_grid = torch.linspace(
        0.0, FINAL_TIME, EVALUATION_NT, device=DEVICE
    )

    x_mesh, t_mesh = torch.meshgrid(
        x_grid, t_grid, indexing="ij"
    )
    x_flat = x_mesh.reshape(-1, 1)
    t_flat = t_mesh.reshape(-1, 1)

    prediction = trial_solution(
        model, x_flat, t_flat
    ).reshape(EVALUATION_NX, EVALUATION_NT)

    reference = exact_solution(
        x_flat, t_flat
    ).reshape(EVALUATION_NX, EVALUATION_NT)

    error = prediction - reference

    return {
        "E_inf": torch.max(torch.abs(error)).item(),
        "E_2": torch.sqrt(torch.mean(error**2)).item(),
        "E_2_rel": (
            torch.linalg.vector_norm(error)
            / torch.linalg.vector_norm(reference)
        ).item(),
    }


# ---------------------------------------------------------------------
# One independent run
# ---------------------------------------------------------------------
def run_one_seed(seed):
    print("\n" + "=" * 82)
    print(
        "Problem 2 multi-run study: "
        f"Burgers--Fisher equation, seed = {seed}"
    )
    print("=" * 82)

    # Standard PINN
    set_seed(seed)

    standard_model = StandardPINN(
        input_dim=2,
        hidden_dim=HIDDEN_UNITS,
        output_dim=1,
    ).to(DEVICE)

    standard_x, standard_t = sample_residual_points()
    initial_x = sample_initial_points()
    boundary_t_standard = sample_boundary_times()

    optimizer = torch.optim.Adam(
        standard_model.parameters(),
        lr=ADAM_RATE,
    )

    standard_history = []
    standard_start = time.perf_counter()

    for iteration in range(NUM_ITERATIONS + 1):
        optimizer.zero_grad(set_to_none=True)

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
            boundary_t_standard,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Standard PINN stopped at iteration "
                f"{iteration} for seed {seed}: invalid loss."
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            standard_model.parameters(),
            max_norm=GRADIENT_CLIP,
        )
        optimizer.step()
        standard_history.append(loss.item())

        if iteration % PRINT_EVERY == 0:
            print(
                f"[Seed {seed} | Standard] "
                f"Iteration {iteration:5d} | "
                f"Loss {loss.item():.6e} | "
                f"PDE {residual_loss.item():.6e} | "
                f"IC {initial_loss.item():.6e} | "
                f"BC {boundary_loss.item():.6e}"
            )

    standard_time = time.perf_counter() - standard_start

    final_standard_loss, _, _, _ = standard_total_loss(
        standard_model,
        standard_x,
        standard_t,
        initial_x,
        boundary_t_standard,
    )

    standard_metrics = evaluate_model(
        standard_model,
        standard_trial_solution,
    )

    # Shared-bias PINN
    set_seed(seed)

    shared_model = SharedBiasPINN(
        input_dim=5,
        hidden_dim=HIDDEN_UNITS,
        output_dim=1,
    ).to(DEVICE)

    shared_x, shared_t = sample_residual_points()
    boundary_t_shared = sample_boundary_times()

    shared_history = []
    shared_start = time.perf_counter()

    for iteration in range(NUM_ITERATIONS + 1):
        if iteration % RESAMPLING_INTERVAL == 0:
            shared_x, shared_t = sample_residual_points()

        shared_model.zero_grad(set_to_none=True)

        loss, residual_loss, boundary_loss = shared_total_loss(
            shared_model,
            shared_x,
            shared_t,
            boundary_t_shared,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Shared-bias PINN stopped at iteration "
                f"{iteration} for seed {seed}: invalid loss."
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            shared_model.parameters(),
            max_norm=GRADIENT_CLIP,
        )
        apply_conformable_gradient(shared_model, iteration)
        shared_history.append(loss.item())

        if iteration % PRINT_EVERY == 0:
            print(
                f"[Seed {seed} | Shared-bias] "
                f"Iteration {iteration:5d} | "
                f"Loss {loss.item():.6e} | "
                f"PDE {residual_loss.item():.6e} | "
                f"BC {boundary_loss.item():.6e} | "
                f"alpha {conformable_order(iteration):.6e} | "
                f"LR {effective_learning_rate(iteration):.6e}"
            )

    shared_time = time.perf_counter() - shared_start

    final_shared_loss, _, _ = shared_total_loss(
        shared_model,
        shared_x,
        shared_t,
        boundary_t_shared,
    )

    shared_metrics = evaluate_model(
        shared_model,
        shared_trial_solution,
    )

    standard_row = {
        "seed": seed,
        "method": "Standard PINN + Adam",
        **standard_metrics,
        "final_loss": final_standard_loss.detach().item(),
        "cpu_time": standard_time,
    }

    shared_row = {
        "seed": seed,
        "method": "Shared-bias PINN + conformable GD",
        **shared_metrics,
        "final_loss": final_shared_loss.detach().item(),
        "cpu_time": shared_time,
    }

    print(
        f"\nSeed {seed} completed:\n"
        f"  Standard E_2     = {standard_row['E_2']:.6e}\n"
        f"  Shared-bias E_2  = {shared_row['E_2']:.6e}"
    )

    return (
        standard_row,
        shared_row,
        standard_history,
        shared_history,
    )


# ---------------------------------------------------------------------
# Summary statistics and output files
# ---------------------------------------------------------------------
def summarize_runs(individual_rows):
    metric_names = [
        "E_inf",
        "E_2",
        "E_2_rel",
        "final_loss",
        "cpu_time",
    ]

    methods = [
        "Standard PINN + Adam",
        "Shared-bias PINN + conformable GD",
    ]

    summaries = []

    for method in methods:
        method_rows = [
            row for row in individual_rows
            if row["method"] == method
        ]

        summary = {"method": method}

        for metric in metric_names:
            values = np.asarray(
                [row[metric] for row in method_rows],
                dtype=float,
            )
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(
                np.std(values, ddof=1)
            )
            summary[f"{metric}_var"] = float(
                np.var(values, ddof=1)
            )

        summaries.append(summary)

    return summaries, metric_names


def save_csv_files(individual_rows, summary_rows, metric_names):
    individual_path = OUTPUT_DIR / "problem2_multirun_individual.csv"
    summary_path = OUTPUT_DIR / "problem2_multirun_summary.csv"

    with individual_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "seed",
                "method",
                "E_inf",
                "E_2",
                "E_2_rel",
                "final_loss",
                "cpu_time",
            ],
        )
        writer.writeheader()
        writer.writerows(individual_rows)

    summary_fields = ["method"]
    for metric in metric_names:
        summary_fields.extend(
            [
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_var",
            ]
        )

    with summary_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=summary_fields,
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    return individual_path, summary_path


def print_individual_results(individual_rows):
    print("\nIndividual five-seed results:\n")

    header = (
        f"{'Seed':>6}"
        f"{'Method':<40}"
        f"{'E_inf':>15}"
        f"{'E_2':>15}"
        f"{'E_2_rel':>15}"
        f"{'Final loss':>15}"
        f"{'CPU time (s)':>15}"
    )

    print(header)
    print("-" * len(header))

    for row in individual_rows:
        print(
            f"{row['seed']:6d}"
            f"{row['method']:<40}"
            f"{row['E_inf']:15.6e}"
            f"{row['E_2']:15.6e}"
            f"{row['E_2_rel']:15.6e}"
            f"{row['final_loss']:15.6e}"
            f"{row['cpu_time']:15.3f}"
        )


def print_summary(summary_rows, metric_names):
    print(
        "\nFive-seed summary: mean, sample standard deviation, "
        "and sample variance\n"
    )

    for summary in summary_rows:
        print(summary["method"])

        for metric in metric_names:
            print(
                f"  {metric:<10} "
                f"mean = {summary[f'{metric}_mean']:.6e}, "
                f"std = {summary[f'{metric}_std']:.6e}, "
                f"variance = {summary[f'{metric}_var']:.6e}"
            )

        print()


def latex_mean_std(mean, standard_deviation, digits=3):
    return (
        f"${mean:.{digits}e}"
        rf"\pm{standard_deviation:.{digits}e}$"
    )


def print_latex_table(summary_rows):
    seed_text = ", ".join(str(seed) for seed in SEEDS)

    print("\nLaTeX five-seed table:\n")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(
        r"\caption{Five-run results for the manufactured "
        r"time-conformable Burgers--Fisher problem. Each entry is "
        r"the mean $\pm$ sample standard deviation over the seeds "
        rf"$\{{{seed_text}\}}$."
        r"}"
    )
    print(r"\label{tab:burgers_fisher_multirun}")
    print(r"\scriptsize")
    print(r"\setlength{\tabcolsep}{3.5pt}")
    print(r"\resizebox{\textwidth}{!}{")
    print(r"\begin{tabular}{lccccc}")
    print(r"\hline")
    print(
        r"Method & $E_{\infty}$ & $E_2$ & "
        r"$E_{2,\mathrm{rel}}$ & Final loss & CPU time (s) \\"
    )
    print(r"\hline")

    for summary in summary_rows:
        print(
            f"{summary['method']} & "
            f"{latex_mean_std(summary['E_inf_mean'], summary['E_inf_std'])} & "
            f"{latex_mean_std(summary['E_2_mean'], summary['E_2_std'])} & "
            f"{latex_mean_std(summary['E_2_rel_mean'], summary['E_2_rel_std'])} & "
            f"{latex_mean_std(summary['final_loss_mean'], summary['final_loss_std'])} & "
            f"${summary['cpu_time_mean']:.3f}"
            rf"\pm{summary['cpu_time_std']:.3f}$ \\"
        )

    print(r"\hline")
    print(r"\end{tabular}")
    print(r"}")
    print(r"\end{table}")


# ---------------------------------------------------------------------
# Mean loss-history figure
# ---------------------------------------------------------------------
def save_loss_figure(standard_histories, shared_histories):
    standard_array = np.asarray(standard_histories, dtype=float)
    shared_array = np.asarray(shared_histories, dtype=float)

    standard_mean = np.mean(standard_array, axis=0)
    standard_sd = np.std(standard_array, axis=0, ddof=1)

    shared_mean = np.mean(shared_array, axis=0)
    shared_sd = np.std(shared_array, axis=0, ddof=1)

    iterations = np.arange(standard_mean.size)
    loss_floor = 1.0e-8

    standard_lower = np.maximum(
        standard_mean - standard_sd, loss_floor
    )
    standard_upper = standard_mean + standard_sd

    shared_lower = np.maximum(
        shared_mean - shared_sd, loss_floor
    )
    shared_upper = shared_mean + shared_sd

    figure, axis = plt.subplots(figsize=(7.8, 4.6), dpi=240)

    axis.semilogy(
        iterations,
        standard_mean,
        color=COLOR_STANDARD,
        label="Standard PINN + Adam",
    )
    axis.fill_between(
        iterations,
        standard_lower,
        standard_upper,
        color=COLOR_STANDARD,
        alpha=0.18,
        rasterized=True,
    )

    axis.semilogy(
        iterations,
        shared_mean,
        color=COLOR_SHARED,
        linestyle="--",
        label="Shared-bias PINN + conformable GD",
    )
    axis.fill_between(
        iterations,
        shared_lower,
        shared_upper,
        color=COLOR_SHARED,
        alpha=0.18,
        rasterized=True,
    )

    axis.set_xlabel("Iteration")
    axis.set_ylabel("Training loss")
    axis.set_title(
        "Five-run loss histories for the Burgers--Fisher problem"
    )
    axis.set_xlim(left=0)
    axis.set_ylim(1.0e-6, 1.0e3)
    axis.margins(x=0)
    axis.legend()

    figure.tight_layout()

    figure_path = OUTPUT_DIR / "burgers_fisher_multirun_loss.pdf"
    figure.savefig(
        figure_path,
        format="pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(figure)

    return figure_path


# ---------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------
def main():
    individual_rows = []
    standard_histories = []
    shared_histories = []

    for seed in SEEDS:
        (
            standard_row,
            shared_row,
            standard_history,
            shared_history,
        ) = run_one_seed(seed)

        individual_rows.extend([standard_row, shared_row])
        standard_histories.append(standard_history)
        shared_histories.append(shared_history)

    summary_rows, metric_names = summarize_runs(individual_rows)

    individual_csv, summary_csv = save_csv_files(
        individual_rows,
        summary_rows,
        metric_names,
    )

    print_individual_results(individual_rows)
    print_summary(summary_rows, metric_names)
    print_latex_table(summary_rows)

    loss_figure = save_loss_figure(
        standard_histories,
        shared_histories,
    )

    print(f"\nIndividual-run CSV:\n{individual_csv}")
    print(f"\nSummary CSV:\n{summary_csv}")
    print(f"\nMulti-run loss figure:\n{loss_figure}")


if __name__ == "__main__":
    main()
