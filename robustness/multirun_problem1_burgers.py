"""
Five-run study for Problem 1: the manufactured time-conformable
Burgers equation.

The script compares a standard PINN trained with Adam against a
shared-bias PINN trained with the conformable-gradient update. Each
model is trained with the seeds 0, 1, 2, 3, and 4. The script reports
individual results, sample statistics, CSV files, a LaTeX table, and a
mean loss plot with one-standard-deviation bands.
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
except Exception:
    pass


# -----------------------------------------------------------------------------
# Numerical and plotting settings
# -----------------------------------------------------------------------------
DEVICE = "cpu"
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)

SEEDS = [0, 1, 2, 3, 4]

plt.rcParams.update(
    {
        "figure.dpi": 180,
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


# -----------------------------------------------------------------------------
# Problem definition
# -----------------------------------------------------------------------------
Q_ORDER = 0.7
OMEGA = 5.0

if OMEGA not in {5.0, 10.0, 20.0}:
    raise ValueError("OMEGA must be 5.0, 10.0, or 20.0.")

OMEGA_TAG = str(int(OMEGA))
VISCOSITY = 0.01 / math.pi
X_LEFT = 0.0
X_RIGHT = 2.0 * math.pi
FINAL_TIME = 1.0

HIDDEN_UNITS = 80
NUM_RESIDUAL = 200
NUM_INITIAL = 100
NUM_BOUNDARY = 100
NUM_ITERATIONS = int(os.environ.get("MULTIRUN_NUM_ITERS", "10000"))

ADAM_RATE = 1.0e-3
BASE_RATE = 1.0e-2
CLIP_VALUE = 10.0
MIN_TIME = 1.0e-4
PRINT_EVERY = max(1, min(1000, NUM_ITERATIONS))

EVALUATION_NX = 256
EVALUATION_NT = 201


# -----------------------------------------------------------------------------
# Output folder
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs" / f"omega_{OMEGA_TAG}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Exact solution, forcing term, and input maps
# -----------------------------------------------------------------------------
def exact_solution(x, t):
    """Exact solution u(x,t) = exp(-t) sin(omega x)."""
    return torch.exp(-t) * torch.sin(OMEGA * x)


def forcing_term(x, t, q):
    """Manufactured source term for the Burgers equation."""
    exponential = torch.exp(-t)
    sine = torch.sin(OMEGA * x)
    cosine = torch.cos(OMEGA * x)

    return (
        -t ** (1.0 - q) * exponential * sine
        + OMEGA * exponential**2 * sine * cosine
        + VISCOSITY * OMEGA**2 * exponential * sine
    )


def initial_condition(x):
    return torch.sin(OMEGA * x)


def standard_features(x, t):
    return torch.cat((x, t), dim=1)


def periodic_features(x, t):
    return torch.cat(
        (torch.sin(OMEGA * x), torch.cos(OMEGA * x), t),
        dim=1,
    )


# -----------------------------------------------------------------------------
# Neural networks
# -----------------------------------------------------------------------------
class StandardPINN(nn.Module):
    """Single-hidden-layer PINN with ordinary neuron-wise biases."""

    def __init__(self, input_dim=2, hidden_dim=80, output_dim=1):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_normal_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, inputs):
        hidden_values = torch.tanh(self.hidden(inputs))
        return self.output(hidden_values)


class SharedBiasPINN(nn.Module):
    """Single-hidden-layer network with one common hidden bias."""

    def __init__(self, input_dim=3, hidden_dim=80, output_dim=1):
        super().__init__()
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
            torch.randn(1, output_dim) * math.sqrt(1.0 / output_dim)
        )

    def forward(self, inputs):
        hidden_values = torch.tanh(inputs @ self.W1 + self.b1)
        return hidden_values @ self.W2 + self.b2


# -----------------------------------------------------------------------------
# Trial solutions and differential operators
# -----------------------------------------------------------------------------
def standard_trial_solution(model, x, t):
    return initial_condition(x) + t * model(standard_features(x, t))


def shared_trial_solution(model, x, t):
    return initial_condition(x) + t * model(periodic_features(x, t))


def derivative(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def burgers_residual(model, x, t, q, trial_solution):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    u = trial_solution(model, x, t)
    u_t = derivative(u, t)
    u_x = derivative(u, x)
    u_xx = derivative(u_x, x)

    conformable_time_term = t ** (1.0 - q) * u_t
    source = forcing_term(x, t, q)

    return conformable_time_term + u * u_x - VISCOSITY * u_xx - source


def initial_loss(model, x_initial, trial_solution):
    t_initial = torch.zeros_like(x_initial)
    mismatch = (
        trial_solution(model, x_initial, t_initial)
        - initial_condition(x_initial)
    )
    return torch.mean(mismatch**2)


def periodic_boundary_loss(model, boundary_times, trial_solution):
    x_left = torch.full_like(
        boundary_times,
        X_LEFT,
        requires_grad=True,
    )
    x_right = torch.full_like(
        boundary_times,
        X_RIGHT,
        requires_grad=True,
    )
    t_left = boundary_times.clone().detach().requires_grad_(True)
    t_right = boundary_times.clone().detach().requires_grad_(True)

    u_left = trial_solution(model, x_left, t_left)
    u_right = trial_solution(model, x_right, t_right)
    ux_left = derivative(u_left, x_left)
    ux_right = derivative(u_right, x_right)

    value_loss = torch.mean((u_left - u_right) ** 2)
    derivative_loss = torch.mean((ux_left - ux_right) ** 2)
    return value_loss + derivative_loss


def total_loss(
    model,
    x_residual,
    t_residual,
    x_initial,
    boundary_times,
    q,
    trial_solution,
):
    residual = burgers_residual(
        model,
        x_residual,
        t_residual,
        q,
        trial_solution,
    )
    residual_loss = torch.mean(residual**2)
    ic_loss = initial_loss(model, x_initial, trial_solution)
    bc_loss = periodic_boundary_loss(
        model,
        boundary_times,
        trial_solution,
    )
    loss = residual_loss + ic_loss + bc_loss
    return loss, residual_loss, ic_loss, bc_loss


# -----------------------------------------------------------------------------
# Conformable-gradient update
# -----------------------------------------------------------------------------
def conformable_order(iteration):
    return 1.5 - 2.0 * (
        math.atan(iteration) + math.pi / 2.0
    ) / math.pi / 2.0


def scheduled_rate(iteration, base_rate=BASE_RATE):
    alpha = max(conformable_order(iteration), 1.0e-8)
    return base_rate * alpha


@torch.no_grad()
def apply_conformable_gradient(model, iteration, base_rate=BASE_RATE):
    alpha = max(conformable_order(iteration), 1.0e-14)
    learning_rate = scheduled_rate(iteration, base_rate)

    for parameter in model.parameters():
        if parameter.grad is None:
            continue

        parameter_norm = torch.norm(parameter).clamp_min(1.0e-14)
        scaled_gradient = (
            parameter_norm**alpha
            * parameter.grad
            / alpha
        )
        parameter -= learning_rate * scaled_gradient


# -----------------------------------------------------------------------------
# Sampling and evaluation
# -----------------------------------------------------------------------------
def make_training_points():
    """Generate the fixed points used by both models in one seed."""
    t_residual = MIN_TIME + (
        FINAL_TIME - MIN_TIME
    ) * torch.rand(NUM_RESIDUAL, 1, device=DEVICE)
    x_residual = X_LEFT + (
        X_RIGHT - X_LEFT
    ) * torch.rand(NUM_RESIDUAL, 1, device=DEVICE)

    x_initial = torch.linspace(
        X_LEFT,
        X_RIGHT,
        NUM_INITIAL,
        device=DEVICE,
    ).reshape(-1, 1)
    boundary_times = torch.linspace(
        MIN_TIME,
        FINAL_TIME,
        NUM_BOUNDARY,
        device=DEVICE,
    ).reshape(-1, 1)

    return x_residual, t_residual, x_initial, boundary_times


@torch.no_grad()
def evaluate_model(model, trial_solution):
    x_grid = torch.linspace(
        X_LEFT,
        X_RIGHT,
        EVALUATION_NX,
        device=DEVICE,
    )
    t_grid = torch.linspace(
        0.0,
        FINAL_TIME,
        EVALUATION_NT,
        device=DEVICE,
    )

    x_mesh, t_mesh = torch.meshgrid(x_grid, t_grid, indexing="ij")
    x_flat = x_mesh.reshape(-1, 1)
    t_flat = t_mesh.reshape(-1, 1)

    prediction = trial_solution(
        model,
        x_flat,
        t_flat,
    ).reshape(EVALUATION_NX, EVALUATION_NT)
    exact = exact_solution(
        x_flat,
        t_flat,
    ).reshape(EVALUATION_NX, EVALUATION_NT)

    error = prediction - exact
    return {
        "E_inf": torch.max(torch.abs(error)).item(),
        "E_2": torch.sqrt(torch.mean(error**2)).item(),
        "E_2_rel": (torch.norm(error) / torch.norm(exact)).item(),
    }


# -----------------------------------------------------------------------------
# Training routines
# -----------------------------------------------------------------------------
def train_standard_model(
    model,
    x_residual,
    t_residual,
    x_initial,
    boundary_times,
    seed,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=ADAM_RATE)
    history = []
    start_time = time.perf_counter()

    for iteration in range(NUM_ITERATIONS + 1):
        optimizer.zero_grad(set_to_none=True)

        loss, residual_loss, ic_loss, bc_loss = total_loss(
            model,
            x_residual,
            t_residual,
            x_initial,
            boundary_times,
            Q_ORDER,
            standard_trial_solution,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Standard PINN stopped at iteration "
                f"{iteration} for seed {seed}: invalid loss."
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=CLIP_VALUE,
        )
        optimizer.step()
        history.append(loss.item())

        if iteration % PRINT_EVERY == 0:
            print(
                f"[Seed {seed} | Standard] Iteration {iteration:5d} | "
                f"Loss {loss.item():.6e} | "
                f"PDE {residual_loss.item():.6e} | "
                f"IC {ic_loss.item():.6e} | "
                f"BC {bc_loss.item():.6e}"
            )

    return history, time.perf_counter() - start_time


def train_shared_model(
    model,
    x_residual,
    t_residual,
    x_initial,
    boundary_times,
    seed,
):
    history = []
    start_time = time.perf_counter()

    for iteration in range(NUM_ITERATIONS + 1):
        model.zero_grad(set_to_none=True)

        loss, residual_loss, ic_loss, bc_loss = total_loss(
            model,
            x_residual,
            t_residual,
            x_initial,
            boundary_times,
            Q_ORDER,
            shared_trial_solution,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Shared-bias PINN stopped at iteration "
                f"{iteration} for seed {seed}: invalid loss."
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=CLIP_VALUE,
        )
        apply_conformable_gradient(
            model,
            iteration,
            base_rate=BASE_RATE,
        )
        history.append(loss.item())

        if iteration % PRINT_EVERY == 0:
            print(
                f"[Seed {seed} | Shared-bias] Iteration {iteration:5d} | "
                f"Loss {loss.item():.6e} | "
                f"PDE {residual_loss.item():.6e} | "
                f"IC {ic_loss.item():.6e} | "
                f"BC {bc_loss.item():.6e} | "
                f"alpha {conformable_order(iteration):.6e} | "
                f"LR {scheduled_rate(iteration):.6e}"
            )

    return history, time.perf_counter() - start_time


def final_training_loss(
    model,
    x_residual,
    t_residual,
    x_initial,
    boundary_times,
    trial_solution,
):
    loss, _, _, _ = total_loss(
        model,
        x_residual,
        t_residual,
        x_initial,
        boundary_times,
        Q_ORDER,
        trial_solution,
    )
    return loss.detach().item()


def run_seed(seed):
    print("\n" + "=" * 78)
    print(f"Problem 1 multi-run study: omega = {OMEGA_TAG}, seed = {seed}")
    print("=" * 78)

    torch.manual_seed(seed)
    np.random.seed(seed)

    # The original initialization and sampling order is kept here.
    shared_model = SharedBiasPINN(
        input_dim=3,
        hidden_dim=HIDDEN_UNITS,
        output_dim=1,
    ).to(DEVICE)

    training_points = make_training_points()
    x_residual, t_residual, x_initial, boundary_times = training_points

    torch.manual_seed(seed)
    standard_model = StandardPINN(
        input_dim=2,
        hidden_dim=HIDDEN_UNITS,
        output_dim=1,
    ).to(DEVICE)

    standard_history, standard_time = train_standard_model(
        standard_model,
        x_residual,
        t_residual,
        x_initial,
        boundary_times,
        seed,
    )
    shared_history, shared_time = train_shared_model(
        shared_model,
        x_residual,
        t_residual,
        x_initial,
        boundary_times,
        seed,
    )

    standard_metrics = evaluate_model(
        standard_model,
        standard_trial_solution,
    )
    shared_metrics = evaluate_model(
        shared_model,
        shared_trial_solution,
    )

    standard_row = {
        "seed": seed,
        "method": "Standard PINN + Adam",
        **standard_metrics,
        "final_loss": final_training_loss(
            standard_model,
            x_residual,
            t_residual,
            x_initial,
            boundary_times,
            standard_trial_solution,
        ),
        "cpu_time": standard_time,
    }
    shared_row = {
        "seed": seed,
        "method": "Shared-bias PINN + conformable GD",
        **shared_metrics,
        "final_loss": final_training_loss(
            shared_model,
            x_residual,
            t_residual,
            x_initial,
            boundary_times,
            shared_trial_solution,
        ),
        "cpu_time": shared_time,
    }

    print(
        f"\nSeed {seed} completed:\n"
        f"  Standard E_2     = {standard_row['E_2']:.6e}\n"
        f"  Shared-bias E_2  = {shared_row['E_2']:.6e}"
    )

    return standard_row, shared_row, standard_history, shared_history


# -----------------------------------------------------------------------------
# Statistical summaries and output files
# -----------------------------------------------------------------------------
METRICS = ["E_inf", "E_2", "E_2_rel", "final_loss", "cpu_time"]
METHODS = [
    "Standard PINN + Adam",
    "Shared-bias PINN + conformable GD",
]


def summarize_results(rows):
    summaries = []

    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        summary = {"method": method}

        for metric in METRICS:
            values = np.asarray(
                [row[metric] for row in method_rows],
                dtype=float,
            )
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_std"] = float(np.std(values, ddof=1))
            summary[f"{metric}_var"] = float(np.var(values, ddof=1))

        summaries.append(summary)

    return summaries


def save_csv_files(individual_rows, summary_rows):
    individual_path = (
        OUTPUT_DIR
        / f"problem1_multirun_individual_omega{OMEGA_TAG}.csv"
    )
    summary_path = (
        OUTPUT_DIR
        / f"problem1_multirun_summary_omega{OMEGA_TAG}.csv"
    )

    individual_fields = [
        "seed",
        "method",
        "E_inf",
        "E_2",
        "E_2_rel",
        "final_loss",
        "cpu_time",
    ]
    with individual_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=individual_fields)
        writer.writeheader()
        writer.writerows(individual_rows)

    summary_fields = ["method"]
    for metric in METRICS:
        summary_fields.extend(
            [
                f"{metric}_mean",
                f"{metric}_std",
                f"{metric}_var",
            ]
        )

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    return individual_path, summary_path


def print_individual_results(rows):
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

    for row in rows:
        print(
            f"{row['seed']:6d}"
            f"{row['method']:<40}"
            f"{row['E_inf']:15.6e}"
            f"{row['E_2']:15.6e}"
            f"{row['E_2_rel']:15.6e}"
            f"{row['final_loss']:15.6e}"
            f"{row['cpu_time']:15.3f}"
        )


def print_summary_statistics(summary_rows):
    print(
        "\nFive-seed summary: mean, sample standard deviation, "
        "and sample variance\n"
    )

    for summary in summary_rows:
        print(summary["method"])
        for metric in METRICS:
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
        f"\\pm{standard_deviation:.{digits}e}$"
    )


def print_latex_table(summary_rows):
    seed_text = ", ".join(str(seed) for seed in SEEDS)

    print("\nLaTeX five-seed table:\n")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(
        r"\caption{Five-run results for the manufactured "
        rf"time-conformable Burgers problem with $\omega={OMEGA_TAG}$. "
        r"Each entry is the mean $\pm$ sample standard deviation over "
        rf"the seeds $\{{{seed_text}\}}$."
        r"}"
    )
    print(rf"\label{{tab:burgers_omega{OMEGA_TAG}_multirun}}")
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


def save_mean_loss_plot(standard_histories, shared_histories):
    standard_array = np.asarray(standard_histories, dtype=float)
    shared_array = np.asarray(shared_histories, dtype=float)

    standard_mean = np.mean(standard_array, axis=0)
    standard_sd = np.std(standard_array, axis=0, ddof=1)
    shared_mean = np.mean(shared_array, axis=0)
    shared_sd = np.std(shared_array, axis=0, ddof=1)

    iterations = np.arange(standard_mean.size)
    standard_lower = np.maximum(
        standard_mean - standard_sd,
        1.0e-8,
    )
    shared_lower = np.maximum(
        shared_mean - shared_sd,
        1.0e-8,
    )

    fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=240)
    ax.semilogy(
        iterations,
        standard_mean,
        color=COLOR_STANDARD,
        label="Standard PINN + Adam",
    )
    ax.fill_between(
        iterations,
        standard_lower,
        standard_mean + standard_sd,
        color=COLOR_STANDARD,
        alpha=0.18,
        rasterized=True,
    )
    ax.semilogy(
        iterations,
        shared_mean,
        color=COLOR_SHARED,
        linestyle="--",
        label="Shared-bias PINN + conformable GD",
    )
    ax.fill_between(
        iterations,
        shared_lower,
        shared_mean + shared_sd,
        color=COLOR_SHARED,
        alpha=0.18,
        rasterized=True,
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training loss")
    ax.set_title(rf"Five-run loss histories, $\omega={OMEGA_TAG}$")
    ax.set_xlim(left=0)
    ax.set_ylim(1.0e-6, 1.0e3)
    ax.margins(x=0)
    ax.legend()
    fig.tight_layout()

    figure_path = (
        OUTPUT_DIR
        / f"burgers_multirun_loss_omega{OMEGA_TAG}.pdf"
    )
    fig.savefig(
        figure_path,
        format="pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)
    return figure_path


def main():
    individual_rows = []
    standard_histories = []
    shared_histories = []

    for seed in SEEDS:
        standard_row, shared_row, standard_history, shared_history = (
            run_seed(seed)
        )
        individual_rows.extend((standard_row, shared_row))
        standard_histories.append(standard_history)
        shared_histories.append(shared_history)

    summary_rows = summarize_results(individual_rows)
    individual_csv, summary_csv = save_csv_files(
        individual_rows,
        summary_rows,
    )

    print_individual_results(individual_rows)
    print_summary_statistics(summary_rows)
    print_latex_table(summary_rows)

    loss_figure = save_mean_loss_plot(
        standard_histories,
        shared_histories,
    )

    print(f"\nIndividual-run CSV:\n{individual_csv}")
    print(f"\nSummary CSV:\n{summary_csv}")
    print(f"\nMulti-run loss figure:\n{loss_figure}")


if __name__ == "__main__":
    main()
