"""
One-factor-at-a-time sensitivity study for Problem 1.

The benchmark is the manufactured time-conformable Burgers equation

    T_q u + u u_x - nu u_xx = f(x,t),

with exact solution

    u(x,t) = exp(-t) sin(omega x).

The study varies:
    1. The hidden width H.
    2. The base conformable-gradient rate L0.
    3. The hidden-layer activation.

All configurations use the same collocation points and the same seed.
The equation, initialization sequence, training settings, conformable-
gradient update, evaluation grid, CSV filename, and LaTeX output are
preserved.
"""

import csv
import gc
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
# Figure settings
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
# Output folder
# ---------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs" / f"omega_{OMEGA_TAG}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Training and evaluation settings
# ---------------------------------------------------------------------
NUM_RESIDUAL_POINTS = 200
NUM_INITIAL_POINTS = 100
NUM_BOUNDARY_POINTS = 100
NUM_ITERATIONS = 10000
GRADIENT_CLIP = 10.0

EVALUATION_NX = 256
EVALUATION_NT = 201


# ---------------------------------------------------------------------
# Sensitivity configurations
# ---------------------------------------------------------------------
CONFIGURATIONS = [
    {
        "id": "width_40",
        "hidden_width": 40,
        "base_rate": 1.0e-2,
        "activation": "tanh",
    },
    {
        "id": "baseline",
        "hidden_width": 80,
        "base_rate": 1.0e-2,
        "activation": "tanh",
    },
    {
        "id": "width_120",
        "hidden_width": 120,
        "base_rate": 1.0e-2,
        "activation": "tanh",
    },
    {
        "id": "rate_5e-3",
        "hidden_width": 80,
        "base_rate": 5.0e-3,
        "activation": "tanh",
    },
    {
        "id": "rate_2e-2",
        "hidden_width": 80,
        "base_rate": 2.0e-2,
        "activation": "tanh",
    },
    {
        "id": "activation_sine",
        "hidden_width": 80,
        "base_rate": 1.0e-2,
        "activation": "sine",
    },
]


# ---------------------------------------------------------------------
# Exact solution, initial condition, and forcing term
# ---------------------------------------------------------------------
def exact_solution(x, t):
    return torch.exp(-t) * torch.sin(OMEGA * x)


def initial_condition(x):
    return torch.sin(OMEGA * x)


def forcing_term(x, t):
    decay = torch.exp(-t)
    sine = torch.sin(OMEGA * x)
    cosine = torch.cos(OMEGA * x)

    return (
        -t ** (1.0 - Q) * decay * sine
        + OMEGA * decay**2 * sine * cosine
        + NU * OMEGA**2 * decay * sine
    )


# ---------------------------------------------------------------------
# Shared-bias model and periodic features
# ---------------------------------------------------------------------
def periodic_features(x, t):
    return torch.cat(
        (
            torch.sin(OMEGA * x),
            torch.cos(OMEGA * x),
            t,
        ),
        dim=1,
    )


class SharedBiasPINN(nn.Module):
    """One-hidden-layer shared-bias network."""

    def __init__(
        self,
        input_dim=3,
        hidden_width=80,
        output_dim=1,
        activation="tanh",
    ):
        super().__init__()

        if activation not in {"tanh", "sine"}:
            raise ValueError(
                "activation must be 'tanh' or 'sine'."
            )

        self.activation_name = activation

        # The original parameter creation order is preserved.
        self.W1 = nn.Parameter(
            torch.randn(input_dim, hidden_width)
            * math.sqrt(1.0 / input_dim)
        )
        self.W2 = nn.Parameter(
            torch.randn(hidden_width, output_dim)
            * math.sqrt(1.0 / hidden_width)
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

        if self.activation_name == "tanh":
            hidden_output = torch.tanh(hidden_argument)
        else:
            hidden_output = torch.sin(hidden_argument)

        return hidden_output @ self.W2 + self.b2


# ---------------------------------------------------------------------
# Trial solution and automatic differentiation
# ---------------------------------------------------------------------
def trial_solution(model, x, t):
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
def burgers_residual(model, x, t):
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
        - forcing_term(local_x, local_t)
    )


def initial_loss(model, initial_x):
    initial_t = torch.zeros_like(initial_x)
    prediction = trial_solution(
        model,
        initial_x,
        initial_t,
    )

    return torch.mean(
        (prediction - initial_condition(initial_x)) ** 2
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
    residual_x,
    residual_t,
    initial_x,
    boundary_t,
):
    residual = burgers_residual(
        model,
        residual_x,
        residual_t,
    )

    residual_loss = torch.mean(residual**2)
    initial_condition_loss = initial_loss(
        model,
        initial_x,
    )
    boundary_condition_loss = periodic_boundary_loss(
        model,
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
    )


# ---------------------------------------------------------------------
# Conformable-gradient schedule and update
# ---------------------------------------------------------------------
def conformable_order(iteration):
    return (
        1.5
        - 2.0
        * (math.atan(iteration) + math.pi / 2.0)
        / math.pi
        / 2.0
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

    # The blockwise update follows the original W1, b1, W2, b2 order.
    parameter_blocks = (
        model.W1,
        model.b1,
        model.W2,
        model.b2,
    )

    for parameter in parameter_blocks:
        gradient = parameter.grad

        if gradient is None:
            continue

        parameter_norm = torch.norm(
            parameter
        ).clamp_min(1.0e-14)

        scaled_gradient = (
            parameter_norm**alpha
        ) * gradient / alpha

        parameter -= learning_rate * scaled_gradient


# ---------------------------------------------------------------------
# Common collocation points
# ---------------------------------------------------------------------
def create_common_training_points():
    """
    Reproduce the original seed-0 sequence.

    The baseline model is instantiated first. The common point set is
    then generated and reused by every sensitivity configuration.
    """

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    reference_model = SharedBiasPINN(
        input_dim=3,
        hidden_width=80,
        output_dim=1,
        activation="tanh",
    ).to(DEVICE)

    residual_t = MINIMUM_TIME + (
        FINAL_TIME - MINIMUM_TIME
    ) * torch.rand(
        NUM_RESIDUAL_POINTS,
        1,
        device=DEVICE,
    )

    residual_x = X_LEFT + (
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

    del reference_model
    gc.collect()

    return residual_x, residual_t, initial_x, boundary_t


# ---------------------------------------------------------------------
# Evaluation and parameter count
# ---------------------------------------------------------------------
def count_trainable_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


@torch.no_grad()
def evaluate_model(model):
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
    ).reshape(EVALUATION_NX, EVALUATION_NT)

    reference = exact_solution(
        x_flat,
        t_flat,
    ).reshape(EVALUATION_NX, EVALUATION_NT)

    error = prediction - reference

    return (
        torch.max(torch.abs(error)).item(),
        torch.sqrt(torch.mean(error**2)).item(),
        (torch.norm(error) / torch.norm(reference)).item(),
    )


# ---------------------------------------------------------------------
# One sensitivity configuration
# ---------------------------------------------------------------------
def run_configuration(
    configuration,
    residual_x,
    residual_t,
    initial_x,
    boundary_t,
):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    model = SharedBiasPINN(
        input_dim=3,
        hidden_width=configuration["hidden_width"],
        output_dim=1,
        activation=configuration["activation"],
    ).to(DEVICE)

    start_time = time.perf_counter()

    for iteration in range(NUM_ITERATIONS + 1):
        model.zero_grad(set_to_none=True)

        (
            loss,
            residual_loss,
            initial_condition_loss,
            boundary_condition_loss,
        ) = total_loss(
            model,
            residual_x,
            residual_t,
            initial_x,
            boundary_t,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"{configuration['id']} stopped at iteration "
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
            configuration["base_rate"],
        )

        if iteration % 1000 == 0:
            print(
                f"[{configuration['id']}] "
                f"Iteration {iteration:5d} | "
                f"Loss {loss.item():.6e} | "
                f"PDE {residual_loss.item():.6e} | "
                f"IC {initial_condition_loss.item():.6e} | "
                f"BC {boundary_condition_loss.item():.6e}"
            )

    cpu_time = time.perf_counter() - start_time

    final_loss, _, _, _ = total_loss(
        model,
        residual_x,
        residual_t,
        initial_x,
        boundary_t,
    )

    e_inf, e_2, e_2_rel = evaluate_model(model)

    result = {
        "id": configuration["id"],
        "H": configuration["hidden_width"],
        "L0": configuration["base_rate"],
        "activation": configuration["activation"],
        "parameters": count_trainable_parameters(model),
        "E_inf": e_inf,
        "E_2": e_2,
        "E_2_rel": e_2_rel,
        "final_loss": final_loss.detach().item(),
        "cpu_time": cpu_time,
    }

    print(
        f"\nCompleted {configuration['id']}: "
        f"E_inf={e_inf:.6e}, "
        f"E_2={e_2:.6e}, "
        f"E_2_rel={e_2_rel:.6e}, "
        f"loss={result['final_loss']:.6e}, "
        f"CPU={cpu_time:.3f} s\n"
    )

    del model
    gc.collect()

    return result


# ---------------------------------------------------------------------
# CSV and table output
# ---------------------------------------------------------------------
def save_results_csv(results):
    csv_path = (
        OUTPUT_DIR / "problem1_sensitivity_omega5.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "id",
                "H",
                "L0",
                "activation",
                "parameters",
                "E_inf",
                "E_2",
                "E_2_rel",
                "final_loss",
                "cpu_time",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    return csv_path


def print_plain_table(results):
    print("\nProblem 1 sensitivity results, omega = 5:\n")

    header = (
        f"{'Configuration':<20}"
        f"{'H':>7}"
        f"{'L0':>13}"
        f"{'Activation':>13}"
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
            f"{row['id']:<20}"
            f"{row['H']:7d}"
            f"{row['L0']:13.3e}"
            f"{row['activation']:>13}"
            f"{row['parameters']:13d}"
            f"{row['E_inf']:15.6e}"
            f"{row['E_2']:15.6e}"
            f"{row['E_2_rel']:15.6e}"
            f"{row['final_loss']:15.6e}"
            f"{row['cpu_time']:15.3f}"
        )


def print_latex_table(results):
    by_id = {row["id"]: row for row in results}

    table_rows = [
        ("Width", r"$H=40$", by_id["width_40"]),
        ("Width", r"$H=80$", by_id["baseline"]),
        ("Width", r"$H=120$", by_id["width_120"]),
        (
            "Base rate",
            r"$L_0=5.0\times10^{-3}$",
            by_id["rate_5e-3"],
        ),
        (
            "Base rate",
            r"$L_0=1.0\times10^{-2}$",
            by_id["baseline"],
        ),
        (
            "Base rate",
            r"$L_0=2.0\times10^{-2}$",
            by_id["rate_2e-2"],
        ),
        ("Activation", r"$\tanh$", by_id["baseline"]),
        (
            "Activation",
            r"$\sin$",
            by_id["activation_sine"],
        ),
    ]

    print("\nConsolidated LaTeX sensitivity table:\n")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(
        r"\caption{One-factor-at-a-time sensitivity results for the "
        r"proposed method on Problem~1 with $\omega=5$. The baseline "
        r"configuration is $H=80$, $L_0=10^{-2}$, and $\tanh$ activation.}"
    )
    print(r"\label{tab:problem1_sensitivity}")
    print(r"\scriptsize")
    print(r"\setlength{\tabcolsep}{3.5pt}")
    print(r"\resizebox{\textwidth}{!}{")
    print(r"\begin{tabular}{llcccccc}")
    print(r"\hline")
    print(
        r"Factor & Setting & Parameters & $E_{\infty}$ & $E_2$ "
        r"& $E_{2,\mathrm{rel}}$ & Final loss & CPU time (s) \\"
    )
    print(r"\hline")

    for factor, setting, row in table_rows:
        print(
            f"{factor} & {setting} & "
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
# Main program
# ---------------------------------------------------------------------
def main():
    (
        residual_x,
        residual_t,
        initial_x,
        boundary_t,
    ) = create_common_training_points()

    results = [
        run_configuration(
            configuration,
            residual_x,
            residual_t,
            initial_x,
            boundary_t,
        )
        for configuration in CONFIGURATIONS
    ]

    csv_path = save_results_csv(results)

    print_plain_table(results)
    print_latex_table(results)

    print(f"\nCSV file saved in:\n{csv_path}")


if __name__ == "__main__":
    main()
