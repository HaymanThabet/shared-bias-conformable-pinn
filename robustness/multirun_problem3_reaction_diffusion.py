"""
Combined comparison code for Problem 3:
manufactured time-conformable reaction--diffusion equation.

Cases:
1) (omega1, omega2) = (2, 5)
2) (omega1, omega2) = (4, 9)
3) (omega1, omega2) = (6, 12)

Exact solution:
    u_ex(x,t) = exp(-sqrt(t+1e-3))*cos(omega1*x)
                + 0.35*exp(-t)*sin(omega2*x)

Compares:
1) Standard PINN baseline
2) Shared-bias PINN with conformable-gradient training

For each case, the code:
- computes E_inf, E_2, E_{2,rel}, final training loss, and CPU time;
- saves PDF-only solution, error, final-time, loss-history, and heatmap figures.

For the representative Case 1, (omega1, omega2) = (2, 5), the code
also reports the trainable parameter count and peak memory increase.
For a fair comparison, each peak-memory measurement is performed in a
separate fresh Python process from a common baseline before model,
data, and optimizer construction.

At the end, the code prints:
- one consolidated accuracy table for all three cases;
- one efficiency table for the representative Case 1.

Pointwise tables are intentionally omitted.
"""

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import math
import time
import numpy as np
import psutil
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    torch.set_num_threads(1)
except Exception:
    pass

# ============================================================
# Isolated-memory worker arguments
# ============================================================
def parse_command_line():
    parser = argparse.ArgumentParser(
        description=(
            "Problem 3 reaction--diffusion benchmark with "
            "isolated-process efficiency diagnostics."
        )
    )
    parser.add_argument(
        "--memory-method",
        choices=("standard", "shared"),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--memory-output",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--memory-case",
        default="case1",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


ARGS = parse_command_line()
MEMORY_METHOD = ARGS.memory_method
MEMORY_OUTPUT = ARGS.memory_output
MEMORY_CASE = ARGS.memory_case

if MEMORY_METHOD is not None and MEMORY_OUTPUT is None:
    raise ValueError(
        "--memory-output is required when --memory-method is used."
    )


# ============================================================
# Output folder for figures
# ============================================================
output_dir = Path(__file__).resolve().parent / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# Figure quality
# ============================================================
plt.rcParams["figure.dpi"] = 220
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["pdf.compression"] = 9
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 10
plt.rcParams["lines.linewidth"] = 2.2

COLOR_STD = "#1f3b73"
COLOR_SB  = "#d95f02"
COLOR_EX  = "#2e7d32"

# ============================================================
# Settings
# ============================================================
device = "cpu"
dtype = torch.float64
torch.set_default_dtype(dtype)
torch.manual_seed(0)
np.random.seed(0)

# ============================================================
# Problem parameters
# ============================================================
q_frac = 0.7
nu = 0.01 / math.pi
lam = 1.0

x_left = 0.0
x_right = 2.0 * math.pi
T_final = 1.0
eps_reg = 1.0e-3

# ============================================================
# Shared implementation settings
# ============================================================
clip_value = 10.0
hid_dim = 100
num_f = 200
num_x0 = 100
num_tb = 100
num_iters = 20000
LR0 = 5e-2

# Standard PINN uses the same iteration count
adam_iters = num_iters
adam_lr = 5e-2

eps_t = 1.0e-4

# ============================================================
# Cases for Problem 3
# ============================================================
case_list = [
    ("case1", 2.0, 5.0),
    ("case2", 4.0, 9.0),
    ("case3", 6.0, 12.0),
]

# ============================================================
# Utilities
# ============================================================
def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True
    )[0]

def smooth_curve(values, beta=0.98):
    out = []
    avg = 0.0
    for k, v in enumerate(values, 1):
        avg = beta * avg + (1.0 - beta) * v
        out.append(avg / (1.0 - beta**k))
    return out

def save_pdf(fig, stem):
    """Save a PDF without terminating the run when Windows locks an existing file."""
    target_path = output_dir / f"{stem}.pdf"

    try:
        fig.savefig(target_path, bbox_inches="tight")
        return target_path
    except PermissionError:
        # Windows may lock a PDF that is open in Acrobat or in the
        # File Explorer preview pane. Save under a temporary suffix
        # so the completed numerical run is not lost.
        counter = 1
        while True:
            fallback_path = output_dir / f"{stem}_new{counter}.pdf"
            if not fallback_path.exists():
                fig.savefig(fallback_path, bbox_inches="tight")
                print(
                    f"\\nWarning: the target PDF is locked:\\n{target_path}"
                    f"\\nThe figure was saved instead as:\\n{fallback_path}\\n"
                    "Close the original PDF, then replace it with the new file."
                )
                return fallback_path
            counter += 1

# ============================================================
# Efficiency diagnostics
# ============================================================
def count_trainable_parameters(model):
    """Return the number of trainable scalar parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


class PeakRSSMonitor:
    """
    Monitor the maximum increase in process resident-set size.

    The increase is measured relative to the process RSS at monitor
    startup. It includes CPU allocations visible to the operating
    system from Python, NumPy, and PyTorch.
    """

    def __init__(self, sampling_interval=0.01):
        self.sampling_interval = sampling_interval
        self.process = psutil.Process(os.getpid())
        self._stop_event = threading.Event()
        self._thread = None
        self.baseline_rss = 0
        self.peak_rss = 0

    def _sample(self):
        while not self._stop_event.wait(self.sampling_interval):
            current_rss = self.process.memory_info().rss
            if current_rss > self.peak_rss:
                self.peak_rss = current_rss

    def __enter__(self):
        gc.collect()
        self.baseline_rss = self.process.memory_info().rss
        self.peak_rss = self.baseline_rss
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._sample,
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        current_rss = self.process.memory_info().rss
        if current_rss > self.peak_rss:
            self.peak_rss = current_rss

        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

    @property
    def peak_increase_mb(self):
        return max(
            0.0,
            (self.peak_rss - self.baseline_rss) / (1024.0**2),
        )


def write_memory_worker_result(
    method,
    case_name,
    peak_memory_mb,
    parameters,
):
    """Write one isolated memory-worker result to JSON."""
    output_path = Path(MEMORY_OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "method": method,
                "case": case_name,
                "peak_memory_increase_mb": float(peak_memory_mb),
                "trainable_parameters": int(parameters),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def run_isolated_memory_worker(method, case_name):
    """Measure one method and one case in a fresh process."""
    script_path = Path(__file__).resolve()

    with tempfile.TemporaryDirectory(
        prefix=f"problem3_{case_name}_{method}_memory_"
    ) as temporary_directory:
        result_path = (
            Path(temporary_directory)
            / f"{case_name}_{method}_memory.json"
        )

        command = [
            sys.executable,
            str(script_path),
            "--memory-method",
            method,
            "--memory-case",
            case_name,
            "--memory-output",
            str(result_path),
        ]

        worker_environment = os.environ.copy()
        worker_environment["MPLBACKEND"] = "Agg"

        completed = subprocess.run(
            command,
            env=worker_environment,
            capture_output=True,
            text=True,
            check=False,
        )

        if completed.returncode != 0:
            diagnostic = (
                completed.stderr.strip()
                or completed.stdout.strip()
                or "No diagnostic output was produced."
            )
            raise RuntimeError(
                f"The isolated {method} memory worker for "
                f"{case_name} failed.\n{diagnostic[-5000:]}"
            )

        if not result_path.exists():
            raise RuntimeError(
                f"The isolated {method} worker for {case_name} "
                "did not produce its JSON result."
            )

        result = json.loads(result_path.read_text(encoding="utf-8"))

    return float(result["peak_memory_increase_mb"])


# ============================================================
# Standard PINN baseline
# ============================================================
class PINN(nn.Module):
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

# ============================================================
# Shared-bias PINN
# ============================================================
class SharedBiasNet(nn.Module):
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

# ============================================================
# Case-dependent functions
# ============================================================
def make_case_functions(omega1, omega2):
    def G_fun(t):
        return torch.exp(-torch.sqrt(t + eps_reg))

    def Gprime_fun(t):
        s = torch.sqrt(t + eps_reg)
        return -0.5 * torch.exp(-s) / s

    def H_fun(t):
        return torch.exp(-t)

    def Hprime_fun(t):
        return -torch.exp(-t)

    def exact_solution(x, t):
        return G_fun(t) * torch.cos(omega1 * x) + 0.35 * H_fun(t) * torch.sin(omega2 * x)

    def forcing_term(x, t, q):
        G = G_fun(t)
        Gp = Gprime_fun(t)
        H = H_fun(t)
        Hp = Hprime_fun(t)

        u_t = Gp * torch.cos(omega1 * x) + 0.35 * Hp * torch.sin(omega2 * x)
        u_xx = -(omega1 ** 2) * G * torch.cos(omega1 * x) - 0.35 * (omega2 ** 2) * H * torch.sin(omega2 * x)

        return t**(1.0 - q) * u_t - nu * u_xx + lam * exact_solution(x, t)

    def u0(x):
        return exact_solution(x, torch.zeros_like(x))

    def periodic_features(x, t):
        return torch.cat(
            [
                torch.sin(omega1 * x),
                torch.cos(omega1 * x),
                torch.sin(omega2 * x),
                torch.cos(omega2 * x),
                t,
            ],
            dim=1,
        )

    return exact_solution, forcing_term, u0, periodic_features

# ============================================================
# PINN helper functions
# ============================================================
def input_features_std(x, t):
    return torch.cat([x, t], dim=1)

def trial_solution_std(model, x, t, u0):
    X = input_features_std(x, t)
    return u0(x) + t * model(X)

def rd_residual_time_std(model, x, t, q, forcing_term, u0):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    u = trial_solution_std(model, x, t, u0)
    u_t = grad(u, t)
    u_x = grad(u, x)
    u_xx = grad(u_x, x)

    Tq_u = t**(1.0 - q) * u_t
    f = forcing_term(x, t, q)

    return Tq_u - nu * u_xx + lam * u - f

def initial_loss_std(model, x0, u0):
    t0 = torch.zeros_like(x0)
    u_pred = trial_solution_std(model, x0, t0, u0)
    return torch.mean((u_pred - u0(x0))**2)

def periodic_bc_loss_std(model, t_b, u0):
    xL = torch.full_like(t_b, x_left, requires_grad=True)
    xR = torch.full_like(t_b, x_right, requires_grad=True)
    tL = t_b.clone().detach().requires_grad_(True)
    tR = t_b.clone().detach().requires_grad_(True)

    uL = trial_solution_std(model, xL, tL, u0)
    uR = trial_solution_std(model, xR, tR, u0)

    uL_x = grad(uL, xL)
    uR_x = grad(uR, xR)

    loss_u = torch.mean((uL - uR)**2)
    loss_ux = torch.mean((uL_x - uR_x)**2)
    return loss_u + loss_ux

def total_loss_std(model, x_f, t_f, x0, t_b, q, forcing_term, u0, w_pde=1.0, w_ic=1.0, w_bc=1.0):
    res = rd_residual_time_std(model, x_f, t_f, q, forcing_term, u0)
    loss_pde = torch.mean(res**2)
    loss_ic = initial_loss_std(model, x0, u0)
    loss_bc = periodic_bc_loss_std(model, t_b, u0)

    loss = w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc
    return loss, loss_pde, loss_ic, loss_bc

@torch.no_grad()
def evaluate_model_std(model, exact_solution, u0, nx=256, nt=201):
    xg = torch.linspace(x_left, x_right, nx, device=device)
    tg = torch.linspace(0.0, T_final, nt, device=device)

    X, T = torch.meshgrid(xg, tg, indexing="ij")
    x_flat = X.reshape(-1, 1)
    t_flat = T.reshape(-1, 1)

    U_pred = trial_solution_std(model, x_flat, t_flat, u0).reshape(nx, nt)
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
        e_2_rel.item(),
    )

def trial_solution_sb(model, x, t, u0, periodic_features):
    X = periodic_features(x, t)
    return u0(x) + t * model(X)

def rd_residual_time_sb(model, x, t, q, forcing_term, u0, periodic_features):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    u = trial_solution_sb(model, x, t, u0, periodic_features)
    u_t = grad(u, t)
    u_x = grad(u, x)
    u_xx = grad(u_x, x)

    Tq_u = t**(1.0 - q) * u_t
    f = forcing_term(x, t, q)

    return Tq_u - nu * u_xx + lam * u - f

def periodic_bc_loss_sb(model, t_b, u0, periodic_features):
    xL = torch.full_like(t_b, x_left, requires_grad=True)
    xR = torch.full_like(t_b, x_right, requires_grad=True)
    tL = t_b.clone().detach().requires_grad_(True)
    tR = t_b.clone().detach().requires_grad_(True)

    uL = trial_solution_sb(model, xL, tL, u0, periodic_features)
    uR = trial_solution_sb(model, xR, tR, u0, periodic_features)

    uL_x = grad(uL, xL)
    uR_x = grad(uR, xR)

    loss_u = torch.mean((uL - uR) ** 2)
    loss_ux = torch.mean((uL_x - uR_x) ** 2)
    return loss_u + loss_ux

def total_loss_sb(model, x_f, t_f, t_b, q, forcing_term, u0, periodic_features, w_pde=1.0, w_bc=1.0):
    res = rd_residual_time_sb(model, x_f, t_f, q, forcing_term, u0, periodic_features)
    loss_pde = torch.mean(res**2)
    loss_bc = periodic_bc_loss_sb(model, t_b, u0, periodic_features)
    loss = w_pde * loss_pde + w_bc * loss_bc
    return loss, loss_pde, loss_bc

def lr_alpha(i):
    return 1.5 - 2.0 * (math.atan(i) + math.pi / 2.0) / math.pi / 2.0

def LR(i, LR0=3e-2):
    a = max(lr_alpha(i), 1.0e-8)
    return LR0 * a

@torch.no_grad()
def conformable_fractional_variable_gradient_pinn(model, i, LR0=7e-3):
    alpha_opt = max(lr_alpha(i), 1.0e-14)
    Lr = LR(i, LR0=LR0)

    for p in model.parameters():
        if p.grad is None:
            continue
        p_norm = torch.norm(p).clamp_min(1.0e-14)
        scaled_grad = (p_norm ** alpha_opt) * p.grad / alpha_opt
        p -= Lr * scaled_grad

def sample_interior_points(num_f, eps_t):
    t_f = eps_t + (T_final - eps_t) * torch.rand(num_f, 1, device=device)
    x_f = x_left + (x_right - x_left) * torch.rand(num_f, 1, device=device)
    return x_f, t_f

def sample_boundary_points(num_tb, eps_t):
    return torch.linspace(eps_t, T_final, num_tb, device=device).reshape(-1, 1)

@torch.no_grad()
def evaluate_model_sb(model, exact_solution, u0, periodic_features, nx=256, nt=201):
    xg = torch.linspace(x_left, x_right, nx, device=device)
    tg = torch.linspace(0.0, T_final, nt, device=device)

    X, T = torch.meshgrid(xg, tg, indexing="ij")
    x_flat = X.reshape(-1, 1)
    t_flat = T.reshape(-1, 1)

    U_pred = trial_solution_sb(model, x_flat, t_flat, u0, periodic_features).reshape(nx, nt)
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
        e_2_rel.item(),
    )

# ============================================================
# Tables
# ============================================================
def print_global_error_table(rows):
    print("\nOverall global error table for Problem 3:\n")
    header = (
        f"{'case':>8} {'(omega1,omega2)':>18} {'method':>18} "
        f"{'E_inf':>16} {'E_2':>16} {'E_2_rel':>16} "
        f"{'final_loss':>16} {'time_sec':>14}"
    )
    print(header)
    print("-" * len(header))

    for (
        case_name,
        omega1,
        omega2,
        method,
        e_inf,
        e_2,
        e_2_rel,
        final_loss,
        elapsed,
    ) in rows:
        pair = f"({int(omega1)},{int(omega2)})"
        print(
            f"{case_name:>8} {pair:>18} {method:>18} "
            f"{e_inf:16.8e} {e_2:16.8e} {e_2_rel:16.8e} "
            f"{final_loss:16.8e} {elapsed:14.6f}"
        )


def latex_global_error_table(rows):
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(
        "\\caption{Global errors, final training loss, and CPU time "
        "for the three manufactured time-conformable reaction--diffusion cases.}"
    )
    lines.append("\\label{tab:reaction_diffusion_global_comparison}")
    lines.append("\\scriptsize")
    lines.append("\\setlength{\\tabcolsep}{4pt}")
    lines.append("\\begin{tabular}{cclccccc}")
    lines.append("\\hline")
    lines.append(
        "Case & $(\\omega_1,\\omega_2)$ & Method & $E_{\\infty}$ "
        "& $E_2$ & $E_{2,\\mathrm{rel}}$ & Final loss & CPU time (s) \\\\"
    )
    lines.append("\\hline")

    for (
        case_name,
        omega1,
        omega2,
        method,
        e_inf,
        e_2,
        e_2_rel,
        final_loss,
        elapsed,
    ) in rows:
        lines.append(
            f"{case_name} & $({int(omega1)},{int(omega2)})$ & {method} "
            f"& {e_inf:.6e} & {e_2:.6e} & {e_2_rel:.6e} "
            f"& {final_loss:.6e} & {elapsed:.3f} \\\\"
        )

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def print_efficiency_table(rows):
    print(
        "\nComputational-efficiency table for the "
        "representative Problem 3 case:\n"
    )
    header = (
        f"{'Method':<38}"
        f"{'Parameters':>13}"
        f"{'E_inf':>15}"
        f"{'E_2':>15}"
        f"{'E_2_rel':>15}"
        f"{'Final loss':>15}"
        f"{'CPU time (s)':>15}"
        f"{'Peak memory increase (MB)':>27}"
    )
    print(header)
    print("-" * len(header))

    for (
        method,
        parameters,
        e_inf,
        e_2,
        e_2_rel,
        final_loss,
        elapsed,
        peak_memory,
    ) in rows:
        print(
            f"{method:<38}"
            f"{parameters:13,d}"
            f"{e_inf:15.6e}"
            f"{e_2:15.6e}"
            f"{e_2_rel:15.6e}"
            f"{final_loss:15.6e}"
            f"{elapsed:15.3f}"
            f"{peak_memory:27.3f}"
        )


def latex_efficiency_table(rows):
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Accuracy and computational-efficiency comparison "
        r"for the representative manufactured time-conformable "
        r"reaction--diffusion problem with "
        r"$(\omega_1,\omega_2)=(2,5)$. The peak memory increase of "
        r"each method is measured in a separate fresh process from a "
        r"common baseline before model, data, and optimizer construction.}"
    )
    lines.append(r"\label{tab:reaction_diffusion_efficiency}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"\begin{tabular}{lccccccc}")
    lines.append(r"\hline")
    lines.append(
        r"Method & Parameters & $E_{\infty}$ & $E_2$ & "
        r"$E_{2,\mathrm{rel}}$ & Final loss & CPU time (s) "
        r"& Peak memory increase (MB) \\"
    )
    lines.append(r"\hline")

    for (
        method,
        parameters,
        e_inf,
        e_2,
        e_2_rel,
        final_loss,
        elapsed,
        peak_memory,
    ) in rows:
        lines.append(
            f"{method} & {parameters:d} & {e_inf:.6e} & "
            f"{e_2:.6e} & {e_2_rel:.6e} & "
            f"{final_loss:.6e} & {elapsed:.3f} & "
            f"{peak_memory:.3f} \\\\"
        )

    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")
    return "\n".join(lines)



# ============================================================
# Five-seed study for the representative case
# ============================================================
CASE_NAME = "case1"
OMEGA1 = 2.0
OMEGA2 = 5.0
SEEDS = [0, 1, 2, 3, 4]
PRINT_EVERY = 2000
EVALUATION_NX = 256
EVALUATION_NT = 201

exact_solution, forcing_term, u0, periodic_features = make_case_functions(
    OMEGA1,
    OMEGA2,
)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def recompute_final_loss_std(model, x_f, t_f, x0, t_b):
    loss, _, _, _ = total_loss_std(
        model,
        x_f,
        t_f,
        x0,
        t_b,
        q_frac,
        forcing_term,
        u0,
    )
    return loss.detach().item()


def recompute_final_loss_sb(model, x_f, t_f, t_b):
    loss, _, _ = total_loss_sb(
        model,
        x_f,
        t_f,
        t_b,
        q_frac,
        forcing_term,
        u0,
        periodic_features,
        w_pde=1.0,
        w_bc=1.0,
    )
    return loss.detach().item()


def run_one_seed(seed):
    print("\n" + "=" * 84)
    print(
        "Problem 3 multi-run study: "
        f"(omega1, omega2) = ({int(OMEGA1)}, {int(OMEGA2)}), "
        f"seed = {seed}"
    )
    print("=" * 84)

    # Standard PINN + Adam
    set_seed(seed)
    std_model = PINN(
        input_dim=2,
        hidden_dim=hid_dim,
        output_dim=1,
    ).to(device)

    t_f_std = eps_t + (T_final - eps_t) * torch.rand(
        num_f, 1, device=device
    )
    x_f_std = x_left + (x_right - x_left) * torch.rand(
        num_f, 1, device=device
    )
    x0_std = torch.linspace(
        x_left, x_right, num_x0, device=device
    ).reshape(-1, 1)
    t_b_std = torch.linspace(
        eps_t, T_final, num_tb, device=device
    ).reshape(-1, 1)

    optimizer_std = torch.optim.Adam(
        std_model.parameters(),
        lr=adam_lr,
    )
    loss_history_std = []
    start_std = time.perf_counter()

    for iteration in range(adam_iters + 1):
        optimizer_std.zero_grad(set_to_none=True)
        loss_std, loss_pde_std, loss_ic_std, loss_bc_std = total_loss_std(
            std_model,
            x_f_std,
            t_f_std,
            x0_std,
            t_b_std,
            q_frac,
            forcing_term,
            u0,
        )
        if not torch.isfinite(loss_std):
            raise RuntimeError(
                "Standard PINN stopped at iteration "
                f"{iteration} for seed {seed}: invalid loss."
            )
        loss_std.backward()
        torch.nn.utils.clip_grad_norm_(
            std_model.parameters(),
            max_norm=clip_value,
        )
        optimizer_std.step()
        loss_history_std.append(loss_std.item())

        if iteration % PRINT_EVERY == 0:
            print(
                f"[Seed {seed} | Standard] Iteration {iteration:5d} | "
                f"Loss {loss_std.item():.6e} | "
                f"PDE {loss_pde_std.item():.6e} | "
                f"IC {loss_ic_std.item():.6e} | "
                f"BC {loss_bc_std.item():.6e}"
            )

    elapsed_std = time.perf_counter() - start_std
    final_loss_std = recompute_final_loss_std(
        std_model,
        x_f_std,
        t_f_std,
        x0_std,
        t_b_std,
    )
    _, _, _, _, e_inf_std, e_2_std, e_2_rel_std = evaluate_model_std(
        std_model,
        exact_solution,
        u0,
        nx=EVALUATION_NX,
        nt=EVALUATION_NT,
    )

    # Shared-bias PINN + conformable GD
    set_seed(seed)
    sb_model = SharedBiasNet(
        input_dim=5,
        hid_dim=hid_dim,
        output_dim=1,
    ).to(device)
    x_f_sb, t_f_sb = sample_interior_points(
        num_f=num_f,
        eps_t=eps_t,
    )
    t_b_sb = sample_boundary_points(
        num_tb=num_tb,
        eps_t=eps_t,
    )

    loss_history_sb = []
    start_sb = time.perf_counter()

    for iteration in range(num_iters + 1):
        if iteration % 200 == 0:
            x_f_sb, t_f_sb = sample_interior_points(
                num_f=num_f,
                eps_t=eps_t,
            )
        sb_model.zero_grad(set_to_none=True)
        loss_sb, loss_pde_sb, loss_bc_sb = total_loss_sb(
            sb_model,
            x_f_sb,
            t_f_sb,
            t_b_sb,
            q_frac,
            forcing_term,
            u0,
            periodic_features,
            w_pde=1.0,
            w_bc=1.0,
        )
        if not torch.isfinite(loss_sb):
            raise RuntimeError(
                "Shared-bias PINN stopped at iteration "
                f"{iteration} for seed {seed}: invalid loss."
            )
        loss_sb.backward()
        torch.nn.utils.clip_grad_norm_(
            sb_model.parameters(),
            max_norm=clip_value,
        )
        conformable_fractional_variable_gradient_pinn(
            sb_model,
            iteration,
            LR0=LR0,
        )
        loss_history_sb.append(loss_sb.item())

        if iteration % PRINT_EVERY == 0:
            print(
                f"[Seed {seed} | Shared-bias] Iteration {iteration:5d} | "
                f"Loss {loss_sb.item():.6e} | "
                f"PDE {loss_pde_sb.item():.6e} | "
                f"BC {loss_bc_sb.item():.6e} | "
                f"alpha {lr_alpha(iteration):.6e} | "
                f"LR {LR(iteration, LR0):.6e}"
            )

    elapsed_sb = time.perf_counter() - start_sb
    final_loss_sb = recompute_final_loss_sb(
        sb_model,
        x_f_sb,
        t_f_sb,
        t_b_sb,
    )
    _, _, _, _, e_inf_sb, e_2_sb, e_2_rel_sb = evaluate_model_sb(
        sb_model,
        exact_solution,
        u0,
        periodic_features,
        nx=EVALUATION_NX,
        nt=EVALUATION_NT,
    )

    row_std = {
        "seed": seed,
        "method": "Standard PINN + Adam",
        "E_inf": e_inf_std,
        "E_2": e_2_std,
        "E_2_rel": e_2_rel_std,
        "final_loss": final_loss_std,
        "cpu_time": elapsed_std,
    }
    row_sb = {
        "seed": seed,
        "method": "Shared-bias PINN + conformable GD",
        "E_inf": e_inf_sb,
        "E_2": e_2_sb,
        "E_2_rel": e_2_rel_sb,
        "final_loss": final_loss_sb,
        "cpu_time": elapsed_sb,
    }

    print(
        f"\nSeed {seed} completed:\n"
        f"  Standard E_2     = {e_2_std:.6e}\n"
        f"  Shared-bias E_2  = {e_2_sb:.6e}"
    )
    return row_std, row_sb, loss_history_std, loss_history_sb


individual_rows = []
all_loss_std = []
all_loss_sb = []

for seed in SEEDS:
    row_std, row_sb, loss_history_std, loss_history_sb = run_one_seed(seed)
    individual_rows.extend([row_std, row_sb])
    all_loss_std.append(loss_history_std)
    all_loss_sb.append(loss_history_sb)

metric_names = [
    "E_inf",
    "E_2",
    "E_2_rel",
    "final_loss",
    "cpu_time",
]
method_names = [
    "Standard PINN + Adam",
    "Shared-bias PINN + conformable GD",
]
summary_rows = []

for method in method_names:
    method_rows = [row for row in individual_rows if row["method"] == method]
    summary = {"method": method}
    for metric in metric_names:
        values = np.asarray([row[metric] for row in method_rows], dtype=float)
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values, ddof=1))
        summary[f"{metric}_var"] = float(np.var(values, ddof=1))
    summary_rows.append(summary)

individual_csv = output_dir / "problem3_case1_multirun_individual.csv"
summary_csv = output_dir / "problem3_case1_multirun_summary.csv"

with individual_csv.open("w", newline="", encoding="utf-8") as file:
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

summary_fieldnames = ["method"]
for metric in metric_names:
    summary_fieldnames.extend(
        [f"{metric}_mean", f"{metric}_std", f"{metric}_var"]
    )

with summary_csv.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=summary_fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)

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


def latex_mean_std(mean, std, digits=3):
    return f"${mean:.{digits}e}" rf"\pm{std:.{digits}e}$"


print("\nLaTeX five-seed table:\n")
print(r"\begin{table}[htbp]")
print(r"\centering")
print(
    r"\caption{Five-run results for the representative manufactured "
    r"time-conformable reaction--diffusion problem with "
    r"$(\omega_1,\omega_2)=(2,5)$. Each entry is the mean "
    r"$\pm$ sample standard deviation over the seeds "
    rf"$\{{{', '.join(str(seed) for seed in SEEDS)}\}}$."
    r"}"
)
print(r"\label{tab:reaction_diffusion_case1_multirun}")
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
        f"{summary['cpu_time_mean']:.3f}"
        rf"$\pm${summary['cpu_time_std']:.3f} \\"
    )
print(r"\hline")
print(r"\end{tabular}")
print(r"}")
print(r"\end{table}")

loss_array_std = np.asarray(all_loss_std, dtype=float)
loss_array_sb = np.asarray(all_loss_sb, dtype=float)
loss_mean_std = np.mean(loss_array_std, axis=0)
loss_sd_std = np.std(loss_array_std, axis=0, ddof=1)
loss_mean_sb = np.mean(loss_array_sb, axis=0)
loss_sd_sb = np.std(loss_array_sb, axis=0, ddof=1)
iterations = np.arange(loss_mean_std.size)
loss_floor = 1.0e-8
lower_std = np.maximum(loss_mean_std - loss_sd_std, loss_floor)
upper_std = loss_mean_std + loss_sd_std
lower_sb = np.maximum(loss_mean_sb - loss_sd_sb, loss_floor)
upper_sb = loss_mean_sb + loss_sd_sb

fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=240)
ax.semilogy(
    iterations,
    loss_mean_std,
    color=COLOR_STD,
    label="Standard PINN + Adam",
)
ax.fill_between(
    iterations,
    lower_std,
    upper_std,
    color=COLOR_STD,
    alpha=0.18,
    rasterized=True,
)
ax.semilogy(
    iterations,
    loss_mean_sb,
    color=COLOR_SB,
    linestyle="--",
    label="Shared-bias PINN + conformable GD",
)
ax.fill_between(
    iterations,
    lower_sb,
    upper_sb,
    color=COLOR_SB,
    alpha=0.18,
    rasterized=True,
)
ax.set_xlabel("Iteration")
ax.set_ylabel("Training loss")
ax.set_title(r"Five-run loss histories, $(\omega_1,\omega_2)=(2,5)$")
ax.set_xlim(left=0)
ax.set_ylim(1.0e-6, 1.0e3)
ax.margins(x=0)
ax.legend()
fig.tight_layout()

loss_figure = output_dir / "reaction_diffusion_case1_multirun_loss.pdf"
fig.savefig(
    loss_figure,
    format="pdf",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.03,
)
plt.close(fig)

print(f"\nIndividual-run CSV:\n{individual_csv}")
print(f"\nSummary CSV:\n{summary_csv}")
print(f"\nMulti-run loss figure:\n{loss_figure}")
