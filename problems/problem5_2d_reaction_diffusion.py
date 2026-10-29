"""
Two-dimensional reaction--diffusion benchmark used in Section 4.6.

The experiment compares a standard PINN and a shared-bias PINN at nearly
matched parameter counts.  Both methods use the same interior collocation
blocks and the same resampling schedule.  The script reports the errors,
independent residual, convergence diagnostic, parameter counts, and CPU time
used in the manuscript, and saves the corresponding figures as PDF files.
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

# Output folder
script_dir = Path(__file__).resolve().parent
output_dir = script_dir / "Figs" / "Ex5_2D_Reaction_Diffusion"
output_dir.mkdir(parents=True, exist_ok=True)

# Plotting defaults
plt.rcParams["figure.dpi"] = 220
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

# Settings
device = "cpu"
dtype = torch.float64
torch.set_default_dtype(dtype)

torch.manual_seed(0)
np.random.seed(0)

# Problem parameters
q_frac = 0.7
nu = 0.01 / math.pi
lam = 1.0

omega1 = 2.0
omega2 = 5.0

x_left = 0.0
x_right = 2.0 * math.pi
y_left = 0.0
y_right = 2.0 * math.pi
T_final = 1.0
eps_reg = 1.0e-3

# Training and evaluation settings
clip_value = 10.0
hid_dim_std = 100
hid_dim_sb = 50
num_f = 1000
num_x0 = 400
num_tb = 400
num_iters = 20000
LR0 = 1.0e-2

adam_iters = num_iters
adam_lr = 1.0e-3

eps_t = 1.0e-4
resample_every = 200
loss_tolerance = 1.0e-3
model_seed = 0
sampling_seed = 0

# Utilities
def grad(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
    )[0]


def smooth_curve(values, beta=0.98):
    out = []
    avg = 0.0
    for k, v in enumerate(values, 1):
        avg = beta * avg + (1.0 - beta) * v
        out.append(avg / (1.0 - beta**k))
    return out


def save_pdf(fig, stem):
    """Save a PDF without terminating the run if Windows locks a file."""
    target_path = output_dir / f"{stem}.pdf"

    try:
        fig.savefig(target_path, format="pdf", bbox_inches="tight")
        return target_path
    except PermissionError:
        counter = 1
        while True:
            fallback_path = output_dir / f"{stem}_new{counter}.pdf"
            if not fallback_path.exists():
                fig.savefig(fallback_path, format="pdf", bbox_inches="tight")
                print(
                    f"\nWarning: the target PDF is locked:\n{target_path}\n"
                    f"The figure was saved instead as:\n{fallback_path}\n"
                )
                return fallback_path
            counter += 1


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def first_iteration_below(loss_history, tolerance=1.0e-3):
    for iteration, value in enumerate(loss_history):
        if value <= tolerance:
            return iteration
    return None

# ============================================================
# Standard PINN baseline
# ============================================================
class PINN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=100, output_dim=1):
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
# Width 50 gives 502 trainable parameters, nearly matching the
# 501-parameter standard PINN despite the richer periodic input map.
# ============================================================
class SharedBiasNet(nn.Module):
    def __init__(self, input_dim=9, hid_dim=50, output_dim=1):
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
        return H @ self.W2 + self.b2

# Manufactured 2D reaction--diffusion problem
def G_fun(t):
    return torch.exp(-torch.sqrt(t + eps_reg))


def Gprime_fun(t):
    s = torch.sqrt(t + eps_reg)
    return -0.5 * torch.exp(-s) / s


def H_fun(t):
    return torch.exp(-t)


def Hprime_fun(t):
    return -torch.exp(-t)


def exact_solution(x, y, t):
    return (
        G_fun(t) * torch.cos(omega1 * x) * torch.cos(omega1 * y)
        + 0.35
        * H_fun(t)
        * torch.sin(omega2 * x)
        * torch.sin(omega2 * y)
    )


def forcing_term(x, y, t, q):
    G = G_fun(t)
    Gp = Gprime_fun(t)
    H = H_fun(t)
    Hp = Hprime_fun(t)

    c1x = torch.cos(omega1 * x)
    c1y = torch.cos(omega1 * y)
    s2x = torch.sin(omega2 * x)
    s2y = torch.sin(omega2 * y)

    u = G * c1x * c1y + 0.35 * H * s2x * s2y
    u_t = Gp * c1x * c1y + 0.35 * Hp * s2x * s2y

    # Laplacian of the exact solution:
    # Delta u = -2*omega1^2*G*cos(omega1 x)*cos(omega1 y)
    #           -0.70*omega2^2*H*sin(omega2 x)*sin(omega2 y).
    lap_u = (
        -2.0 * omega1**2 * G * c1x * c1y
        - 0.70 * omega2**2 * H * s2x * s2y
    )

    return t ** (1.0 - q) * u_t - nu * lap_u + lam * u


def u0(x, y):
    return exact_solution(x, y, torch.zeros_like(x))


def periodic_features(x, y, t):
    return torch.cat(
        [
            torch.sin(omega1 * x),
            torch.cos(omega1 * x),
            torch.sin(omega2 * x),
            torch.cos(omega2 * x),
            torch.sin(omega1 * y),
            torch.cos(omega1 * y),
            torch.sin(omega2 * y),
            torch.cos(omega2 * y),
            t,
        ],
        dim=1,
    )

# Sampling geometry
def sample_interior_block(block_index, num_points=num_f, eps_time=eps_t):
    """Return one deterministic random collocation block.

    The block is generated from a dedicated seed that depends only on
    block_index. Therefore the standard and shared-bias PINNs receive
    exactly the same interior points at the same training iterations,
    independent of model initialization or optimizer state.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(sampling_seed + int(block_index))

    # Preserve the validated Problem-3 draw order: time, then x, then y.
    t_f = eps_time + (T_final - eps_time) * torch.rand(
        num_points, 1, generator=generator, device=device
    )
    x_f = x_left + (x_right - x_left) * torch.rand(
        num_points, 1, generator=generator, device=device
    )
    y_f = y_left + (y_right - y_left) * torch.rand(
        num_points, 1, generator=generator, device=device
    )
    return x_f, y_f, t_f


def deterministic_square_grid(num_points, a, b):
    n_side = int(round(math.sqrt(num_points)))
    if n_side * n_side != num_points:
        raise ValueError(
            "For the locked 2D protocol, num_points must be a perfect square."
        )

    values = torch.linspace(a, b, n_side, device=device)
    A, B = torch.meshgrid(values, values, indexing="ij")
    return A.reshape(-1, 1), B.reshape(-1, 1)


def initial_grid(num_points):
    return deterministic_square_grid(num_points, x_left, x_right)


def boundary_parameter_grid(num_points, eps_time):
    n_side = int(round(math.sqrt(num_points)))
    if n_side * n_side != num_points:
        raise ValueError(
            "For the locked 2D protocol, num_tb must be a perfect square."
        )

    s = torch.linspace(x_left, x_right, n_side, device=device)
    t = torch.linspace(eps_time, T_final, n_side, device=device)
    S, T = torch.meshgrid(s, t, indexing="ij")
    return S.reshape(-1, 1), T.reshape(-1, 1)

# Standard PINN helper functions
def input_features_std(x, y, t):
    return torch.cat([x, y, t], dim=1)


def trial_solution_std(model, x, y, t):
    X = input_features_std(x, y, t)
    return u0(x, y) + t * model(X)


def rd_residual_std(model, x, y, t, q):
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    u = trial_solution_std(model, x, y, t)
    u_t = grad(u, t)
    u_x = grad(u, x)
    u_xx = grad(u_x, x)
    u_y = grad(u, y)
    u_yy = grad(u_y, y)

    Tq_u = t ** (1.0 - q) * u_t
    f = forcing_term(x, y, t, q)

    return Tq_u - nu * (u_xx + u_yy) + lam * u - f


def initial_loss_std(model, x0, y0):
    t0 = torch.zeros_like(x0)
    u_pred = trial_solution_std(model, x0, y0, t0)
    return torch.mean((u_pred - u0(x0, y0)) ** 2)


def periodic_bc_loss_std(model, s_b, t_b):
    # x-periodic pair: y = s_b
    xL = torch.full_like(s_b, x_left, requires_grad=True)
    xR = torch.full_like(s_b, x_right, requires_grad=True)
    yL = s_b.clone().detach()
    yR = s_b.clone().detach()
    tL = t_b.clone().detach()
    tR = t_b.clone().detach()

    u_xL = trial_solution_std(model, xL, yL, tL)
    u_xR = trial_solution_std(model, xR, yR, tR)
    du_xL = grad(u_xL, xL)
    du_xR = grad(u_xR, xR)

    loss_x = torch.mean((u_xL - u_xR) ** 2) + torch.mean(
        (du_xL - du_xR) ** 2
    )

    # y-periodic pair: x = s_b
    yB = torch.full_like(s_b, y_left, requires_grad=True)
    yT = torch.full_like(s_b, y_right, requires_grad=True)
    xB = s_b.clone().detach()
    xT = s_b.clone().detach()
    tB = t_b.clone().detach()
    tT = t_b.clone().detach()

    u_yB = trial_solution_std(model, xB, yB, tB)
    u_yT = trial_solution_std(model, xT, yT, tT)
    du_yB = grad(u_yB, yB)
    du_yT = grad(u_yT, yT)

    loss_y = torch.mean((u_yB - u_yT) ** 2) + torch.mean(
        (du_yB - du_yT) ** 2
    )

    return loss_x + loss_y


def total_loss_std(
    model,
    x_f,
    y_f,
    t_f,
    x0,
    y0,
    s_b,
    t_b,
    q,
    w_pde=1.0,
    w_ic=1.0,
    w_bc=1.0,
):
    residual = rd_residual_std(model, x_f, y_f, t_f, q)
    loss_pde = torch.mean(residual**2)
    loss_ic = initial_loss_std(model, x0, y0)
    loss_bc = periodic_bc_loss_std(model, s_b, t_b)

    loss = w_pde * loss_pde + w_ic * loss_ic + w_bc * loss_bc
    return loss, loss_pde, loss_ic, loss_bc

# Shared-bias PINN helper functions
def trial_solution_sb(model, x, y, t):
    X = periodic_features(x, y, t)
    return u0(x, y) + t * model(X)


def rd_residual_sb(model, x, y, t, q):
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    u = trial_solution_sb(model, x, y, t)
    u_t = grad(u, t)
    u_x = grad(u, x)
    u_xx = grad(u_x, x)
    u_y = grad(u, y)
    u_yy = grad(u_y, y)

    Tq_u = t ** (1.0 - q) * u_t
    f = forcing_term(x, y, t, q)

    return Tq_u - nu * (u_xx + u_yy) + lam * u - f


def periodic_bc_loss_sb(model, s_b, t_b):
    # x-periodic pair: y = s_b
    xL = torch.full_like(s_b, x_left, requires_grad=True)
    xR = torch.full_like(s_b, x_right, requires_grad=True)
    yL = s_b.clone().detach()
    yR = s_b.clone().detach()
    tL = t_b.clone().detach()
    tR = t_b.clone().detach()

    u_xL = trial_solution_sb(model, xL, yL, tL)
    u_xR = trial_solution_sb(model, xR, yR, tR)
    du_xL = grad(u_xL, xL)
    du_xR = grad(u_xR, xR)

    loss_x = torch.mean((u_xL - u_xR) ** 2) + torch.mean(
        (du_xL - du_xR) ** 2
    )

    # y-periodic pair: x = s_b
    yB = torch.full_like(s_b, y_left, requires_grad=True)
    yT = torch.full_like(s_b, y_right, requires_grad=True)
    xB = s_b.clone().detach()
    xT = s_b.clone().detach()
    tB = t_b.clone().detach()
    tT = t_b.clone().detach()

    u_yB = trial_solution_sb(model, xB, yB, tB)
    u_yT = trial_solution_sb(model, xT, yT, tT)
    du_yB = grad(u_yB, yB)
    du_yT = grad(u_yT, yT)

    loss_y = torch.mean((u_yB - u_yT) ** 2) + torch.mean(
        (du_yB - du_yT) ** 2
    )

    return loss_x + loss_y


def total_loss_sb(
    model,
    x_f,
    y_f,
    t_f,
    s_b,
    t_b,
    q,
    w_pde=1.0,
    w_bc=1.0,
):
    residual = rd_residual_sb(model, x_f, y_f, t_f, q)
    loss_pde = torch.mean(residual**2)
    loss_bc = periodic_bc_loss_sb(model, s_b, t_b)
    loss = w_pde * loss_pde + w_bc * loss_bc
    return loss, loss_pde, loss_bc

# Conformable-gradient optimizer
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
        scaled_grad = (p_norm**alpha_opt) * p.grad / alpha_opt
        p -= Lr * scaled_grad

# Manufactured-problem check
def verify_manufactured_problem():
    x = torch.tensor(
        [0.17, 0.74, 1.31, 2.05, 3.14, 4.02, 5.11, 5.86],
        dtype=dtype,
        device=device,
    ).reshape(-1, 1).requires_grad_(True)

    y = torch.tensor(
        [0.29, 1.08, 1.77, 2.46, 3.35, 4.14, 5.03, 5.72],
        dtype=dtype,
        device=device,
    ).reshape(-1, 1).requires_grad_(True)

    t = torch.tensor(
        [0.04, 0.13, 0.24, 0.37, 0.52, 0.68, 0.83, 0.96],
        dtype=dtype,
        device=device,
    ).reshape(-1, 1).requires_grad_(True)

    u = exact_solution(x, y, t)
    u_t = grad(u, t)
    u_x = grad(u, x)
    u_xx = grad(u_x, x)
    u_y = grad(u, y)
    u_yy = grad(u_y, y)

    residual = (
        t ** (1.0 - q_frac) * u_t
        - nu * (u_xx + u_yy)
        + lam * u
        - forcing_term(x, y, t, q_frac)
    )

    x0_v, y0_v = deterministic_square_grid(121, x_left, x_right)
    t0_v = torch.zeros_like(x0_v)
    ic_error = torch.max(
        torch.abs(exact_solution(x0_v, y0_v, t0_v) - u0(x0_v, y0_v))
    )

    s_v, t_v = boundary_parameter_grid(100, eps_t)

    # x-periodicity of exact solution and x derivative
    xL = torch.full_like(s_v, x_left, requires_grad=True)
    xR = torch.full_like(s_v, x_right, requires_grad=True)
    yL = s_v.clone().detach()
    yR = s_v.clone().detach()
    txL = t_v.clone().detach()
    txR = t_v.clone().detach()

    uxL = exact_solution(xL, yL, txL)
    uxR = exact_solution(xR, yR, txR)
    duxL = grad(uxL, xL)
    duxR = grad(uxR, xR)

    # y-periodicity of exact solution and y derivative
    yB = torch.full_like(s_v, y_left, requires_grad=True)
    yT = torch.full_like(s_v, y_right, requires_grad=True)
    xB = s_v.clone().detach()
    xT = s_v.clone().detach()
    tyB = t_v.clone().detach()
    tyT = t_v.clone().detach()

    uyB = exact_solution(xB, yB, tyB)
    uyT = exact_solution(xT, yT, tyT)
    duyB = grad(uyB, yB)
    duyT = grad(uyT, yT)

    print("\nManufactured-problem verification")
    print(f"  max PDE residual      = {torch.max(torch.abs(residual)).item():.3e}")
    print(f"  max IC error          = {ic_error.item():.3e}")
    print(f"  periodic x: u error   = {torch.max(torch.abs(uxL - uxR)).item():.3e}")
    print(f"  periodic x: ux error  = {torch.max(torch.abs(duxL - duxR)).item():.3e}")
    print(f"  periodic y: u error   = {torch.max(torch.abs(uyB - uyT)).item():.3e}")
    print(f"  periodic y: uy error  = {torch.max(torch.abs(duyB - duyT)).item():.3e}")

# Global error evaluation on an independent deterministic 3D grid
@torch.no_grad()
def evaluate_global_errors(model, method, nx=61, ny=61, nt=81, batch_size=8192):
    xg = torch.linspace(x_left, x_right, nx, device=device)
    yg = torch.linspace(y_left, y_right, ny, device=device)
    tg = torch.linspace(0.0, T_final, nt, device=device)

    X, Y, T = torch.meshgrid(xg, yg, tg, indexing="ij")
    xf = X.reshape(-1, 1)
    yf = Y.reshape(-1, 1)
    tf = T.reshape(-1, 1)

    max_error = 0.0
    sum_sq_error = 0.0
    sum_sq_exact = 0.0
    count = 0

    for start in range(0, xf.shape[0], batch_size):
        stop = min(start + batch_size, xf.shape[0])
        xb = xf[start:stop]
        yb = yf[start:stop]
        tb = tf[start:stop]

        if method == "standard":
            pred = trial_solution_std(model, xb, yb, tb)
        elif method == "shared":
            pred = trial_solution_sb(model, xb, yb, tb)
        else:
            raise ValueError("method must be 'standard' or 'shared'.")

        exact = exact_solution(xb, yb, tb)
        error = pred - exact

        max_error = max(max_error, torch.max(torch.abs(error)).item())
        sum_sq_error += torch.sum(error**2).item()
        sum_sq_exact += torch.sum(exact**2).item()
        count += error.numel()

    e_2 = math.sqrt(sum_sq_error / count)
    e_2_rel = math.sqrt(sum_sq_error / sum_sq_exact)

    return max_error, e_2, e_2_rel

# Independent residual RMS
def independent_residual_rms(
    model,
    method,
    nx=25,
    ny=25,
    nt=31,
    batch_size=2048,
):
    xg = torch.linspace(x_left, x_right, nx, device=device)
    yg = torch.linspace(y_left, y_right, ny, device=device)
    tg = torch.linspace(eps_t, T_final, nt, device=device)

    X, Y, T = torch.meshgrid(xg, yg, tg, indexing="ij")
    xf = X.reshape(-1, 1)
    yf = Y.reshape(-1, 1)
    tf = T.reshape(-1, 1)

    sum_sq = 0.0
    count = 0

    for start in range(0, xf.shape[0], batch_size):
        stop = min(start + batch_size, xf.shape[0])
        xb = xf[start:stop]
        yb = yf[start:stop]
        tb = tf[start:stop]

        if method == "standard":
            residual = rd_residual_std(model, xb, yb, tb, q_frac)
        elif method == "shared":
            residual = rd_residual_sb(model, xb, yb, tb, q_frac)
        else:
            raise ValueError("method must be 'standard' or 'shared'.")

        residual_detached = residual.detach()
        sum_sq += torch.sum(residual_detached**2).item()
        count += residual_detached.numel()

    return math.sqrt(sum_sq / count)

# Plotting evaluations
@torch.no_grad()
def evaluate_final_time_field(model, method, n=141):
    xg = torch.linspace(x_left, x_right, n, device=device)
    yg = torch.linspace(y_left, y_right, n, device=device)
    X, Y = torch.meshgrid(xg, yg, indexing="ij")

    xf = X.reshape(-1, 1)
    yf = Y.reshape(-1, 1)
    tf = torch.full_like(xf, T_final)

    exact = exact_solution(xf, yf, tf).reshape(n, n)

    if method == "standard":
        pred = trial_solution_std(model, xf, yf, tf).reshape(n, n)
    elif method == "shared":
        pred = trial_solution_sb(model, xf, yf, tf).reshape(n, n)
    else:
        raise ValueError("method must be 'standard' or 'shared'.")

    return (
        xg.cpu().numpy(),
        yg.cpu().numpy(),
        exact.cpu().numpy(),
        pred.cpu().numpy(),
    )


@torch.no_grad()
def evaluate_xt_slice(model, method, y_value, nx=181, nt=141):
    xg = torch.linspace(x_left, x_right, nx, device=device)
    tg = torch.linspace(0.0, T_final, nt, device=device)
    X, T = torch.meshgrid(xg, tg, indexing="ij")

    xf = X.reshape(-1, 1)
    tf = T.reshape(-1, 1)
    yf = torch.full_like(xf, y_value)

    exact = exact_solution(xf, yf, tf).reshape(nx, nt)

    if method == "standard":
        pred = trial_solution_std(model, xf, yf, tf).reshape(nx, nt)
    elif method == "shared":
        pred = trial_solution_sb(model, xf, yf, tf).reshape(nx, nt)
    else:
        raise ValueError("method must be 'standard' or 'shared'.")

    return (
        xg.cpu().numpy(),
        tg.cpu().numpy(),
        exact.cpu().numpy(),
        pred.cpu().numpy(),
    )

# Training
def main():
    print("\n" + "=" * 88)
    print("Benchmark 5: 2D time-conformable reaction--diffusion equation")
    print("=" * 88)
    print(
        f"q={q_frac}, nu={nu:.8e}, lambda={lam}, "
        f"(omega1,omega2)=({int(omega1)},{int(omega2)})"
    )
    print(
        f"domain=[0,2pi]^2 x [0,1], H_std={hid_dim_std}, H_sb={hid_dim_sb}, "
        f"N_f={num_f}, N_0={num_x0}, N_b={num_tb} per periodic direction"
    )
    print(
        f"Adam lr={adam_lr:.1e}, conformable-GD L0={LR0:.1e}, "
        f"common resampling interval={resample_every}"
    )

    verify_manufactured_problem()

    # Confirm the shared collocation schedule before training.
    x_a, y_a, t_a = sample_interior_block(0)
    x_b, y_b, t_b = sample_interior_block(0)
    if not (
        torch.equal(x_a, x_b)
        and torch.equal(y_a, y_b)
        and torch.equal(t_a, t_b)
    ):
        raise RuntimeError("Deterministic collocation schedule check failed.")
    print(
        f"  matched parameter target = 501 vs 502\n"
        f"  shared collocation block 0 verified: {num_f} identical points"
    )

    # --------------------------------------------------------
    # Standard PINN + Adam
    # --------------------------------------------------------
    torch.manual_seed(model_seed)
    std_model = PINN(
        input_dim=3,
        hidden_dim=hid_dim_std,
        output_dim=1,
    ).to(device)
    params_std = count_parameters(std_model)

    x0_std, y0_std = initial_grid(num_x0)
    s_b_std, t_b_std = boundary_parameter_grid(num_tb, eps_t)

    optimizer_std = torch.optim.Adam(std_model.parameters(), lr=adam_lr)

    Loss_std = []
    x_f_std = y_f_std = t_f_std = None
    t0_wall_std = time.time()

    for i in range(adam_iters + 1):
        if i % resample_every == 0:
            block_index = i // resample_every
            x_f_std, y_f_std, t_f_std = sample_interior_block(block_index)

        optimizer_std.zero_grad()

        loss_std, loss_pde_std, loss_ic_std, loss_bc_std = total_loss_std(
            std_model,
            x_f_std,
            y_f_std,
            t_f_std,
            x0_std,
            y0_std,
            s_b_std,
            t_b_std,
            q_frac,
        )

        if torch.isnan(loss_std) or torch.isinf(loss_std):
            print(f"Standard PINN stopped: invalid loss at iteration {i}")
            break

        if i % 200 == 0:
            print(
                f"[Standard PINN] Iteration: {i:5d} | "
                f"Loss: {loss_std.item():.6e} | "
                f"PDE: {loss_pde_std.item():.6e} | "
                f"IC: {loss_ic_std.item():.6e} | "
                f"BC: {loss_bc_std.item():.6e}"
            )

        loss_std.backward()
        torch.nn.utils.clip_grad_norm_(
            std_model.parameters(), max_norm=clip_value
        )
        optimizer_std.step()

        Loss_std.append(loss_std.item())

    elapsed_std = time.time() - t0_wall_std

    # Recompute the final loss after the last optimizer update.
    final_loss_std_tensor, _, _, _ = total_loss_std(
        std_model,
        x_f_std,
        y_f_std,
        t_f_std,
        x0_std,
        y0_std,
        s_b_std,
        t_b_std,
        q_frac,
    )
    final_loss_std = final_loss_std_tensor.detach().item()

    # --------------------------------------------------------
    # Shared-bias PINN + conformable-gradient training
    # --------------------------------------------------------
    torch.manual_seed(model_seed)
    sb_model = SharedBiasNet(
        input_dim=9,
        hid_dim=hid_dim_sb,
        output_dim=1,
    ).to(device)
    params_sb = count_parameters(sb_model)

    s_b_sb, t_b_sb = boundary_parameter_grid(num_tb, eps_t)

    Loss_sb = []
    x_f_sb = y_f_sb = t_f_sb = None
    t0_wall_sb = time.time()

    for i in range(num_iters + 1):
        if i % resample_every == 0:
            block_index = i // resample_every
            x_f_sb, y_f_sb, t_f_sb = sample_interior_block(block_index)

        sb_model.zero_grad(set_to_none=True)

        loss_sb, loss_pde_sb, loss_bc_sb = total_loss_sb(
            sb_model,
            x_f_sb,
            y_f_sb,
            t_f_sb,
            s_b_sb,
            t_b_sb,
            q_frac,
            w_pde=1.0,
            w_bc=1.0,
        )

        if torch.isnan(loss_sb) or torch.isinf(loss_sb):
            print(f"Shared-bias PINN stopped: invalid loss at iteration {i}")
            break

        if i % 200 == 0:
            alpha_opt = max(lr_alpha(i), 1.0e-14)
            current_lr = LR(i, LR0=LR0)
            print(
                f"[Shared-bias PINN] Iteration: {i:5d} | "
                f"Loss: {loss_sb.item():.6e} | "
                f"PDE: {loss_pde_sb.item():.6e} | "
                f"IC: {0.0:.6e} | "
                f"BC: {loss_bc_sb.item():.6e} | "
                f"alpha_opt: {alpha_opt:.6e} | LR: {current_lr:.6e}"
            )

        loss_sb.backward()
        torch.nn.utils.clip_grad_norm_(
            sb_model.parameters(), max_norm=clip_value
        )
        conformable_fractional_variable_gradient_pinn(
            sb_model, i, LR0=LR0
        )

        Loss_sb.append(loss_sb.item())

    elapsed_sb = time.time() - t0_wall_sb

    # Recompute the final loss after the last conformable-gradient update.
    final_loss_sb_tensor, _, _ = total_loss_sb(
        sb_model,
        x_f_sb,
        y_f_sb,
        t_f_sb,
        s_b_sb,
        t_b_sb,
        q_frac,
        w_pde=1.0,
        w_bc=1.0,
    )
    final_loss_sb = final_loss_sb_tensor.detach().item()

    # --------------------------------------------------------
    # Independent post-training diagnostics
    # --------------------------------------------------------
    e_inf_std, e_2_std, e_2_rel_std = evaluate_global_errors(
        std_model, "standard"
    )
    e_inf_sb, e_2_sb, e_2_rel_sb = evaluate_global_errors(
        sb_model, "shared"
    )

    e_r_std = independent_residual_rms(std_model, "standard")
    e_r_sb = independent_residual_rms(sb_model, "shared")

    n_tol_std = first_iteration_below(Loss_std, loss_tolerance)
    n_tol_sb = first_iteration_below(Loss_sb, loss_tolerance)

    # Locked parameter-count check.
    expected_std_params = 501
    expected_sb_params = 502
    if params_std != expected_std_params or params_sb != expected_sb_params:
        raise RuntimeError(
            "Unexpected parameter count. "
            f"Got standard={params_std}, shared={params_sb}; "
            f"expected {expected_std_params} and {expected_sb_params}."
        )

    # --------------------------------------------------------
    # Results table
    # --------------------------------------------------------
    results = [
        (
            "Standard PINN + Adam",
            params_std,
            e_inf_std,
            e_2_std,
            e_2_rel_std,
            e_r_std,
            final_loss_std,
            n_tol_std,
            elapsed_std,
        ),
        (
            "Shared-bias PINN + conformable GD",
            params_sb,
            e_inf_sb,
            e_2_sb,
            e_2_rel_sb,
            e_r_sb,
            final_loss_sb,
            n_tol_sb,
            elapsed_sb,
        ),
    ]

    print("\n2D reaction--diffusion benchmark summary:\n")
    header = (
        f"{'Method':<38}"
        f"{'Params':>10}"
        f"{'E_inf':>15}"
        f"{'E_2':>15}"
        f"{'E_2_rel':>15}"
        f"{'E_r':>15}"
        f"{'Final loss':>15}"
        f"{'N_tol':>10}"
        f"{'CPU (s)':>12}"
    )
    print(header)
    print("-" * len(header))

    for (
        method,
        params,
        e_inf,
        e_2,
        e_2_rel,
        e_r,
        final_loss,
        n_tol,
        cpu,
    ) in results:
        n_tol_text = str(n_tol) if n_tol is not None else ">20000"
        print(
            f"{method:<38}"
            f"{params:10d}"
            f"{e_inf:15.6e}"
            f"{e_2:15.6e}"
            f"{e_2_rel:15.6e}"
            f"{e_r:15.6e}"
            f"{final_loss:15.6e}"
            f"{n_tol_text:>10}"
            f"{cpu:12.3f}"
        )

    print("\nLaTeX table:\n")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(
        r"\caption{Accuracy, independent residual, convergence, and "
        r"computational-cost comparison for the two-dimensional "
        r"time-conformable reaction--diffusion benchmark.}"
    )
    print(r"\label{tab:2d_reaction_diffusion_comparison}")
    print(r"\scriptsize")
    print(r"\setlength{\tabcolsep}{3.0pt}")
    print(r"\resizebox{\textwidth}{!}{%")
    print(r"\begin{tabular}{lcccccccc}")
    print(r"\hline")
    print(
        r"Method & Parameters & $E_{\infty}$ & $E_2$ & "
        r"$E_{2,\mathrm{rel}}$ & $E_r$ & Final loss & "
        r"$N_{\mathrm{tol}}$ & CPU time (s) \\"
    )
    print(r"\hline")

    for (
        method,
        params,
        e_inf,
        e_2,
        e_2_rel,
        e_r,
        final_loss,
        n_tol,
        cpu,
    ) in results:
        n_tol_text = str(n_tol) if n_tol is not None else r"$>20{,}000$"
        print(
            f"{method} & {params} & {e_inf:.6e} & {e_2:.6e} & "
            f"{e_2_rel:.6e} & {e_r:.6e} & {final_loss:.6e} & "
            f"{n_tol_text} & {cpu:.3f} \\\\"
        )

    print(r"\hline")
    print(r"\end{tabular}%")
    print(r"}")
    print(r"\end{table}")

    # --------------------------------------------------------
    # Graphical diagnostics
    # --------------------------------------------------------
    Loss_std_s = smooth_curve(Loss_std)
    Loss_sb_s = smooth_curve(Loss_sb)

    # Final-time fields
    xg, yg, U_exact_final, U_std_final = evaluate_final_time_field(
        std_model, "standard"
    )
    _, _, _, U_sb_final = evaluate_final_time_field(sb_model, "shared")
    Xp, Yp = np.meshgrid(xg, yg, indexing="ij")

    # 1) Final-time solution surfaces
    fig1 = plt.figure(figsize=(21.0, 6.2), dpi=240, constrained_layout=True)

    ax1 = fig1.add_subplot(1, 3, 1, projection="3d")
    surf1 = ax1.plot_surface(
        Xp,
        Yp,
        U_exact_final,
        cmap="viridis",
        rcount=120,
        ccount=120,
        linewidth=0,
        antialiased=True,
        rasterized=True,
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("u(x,y,1)")
    ax1.set_title("Exact solution")
    ax1.view_init(elev=28, azim=-135)
    fig1.colorbar(surf1, ax=ax1, shrink=0.72, pad=0.08)

    ax2 = fig1.add_subplot(1, 3, 2, projection="3d")
    surf2 = ax2.plot_surface(
        Xp,
        Yp,
        U_std_final,
        cmap="viridis",
        rcount=120,
        ccount=120,
        linewidth=0,
        antialiased=True,
        rasterized=True,
    )
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("u(x,y,1)")
    ax2.set_title("Standard PINN")
    ax2.view_init(elev=28, azim=-135)
    fig1.colorbar(surf2, ax=ax2, shrink=0.72, pad=0.08)

    ax3 = fig1.add_subplot(1, 3, 3, projection="3d")
    surf3 = ax3.plot_surface(
        Xp,
        Yp,
        U_sb_final,
        cmap="viridis",
        rcount=120,
        ccount=120,
        linewidth=0,
        antialiased=True,
        rasterized=True,
    )
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("u(x,y,1)")
    ax3.set_title("Shared-bias PINN")
    ax3.view_init(elev=28, azim=-135)
    fig1.colorbar(surf3, ax=ax3, shrink=0.72, pad=0.08)

    save_pdf(fig1, "compare_final_time_solution_surfaces_prob5_2d")
    plt.close(fig1)

    # 2) Final-time absolute-error surfaces
    fig2 = plt.figure(figsize=(14.5, 6.0), dpi=240, constrained_layout=True)

    ax4 = fig2.add_subplot(1, 2, 1, projection="3d")
    err_std_final = np.abs(U_std_final - U_exact_final)
    surf4 = ax4.plot_surface(
        Xp,
        Yp,
        err_std_final,
        cmap="magma",
        rcount=120,
        ccount=120,
        linewidth=0,
        antialiased=True,
        rasterized=True,
    )
    ax4.set_xlabel("x")
    ax4.set_ylabel("y")
    ax4.set_zlabel("|error|")
    ax4.set_title("Standard PINN error")
    ax4.view_init(elev=28, azim=-135)
    fig2.colorbar(surf4, ax=ax4, shrink=0.75, pad=0.08)

    ax5 = fig2.add_subplot(1, 2, 2, projection="3d")
    err_sb_final = np.abs(U_sb_final - U_exact_final)
    surf5 = ax5.plot_surface(
        Xp,
        Yp,
        err_sb_final,
        cmap="magma",
        rcount=120,
        ccount=120,
        linewidth=0,
        antialiased=True,
        rasterized=True,
    )
    ax5.set_xlabel("x")
    ax5.set_ylabel("y")
    ax5.set_zlabel("|error|")
    ax5.set_title("Shared-bias PINN error")
    ax5.view_init(elev=28, azim=-135)
    fig2.colorbar(surf5, ax=ax5, shrink=0.75, pad=0.08)

    save_pdf(fig2, "compare_final_time_error_surfaces_prob5_2d")
    plt.close(fig2)

    # 3) Final-time solution heatmaps
    fig3, axes3 = plt.subplots(
        1, 3, figsize=(15.8, 4.8), dpi=240, constrained_layout=True
    )
    fields = [U_exact_final, U_std_final, U_sb_final]
    titles = ["Exact", "Standard PINN", "Shared-bias PINN"]
    for ax, field, title in zip(axes3, fields, titles):
        im = ax.imshow(
            field.T,
            extent=[x_left, x_right, y_left, y_right],
            origin="lower",
            aspect="auto",
            cmap="viridis",
            rasterized=True,
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig3.colorbar(im, ax=ax, shrink=0.88)
    save_pdf(fig3, "compare_final_time_solution_heatmaps_prob5_2d")
    plt.close(fig3)

    # 4) Final-time error heatmaps
    fig4, axes4 = plt.subplots(
        1, 2, figsize=(11.2, 4.8), dpi=240, constrained_layout=True
    )
    for ax, field, title in zip(
        axes4,
        [err_std_final, err_sb_final],
        ["Standard PINN error", "Shared-bias PINN error"],
    ):
        im = ax.imshow(
            field.T,
            extent=[x_left, x_right, y_left, y_right],
            origin="lower",
            aspect="auto",
            cmap="magma",
            rasterized=True,
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig4.colorbar(im, ax=ax, shrink=0.88)
    save_pdf(fig4, "compare_final_time_error_heatmaps_prob5_2d")
    plt.close(fig4)

    # 5) Final-time 1D slice and training-loss history
    # Choose y = pi/(2*omega2), so both modal components are present.
    y_slice = math.pi / (2.0 * omega2)
    x_line = torch.linspace(x_left, x_right, 500, device=device).reshape(-1, 1)
    y_line = torch.full_like(x_line, y_slice)
    t_line = torch.full_like(x_line, T_final)

    with torch.no_grad():
        exact_line = exact_solution(x_line, y_line, t_line).cpu().numpy().ravel()
        std_line = trial_solution_std(std_model, x_line, y_line, t_line).cpu().numpy().ravel()
        sb_line = trial_solution_sb(sb_model, x_line, y_line, t_line).cpu().numpy().ravel()
        x_line_np = x_line.cpu().numpy().ravel()

    fig5, axes5 = plt.subplots(1, 2, figsize=(12.4, 4.6), dpi=240, constrained_layout=True)
    axes5[0].plot(x_line_np, exact_line, color=COLOR_EX, label="Exact")
    axes5[0].plot(x_line_np, std_line, "--", color=COLOR_STD, label="Standard PINN")
    axes5[0].plot(x_line_np, sb_line, "-.", color=COLOR_SB, label="Shared-bias PINN")
    axes5[0].set_xlabel("x")
    axes5[0].set_ylabel("u(x,y,1)")
    axes5[0].set_title(rf"Final-time slice, $y=\pi/(2\omega_2)$")
    axes5[0].legend()

    axes5[1].semilogy(Loss_std_s, color=COLOR_STD, label="Standard PINN")
    axes5[1].semilogy(Loss_sb_s, color=COLOR_SB, label="Shared-bias PINN")
    axes5[1].axhline(loss_tolerance, linestyle="--", linewidth=1.2, label=r"$10^{-3}$ tolerance")
    axes5[1].set_xlabel("Iteration")
    axes5[1].set_ylabel("Loss")
    axes5[1].set_title("Training-loss comparison")
    axes5[1].legend()

    save_pdf(fig5, "compare_final_time_slice_and_loss_prob5_2d")
    plt.close(fig5)

    # 6) x--t slice through the 2D solution
    xg_slice, tg_slice, U_exact_slice, U_std_slice = evaluate_xt_slice(
        std_model, "standard", y_slice
    )
    _, _, _, U_sb_slice = evaluate_xt_slice(sb_model, "shared", y_slice)

    fig6, axes6 = plt.subplots(
        1, 3, figsize=(15.8, 4.8), dpi=240, constrained_layout=True
    )
    slice_fields = [U_exact_slice, U_std_slice, U_sb_slice]
    for ax, field, title in zip(axes6, slice_fields, titles):
        im = ax.imshow(
            field.T,
            extent=[x_left, x_right, 0.0, T_final],
            origin="lower",
            aspect="auto",
            cmap="viridis",
            rasterized=True,
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig6.colorbar(im, ax=ax, shrink=0.88)
    save_pdf(fig6, "compare_xt_slice_solution_heatmaps_prob5_2d")
    plt.close(fig6)

    fig7, axes7 = plt.subplots(
        1, 2, figsize=(11.2, 4.8), dpi=240, constrained_layout=True
    )
    slice_errors = [
        np.abs(U_std_slice - U_exact_slice),
        np.abs(U_sb_slice - U_exact_slice),
    ]
    for ax, field, title in zip(
        axes7,
        slice_errors,
        ["Standard PINN error", "Shared-bias PINN error"],
    ):
        im = ax.imshow(
            field.T,
            extent=[x_left, x_right, 0.0, T_final],
            origin="lower",
            aspect="auto",
            cmap="magma",
            rasterized=True,
        )
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        fig7.colorbar(im, ax=ax, shrink=0.88)
    save_pdf(fig7, "compare_xt_slice_error_heatmaps_prob5_2d")
    plt.close(fig7)

    print(f"\nPDF figures saved in:\n{output_dir}")


if __name__ == "__main__":
    main()
