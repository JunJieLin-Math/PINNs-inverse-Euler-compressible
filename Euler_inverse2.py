"""
Compressible-Euler PINN inverse problem (2-D ideal gas)
======================================================
Unknowns: d (vertical offset), M (mass ratio), R (radius ratio), T (snapshot time).
Observation: a synthetic X-ray-like proxy I = rho**2 at one unknown time.

Dependencies:
    pip install numpy scipy matplotlib torch

Run:
    python compressible_euler_inverse.py

This script has two stages:
  1. Generate a synthetic observation using a 2-D finite-volume Rusanov solver.
  2. Fit a strong-form PINN to that observation.

Important: strong-form PINNs are not robust for strong shocks. Start with low
Mach number / smooth cases; for strong shocks use a shock-aware weak-form PINN
or a differentiable finite-volume inverse solver.
"""
import os, time, csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ------------------------- Configuration -------------------------
SEED = 1234
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32
OUT = Path("results_compressible_euler")
OUT.mkdir(exist_ok=True)

# Dimensionless domain and reference scales
LX, LY = 4.0, 2.5
NX, NY = 128, 80                 # finite-volume grid
GAMMA = 1.4
RHO_FLOOR = 0.03
P0 = 1.0
SIGMA1 = 0.22
CENTER_X = 0.72
V0 = 0.35                        # reduce for smoother, easier PINN experiments
T_MAX = 0.75

# Ground truth used ONLY to create synthetic data
TRUE = {"d": 0.42, "M": 0.72, "R": 0.82, "T": 0.48}
# Bounds must include the expected solution
BOUNDS = {"d": (0.15, 0.80), "M": (0.35, 1.40),
          "R": (0.55, 1.25), "T": (0.12, T_MAX)}

# PINN settings (computationally expensive; reduce for a smoke test)
WIDTH, DEPTH = 96, 5
N_F = 2500
N_IC = 2500
ADAM_STEPS = 12000
LBFGS_STEPS = 500
LOG_EVERY = 200
LAMBDA_PDE, LAMBDA_IC, LAMBDA_DATA = 1.0, 10.0, 10.0
OBS_FILE = OUT / "synthetic_observation.npz"

torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ------------------------- Initial conditions -------------------------
def gaussian(x, y, cx, cy, sigma):
    return np.exp(-((x-cx)**2 + (y-cy)**2) / (2.0*sigma**2))

def initial_primitive_np(x, y, d, M, R):
    """
    Cluster 1: center (-CENTER_X,-d/2), mass scale 1, radius SIGMA1.
    Cluster 2: center (+CENTER_X,+d/2), mass scale M, radius R*SIGMA1.
    Gaussian amplitudes are chosen so integrated 2-D excess masses have ratio M:
      A2/A1 = M/R^2.
    The two clusters move toward one another along x.
    """
    g1 = gaussian(x, y, -CENTER_X, -d/2, SIGMA1)
    s2 = R * SIGMA1
    g2 = gaussian(x, y, +CENTER_X, +d/2, s2)
    rho1 = g1
    rho2 = (M / (R*R)) * g2
    rho = RHO_FLOOR + rho1 + rho2
    # Density-weighted velocity prevents an artificial discontinuity where
    # the Gaussian tails overlap: cluster 1 moves +x, cluster 2 moves -x.
    u = V0 * (rho1 - rho2) / np.maximum(rho, 1.e-8)
    v = np.zeros_like(u)
    p = np.full_like(rho, P0)
    return rho, u, v, p

def primitive_to_conserved(rho, u, v, p):
    E = p/(GAMMA-1.0) + 0.5*rho*(u*u+v*v)
    return np.stack([rho, rho*u, rho*v, E], axis=0)

def conserved_to_primitive(U):
    rho = np.maximum(U[0], 1.e-6)
    u, v = U[1]/rho, U[2]/rho
    E = np.maximum(U[3], 1.e-6)
    p = np.maximum((GAMMA-1.0)*(E-0.5*rho*(u*u+v*v)), 1.e-6)
    return rho, u, v, p

# ------------------------- Finite-volume synthetic-data solver -------------------------
def flux_x(U):
    rho, u, v, p = conserved_to_primitive(U)
    return np.stack([rho*u, rho*u*u+p, rho*u*v, (U[3]+p)*u], axis=0)

def flux_y(U):
    rho, u, v, p = conserved_to_primitive(U)
    return np.stack([rho*v, rho*u*v, rho*v*v+p, (U[3]+p)*v], axis=0)

def rusanov_flux(UL, UR, axis):
    if axis == 0:
        FL, FR = flux_x(UL), flux_x(UR)
        _, uL, vL, pL = conserved_to_primitive(UL)
        _, uR, vR, pR = conserved_to_primitive(UR)
        cL = np.sqrt(GAMMA*pL/np.maximum(UL[0], 1.e-6))
        cR = np.sqrt(GAMMA*pR/np.maximum(UR[0], 1.e-6))
        a = np.maximum(np.abs(uL)+cL, np.abs(uR)+cR)
    else:
        FL, FR = flux_y(UL), flux_y(UR)
        _, uL, vL, pL = conserved_to_primitive(UL)
        _, uR, vR, pR = conserved_to_primitive(UR)
        cL = np.sqrt(GAMMA*pL/np.maximum(UL[0], 1.e-6))
        cR = np.sqrt(GAMMA*pR/np.maximum(UR[0], 1.e-6))
        a = np.maximum(np.abs(vL)+cL, np.abs(vR)+cR)
    return 0.5*(FL+FR) - 0.5*a[None, ...]*(UR-UL)

def simulate_observation(d, M, R, T, nx=NX, ny=NY):
    """First-order finite-volume solver with transmissive (zero-gradient) edges."""
    dx, dy = LX/nx, LY/ny
    x = np.linspace(-LX/2+dx/2, LX/2-dx/2, nx)
    y = np.linspace(-LY/2+dy/2, LY/2-dy/2, ny)
    X, Y = np.meshgrid(x, y)
    prim = initial_primitive_np(X, Y, d, M, R)
    U = primitive_to_conserved(*prim)
    t, nsteps = 0.0, 0
    while t < T:
        rho, u, v, p = conserved_to_primitive(U)
        c = np.sqrt(GAMMA*p/np.maximum(rho, 1.e-6))
        dt = 0.38 * min(dx, dy) / (np.max(np.abs(u)+c) + np.max(np.abs(v)+c) + 1.e-8)
        dt = min(dt, T-t)
        # x faces, with transmissive ghost cells
        Ux = np.pad(U, ((0,0),(0,0),(1,1)), mode="edge")
        Fx = rusanov_flux(Ux[:, :, :-1], Ux[:, :, 1:], axis=0)
        # y faces
        Uy = np.pad(U, ((0,0),(1,1),(0,0)), mode="edge")
        Fy = rusanov_flux(Uy[:, :-1, :], Uy[:, 1:, :], axis=1)
        U = U - dt/dx*(Fx[:, :, 1:] - Fx[:, :, :-1]) \
              - dt/dy*(Fy[:, 1:, :] - Fy[:, :-1, :])
        # Protect the synthetic generator from tiny roundoff-level negative states.
        rho, u, v, p = conserved_to_primitive(U)
        U = primitive_to_conserved(rho, u, v, p)
        t += dt
        nsteps += 1
        if nsteps > 20000:
            raise RuntimeError("Too many FV steps; reduce T or increase grid spacing.")
    rho, u, v, p = conserved_to_primitive(U)
    # X-ray-like proxy. Real X-ray emissivity also depends on temperature/bandpass.
    intensity = rho**2
    intensity = intensity / intensity.max()
    return X, Y, intensity.astype(np.float32)

def make_or_load_observation():
    if OBS_FILE.exists():
        z = np.load(OBS_FILE)
        print(f"Loaded observation: {OBS_FILE}")
        return z["x"], z["y"], z["intensity"]
    print("Generating synthetic observation with finite-volume compressible Euler solver...")
    X, Y, I = simulate_observation(**TRUE)
    np.savez(OBS_FILE, x=X, y=Y, intensity=I,
             true_d=TRUE["d"], true_M=TRUE["M"], true_R=TRUE["R"], true_T=TRUE["T"])
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(I, origin="lower", extent=[-LX/2,LX/2,-LY/2,LY/2], aspect="auto")
    ax.set_title("Synthetic observation: normalized $I\\propto\\rho^2$")
    ax.set_xlabel("x"); ax.set_ylabel("y"); fig.colorbar(im, ax=ax)
    fig.tight_layout(); fig.savefig(OUT/"synthetic_observation.png", dpi=160); plt.close(fig)
    return X, Y, I


# ------------------------- PINN -------------------------
def grad(y, x):
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True,
                               retain_graph=True)[0]

class MLP(nn.Module):
    def __init__(self, width=WIDTH, depth=DEPTH):
        super().__init__()
        layers = [nn.Linear(3, width), nn.Tanh()]
        for _ in range(depth-1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, 4)]
        self.net = nn.Sequential(*layers)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, z):
        raw = self.net(z)
        # Positive density and pressure; velocities remain unconstrained.
        rho = torch.nn.functional.softplus(raw[:, 0:1]) + 1.e-5
        u, v = raw[:, 1:2], raw[:, 2:3]
        p = torch.nn.functional.softplus(raw[:, 3:4]) + 1.e-5
        return rho, u, v, p

class InverseParameters(nn.Module):
    def __init__(self):
        super().__init__()
        # Sigmoid maps unconstrained trainable variables into physical bounds.
        self.z = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(0.0)) for k in BOUNDS
        })
        # z=0 maps to the midpoint of each admissible interval. Do not
        # initialize from TRUE: those values are used only by the data generator.

    def physical(self):
        return {k: lo+(hi-lo)*torch.sigmoid(self.z[k])
                for k, (lo, hi) in BOUNDS.items()}

def initial_torch(x, y, pars):
    d, M, R = pars["d"], pars["M"], pars["R"]
    g1 = torch.exp(-((x+CENTER_X)**2+(y+d/2)**2)/(2*SIGMA1**2))
    s2 = R*SIGMA1
    g2 = torch.exp(-((x-CENTER_X)**2+(y-d/2)**2)/(2*s2**2))
    rho1, rho2 = g1, (M/(R*R))*g2
    rho = RHO_FLOOR + rho1 + rho2
    u = V0*(rho1-rho2)/rho
    v = torch.zeros_like(u)
    p = torch.ones_like(rho)*P0
    return rho, u, v, p

def to_conserved(rho, u, v, p):
    E = p/(GAMMA-1.0) + 0.5*rho*(u*u+v*v)
    return rho, rho*u, rho*v, E

def pde_residual(model, coords):
    coords = coords.detach().clone().requires_grad_(True)
    rho, u, v, p = model(coords)
    U = to_conserved(rho, u, v, p)
    E = U[3]
    Fx = (rho*u, rho*u*u+p, rho*u*v, (E+p)*u)
    Fy = (rho*v, rho*u*v, rho*v*v+p, (E+p)*v)
    residuals = []
    # The network input is normalized: X=2x/LX, Y=2y/LY, tau=t/T_MAX.
    # Convert derivatives back to physical coordinates before evaluating PDE.
    for q, fx, fy in zip(U, Fx, Fy):
        dq = grad(q, coords)
        dfx = grad(fx, coords)
        dfy = grad(fy, coords)
        qt = dq[:, 2:3] / T_MAX
        fxx = dfx[:, 0:1] * (2.0 / LX)
        fyy = dfy[:, 1:2] * (2.0 / LY)
        residuals.append(qt + fxx + fyy)
    return torch.cat(residuals, dim=1)

def sample_xy(n, device):
    x = (torch.rand(n, 1, device=device)-0.5)*LX
    y = (torch.rand(n, 1, device=device)-0.5)*LY
    return x, y

def train_inverse(X_np, Y_np, I_np):
    model = MLP().to(device=DEVICE, dtype=DTYPE)
    pars = InverseParameters().to(device=DEVICE, dtype=DTYPE)
    X = torch.tensor(X_np.reshape(-1,1), device=DEVICE, dtype=DTYPE)
    Y = torch.tensor(Y_np.reshape(-1,1), device=DEVICE, dtype=DTYPE)
    Iobs = torch.tensor(I_np.reshape(-1,1), device=DEVICE, dtype=DTYPE)
    # Normalize observed image and compare density-squared proxy.
    I_scale = Iobs.max().detach().clamp_min(1.e-8)
    Iobs = Iobs / I_scale
    obs_tau = torch.ones_like(X)
    obs_coords = torch.cat([X, Y, obs_tau], dim=1)

    # Scale coordinates and physical time to O(1). PINN input uses tau=t/T_MAX.
    def network_coords(x, y, t):
        return torch.cat([2*x/LX, 2*y/LY, t/T_MAX], dim=1)

    opt = torch.optim.Adam([
        {"params": model.parameters(), "lr": 1.e-3},
        {"params": pars.parameters(), "lr": 2.e-3},
    ])
    history = []
    t0 = time.time()

    def losses():
        pp = pars.physical()
        xf, yf = sample_xy(N_F, DEVICE)
        tf = torch.rand(N_F, 1, device=DEVICE, dtype=DTYPE)*T_MAX
        cf = network_coords(xf, yf, tf)
        r = pde_residual(model, cf)
        # Residual normalization by representative scales.
        lpde = torch.mean((r / torch.tensor([1.0, 1.0, 1.0, 3.0], device=DEVICE))**2)

        xi, yi = sample_xy(N_IC, DEVICE)
        ci = network_coords(xi, yi, torch.zeros_like(xi))
        pred = model(ci)
        target = initial_torch(xi, yi, pp)
        lic = sum(torch.mean((a-b)**2) for a,b in zip(pred, target))

        # tau=1 corresponds to physical time T_MAX; the unknown snapshot time
        # is handled by evaluating the network at tau=T/T_MAX.
        tau = (pp["T"]/T_MAX).expand_as(X)
        co = network_coords(X, Y, tau)
        ro, uo, vo, po = model(co)
        Ipred = ro**2
        Ipred = Ipred / Ipred.max().detach().clamp_min(1.e-8)
        ldata = torch.mean((Ipred-Iobs)**2)
        total = LAMBDA_PDE*lpde + LAMBDA_IC*lic + LAMBDA_DATA*ldata
        return total, lpde, lic, ldata

    for step in range(1, ADAM_STEPS+1):
        opt.zero_grad(set_to_none=True)
        total, lpde, lic, ldata = losses()
        total.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(pars.parameters()), 100.0)
        opt.step()
        if step == 1 or step % LOG_EVERY == 0 or step == ADAM_STEPS:
            vals = {k: float(v.detach().cpu()) for k,v in pars.physical().items()}
            row = {"step": step, "total": total.item(), "pde": lpde.item(),
                   "ic": lic.item(), "data": ldata.item(), **vals}
            history.append(row)
            print(f"Adam {step:6d}/{ADAM_STEPS} L={total.item():.3e} "
                  f"PDE={lpde.item():.2e} IC={lic.item():.2e} Data={ldata.item():.2e} "
                  f"d={vals['d']:.4f} M={vals['M']:.4f} R={vals['R']:.4f} T={vals['T']:.4f}")

    # Fixed collocation points for the quasi-Newton stage.
    xf, yf = sample_xy(N_F, DEVICE)
    tf = torch.rand(N_F, 1, device=DEVICE, dtype=DTYPE)*T_MAX
    cf = network_coords(xf, yf, tf)
    xi, yi = sample_xy(N_IC, DEVICE)
    ci = network_coords(xi, yi, torch.zeros_like(xi))

    def fixed_losses():
        pp = pars.physical()
        r = pde_residual(model, cf)
        lpde = torch.mean((r / torch.tensor([1.0, 1.0, 1.0, 3.0], device=DEVICE))**2)
        pred = model(ci); target = initial_torch(xi, yi, pp)
        lic = sum(torch.mean((a-b)**2) for a,b in zip(pred, target))
        tau = (pp["T"]/T_MAX).expand_as(X)
        ro, _, _, _ = model(network_coords(X, Y, tau))
        Ipred = ro**2; Ipred = Ipred/Ipred.max().detach().clamp_min(1.e-8)
        ldata = torch.mean((Ipred-Iobs)**2)
        return LAMBDA_PDE*lpde+LAMBDA_IC*lic+LAMBDA_DATA*ldata, lpde, lic, ldata

    lbfgs = torch.optim.LBFGS(list(model.parameters())+list(pars.parameters()),
                              lr=0.8, max_iter=1, max_eval=20,
                              history_size=50, line_search_fn="strong_wolfe")
    for k in range(1, LBFGS_STEPS+1):
        def closure():
            lbfgs.zero_grad(set_to_none=True)
            L, *_ = fixed_losses()
            L.backward()
            return L
        lbfgs.step(closure)
        if k == 1 or k % 25 == 0 or k == LBFGS_STEPS:
            L, lp, li, ld = fixed_losses()
            vals = {name: float(value.detach().cpu()) for name,value in pars.physical().items()}
            row = {"step": ADAM_STEPS+k, "total": L.item(), "pde": lp.item(),
                   "ic": li.item(), "data": ld.item(), **vals}
            history.append(row)
            print(f"LBFGS {k:5d}/{LBFGS_STEPS} L={L.item():.3e} "
                  f"PDE={lp.item():.2e} IC={li.item():.2e} Data={ld.item():.2e} "
                  f"d={vals['d']:.4f} M={vals['M']:.4f} R={vals['R']:.4f} T={vals['T']:.4f}")

    elapsed = time.time()-t0
    estimates = {k: float(v.detach().cpu()) for k,v in pars.physical().items()}
    print("\nEstimated parameters:")
    for k in ("d","M","R","T"):
        print(f"  {k}: estimate={estimates[k]:.6f} | truth={TRUE[k]:.6f}")
    print(f"Elapsed training time: {elapsed:.1f} s")

    with open(OUT/"training_history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader(); w.writerows(history)
    torch.save({"model": model.state_dict(), "parameters": estimates}, OUT/"checkpoint.pt")

    # Save reconstructed image at the estimated snapshot time.
    model.eval()
    with torch.no_grad():
        tau = torch.full_like(X, estimates["T"]/T_MAX)
        rho_pred, _, _, _ = model(network_coords(X, Y, tau))
        Ipred = (rho_pred**2).reshape(I_np.shape).cpu().numpy()
        Ipred /= max(Ipred.max(), 1.e-8)
    fig, ax = plt.subplots(1,3,figsize=(13,4))
    extent=[-LX/2,LX/2,-LY/2,LY/2]
    im=ax[0].imshow(I_np,origin="lower",extent=extent,aspect="auto")
    ax[0].set_title("Observation"); fig.colorbar(im,ax=ax[0])
    im=ax[1].imshow(Ipred,origin="lower",extent=extent,aspect="auto")
    ax[1].set_title("PINN reconstruction"); fig.colorbar(im,ax=ax[1])
    im=ax[2].imshow(np.abs(Ipred-I_np),origin="lower",extent=extent,aspect="auto")
    ax[2].set_title("Absolute error"); fig.colorbar(im,ax=ax[2])
    for a in ax: a.set_xlabel("x"); a.set_ylabel("y")
    fig.tight_layout(); fig.savefig(OUT/"reconstruction.png",dpi=160); plt.close(fig)

    steps=np.array([r["step"] for r in history])
    fig,ax=plt.subplots(figsize=(8,4.5))
    for key in ("total","pde","ic","data"):
        ax.plot(steps,[max(r[key],1.e-16) for r in history],label=key)
    ax.set_yscale("log"); ax.set_xlabel("optimization step"); ax.set_ylabel("loss")
    ax.legend(); ax.grid(True,alpha=.3); fig.tight_layout()
    fig.savefig(OUT/"loss_history.png",dpi=160); plt.close(fig)
    return estimates


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    X, Y, I = make_or_load_observation()
    train_inverse(X, Y, I)
