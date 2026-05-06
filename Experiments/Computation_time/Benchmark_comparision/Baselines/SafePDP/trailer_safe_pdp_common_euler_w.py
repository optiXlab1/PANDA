import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
from casadi import SX, DM, vertcat, dot, sin, cos, fmax
from SafePDP import SafePDP


@dataclass
class TrailerScene:
    x0: np.ndarray
    xref: np.ndarray
    circle_c: np.ndarray
    horizon: int
    dt: float = 0.1
    L: float = 0.5
    umax: Tuple[float, float] = (0.8, 0.8)


@dataclass
class TrailerTheta:
    q: np.ndarray
    r: np.ndarray
    w: np.ndarray
    eta_circle: float
    circle_r: float

    def encode(self) -> np.ndarray:
        return np.array([
            float(self.q[0]), float(self.q[1]), float(self.q[2]),
            float(self.r[0]), float(self.r[1]),
            float(self.w[0]), float(self.w[1]), float(self.w[2]),
            float(self.eta_circle), float(self.circle_r)
        ], dtype=float)

    @staticmethod
    def decode(theta: np.ndarray) -> "TrailerTheta":
        theta = np.asarray(theta, dtype=float).reshape(-1)
        return TrailerTheta(
            q=theta[0:3].copy(),
            r=theta[3:5].copy(),
            w=theta[5:8].copy(),
            eta_circle=float(theta[8]),
            circle_r=float(theta[9]),
        )


def teacher_theta_default() -> TrailerTheta:
    return TrailerTheta(
        q=np.array([0.6, 0.6, 0.03], dtype=float),
        r=np.array([0.02, 0.04], dtype=float),
        w=np.array([4.0, 4.0, 0.8], dtype=float),
        eta_circle=100.0,
        circle_r=0.70,
    )


def student_theta_default() -> TrailerTheta:
    return TrailerTheta(
        q=np.array([0.34, 0.91, 0.12], dtype=float),
        r=np.array([0.07, 0.02], dtype=float),
        w=np.array([5.6, 3.3, 0.4], dtype=float),
        eta_circle=100.0,
        circle_r=0.50,
    )


def trailer_continuous_dynamics(x, u, L):
    theta = x[2]
    ux = u[0]
    uy = u[1]

    thetadot = (uy * cos(theta) - ux * sin(theta)) / L
    pxdot = ux + L * sin(theta) * thetadot
    pydot = uy - L * cos(theta) * thetadot
    return vertcat(pxdot, pydot, thetadot)


def euler_step(x, u, dt, L):
    return x + dt * trailer_continuous_dynamics(x, u, L)


def build_trailer_coc(scene: TrailerScene, gamma: float = 1e-2):
    X = SX.sym('X', 3)
    U = SX.sym('U', 2)
    aux = SX.sym('aux', 10)

    q1, q2, q3 = aux[0], aux[1], aux[2]
    r1, r2 = aux[3], aux[4]
    w1, w2, w3 = aux[5], aux[6], aux[7]
    eta_circle = aux[8]
    circle_r = aux[9]

    xref = DM(np.asarray(scene.xref, dtype=float).reshape(3))
    circle_c = DM(np.asarray(scene.circle_c, dtype=float).reshape(2))

    dyn = euler_step(X, U, scene.dt, scene.L)

    dx = X - xref
    z = X[0:2]
    dz = z - circle_c
    h = 1 - dot(dz, dz) / (circle_r ** 2)
    obs_penalty = 0.5 * eta_circle * (fmax(h, 0.0) ** 2)

    path_cost = (
        q1 * dx[0] ** 2
        + q2 * dx[1] ** 2
        + q3 * dx[2] ** 2
        + r1 * U[0] ** 2
        + r2 * U[1] ** 2
        + obs_penalty
    )
    final_cost = w1 * dx[0] ** 2 + w2 * dx[1] ** 2 + w3 * dx[2] ** 2

    umax_x, umax_y = scene.umax
    path_inequ = vertcat(
        U[0] - umax_x,
        -U[0] - umax_x,
        U[1] - umax_y,
        -U[1] - umax_y,
    )

    coc = SafePDP.COCsys()
    coc.setAuxvarVariable(aux)
    coc.setStateVariable(X)
    coc.setControlVariable(U)
    coc.setPathInequCstr(path_inequ)
    coc.setDyn(dyn)
    coc.setPathCost(path_cost)
    coc.setFinalCost(final_cost)
    coc.diffCPMP()
    coc.convert2BarrierOC(gamma=gamma)
    return coc


def build_default_scene(horizon: int = 40) -> TrailerScene:
    return TrailerScene(
        x0=np.array([0.0, 0.0, np.pi / 5.0], dtype=float),
        xref=np.array([3.5, 1.5, 0.0], dtype=float),
        circle_c=np.array([1.55, 1.45], dtype=float),
        horizon=int(horizon),
        dt=0.1,
        L=0.5,
        umax=(0.8, 0.8),
    )


def build_default_teacher_theta() -> TrailerTheta:
    return teacher_theta_default()


def build_default_student_theta() -> TrailerTheta:
    return student_theta_default()


def build_default_imitation_weights():
    wx = np.array([10.0, 10.0, 2.0], dtype=float)
    wu = np.array([2.0, 2.0], dtype=float)
    return wx, wu


def solve_teacher_demo(
    coc,
    scene: TrailerScene,
    theta_teacher: np.ndarray,
    wx: Optional[np.ndarray] = None,
    wu: Optional[np.ndarray] = None,
) -> Dict:
    if wx is None or wu is None:
        wx_default, wu_default = build_default_imitation_weights()
        if wx is None:
            wx = wx_default
        if wu is None:
            wu = wu_default

    sol = coc.ocSolver(
        horizon=scene.horizon,
        init_state=np.asarray(scene.x0, dtype=float).reshape(-1),
        auxvar_value=np.asarray(theta_teacher, dtype=float).reshape(-1),
    )
    return {
        'state_traj_opt': np.asarray(sol['state_traj_opt'], dtype=float),
        'control_traj_opt': np.asarray(sol['control_traj_opt'], dtype=float),
        'init_state': np.asarray(scene.x0, dtype=float).reshape(-1),
        'horizon': int(scene.horizon),
        'cost': float(np.asarray(sol['cost']).squeeze()),
        'wx': np.asarray(wx, dtype=float).reshape(-1),
        'wu': np.asarray(wu, dtype=float).reshape(-1),
    }


def imitation_l2_loss_and_grad(
    demo: Dict,
    traj: Dict,
    aux_sol: Dict,
    wx: Optional[np.ndarray] = None,
    wu: Optional[np.ndarray] = None,
):
    Xd = np.asarray(demo['state_traj_opt'], dtype=float)
    Ud = np.asarray(demo['control_traj_opt'], dtype=float)
    X = np.asarray(traj['state_traj_opt'], dtype=float)
    U = np.asarray(traj['control_traj_opt'], dtype=float)

    dX = [np.asarray(v, dtype=float) for v in aux_sol['state_traj_opt']]
    dU = [np.asarray(v, dtype=float) for v in aux_sol['control_traj_opt']]

    nx = X.shape[1]
    nu = U.shape[1]
    n_theta = dX[0].shape[1]

    if wx is None:
        wx = np.asarray(demo.get('wx', np.ones(nx)), dtype=float).reshape(nx)
    else:
        wx = np.asarray(wx, dtype=float).reshape(nx)

    if wu is None:
        wu = np.asarray(demo.get('wu', np.ones(nu)), dtype=float).reshape(nu)
    else:
        wu = np.asarray(wu, dtype=float).reshape(nu)

    loss = 0.0
    grad = np.zeros(n_theta, dtype=float)

    for t in range(U.shape[0]):
        eu = U[t] - Ud[t]
        loss += float(np.sum(wu * eu ** 2))
        grad += (2.0 * wu * eu) @ dU[t]

        ex_next = X[t + 1] - Xd[t + 1]
        loss += float(np.sum(wx * ex_next ** 2))
        grad += (2.0 * wx * ex_next) @ dX[t + 1]

    return float(loss), grad


def backward_once(coc, demo: Dict, theta: np.ndarray, strategy: str = 'COC'):
    theta = np.asarray(theta, dtype=float).reshape(-1)
    init_state = np.asarray(demo['init_state'], dtype=float).reshape(-1)
    horizon = int(demo['horizon'])

    if strategy.upper() == 'COC':
        clqr = SafePDP.EQCLQR()
        traj = coc.ocSolver(horizon=horizon, init_state=init_state, auxvar_value=theta)
        auxsys = coc.getAuxSys(opt_sol=traj, threshold=1e-5)
        clqr.auxsys2Eqctlqr(auxsys=auxsys)
        aux_sol = clqr.eqctlqrSolver(threshold=1e-5)
        return imitation_l2_loss_and_grad(demo, traj, aux_sol)

    if strategy.upper() == 'BARRIER':
        traj = coc.solveBarrierOC(horizon=horizon, init_state=init_state, auxvar_value=theta)
        aux_sol = coc.auxSysBarrierOC(opt_sol=traj)
        return imitation_l2_loss_and_grad(demo, traj, aux_sol)

    if strategy.upper() == 'HYBRID':
        traj_coc = coc.ocSolver(horizon=horizon, init_state=init_state, auxvar_value=theta)
        traj_bar = coc.solveBarrierOC(horizon=horizon, init_state=init_state, auxvar_value=theta)
        aux_sol = coc.auxSysBarrierOC(opt_sol=traj_bar)
        return imitation_l2_loss_and_grad(demo, traj_coc, aux_sol)

    raise ValueError(f'Unknown strategy: {strategy}')


def profile_call(fun, *args, **kwargs):
    t0 = time.perf_counter()
    out = fun(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    stats = {
        'elapsed_s': float(elapsed),
    }
    return out, stats


def profile_backward_call(fun, *args, **kwargs):
    return profile_call(fun, *args, **kwargs)