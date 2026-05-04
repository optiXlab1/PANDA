import os
import argparse
import inspect
import numpy as np

from SafePDP import SafePDP

from trailer_safe_pdp_common_euler_w import (
    TrailerScene,
    build_trailer_coc,
    imitation_l2_loss_and_grad,
    build_default_student_theta,
)

try:
    import psutil
except ImportError:
    psutil = None


DEFAULT_DEMO = './Demos/TrailerObstacle_RollingRecord_EulerW.npy'


def get_process_memory_mb():
    if psutil is None:
        return float("nan")
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2


def normalize_state_traj(X, horizon):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f'state trajectory must be 2D, got shape={X.shape}')
    if X.shape[0] == horizon + 1 and X.shape[1] == 3:
        return X
    if X.shape[0] == 3 and X.shape[1] == horizon + 1:
        return X.T
    raise ValueError(f'Unexpected state trajectory shape {X.shape} for horizon={horizon}')


def normalize_control_traj(U, horizon):
    U = np.asarray(U, dtype=float)
    if U.ndim != 2:
        raise ValueError(f'control trajectory must be 2D, got shape={U.shape}')
    if U.shape[0] == horizon and U.shape[1] == 2:
        return U
    if U.shape[0] == 2 and U.shape[1] == horizon:
        return U.T
    raise ValueError(f'Unexpected control trajectory shape {U.shape} for horizon={horizon}')


def make_scene_for_coc(scene_raw):
    kwargs = dict(
        xref=np.asarray(scene_raw['xref'], dtype=float),
        circle_c=np.asarray(scene_raw['circle_c'], dtype=float),
        horizon=int(scene_raw['horizon']),
        dt=float(scene_raw['dt']),
        L=float(scene_raw['L']),
        umax=tuple(np.asarray(scene_raw['umax'], dtype=float).reshape(-1).tolist()),
    )
    sig = inspect.signature(TrailerScene)
    if 'x0' in sig.parameters:
        kwargs['x0'] = np.asarray(scene_raw['x0'], dtype=float)
    return TrailerScene(**kwargs)


def record_item_to_demo(rec, wx, wu):
    scene = rec['scene']
    horizon = int(scene['horizon'])
    return {
        'init_state': np.asarray(scene['x0'], dtype=float).reshape(-1),
        'horizon': horizon,
        'state_traj_opt': normalize_state_traj(rec['X_teacher'], horizon),
        'control_traj_opt': normalize_control_traj(rec['U_teacher'], horizon),
        'wx': np.asarray(wx, dtype=float).reshape(-1),
        'wu': np.asarray(wu, dtype=float).reshape(-1),
    }


def load_batch_package(path: str, batch_stride: int):
    pkg = np.load(path, allow_pickle=True).item()
    teacher_theta = np.asarray(pkg['teacher_theta'], dtype=float).reshape(-1)
    record = list(pkg['record'])
    wx = np.asarray(pkg.get('wx'), dtype=float).reshape(-1)
    wu = np.asarray(pkg.get('wu'), dtype=float).reshape(-1)
    total_len = len(record)
    batch_idx = list(range(0, total_len, batch_stride))
    demos = [record_item_to_demo(record[i], wx, wu) for i in batch_idx]
    scene_raw = pkg.get('scene', pkg.get('scene0', record[0]['scene']))
    return {
        'scene_raw': scene_raw,
        'teacher_theta': teacher_theta,
        'wx': wx,
        'wu': wu,
        'demos': demos,
        'batch_idx': np.asarray(batch_idx, dtype=int),
        'total_len': total_len,
    }


def forward_only(coc, demo, theta, strategy='COC', ep=None, b=None):
    theta = np.asarray(theta, dtype=float).reshape(-1)
    init_state = np.asarray(demo['init_state'], dtype=float).reshape(-1)
    horizon = int(demo['horizon'])
    print(horizon)

    tag = f"[ep={ep:03d}, b={b:04d}]" if ep is not None and b is not None else "[forward]"

    mem_fwd_before = get_process_memory_mb()
    print(f"{tag} before forward | mem = {mem_fwd_before:.2f} MB")

    if strategy.upper() == 'COC':
        traj = coc.ocSolver(horizon=horizon, init_state=init_state, auxvar_value=theta)
    elif strategy.upper() == 'BARRIER':
        traj = coc.solveBarrierOC(horizon=horizon, init_state=init_state, auxvar_value=theta)
    elif strategy.upper() == 'HYBRID':
        traj = coc.ocSolver(horizon=horizon, init_state=init_state, auxvar_value=theta)
    else:
        raise ValueError(f'Unknown strategy: {strategy}')

    mem_fwd_after = get_process_memory_mb()
    print(
        f"{tag} after  forward | "
        f"mem = {mem_fwd_after:.2f} MB | "
        f"delta = {mem_fwd_after - mem_fwd_before:+.2f} MB"
    )

    return traj


def backward_only_from_traj(coc, demo, theta, traj_fwd, strategy='COC', ep=None, b=None):
    theta = np.asarray(theta, dtype=float).reshape(-1)
    init_state = np.asarray(demo['init_state'], dtype=float).reshape(-1)
    horizon = int(demo['horizon'])

    tag = f"[ep={ep:03d}, b={b:04d}]" if ep is not None and b is not None else "[backward]"

    if strategy.upper() == 'COC':
        clqr = SafePDP.EQCLQR()
        auxsys = coc.getAuxSys(opt_sol=traj_fwd, threshold=1e-5)
        clqr.auxsys2Eqctlqr(auxsys=auxsys)

        mem_bwd_before = get_process_memory_mb()
        print(f"{tag} before aux_sol | mem = {mem_bwd_before:.2f} MB")

        aux_sol = clqr.eqctlqrSolver(threshold=1e-5)

        mem_bwd_after = get_process_memory_mb()
        print(
            f"{tag} after  aux_sol | "
            f"mem = {mem_bwd_after:.2f} MB | "
            f"delta = {mem_bwd_after - mem_bwd_before:+.2f} MB"
        )

        return imitation_l2_loss_and_grad(demo, traj_fwd, aux_sol)

    if strategy.upper() == 'BARRIER':
        mem_bwd_before = get_process_memory_mb()
        print(f"{tag} before aux_sol | mem = {mem_bwd_before:.2f} MB")

        aux_sol = coc.auxSysBarrierOC(opt_sol=traj_fwd)

        mem_bwd_after = get_process_memory_mb()
        print(
            f"{tag} after  aux_sol | "
            f"mem = {mem_bwd_after:.2f} MB | "
            f"delta = {mem_bwd_after - mem_bwd_before:+.2f} MB"
        )

        return imitation_l2_loss_and_grad(demo, traj_fwd, aux_sol)

    if strategy.upper() == 'HYBRID':
        traj_bar = coc.solveBarrierOC(
            horizon=horizon,
            init_state=init_state,
            auxvar_value=theta,
        )

        mem_bwd_before = get_process_memory_mb()
        print(f"{tag} before aux_sol | mem = {mem_bwd_before:.2f} MB")

        aux_sol = coc.auxSysBarrierOC(opt_sol=traj_bar)

        mem_bwd_after = get_process_memory_mb()
        print(
            f"{tag} after  aux_sol | "
            f"mem = {mem_bwd_after:.2f} MB | "
            f"delta = {mem_bwd_after - mem_bwd_before:+.2f} MB"
        )

        return imitation_l2_loss_and_grad(demo, traj_fwd, aux_sol)

    raise ValueError(f'Unknown strategy: {strategy}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_path', type=str, default=DEFAULT_DEMO)
    parser.add_argument('--strategy', type=str, default='BARRIER', choices=['COC', 'BARRIER', 'HYBRID'])
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_stride', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.008)
    args = parser.parse_args()

    if psutil is None:
        print("Warning: psutil is not installed. Memory output will be NaN. Install it by: pip install psutil")

    pkg = load_batch_package(args.demo_path, batch_stride=args.batch_stride)
    scene_raw = pkg['scene_raw']
    teacher_theta = pkg['teacher_theta']
    demos = pkg['demos']
    batch_idx = pkg['batch_idx']
    total_len = pkg['total_len']
    batch_size = len(demos)

    print(f'Total record length = {total_len}')
    print(f'Batch stride        = {args.batch_stride}')
    print(f'Batch size          = {batch_size}')
    print(f'Batch indices       = {batch_idx.tolist()}')

    scene = make_scene_for_coc(scene_raw)
    coc = build_trailer_coc(scene, gamma=args.gamma)

    theta = build_default_student_theta().encode()
    theta_init = theta.copy()

    if theta.shape != teacher_theta.shape:
        raise RuntimeError(
            f'theta_init shape {theta.shape} does not match teacher_theta shape {teacher_theta.shape}'
        )

    for ep in range(args.epochs):
        grad_acc = np.zeros_like(theta)
        loss_acc = 0.0

        for b, demo in enumerate(demos):
            traj_fwd = forward_only(
                coc,
                demo,
                theta,
                strategy=args.strategy,
                ep=ep + 1,
                b=b + 1,
            )

            loss_val, grad_val = backward_only_from_traj(
                coc,
                demo,
                theta,
                traj_fwd,
                strategy=args.strategy,
                ep=ep + 1,
                b=b + 1,
            )

            grad_val = np.asarray(grad_val, dtype=float).reshape(-1)
            grad_acc += grad_val
            loss_acc += float(loss_val)

        grad_acc = grad_acc / batch_size
        loss_acc = loss_acc / batch_size

        theta = theta - args.lr * grad_acc

        print(f"ep={ep + 1:03d} | L={loss_acc:.8f}")
        print(f"  theta = {np.array2string(theta, precision=6, separator=', ')}")

    print('\nMemory-only SafePDP test finished. No result files were written.')
    print('initial theta =', theta_init)
    print('final theta   =', theta)


if __name__ == '__main__':
    main()