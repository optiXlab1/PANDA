import os
import csv
import argparse
import inspect
import numpy as np

from SafePDP import SafePDP

from trailer_safe_pdp_common_euler_w import (
    TrailerScene,
    build_trailer_coc,
    imitation_l2_loss_and_grad,
    profile_call,
    build_default_student_theta,
)

DEFAULT_DEMO = './Demos/TrailerObstacle_RollingRecord_EulerW.npy'
DEFAULT_OUT = './Results/TrailerSafePDP_EulerW_batch_benchmark.npy'


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


def forward_only(coc, demo, theta, strategy='COC'):
    theta = np.asarray(theta, dtype=float).reshape(-1)
    init_state = np.asarray(demo['init_state'], dtype=float).reshape(-1)
    horizon = int(demo['horizon'])
    if strategy.upper() == 'COC':
        return coc.ocSolver(horizon=horizon, init_state=init_state, auxvar_value=theta)
    if strategy.upper() == 'BARRIER':
        return coc.solveBarrierOC(horizon=horizon, init_state=init_state, auxvar_value=theta)
    if strategy.upper() == 'HYBRID':
        return coc.ocSolver(horizon=horizon, init_state=init_state, auxvar_value=theta)
    raise ValueError(f'Unknown strategy: {strategy}')


def backward_only_from_traj(coc, demo, theta, traj_fwd, strategy='COC'):
    theta = np.asarray(theta, dtype=float).reshape(-1)
    init_state = np.asarray(demo['init_state'], dtype=float).reshape(-1)
    horizon = int(demo['horizon'])

    if strategy.upper() == 'COC':
        clqr = SafePDP.EQCLQR()
        auxsys = coc.getAuxSys(opt_sol=traj_fwd, threshold=1e-4)
        clqr.auxsys2Eqctlqr(auxsys=auxsys)
        aux_sol = clqr.eqctlqrSolver(threshold=1e-4)
        return imitation_l2_loss_and_grad(demo, traj_fwd, aux_sol)

    if strategy.upper() == 'BARRIER':
        aux_sol = coc.auxSysBarrierOC(opt_sol=traj_fwd)
        return imitation_l2_loss_and_grad(demo, traj_fwd, aux_sol)

    if strategy.upper() == 'HYBRID':
        traj_bar = coc.solveBarrierOC(
            horizon=horizon,
            init_state=init_state,
            auxvar_value=theta,
        )
        aux_sol = coc.auxSysBarrierOC(opt_sol=traj_bar)
        return imitation_l2_loss_and_grad(demo, traj_fwd, aux_sol)

    raise ValueError(f'Unknown strategy: {strategy}')


def save_csv(csv_path, records):
    fieldnames = [
        'epoch', 'loss',
        'forward_total_time', 'backward_total_time',
        'forward_mean_time', 'backward_mean_time',
        'grad_norm', 'theta_step'
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_path', type=str, default=DEFAULT_DEMO)
    parser.add_argument('--out_path', type=str, default=DEFAULT_OUT)
    parser.add_argument('--strategy', type=str, default='BARRIER', choices=['COC', 'BARRIER', 'HYBRID'])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_stride', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.008)
    parser.add_argument('--warmup', action='store_true')
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out_path)
    if out_dir != '':
        os.makedirs(out_dir, exist_ok=True)

    pkg = load_batch_package(args.demo_path, batch_stride=args.batch_stride)
    scene_raw = pkg['scene_raw']
    teacher_theta = pkg['teacher_theta']
    wx = pkg['wx']
    wu = pkg['wu']
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

    if args.warmup and batch_size > 0:
        traj0, _ = profile_call(
            forward_only, coc, demos[0], theta, strategy=args.strategy
        )
        _ = profile_call(
            backward_only_from_traj, coc, demos[0], theta, traj0, strategy=args.strategy
        )

    loss_hist = []
    t_forward_total_hist = []
    t_forward_mean_hist = []
    t_backward_total_hist = []
    t_backward_mean_hist = []

    grad_norm_hist = []
    theta_step_hist = []
    theta_hist = []
    csv_rows = []

    for ep in range(args.epochs):
        grad_acc = np.zeros_like(theta)
        loss_acc = 0.0

        t_forward_samples = []
        t_backward_samples = []

        for demo in demos:
            traj_fwd, stat_fwd = profile_call(
                forward_only,
                coc, demo, theta,
                strategy=args.strategy
            )

            (loss_val, grad_val), stat_bwd = profile_call(
                backward_only_from_traj,
                coc, demo, theta, traj_fwd,
                strategy=args.strategy
            )

            grad_val = np.asarray(grad_val, dtype=float).reshape(-1)
            grad_acc += grad_val
            loss_acc += float(loss_val)

            t_forward_samples.append(float(stat_fwd['elapsed_s']))
            t_backward_samples.append(float(stat_bwd['elapsed_s']))

        grad_acc = grad_acc / batch_size
        loss_acc = loss_acc / batch_size
        grad_norm = float(np.linalg.norm(grad_acc))
        theta_old = theta.copy()
        theta = theta - args.lr * grad_acc
        theta_step = float(np.linalg.norm(theta - theta_old))

        theta_hist.append(theta_old.copy())
        loss_hist.append(float(loss_acc))

        fwd_total = float(np.sum(t_forward_samples))
        fwd_mean = float(np.mean(t_forward_samples))
        bwd_total = float(np.sum(t_backward_samples))
        bwd_mean = float(np.mean(t_backward_samples))

        t_forward_total_hist.append(fwd_total)
        t_forward_mean_hist.append(fwd_mean)
        t_backward_total_hist.append(bwd_total)
        t_backward_mean_hist.append(bwd_mean)

        grad_norm_hist.append(grad_norm)
        theta_step_hist.append(theta_step)

        csv_rows.append({
            'epoch': ep + 1,
            'loss': float(loss_acc),
            'forward_total_time': fwd_total,
            'backward_total_time': bwd_total,
            'forward_mean_time': fwd_mean,
            'backward_mean_time': bwd_mean,
            'grad_norm': grad_norm,
            'theta_step': theta_step,
        })

        print(
            f"ep={ep:02d} | L={loss_acc:.8f} | "
            f"fwd_total={fwd_total:.4f}s | "
            f"fwd_mean={fwd_mean:.4f}s | "
            f"bwd_total={bwd_total:.4f}s | "
            f"bwd_mean={bwd_mean:.4f}s | "
            f"||grad||={grad_norm:.3e} | "
            f"||dtheta||={theta_step:.3e}"
        )

    result = {
        'strategy': args.strategy,
        'demo_path': args.demo_path,
        'batch_stride': args.batch_stride,
        'batch_idx': batch_idx,
        'batch_size': batch_size,
        'total_record_length': total_len,
        'theta_init': theta_init,
        'theta_final': theta,
        'teacher_theta': teacher_theta,
        'theta_hist': np.asarray(theta_hist, dtype=float),
        'loss_hist': np.asarray(loss_hist, dtype=float),
        't_forward_total_hist': np.asarray(t_forward_total_hist, dtype=float),
        't_forward_mean_hist': np.asarray(t_forward_mean_hist, dtype=float),
        't_backward_total_hist': np.asarray(t_backward_total_hist, dtype=float),
        't_backward_mean_hist': np.asarray(t_backward_mean_hist, dtype=float),
        'grad_norm_hist': np.asarray(grad_norm_hist, dtype=float),
        'theta_step_hist': np.asarray(theta_step_hist, dtype=float),
        'scene': scene_raw,
        'wx': wx,
        'wu': wu,
        'epochs': args.epochs,
        'lr': args.lr,
        'gamma': args.gamma,
        'discretization': 'euler',
    }
    np.save(args.out_path, result)

    csv_path = os.path.splitext(args.out_path)[0] + '_training_log.csv'
    save_csv(csv_path, csv_rows)

    print('\n========== summary ==========')
    print(f'strategy                  : {args.strategy}')
    print(f'mean forward total / ep   : {np.mean(t_forward_total_hist):.6f} s')
    print(f'mean forward mean / samp  : {np.mean(t_forward_mean_hist):.6f} s')
    print(f'mean backward total / ep  : {np.mean(t_backward_total_hist):.6f} s')
    print(f'mean backward mean / samp : {np.mean(t_backward_mean_hist):.6f} s')
    print(f'final loss                : {loss_hist[-1]:.8f}')
    print(f'saved npy                 : {args.out_path}')
    print(f'saved csv                 : {csv_path}')


if __name__ == '__main__':
    main()