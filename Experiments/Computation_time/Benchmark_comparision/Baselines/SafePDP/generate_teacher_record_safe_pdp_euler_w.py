import os
import time
import copy
import argparse
import numpy as np
import matplotlib.pyplot as plt

from trailer_safe_pdp_common_euler_w import (
    build_default_scene,
    build_default_teacher_theta,
    build_default_imitation_weights,
    build_trailer_coc,
    solve_teacher_demo,
)

OUT_DIR = './Demos'
OUT_FILE = os.path.join(OUT_DIR, 'TrailerObstacle_RollingRecord_EulerW.npy')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type=int, default=40)
    parser.add_argument('--gamma', type=float, default=1e-2)
    parser.add_argument('--Tsim', type=int, default=50)
    parser.add_argument('--out_path', type=str, default=OUT_FILE)
    args = parser.parse_args()

    out_dir = os.path.dirname(args.out_path)
    if out_dir != '':
        os.makedirs(out_dir, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    scene0 = build_default_scene(horizon=args.horizon)
    teacher = build_default_teacher_theta()
    wx, wu = build_default_imitation_weights()

    teacher_theta = teacher.encode()
    coc = build_trailer_coc(scene0, gamma=args.gamma)

    Tsim = int(args.Tsim)
    record = []
    demos = []
    xcur = np.asarray(scene0.x0, dtype=float).copy()

    print('========== rolling teacher generation (SafePDP / Euler / w) ==========' )
    print(f'x0       = {scene0.x0}')
    print(f'xref     = {scene0.xref}')
    print(f'circle_c = {scene0.circle_c}')
    print(f'teacher q = {teacher.q}')
    print(f'teacher r = {teacher.r}')
    print(f'teacher w = {teacher.w}')
    print(f'circle_r = {teacher.circle_r}')
    print(f'horizon  = {scene0.horizon}')
    print(f'Tsim     = {Tsim}')
    print('')

    for t in range(Tsim):
        scene_t = copy.deepcopy(scene0)
        scene_t.x0 = np.asarray(xcur, dtype=float).copy()

        t0 = time.perf_counter()
        demo_t = solve_teacher_demo(coc, scene_t, teacher_theta, wx=wx, wu=wu)
        solve_time = time.perf_counter() - t0

        X_teacher = np.asarray(demo_t['state_traj_opt'], dtype=float)
        U_teacher = np.asarray(demo_t['control_traj_opt'], dtype=float)

        if X_teacher.shape[0] < 2:
            raise RuntimeError(
                f'Invalid teacher trajectory at step {t}: '
                f'state_traj_opt should have at least 2 states, got shape={X_teacher.shape}'
            )
        if U_teacher.shape[0] < 1:
            raise RuntimeError(
                f'Invalid teacher control at step {t}: '
                f'control_traj_opt should have at least 1 control, got shape={U_teacher.shape}'
            )

        u_apply = U_teacher[0].copy()
        x_next = X_teacher[1].copy()

        demo_t['t'] = t + 1
        demo_t['scene'] = {
            'x0': scene_t.x0.copy(),
            'xref': scene_t.xref.copy(),
            'circle_c': scene_t.circle_c.copy(),
            'horizon': int(scene_t.horizon),
            'dt': float(scene_t.dt),
            'L': float(scene_t.L),
            'umax': np.array(scene_t.umax, dtype=float),
        }
        demo_t['u_apply'] = u_apply.copy()
        demo_t['x_next'] = x_next.copy()
        demo_t['solve_time_s'] = float(solve_time)

        demos.append(demo_t)

        record_t = {
            't': t + 1,
            'scene': demo_t['scene'],
            'X_teacher': X_teacher.copy(),
            'U_teacher': U_teacher.copy(),
            'u_apply': u_apply.copy(),
            'x_next': x_next.copy(),
            'cost': float(demo_t['cost']),
            'solve_time_s': float(solve_time),
            'wx': np.asarray(wx, dtype=float).copy(),
            'wu': np.asarray(wu, dtype=float).copy(),
        }
        record.append(record_t)
        xcur = x_next.copy()

        print(
            f'record {t+1:03d}/{Tsim:03d} | '
            f'cost={demo_t["cost"]:.6f} | '
            f'time={solve_time:.4f}s | '
            f'x=[{xcur[0]:.3f}, {xcur[1]:.3f}, {xcur[2]:.3f}]'
        )

    X_cl = np.zeros((Tsim + 1, 3), dtype=float)
    X_cl[0] = np.asarray(scene0.x0, dtype=float)
    for t in range(Tsim):
        X_cl[t + 1] = np.asarray(record[t]['x_next'], dtype=float)

    save_dict = {
        'scene': {
            'x0': scene0.x0,
            'xref': scene0.xref,
            'circle_c': scene0.circle_c,
            'horizon': scene0.horizon,
            'dt': scene0.dt,
            'L': scene0.L,
            'umax': np.array(scene0.umax, dtype=float),
        },
        'scene0': {
            'x0': scene0.x0,
            'xref': scene0.xref,
            'circle_c': scene0.circle_c,
            'horizon': scene0.horizon,
            'dt': scene0.dt,
            'L': scene0.L,
            'umax': np.array(scene0.umax, dtype=float),
        },
        'teacher_theta': teacher_theta,
        'teacher_theta_named': {
            'q': teacher.q,
            'r': teacher.r,
            'w': teacher.w,
            'eta_circle': teacher.eta_circle,
            'circle_r': teacher.circle_r,
        },
        'wx': wx,
        'wu': wu,
        'Tsim': Tsim,
        'record': record,
        'demos': demos,
        'X_cl': X_cl,
        'discretization': 'euler',
    }
    np.save(args.out_path, save_dict)

    print('\n========== saved ==========')
    print(f'Saved rolling record to: {args.out_path}')
    print(f'Mean teacher cost      : {np.mean([r["cost"] for r in record]):.6f}')
    print(f'Mean solve time        : {np.mean([r["solve_time_s"] for r in record]):.6f} s')

    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    th = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(
        scene0.circle_c[0] + teacher.circle_r * np.cos(th),
        scene0.circle_c[1] + teacher.circle_r * np.sin(th),
        'g-', linewidth=2.0, label=f'circle obstacle (r={teacher.circle_r:.2f})'
    )
    ax.plot(X_cl[:, 0], X_cl[:, 1], 'b-', linewidth=2.3, label='closed-loop trajectory')
    ax.plot(scene0.x0[0], scene0.x0[1], 'ko', markersize=5, markerfacecolor='k', label='start')
    ax.plot(scene0.xref[0], scene0.xref[1], 'rx', markersize=9, mew=2, label='goal')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    ax.set_xlabel('p_x')
    ax.set_ylabel('p_y')
    ax.set_title('Rolling teacher record for trailer obstacle avoidance')
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    preview_path = os.path.splitext(args.out_path)[0] + '_preview.png'
    fig.savefig(preview_path, bbox_inches='tight')
    print(f'Saved preview figure to: {preview_path}')
    plt.show()


if __name__ == '__main__':
    main()
