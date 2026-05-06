import csv
import numpy as np
import time
from casadi import *
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


DT = 0.1
L_TRAILER = 0.5

TEACHER_THETA = np.array([
    0.6, 0.6, 0.03,
    0.02, 0.04,
    4.0, 4.0, 0.8,
    100.0, 0.70,
], dtype=float)

STUDENT_THETA_INIT = np.array([
    0.34, 0.91, 0.12,
    0.07, 0.02,
    5.6, 3.3, 0.4,
    100.0, 0.50,
], dtype=float)

# full-sequence imitation weights
WX_DEFAULT = np.array([10.0, 10.0, 2.0], dtype=float)
WU_DEFAULT = np.array([2.0, 2.0], dtype=float)

X0_DEFAULT = np.array([0.0, 0.0, np.pi / 5], dtype=float)
XREF_DEFAULT = np.array([3.5, 1.5, 0.0], dtype=float)
CIRCLE_C_DEFAULT = np.array([1.55, 1.45], dtype=float)


def trailer_model(dt=DT, L=L_TRAILER):
    model = AcadosModel()
    model.name = "trailer_two_solver_sens_euler_w_fullseq"

    px = MX.sym("px")
    py = MX.sym("py")
    theta = MX.sym("theta")
    x = vertcat(px, py, theta)

    ux = MX.sym("ux")
    uy = MX.sym("uy")
    u = vertcat(ux, uy)

    p = MX.sym("p", 5)
    xref = p[0:3]
    circle_c = p[3:5]

    xdot = MX.sym("xdot", 3)

    theta_dot = (uy * cos(theta) - ux * sin(theta)) / L
    px_dot = ux + L * sin(theta) * theta_dot
    py_dot = uy - L * cos(theta) * theta_dot

    f_expl = vertcat(px_dot, py_dot, theta_dot)

    model.x = x
    model.xdot = xdot
    model.u = u
    model.p = p
    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot - f_expl
    model.disc_dyn_expr = x + dt * f_expl

    p_global = MX.sym("p_global", 10)
    q1, q2, q3, r1, r2, w1, w2, w3, eta_circle, circle_r = [
        p_global[i] for i in range(10)
    ]
    model.p_global = p_global

    dx = x - xref
    z = x[0:2]

    h_circle = 1 - dot(z - circle_c, z - circle_c) / (circle_r ** 2)
    circle_penalty = 0.5 * eta_circle * fmax(h_circle, 0) ** 2

    stage_cost = (
        q1 * dx[0] ** 2
        + q2 * dx[1] ** 2
        + q3 * dx[2] ** 2
        + r1 * u[0] ** 2
        + r2 * u[1] ** 2
        + circle_penalty
    )

    terminal_cost = (
        w1 * dx[0] ** 2
        + w2 * dx[1] ** 2
        + w3 * dx[2] ** 2
    )

    model.cost_expr_ext_cost = stage_cost
    model.cost_expr_ext_cost_e = terminal_cost

    return model


def create_ocp_forward(N, p_global_values=None):
    model = trailer_model()

    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = N

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    ocp.parameter_values = np.concatenate([XREF_DEFAULT, CIRCLE_C_DEFAULT]).astype(float)

    if p_global_values is None:
        p_global_values = TEACHER_THETA.copy()

    ocp.p_global_values = np.asarray(p_global_values, dtype=float)

    ocp.constraints.lbu = np.array([-0.8, -0.8], dtype=float)
    ocp.constraints.ubu = np.array([0.8, 0.8], dtype=float)
    ocp.constraints.idxbu = np.array([0, 1], dtype=int)
    ocp.constraints.x0 = X0_DEFAULT.copy()

    ocp.solver_options.tf = N * DT
    ocp.solver_options.N_horizon = N
    ocp.solver_options.integrator_type = "DISCRETE"

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_ric_alg = 0
    ocp.solver_options.qp_solver_cond_ric_alg = 0

    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    ocp.solver_options.regularize_method = "MIRROR"

    ocp.solver_options.nlp_solver_max_iter = 2000
    ocp.solver_options.tol = 1e-4
    ocp.solver_options.print_level = 0

    return ocp


def create_ocp_sensitivity(N, p_global_values=None):
    model = trailer_model()

    ocp = AcadosOcp()
    ocp.model = model
    ocp.dims.N = N

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    ocp.parameter_values = np.concatenate([XREF_DEFAULT, CIRCLE_C_DEFAULT]).astype(float)

    if p_global_values is None:
        p_global_values = TEACHER_THETA.copy()

    ocp.p_global_values = np.asarray(p_global_values, dtype=float)

    ocp.constraints.lbu = np.array([-0.8, -0.8], dtype=float)
    ocp.constraints.ubu = np.array([0.8, 0.8], dtype=float)
    ocp.constraints.idxbu = np.array([0, 1], dtype=int)
    ocp.constraints.x0 = X0_DEFAULT.copy()

    ocp.solver_options.tf = N * DT
    ocp.solver_options.N_horizon = N
    ocp.solver_options.integrator_type = "DISCRETE"

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_ric_alg = 0
    ocp.solver_options.qp_solver_cond_ric_alg = 0

    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.hessian_approx = "EXACT"
    ocp.solver_options.with_solution_sens_wrt_params = True
    ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    ocp.solver_options.regularize_method = "NO_REGULARIZE"

    ocp.solver_options.nlp_solver_max_iter = 2000
    ocp.solver_options.tol = 1e-4
    ocp.solver_options.print_level = 0

    return ocp


def create_two_solvers(
    N,
    build=True,
    generate=True,
    verbose=False,
    forward_json="trailer_forward_euler_w_fullseq.json",
    sens_json="sensitivity_solver_euler_w_fullseq.json",
):
    ocp_fwd = create_ocp_forward(N)

    ocp_solver = AcadosOcpSolver(
        ocp_fwd,
        build=build,
        generate=generate,
        json_file=forward_json,
        verbose=verbose,
    )

    ocp_sens = create_ocp_sensitivity(N)
    ocp_sens.model.name = "sensitivity_solver_euler_w_fullseq"
    ocp_sens.code_export_directory = f"c_generated_code_{ocp_sens.model.name}"

    sensitivity_solver = AcadosOcpSolver(
        ocp_sens,
        build=build,
        generate=generate,
        json_file=sens_json,
        verbose=verbose,
    )

    return ocp_solver, sensitivity_solver


def euler_step(x, u, dt=DT, L=L_TRAILER):
    px, py, th = x
    ux, uy = u

    th_dot = (uy * np.cos(th) - ux * np.sin(th)) / L
    px_dot = ux + L * np.sin(th) * th_dot
    py_dot = uy - L * np.cos(th) * th_dot

    return x + dt * np.array([px_dot, py_dot, th_dot], dtype=float)


def set_online_data(solver, N, x0, xref, circle_c, theta_global):
    p_stage = np.concatenate([xref, circle_c])

    for k in range(N + 1):
        solver.set(k, "p", p_stage)

    solver.set(0, "lbx", x0)
    solver.set(0, "ubx", x0)

    solver.set_p_global_and_precompute_dependencies(
        np.asarray(theta_global, dtype=float)
    )


def cold_start(solver, N, x0, xref):
    for k in range(N + 1):
        alpha = k / N
        x_guess = (1 - alpha) * x0 + alpha * xref
        solver.set(k, "x", x_guess)

    for k in range(N):
        solver.set(k, "u", np.zeros(2))


def get_iterate_obj(solver):
    if hasattr(solver, "get_flat_iterate"):
        return solver.get_flat_iterate()

    return solver.store_iterate_to_flat_obj()


def load_iterate_obj(solver, iterate_obj):
    solver.load_iterate_from_flat_obj(iterate_obj)


def generate_target_record(
    Tsim=50,
    N=40,
    x0=X0_DEFAULT.copy(),
    xref=XREF_DEFAULT.copy(),
    circle_c=CIRCLE_C_DEFAULT.copy(),
    theta=TEACHER_THETA,
    save_path="target_record_teacher_euler_w_fullseq.npz",
):
    forward_solver, _ = create_two_solvers(
        N=N,
        build=True,
        generate=True,
        verbose=False,
        forward_json="trailer_target_forward_euler_w_fullseq.json",
        sens_json="trailer_target_sens_unused_euler_w_fullseq.json",
    )

    X_hist = np.zeros((Tsim + 1, 3))
    U_hist = np.zeros((Tsim, 2))
    X_opt_hist = np.zeros((Tsim, N + 1, 3))
    U_opt_hist = np.zeros((Tsim, N, 2))
    status_hist = np.zeros(Tsim, dtype=int)

    x_cur = x0.copy()
    X_hist[0] = x_cur
    theta = np.asarray(theta, dtype=float)

    for t in range(Tsim):
        set_online_data(forward_solver, N, x_cur, xref, circle_c, theta)
        cold_start(forward_solver, N, x_cur, xref)

        status = forward_solver.solve()
        status_hist[t] = status

        if status != 0:
            print(f"[target step {t}] forward solve failed, status = {status}")

        X_opt = forward_solver.get_flat("x").reshape(N + 1, 3)
        U_opt = forward_solver.get_flat("u").reshape(N, 2)

        X_opt_hist[t] = X_opt
        U_opt_hist[t] = U_opt

        u_apply = U_opt[0]
        U_hist[t] = u_apply

        x_cur = euler_step(x_cur, u_apply)
        X_hist[t + 1] = x_cur

    rec = {
        "X_hist": X_hist,
        "U_hist": U_hist,
        "X_opt_hist": X_opt_hist,
        "U_opt_hist": U_opt_hist,
        "status_hist": status_hist,
        "x0": x0.copy(),
        "xref": xref.copy(),
        "circle_c": circle_c.copy(),
        "theta": theta.copy(),
        "Tsim": Tsim,
        "N": N,
    }

    np.savez(save_path, **rec)
    return rec


def solve_one_step_and_prepare_sens(
    ocp_solver,
    sensitivity_solver,
    x_cur,
    xref,
    circle_c,
    theta,
    N,
):
    set_online_data(ocp_solver, N, x_cur, xref, circle_c, theta)
    set_online_data(sensitivity_solver, N, x_cur, xref, circle_c, theta)

    cold_start(ocp_solver, N, x_cur, xref)

    status = ocp_solver.solve()

    if status != 0:
        print(f"forward solver failed with status {status}")

    X_opt = ocp_solver.get_flat("x").reshape(N + 1, 3)
    U_opt = ocp_solver.get_flat("u").reshape(N, 2)

    iterate = get_iterate_obj(ocp_solver)
    load_iterate_obj(sensitivity_solver, iterate)
    sensitivity_solver.setup_qp_matrices_and_factorize()

    return status, X_opt, U_opt


def compute_grad_theta_from_sensitivity(
    sensitivity_solver,
    X_opt,
    U_opt,
    X_tar,
    U_tar,
    N,
    wx=WX_DEFAULT,
    wu=WU_DEFAULT,
):
    seed_x = []
    seed_u = []

    for k in range(N):
        gu = (2.0 * wu * (U_opt[k] - U_tar[k])).reshape(-1, 1)
        gx = (2.0 * wx * (X_opt[k + 1] - X_tar[k + 1])).reshape(-1, 1)

        seed_u.append((k, gu))
        seed_x.append((k + 1, gx))

    grad_theta = sensitivity_solver.eval_adjoint_solution_sensitivity(
        seed_x=seed_x,
        seed_u=seed_u,
        with_respect_to="p_global",
        sanity_checks=True,
    )

    loss = 0.0

    for k in range(N):
        loss += np.sum((U_opt[k] - U_tar[k]) ** 2 * wu)
        loss += np.sum((X_opt[k + 1] - X_tar[k + 1]) ** 2 * wx)

    return float(loss), np.asarray(grad_theta).reshape(-1)


def project_theta(theta):
    theta_new = theta.copy()

    for i in range(9):
        theta_new[i] = max(theta_new[i], 1e-6)

    theta_new[9] = max(theta_new[9], 0.05)

    return theta_new


def save_training_logs(
    history,
    summary,
    txt_file="training_log_euler_w_fullseq.txt",
    csv_file="training_history_euler_w_fullseq.csv",
    npz_file="training_summary_euler_w_fullseq.npz",
):
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "epoch",
            "loss_mean",
            "loss_std",
            "grad_norm_mean",
            "grad_norm_std",
            "forward_total_time_s",
            "backward_total_time_s",
            "forward_time_mean_s",
            "backward_time_mean_s",
            "theta_0_q1",
            "theta_1_q2",
            "theta_2_q3",
            "theta_3_r1",
            "theta_4_r2",
            "theta_5_w1",
            "theta_6_w2",
            "theta_7_w3",
            "theta_8_eta_circle",
            "theta_9_circle_r",
        ])

        for row in history:
            writer.writerow([
                row["epoch"],
                row["loss_mean"],
                row["loss_std"],
                row["grad_norm_mean"],
                row["grad_norm_std"],
                row["forward_total_time_s"],
                row["backward_total_time_s"],
                row["forward_time_mean_s"],
                row["backward_time_mean_s"],
                *row["theta"].tolist(),
            ])

    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("Training Summary\n")
        f.write("================\n\n")

        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

        f.write("\nPer-Epoch History\n")
        f.write("=================\n\n")

        for row in history:
            f.write(
                f"Epoch {row['epoch']:03d} | "
                f"loss_mean={row['loss_mean']:.6f}, "
                f"loss_std={row['loss_std']:.6f}, "
                f"grad_norm_mean={row['grad_norm_mean']:.6f}, "
                f"grad_norm_std={row['grad_norm_std']:.6f}, "
                f"fwd_t={row['forward_time_mean_s']:.6f}s, "
                f"bwd_t={row['backward_time_mean_s']:.6f}s, "
                f"theta={np.array2string(row['theta'], precision=6, separator=', ')}\n"
            )

    np.savez(
        npz_file,
        history=np.array(history, dtype=object),
        summary=np.array(summary, dtype=object),
    )


def train_two_solver(
    target_rec,
    theta_train_init,
    epochs=50,
    lr=1e-3,
    wx=WX_DEFAULT,
    wu=WU_DEFAULT,
):
    Tsim = int(target_rec["Tsim"])
    N = int(target_rec["N"])

    x0 = target_rec["x0"].copy()
    xref = target_rec["xref"].copy()
    circle_c = target_rec["circle_c"].copy()

    X_tar_all = target_rec["X_opt_hist"]
    U_tar_all = target_rec["U_opt_hist"]

    theta = theta_train_init.copy()

    ocp_solver, sensitivity_solver = create_two_solvers(
        N=N,
        build=True,
        generate=True,
        verbose=False,
        forward_json="trailer_bench_forward_euler_w_fullseq.json",
        sens_json="sensitivity_solver_euler_w_fullseq.json",
    )

    history = []

    for ep in range(epochs):
        x_cur = x0.copy()

        epoch_fwd_t = 0.0
        epoch_bwd_t = 0.0

        losses = []
        grad_norms = []
        grad_sum = np.zeros_like(theta)

        for t in range(Tsim):
            tf0 = time.perf_counter()

            status, X_opt, U_opt = solve_one_step_and_prepare_sens(
                ocp_solver=ocp_solver,
                sensitivity_solver=sensitivity_solver,
                x_cur=x_cur,
                xref=xref,
                circle_c=circle_c,
                theta=theta,
                N=N,
            )

            tf1 = time.perf_counter()
            epoch_fwd_t += tf1 - tf0

            X_tar = X_tar_all[t]
            U_tar = U_tar_all[t]

            tb0 = time.perf_counter()

            loss, grad_theta = compute_grad_theta_from_sensitivity(
                sensitivity_solver=sensitivity_solver,
                X_opt=X_opt,
                U_opt=U_opt,
                X_tar=X_tar,
                U_tar=U_tar,
                N=N,
                wx=wx,
                wu=wu,
            )

            tb1 = time.perf_counter()
            epoch_bwd_t += tb1 - tb0

            losses.append(loss)
            grad_norms.append(np.linalg.norm(grad_theta))
            grad_sum += grad_theta

            u_apply = U_opt[0]
            x_cur = euler_step(x_cur, u_apply)

        grad_mean = grad_sum / Tsim
        theta = theta - lr * grad_mean
        theta = project_theta(theta)

        epoch_info = {
            "epoch": ep + 1,
            "loss_mean": float(np.mean(losses)),
            "loss_std": float(np.std(losses)),
            "grad_norm_mean": float(np.mean(grad_norms)),
            "grad_norm_std": float(np.std(grad_norms)),
            "forward_total_time_s": float(epoch_fwd_t),
            "backward_total_time_s": float(epoch_bwd_t),
            "forward_time_mean_s": float(epoch_fwd_t / Tsim),
            "backward_time_mean_s": float(epoch_bwd_t / Tsim),
            "theta": theta.copy(),
        }

        history.append(epoch_info)

        print(
            f"Epoch {ep + 1:03d}/{epochs} | "
            f"loss_mean={epoch_info['loss_mean']:.6f} | "
            f"loss_std={epoch_info['loss_std']:.6f} | "
            f"grad_norm_mean={epoch_info['grad_norm_mean']:.6f} | "
            f"grad_norm_std={epoch_info['grad_norm_std']:.6f} | "
            f"fwd_t={epoch_info['forward_time_mean_s']:.6f}s | "
            f"bwd_t={epoch_info['backward_time_mean_s']:.6f}s | "
            f"r={theta[-1]:.6f} | "
            f"theta={np.array2string(theta, precision=6, separator=', ')}"
        )

    summary = {
        "epochs": epochs,
        "learning_rate": lr,
        "initial_theta": theta_train_init.tolist(),
        "final_theta": theta.tolist(),
        "final_circle_r": float(theta[-1]),
        "forward_time_mean_s_over_epochs": float(
            np.mean([h["forward_time_mean_s"] for h in history])
        ),
        "backward_time_mean_s_over_epochs": float(
            np.mean([h["backward_time_mean_s"] for h in history])
        ),
        "loss_mean_over_epochs": float(
            np.mean([h["loss_mean"] for h in history])
        ),
        "loss_std_over_epochs": float(
            np.std([h["loss_mean"] for h in history])
        ),
        "grad_norm_mean_over_epochs": float(
            np.mean([h["grad_norm_mean"] for h in history])
        ),
        "grad_norm_std_over_epochs": float(
            np.std([h["grad_norm_mean"] for h in history])
        ),
    }

    save_training_logs(history, summary)

    return history, summary, theta


if __name__ == "__main__":
    Tsim = 50
    N = 40

    x0 = X0_DEFAULT.copy()
    xref = XREF_DEFAULT.copy()
    circle_c = CIRCLE_C_DEFAULT.copy()

    theta_target = TEACHER_THETA.copy()
    theta_train_init = STUDENT_THETA_INIT.copy()

    target_rec = generate_target_record(
        Tsim=Tsim,
        N=N,
        x0=x0,
        xref=xref,
        circle_c=circle_c,
        theta=theta_target,
        save_path="target_record_teacher_euler_w_fullseq.npz",
    )

    print("\nTarget generated.")
    print("target status unique =", np.unique(target_rec["status_hist"]))
    print("target final state =", target_rec["X_hist"][-1])
    print("target theta =", theta_target)

    history, summary, theta_final = train_two_solver(
        target_rec=target_rec,
        theta_train_init=theta_train_init,
        epochs=50,
        lr=2e-4,
        wx=WX_DEFAULT.copy(),
        wu=WU_DEFAULT.copy(),
    )

    print("\nTraining finished.")
    print("initial theta =", theta_train_init)
    print("final theta   =", theta_final)
    print("final circle_r=", theta_final[-1])

    print("\n===== Final Summary =====")
    for k, v in summary.items():
        print(f"{k}: {v}")