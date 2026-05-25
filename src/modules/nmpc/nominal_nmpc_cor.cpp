#include "nominal_nmpc_cor.hpp"
#include "utility/utility.hpp"
#include <cstdio>
#include <string>

NominalNMPC::NominalNMPC(double hover_ratio, std::string opt_algorithm, int maxeval, double dt, ros::NodeHandle& nh) 
    : quad_dynamic_(hover_ratio),
      opt_(nlopt::LD_SLSQP, u_dim_ * n_step_),
      dt_(dt) {
        nh.param("nmpc/maxVel", maxVel_, -1.0);
        nh.param("nmpc/real_maxVel", real_maxVel_, -1.0);
        nh.param("nmpc/contour_weight", contour_w_, 50.0);
        nh.param("nmpc/yaw_w", yaw_w_, 0.15);
        terminal_vel_w_ = 10.0;
        terminal_radius_ = 3.0;

        for (int i = 0; i < n_step_; ++i) {
            state_[i].setZero();
            state_g_[i].setZero();
            acc_[i].setZero();
            acc_g_[i].setZero();
        }
        opt_.set_min_objective(&NominalNMPC::cost_func_cb, this);
        std::vector<double> tols(n_step_);
        for (int i = 0; i < n_step_; ++i) {
            tols[i] = 1e-3;
        }
        opt_.add_inequality_mconstraint(&NominalNMPC::v_b_constraint_cb, this, tols);
        opt_.add_inequality_mconstraint(&NominalNMPC::ellipsoid_constraint_cb, this, tols);
        opt_.set_xtol_rel(1e-10);
        opt_.set_ftol_rel(1e-8);
        opt_.set_maxeval(maxeval);
        nlopt::opt local_opt(nlopt::LD_LBFGS, u_dim_ * n_step_);
        local_opt.set_xtol_rel(1e-10);
        local_opt.set_ftol_rel(1e-8);
        local_opt.set_maxeval(maxeval);
        opt_.set_local_optimizer(local_opt);
    }

double NominalNMPC::cost_func_cb(const std::vector<double>& x,
                                 std::vector<double>& grad,
                                 void* data) {
  return static_cast<NominalNMPC*>(data)->cost_func_impl(x, grad);
}

void NominalNMPC::v_b_constraint_cb(unsigned m, double* result,
                                    unsigned n, const double* x,
                                    double* grad, void* data) {
  static_cast<NominalNMPC*>(data)->v_b_constraint_impl(m, result, n, x, grad);
}

void NominalNMPC::ellipsoid_constraint_cb(unsigned m, double* result,
                                    unsigned n, const double* x,
                                    double* grad, void* data) {
  static_cast<NominalNMPC*>(data)->ellipsoid_constraint_impl(m, result, n, x, grad);
}

void NominalNMPC::v_b_constraint_impl(unsigned m, double *result, unsigned n, const double *u,
                            double *gradient) {                                
    this->rollout_and_sens(u, this->state_, this->state_g_, this->acc_, this->acc_g_);
    Eigen::Matrix<double, x_dim_, 1> *state = this->state_;
    Eigen::Matrix<double, x_dim_, n_step_ * u_dim_> *state_g = this->state_g_;
    const double &dt = this->dt_;

    for (int k = 0; k < m; k++) {
        result[k] = pow(state[k](VX), 2) + pow(state[k](VY), 2) + pow(state[k](VZ), 2) - pow(this->maxVel_, 2);
        if (gradient) {
            for (int i = 0; i < n_step_ * u_dim_; i++) {
                gradient[k * n + i] = 2 * state[k](VX) * state_g[k](VX, i) + 
                                       2 * state[k](VY) * state_g[k](VY, i) +
                                       2 * state[k](VZ) * state_g[k](VZ, i);
            }
        }
    }
}

void NominalNMPC::ellipsoid_constraint_impl(unsigned m, double *result, unsigned n, const double *u,
    double *gradient) {
    this->rollout_and_sens(u, this->state_, this->state_g_, this->acc_, this->acc_g_);
    Eigen::Matrix<double, x_dim_, 1> *state = this->state_;
    Eigen::Matrix<double, x_dim_, n_step_ * u_dim_> *state_g = this->state_g_;
    auto &ref_data = this->ref_data_;
    
    for (int k = 0; k < m; k++) {
        Eigen::Vector3d diff_pos = state[k].block(0, 0, 3, 1) - ref_data.ref_pos[k + 1];

        Eigen::Matrix3d R = ref_data.ref_fsf[k + 1];
        Eigen::Vector3d t = R.col(0).normalized();
        Eigen::Vector3d n_hat = (R.col(1) - R.col(1).dot(t) * t).normalized();
        Eigen::Vector3d b_hat = t.cross(n_hat);
        Eigen::Matrix<double,3,2> Fs; Fs.col(0)=n_hat; Fs.col(1)=b_hat;

        Eigen::Vector2d x = Fs.transpose() * diff_pos;

        Eigen::Matrix2d E;
        E << ref_data.ref_coeffs[k + 1][0], ref_data.ref_coeffs[k + 1][2],
             ref_data.ref_coeffs[k + 1][2], ref_data.ref_coeffs[k + 1][1];
        Eigen::Vector2d pp(ref_data.ref_coeffs[k + 1][3], ref_data.ref_coeffs[k + 1][4]);
        double g = x.dot(E * x) + pp.dot(x) - 1.0;
        result[k] = std::isfinite(g) ? g : 1e6;
        if (!std::isfinite(g)) continue;
        if (gradient) {
            Eigen::Matrix<double, 3, 1> grad = Fs * (2 * E * x + pp);
            Eigen::Matrix<double, 1, u_dim_ * n_step_> ellipsoid_grad = grad.transpose() * state_g[k].block(0, 0, 3, n_step_ * u_dim_);
            for (int i = 0; i < n_step_ * u_dim_; i++) {
                gradient[k * n + i] = std::isfinite(ellipsoid_grad(i)) ? ellipsoid_grad(i) : 0.0;
            }
        }
    }
}

void NominalNMPC::rollout_and_sens(const double* u,
                                   Eigen::Matrix<double, x_dim_, 1>* state,
                                   Eigen::Matrix<double, x_dim_, n_step_ * u_dim_>* state_g,
                                   Eigen::Matrix<double, 3, 1>* acc,
                                   Eigen::Matrix<double, 3, n_step_ * u_dim_>* acc_g) {
    const double dt = this->dt_;
    for (int k = 0; k < n_step_; ++k) {
        state_g[k].setZero();
        acc_g[k].setZero();
    }

    Eigen::Matrix<double, u_dim_, 1> uvec;
    Eigen::Matrix<double, x_dim_, x_dim_> x1dotx0;
    Eigen::Matrix<double, x_dim_, u_dim_> x1dotu;
    Eigen::Matrix<double, 3, x_dim_>      accdotx0;
    Eigen::Matrix<double, 3, u_dim_>      accdotu;

    uvec << u[0], u[1], u[2], u[3];
    this->quad_dynamic_.rk4_func(this->x0_, uvec, dt,
                                 state[0], x1dotx0, x1dotu,
                                 acc[0],   accdotx0, accdotu);
    state_g[0].block(0, 0, x_dim_, u_dim_) = x1dotu;
    acc_g[0].block(0, 0, 3, u_dim_)   = accdotu;

    for (int k = 1; k < n_step_; ++k) {
        const int off = k * u_dim_;
        uvec << u[off + 0], u[off + 1], u[off + 2], u[off + 3];

        this->quad_dynamic_.rk4_func(state[k - 1], uvec, dt,
                                     state[k], x1dotx0, x1dotu,
                                     acc[k],   accdotx0, accdotu);
        state_g[k].block(0, off, x_dim_, u_dim_) = x1dotu;
        acc_g[k].block(0, off, 3, u_dim_)   = accdotu;

        state_g[k].block(0, 0, x_dim_, off) =
            x1dotx0 * state_g[k - 1].block(0, 0, x_dim_, off);
        acc_g[k].block(0, 0, 3, off) =
            accdotx0 * state_g[k - 1].block(0, 0, x_dim_, off);
    }
}

double NominalNMPC::cost_func_impl(const std::vector<double> &u, std::vector<double> &grad){
    Eigen::Matrix<double, x_dim_, 1> *state = this->state_;
    Eigen::Matrix<double, x_dim_, n_step_ * u_dim_> *state_g = this->state_g_;
    Eigen::Matrix<double, 3, 1> *acc = this->acc_;
    Eigen::Matrix<double, 3, n_step_ * u_dim_> *acc_g = this->acc_g_;
    auto &cost_w = this->cost_w_;
    auto &ref_data = this->ref_data_;

    this->rollout_and_sens(u.data(), state, state_g, acc, acc_g);

    std::vector<Eigen::Vector3d> diff_pos(n_step_ + 1);
    std::vector<Eigen::Vector3d> proj_pos(n_step_ + 1);
    for (int k = 0; k < n_step_ + 1; k++) {
        if (k > 0){
            diff_pos[k] = state[k - 1].block(0, 0, 3, 1) - ref_data.ref_pos[k];
            proj_pos[k] = ref_data.ref_fsf[k].transpose() * diff_pos[k];
        }
        else {
            diff_pos[k] = this->x0_.block(0, 0, 3, 1) - ref_data.ref_pos[k];
            proj_pos[k] = ref_data.ref_fsf[k].transpose() * diff_pos[k];
        }
    }

    Eigen::Matrix<double, 1, u_dim_ * n_step_> pos_err_g;
    pos_err_g.setZero();
    double pos_err = 0.0;

    double progress_reward = 0.0;
    Eigen::Matrix<double, 1, u_dim_ * n_step_> progress_reward_g;
    progress_reward_g.setZero();

    double contouring_cost = 0.0;
    Eigen::Matrix<double, 1, u_dim_ * n_step_> contouring_cost_g;
    contouring_cost_g.setZero();

    const double terminal_radius = terminal_radius_;
    double term_vcost = 0.0;
    Eigen::Matrix<double, 1, u_dim_ * n_step_> term_vcost_g;
    term_vcost_g.setZero();
    for (int k = 0; k < n_step_ + 1; k++) {
        if (k > 0){
            bool near_terminal = (ref_data.ref_pos[k] - ref_data.TerminalPos).norm() < terminal_radius;
            if (near_terminal){
                pos_err += diff_pos[k].squaredNorm();
                pos_err_g += 2 * diff_pos[k].transpose() * state_g[k - 1].block(0, 0, 3, u_dim_ * n_step_);

                if (terminal_vel_w_ > 0.0) {
                    double prox = 1.0 - (ref_data.ref_pos[k] - ref_data.TerminalPos).norm() / terminal_radius;
                    if (prox < 0.0) prox = 0.0;
                    double w_eff = prox * prox;
                    double vx_k = state[k - 1](VX);
                    double vy_k = state[k - 1](VY);
                    double vz_k = state[k - 1](VZ);
                    term_vcost += w_eff * (vx_k * vx_k + vy_k * vy_k + vz_k * vz_k);
                    term_vcost_g += 2.0 * w_eff * (
                          vx_k * state_g[k - 1].block(VX, 0, 1, u_dim_ * n_step_)
                        + vy_k * state_g[k - 1].block(VY, 0, 1, u_dim_ * n_step_)
                        + vz_k * state_g[k - 1].block(VZ, 0, 1, u_dim_ * n_step_));
                }
            } else {
                pos_err += proj_pos[k](0) * proj_pos[k](0);
                Eigen::Vector3d ref_fsf_e1 = ref_data.ref_fsf[k].col(0);
                pos_err_g += 2 * proj_pos[k](0) * ref_fsf_e1.transpose() * state_g[k - 1].block(0, 0, 3, u_dim_ * n_step_);

                progress_reward += proj_pos[k](0);
                progress_reward_g += ref_fsf_e1.transpose() * state_g[k - 1].block(0, 0, 3, u_dim_ * n_step_);
            }
        }
        else {
            pos_err += proj_pos[k](0) * proj_pos[k](0);
        }
    }

    double vcost = 0.0;
    Eigen::Matrix<double, 1, u_dim_ * n_step_> vcost_g;
    vcost_g.setZero();
    for (int k = 0; k < n_step_; k++) {
        double tmp = pow(state[k](VX), 2) + pow(state[k](VY), 2) + pow(state[k](VZ), 2) - pow(this->real_maxVel_, 2);
        if (tmp > 0.0) {
            vcost += tmp;
            vcost_g.block(0, 0, 1, u_dim_ * n_step_) += 
                (2 * state[k](VX) * state_g[k].block(VX, 0, 1, u_dim_ * n_step_) +
                 2 * state[k](VY) * state_g[k].block(VY, 0, 1, u_dim_ * n_step_) +
                 2 * state[k](VZ) * state_g[k].block(VZ, 0, 1, u_dim_ * n_step_));
        }
    }

    double ucost = 0.0;
    Eigen::Matrix<double, 1, u_dim_ * n_step_> ucost_g;
    ucost_g.setZero();
    for (int k = 0; k < n_step_; k++) {
        int offset = k * u_dim_;
        Eigen::Matrix<double, u_dim_, 1> past_u;
        if (k == 0) {
            past_u = this->u0_;
        } else {
            past_u << u[offset - u_dim_], u[offset - u_dim_ + 1], u[offset - u_dim_ + 2], u[offset - u_dim_ + 3];
        }
        
        ucost +=  cost_w(0) * pow(acc[k](0), 2)
                + cost_w(1) * pow(acc[k](1), 2)
                + cost_w(2) * pow(acc[k](2), 2)
                + cost_w(3) * pow(u[offset + 0] - past_u(0), 2)
                + cost_w(4) * pow(u[offset + 1] - past_u(1), 2)
                + cost_w(5) * pow(u[offset + 2] - past_u(2), 2)
                + cost_w(6) * pow(u[offset + 3] - past_u(3), 2);

        ucost_g.block(0, 0, 1, u_dim_ * n_step_) += 
            2 * cost_w(0) * acc[k](0) * acc_g[k].row(0) +
            2 * cost_w(1) * acc[k](1) * acc_g[k].row(1) +
            2 * cost_w(2) * acc[k](2) * acc_g[k].row(2);
        ucost_g[offset + 0] += 2 * cost_w(3) * (u[offset + 0] - past_u(0));
        ucost_g[offset + 1] += 2 * cost_w(4) * (u[offset + 1] - past_u(1));
        ucost_g[offset + 2] += 2 * cost_w(5) * (u[offset + 2] - past_u(2));
        ucost_g[offset + 3] += 2 * cost_w(6) * (u[offset + 3] - past_u(3));

        if (k > 0) {
            ucost_g[offset - 4] -= 2 * cost_w(3) * (u[offset + 0] - past_u(0));
            ucost_g[offset - 3] -= 2 * cost_w(4) * (u[offset + 1] - past_u(1));
            ucost_g[offset - 2] -= 2 * cost_w(5) * (u[offset + 2] - past_u(2));
            ucost_g[offset - 1] -= 2 * cost_w(6) * (u[offset + 3] - past_u(3));
        }
    }

    double yawcost = 0.0;
    Eigen::Matrix<double, 1, u_dim_ * n_step_> yawcost_g;
    yawcost_g.setZero();
    for (int k = 0; k < n_step_; k++) {
        const double& vx_k = state[k](VX);
        const double& vy_k = state[k](VY);
        double vxy2 = vx_k * vx_k + vy_k * vy_k;
        if (vxy2 < 1e-6) continue;

        const double& qw_k = state[k](QW);
        const double& qx_k = state[k](QX);
        const double& qy_k = state[k](QY);
        const double& qz_k = state[k](QZ);
        double tmpx = 1.0 - 2.0 * (qy_k * qy_k + qz_k * qz_k);
        double tmpy = 2.0 * (qw_k * qz_k + qx_k * qy_k);
        double tmpnorm = tmpx * tmpx + tmpy * tmpy;
        double yaw_val = std::atan2(tmpy, tmpx);
        Eigen::Vector2d yaw_g_quat(-tmpy / tmpnorm, tmpx / tmpnorm);
        Eigen::Vector4d yaw_g_q;
        yaw_g_q(0) = yaw_g_quat.y() * 2.0 * qz_k;
        yaw_g_q(1) = yaw_g_quat.y() * 2.0 * qy_k;
        yaw_g_q(2) = yaw_g_quat.x() * (-4.0 * qy_k) + yaw_g_quat.y() * 2.0 * qx_k;
        yaw_g_q(3) = yaw_g_quat.x() * (-4.0 * qz_k) + yaw_g_quat.y() * 2.0 * qw_k;

        double vang = std::atan2(vy_k, vx_k);
        Eigen::Vector2d vang_g_v(-vy_k / vxy2, vx_k / vxy2);

        double diff = yaw_val - vang;
        if (diff > M_PI) diff -= 2.0 * M_PI;
        else if (diff < -M_PI) diff += 2.0 * M_PI;
        double sign = (diff >= 0.0) ? 1.0 : -1.0;

        yawcost += std::abs(diff);
        yawcost_g += sign * (
            yaw_g_q.transpose() * state_g[k].block(QW, 0, 4, u_dim_ * n_step_)
            - vang_g_v.transpose() * state_g[k].block(VX, 0, 2, u_dim_ * n_step_)
        );
    }

    double cost = 0.0;
    double pos_err_w = cost_w(7);
    double ellipsoid_w = cost_w(8);
    double vcost_w = cost_w(9);
    double w_theta = cost_w(10);
    cost = pos_err_w * pos_err - w_theta * progress_reward
           + vcost_w * vcost + ucost + yaw_w_ * yawcost
           + terminal_vel_w_ * term_vcost;

    Eigen::Matrix<double, 1, u_dim_ * n_step_> grad_tmp;
    grad_tmp.setZero();
    grad_tmp += pos_err_w * pos_err_g - w_theta * progress_reward_g
                + vcost_w * vcost_g + ucost_g + yaw_w_ * yawcost_g
                + terminal_vel_w_ * term_vcost_g;

    if (!grad.empty()) {
        for (int i = 0; i < u_dim_ * n_step_; i++) {
            grad[i] = grad_tmp(i);
        }
    }

    return cost;
}

void NominalNMPC::set_w(const Eigen::Matrix<double, 3 + u_dim_ + 4, 1> &cost_w) {
    cost_w_ = cost_w;
}

void NominalNMPC::set_ref_data(const RefData &ref_data) {
    ref_data_ = ref_data;
}

static double dbg_ellipsoid_g(const Eigen::Matrix<double, NominalQuadDynamic::x_dim_, 1>& st,
                              const RefData& ref, int ref_idx) {
    if (ref_idx < 0 || ref_idx >= (int)ref.ref_pos.size()) return 1e6;
    if (ref_idx >= (int)ref.ref_coeffs.size() || (int)ref.ref_coeffs[ref_idx].size() < 5) return 1e6;
    Eigen::Vector3d diff_pos = st.template block<3, 1>(0, 0) - ref.ref_pos[ref_idx];
    const Eigen::Matrix3d& R = ref.ref_fsf[ref_idx];
    Eigen::Vector3d t = R.col(0).normalized();
    Eigen::Vector3d n_raw = R.col(1) - R.col(1).dot(t) * t;
    double n_norm = n_raw.norm();
    if (n_norm < 1e-9) return 1e6;
    Eigen::Vector3d n_hat = n_raw / n_norm;
    Eigen::Vector3d b_hat = t.cross(n_hat);
    Eigen::Matrix<double, 3, 2> Fs;
    Fs.col(0) = n_hat;
    Fs.col(1) = b_hat;
    Eigen::Vector2d x = Fs.transpose() * diff_pos;
    const auto& c = ref.ref_coeffs[ref_idx];
    Eigen::Matrix2d E;
    E << c[0], c[2], c[2], c[1];
    Eigen::Vector2d pp(c[3], c[4]);
    double g = x.dot(E * x) + pp.dot(x) - 1.0;
    return std::isfinite(g) ? g : 1e6;
}

int NominalNMPC::solve(const Eigen::Matrix<double, x_dim_, 1> &state,
    Eigen::Matrix<double, u_dim_, 1> &u0,
    Eigen::Matrix<double, n_step_, u_dim_> &u,
    Eigen::Matrix<double, n_step_, x_dim_> &pred_x){

    std::vector<double> uv(u_dim_ * n_step_);
    for (int k = 0; k < n_step_; k++) {
        uv[k * u_dim_ + 0] = u(k, 0);
        uv[k * u_dim_ + 1] = u(k, 1);
        uv[k * u_dim_ + 2] = u(k, 2);
        uv[k * u_dim_ + 3] = u(k, 3);
    }

    std::vector<double> lb(uv.size()), ub(uv.size());
    for (int k = 0; k < n_step_; k++) {
        lb[k * u_dim_ + 0] = -M_PI * 1.8;
        lb[k * u_dim_ + 1] = -M_PI * 1.8;
        lb[k * u_dim_ + 2] = -M_PI * 1.8;
        lb[k * u_dim_ + 3] = 0.0;

        ub[k * u_dim_ + 0] = M_PI * 1.8;
        ub[k * u_dim_ + 1] = M_PI * 1.8;
        ub[k * u_dim_ + 2] = M_PI * 1.8;
        ub[k * u_dim_ + 3] = 0.8;
    }

    opt_.set_lower_bounds(lb);
    opt_.set_upper_bounds(ub);

    for (int i = 0; i < uv.size(); i++){
        if (uv[i] > ub[i]) {
            uv[i] = ub[i];
        } else if (uv[i] < lb[i]) {
            uv[i] = lb[i];
        }
    }
    
    x0_ = state;
    u0_ = u0;

    {
        int n_g_pos = 0, n_vel_pos = 0;
        for (int k = 0; k < n_step_; ++k) {
            double g = dbg_ellipsoid_g(state_[k], ref_data_, k + 1);
            if (g > 0.0) n_g_pos++;
            double vel2 = state_[k](VX) * state_[k](VX) + state_[k](VY) * state_[k](VY)
                        + state_[k](VZ) * state_[k](VZ) - maxVel_ * maxVel_;
            if (vel2 > 0.0) n_vel_pos++;
        }
        if (n_g_pos > 0 || n_vel_pos > 0) {
            const double hr = quad_dynamic_.hover_ratio_;
            u0 << 0.0, 0.0, 0.0, hr;
            u0_ = u0;
            for (int k = 0; k < n_step_; ++k) {
                u.row(k) << 0.0, 0.0, 0.0, hr;
                uv[k * u_dim_ + 0] = 0.0;
                uv[k * u_dim_ + 1] = 0.0;
                uv[k * u_dim_ + 2] = 0.0;
                uv[k * u_dim_ + 3] = hr;
            }
        }
    }

    double minf;
    int result = -1;

    try {
        result = opt_.optimize(uv, minf);
        if (result < 0) {
            result = -1;
        }

        for (int k = 0; k < n_step_; k++) {
            u(k, 0) = uv[k * u_dim_ + 0];
            u(k, 1) = uv[k * u_dim_ + 1];
            u(k, 2) = uv[k * u_dim_ + 2];
            u(k, 3) = uv[k * u_dim_ + 3];
        }
        for (int k = 0; k < n_step_; k++) {
            pred_x.row(k) = state_[k].transpose();
        }
    } catch (const std::exception& e) {
        result = -1;
    }

    return result;

}