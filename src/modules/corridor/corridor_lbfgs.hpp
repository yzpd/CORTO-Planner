#ifndef CORRIDOR_LBFGS_HPP
#define CORRIDOR_LBFGS_HPP

#include <Eigen/Eigen>
#include <vector>
#include <cmath>
#include <algorithm>

namespace corridor_lbfgs {

inline double softplus(double x) {
    return x > 20.0 ? x : std::log1p(std::exp(x));
}

inline double dsoftplus(double x) {
    return x > 20.0 ? 1.0 : 1.0 / (1.0 + std::exp(-x));
}

inline double inv_softplus(double y) {
    return y > 20.0 ? y : std::log(std::expm1(std::max(y, 1e-8)));
}

inline double smoothAbs(double x, double mu = 1e-3) {
    return std::sqrt(x * x + mu * mu) - mu;
}
inline double dsmoothAbs(double x, double mu = 1e-3) {
    return x / std::sqrt(x * x + mu * mu);
}

struct CorridorProblem {
    int n_segments;
    int degree;
    std::vector<double> sweep_s;
    double total_arc_length;

    Eigen::MatrixXd obs;
    Eigen::MatrixXd zp;

    double w_obs    = 1e4;
    double w_zp     = 1e4;
    double w_smooth = 1e-2;
    double w_delta  = 1.0;
    double eps_psd  = 0.1;

    int ncp() const { return n_segments * (degree + 1); }
    int n_free_per_param() const {
        return (degree + 1) + std::max(0, n_segments - 1) * (degree - 1);
    }
    int n_free_total() const { return 6 * n_free_per_param(); }

    static constexpr int MAX_DEGREE = 64;
    struct EvalPoint {
        int seg;
        double basis[MAX_DEGREE + 1];
    };
    std::vector<EvalPoint> obs_eval, zp_eval;

    void precompute() {
        auto makeEval = [&](const Eigen::MatrixXd &pts) {
            int n = pts.rows();
            std::vector<EvalPoint> ev(n);
            double seg_len = total_arc_length / n_segments;
            for (int i = 0; i < n; ++i) {
                double s = pts(i, 2);
                int seg = std::min(n_segments - 1, std::max(0, (int)(s / seg_len)));
                double t = (s - seg * seg_len) / seg_len;
                t = std::max(0.0, std::min(1.0, t));
                ev[i].seg = seg;
                for (int k = 0; k <= degree; ++k)
                    ev[i].basis[k] = bernstein_val(k, degree, t);
            }
            return ev;
        };
        if (obs.rows() > 0) obs_eval = makeEval(obs);
        if (zp.rows() > 0) zp_eval = makeEval(zp);
    }

    static double bernstein_val(int k, int n, double t) {
        double binom = 1.0;
        for (int i = 0; i < k; ++i) binom *= double(n - i) / (i + 1);
        return binom * std::pow(t, k) * std::pow(1.0 - t, n - k);
    }

    void expandCtrl(const double *free_vals, double *ctrl) const {
        int dp1 = degree + 1;
        int dm1 = degree - 1;

        for (int k = 0; k < dp1; ++k)
            ctrl[k] = free_vals[k];

        for (int seg = 1; seg < n_segments; ++seg) {
            int base = seg * dp1;
            const double *fv = free_vals + dp1 + dm1 * (seg - 1);

            ctrl[base + 0] = ctrl[base - 1];
            double ds_left  = sweep_s[seg] - sweep_s[seg - 1];
            double ds_right = sweep_s[seg + 1] - sweep_s[seg];
            double ratio = (ds_left > 1e-12) ? (ds_right / ds_left) : 1.0;
            ctrl[base + 1] = ctrl[base] + ratio * (ctrl[base - 1] - ctrl[base - 2]);

            for (int k = 0; k < dm1; ++k)
                ctrl[base + 2 + k] = fv[k];
        }
    }

    void backpropCtrl(const double *grad_ctrl_in, double *grad_free) const {
        int dp1 = degree + 1;
        int dm1 = degree - 1;
        int total = ncp();
        std::vector<double> gc(grad_ctrl_in, grad_ctrl_in + total);

        for (int seg = n_segments - 1; seg >= 1; --seg) {
            int base = seg * dp1;
            double ds_left  = sweep_s[seg] - sweep_s[seg - 1];
            double ds_right = sweep_s[seg + 1] - sweep_s[seg];
            double ratio = (ds_left > 1e-12) ? (ds_right / ds_left) : 1.0;

            gc[base + 0] += gc[base + 1] * 1.0;
            gc[base - 1] += gc[base + 1] * ratio;
            gc[base - 2] += gc[base + 1] * (-ratio);

            gc[base - 1] += gc[base + 0];
        }

        for (int k = 0; k < dp1; ++k)
            grad_free[k] = gc[k];

        for (int seg = 1; seg < n_segments; ++seg) {
            int base = seg * dp1;
            int fi = dp1 + dm1 * (seg - 1);
            for (int k = 0; k < dm1; ++k)
                grad_free[fi + k] = gc[base + 2 + k];
        }
    }

    inline double evalAt(const double *ctrl, const EvalPoint &ep) const {
        int base = ep.seg * (degree + 1);
        double v = 0.0;
        for (int k = 0; k <= degree; ++k)
            v += ctrl[base + k] * ep.basis[k];
        return v;
    }

    double costAndGrad(const Eigen::VectorXd &theta, Eigen::VectorXd &grad) const {
        grad.setZero();
        double cost = 0.0;

        int dp1 = degree + 1;
        int cp_total = ncp();
        int nfp = n_free_per_param();

        std::vector<double> all_ctrl(6 * cp_total);
        for (int p = 0; p < 6; ++p)
            expandCtrl(theta.data() + p * nfp, all_ctrl.data() + p * cp_total);

        std::vector<double> grad_ctrl(6 * cp_total, 0.0);

        auto ctrl = [&](int p) -> double* { return all_ctrl.data() + p * cp_total; };
        auto gctrl = [&](int p) -> double* { return grad_ctrl.data() + p * cp_total; };

        for (int i = 0; i < cp_total; ++i) {
            double l11r = ctrl(0)[i], l21v = ctrl(1)[i], l22r = ctrl(2)[i];
            double l11 = softplus(l11r), l22 = softplus(l22r);
            double dl11 = dsoftplus(l11r), dl22 = dsoftplus(l22r);

            double a_v = eps_psd + l11 * l11;
            double b_v = eps_psd + l21v * l21v + l22 * l22;
            cost += a_v + b_v;

            gctrl(0)[i] += 2.0 * l11 * dl11;
            gctrl(1)[i] += 2.0 * l21v;
            gctrl(2)[i] += 2.0 * l22 * dl22;
        }

        for (int i = 0; i < cp_total; ++i) {
            double l31v = ctrl(3)[i], l32v = ctrl(4)[i], l33r = ctrl(5)[i];
            double l33 = softplus(l33r), dl33 = dsoftplus(l33r);

            double delta_v = l31v * l31v + l32v * l32v + l33 * l33;
            cost += w_delta * delta_v;

            gctrl(3)[i] += w_delta * 2.0 * l31v;
            gctrl(4)[i] += w_delta * 2.0 * l32v;
            gctrl(5)[i] += w_delta * 2.0 * l33 * dl33;
        }

        for (int p = 0; p < 6; ++p) {
            for (int i = 0; i < cp_total - 1; ++i) {
                double diff = ctrl(p)[i + 1] - ctrl(p)[i];
                cost += w_smooth * smoothAbs(diff);
                double ds = w_smooth * dsmoothAbs(diff);
                gctrl(p)[i]     -= ds;
                gctrl(p)[i + 1] += ds;
            }
        }

        int n_obs = obs.rows();
        for (int i = 0; i < n_obs; ++i) {
            const auto &ep = obs_eval[i];
            double yi = obs(i, 0), zi = obs(i, 1);
            double lv[6];
            for (int p = 0; p < 6; ++p)
                lv[p] = evalAt(ctrl(p), ep);

            double l11 = softplus(lv[0]);
            double l21 = lv[1], l22 = softplus(lv[2]);
            double l31 = lv[3], l32 = lv[4];

            double a_v = eps_psd + l11 * l11;
            double b_v = eps_psd + l21 * l21 + l22 * l22;
            double c_v = l11 * l21;
            double d_v = 2.0 * l11 * l31;
            double e_v = 2.0 * (l21 * l31 + l22 * l32);

            double g = a_v * yi * yi + b_v * zi * zi + 2.0 * c_v * yi * zi
                     + d_v * yi + e_v * zi;

            if (g >= 1.0) continue;

            double viol = 1.0 - g;
            cost += w_obs * viol * viol;
            double dP = -2.0 * w_obs * viol;
            double dg[5];
            dg[0] = 2.0 * l11 * yi * yi + l21 * 2.0 * yi * zi + 2.0 * l31 * yi;
            dg[1] = 2.0 * l21 * zi * zi + l11 * 2.0 * yi * zi + 2.0 * l31 * zi;
            dg[2] = 2.0 * l22 * zi * zi + 2.0 * l32 * zi;
            dg[3] = 2.0 * l11 * yi + 2.0 * l21 * zi;
            dg[4] = 2.0 * l22 * zi;

            double dlraw[6];
            dlraw[0] = dP * dg[0] * dsoftplus(lv[0]);
            dlraw[1] = dP * dg[1];
            dlraw[2] = dP * dg[2] * dsoftplus(lv[2]);
            dlraw[3] = dP * dg[3];
            dlraw[4] = dP * dg[4];
            dlraw[5] = 0.0;

            int base = ep.seg * dp1;
            for (int p = 0; p < 5; ++p) {
                if (dlraw[p] == 0.0) continue;
                for (int k = 0; k <= degree; ++k)
                    gctrl(p)[base + k] += dlraw[p] * ep.basis[k];
            }
        }

        int n_zp = zp.rows();
        for (int i = 0; i < n_zp; ++i) {
            const auto &ep = zp_eval[i];
            double yi = zp(i, 0), zi = zp(i, 1);

            double lv[6];
            for (int p = 0; p < 6; ++p)
                lv[p] = evalAt(ctrl(p), ep);

            double l11 = softplus(lv[0]);
            double l21 = lv[1], l22 = softplus(lv[2]);
            double l31 = lv[3], l32 = lv[4];

            double a_v = eps_psd + l11 * l11;
            double b_v = eps_psd + l21 * l21 + l22 * l22;
            double c_v = l11 * l21;
            double d_v = 2.0 * l11 * l31;
            double e_v = 2.0 * (l21 * l31 + l22 * l32);

            double g = a_v * yi * yi + b_v * zi * zi + 2.0 * c_v * yi * zi
                     + d_v * yi + e_v * zi;

            if (g >= 1.0) continue;

            double viol = 1.0 - g;
            cost += w_zp * viol * viol;
            double dP = -2.0 * w_zp * viol;

            double dg[5];
            dg[0] = 2.0 * l11 * yi * yi + l21 * 2.0 * yi * zi + 2.0 * l31 * yi;
            dg[1] = 2.0 * l21 * zi * zi + l11 * 2.0 * yi * zi + 2.0 * l31 * zi;
            dg[2] = 2.0 * l22 * zi * zi + 2.0 * l32 * zi;
            dg[3] = 2.0 * l11 * yi + 2.0 * l21 * zi;
            dg[4] = 2.0 * l22 * zi;

            double dlraw[6];
            dlraw[0] = dP * dg[0] * dsoftplus(lv[0]);
            dlraw[1] = dP * dg[1];
            dlraw[2] = dP * dg[2] * dsoftplus(lv[2]);
            dlraw[3] = dP * dg[3];
            dlraw[4] = dP * dg[4];
            dlraw[5] = 0.0;

            int base = ep.seg * dp1;
            for (int p = 0; p < 5; ++p) {
                if (dlraw[p] == 0.0) continue;
                for (int k = 0; k <= degree; ++k)
                    gctrl(p)[base + k] += dlraw[p] * ep.basis[k];
            }
        }

        for (int p = 0; p < 6; ++p)
            backpropCtrl(gctrl(p), grad.data() + p * nfp);

        return cost;
    }

    void evalEllipseAt(const double *all_ctrl, double s,
                       double &a_out, double &b_out, double &c_out,
                       double &d_out, double &e_out) const {
        int cp_total_val = ncp();
        double seg_len = total_arc_length / n_segments;
        int seg = std::min(n_segments - 1, std::max(0, (int)(s / seg_len)));
        double t = (s - seg * seg_len) / seg_len;
        t = std::max(0.0, std::min(1.0, t));

        double lv[6] = {};
        int base = seg * (degree + 1);
        for (int k = 0; k <= degree; ++k) {
            double bk = bernstein_val(k, degree, t);
            for (int p = 0; p < 6; ++p)
                lv[p] += all_ctrl[p * cp_total_val + base + k] * bk;
        }

        double l11 = softplus(lv[0]);
        double l21 = lv[1], l22 = softplus(lv[2]);
        double l31 = lv[3], l32 = lv[4];

        a_out = eps_psd + l11 * l11;
        b_out = eps_psd + l21 * l21 + l22 * l22;
        c_out = l11 * l21;
        d_out = 2.0 * l11 * l31;
        e_out = 2.0 * (l21 * l31 + l22 * l32);
    }
};

inline double lbfgsCostCallback(void *instance,
                                 const Eigen::VectorXd &x,
                                 Eigen::VectorXd &g) {
    return static_cast<CorridorProblem *>(instance)->costAndGrad(x, g);
}

}

#endif // CORRIDOR_LBFGS_HPP
