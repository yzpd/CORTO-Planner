#ifndef _UTILITY_HPP_
#define _UTILITY_HPP_

#include <Eigen/Eigen>
#include <vector>
#include "nmpc/nominal_nmpc_cor.hpp"
#include "cubic_spline/arc_length_spline.hpp"
#include "corridor/corGen.hpp"
#include <ros/ros.h>
#include <cmath>
#include <string>

struct InitData4NMPC
{
    std::vector<double> cumulative_times;
    RefData ref_data;              
    double spline_len{0.0};    
};

inline double projectOnSpline(const ArcLengthSpline& spline, const Eigen::Vector3d& point) {
    const PathData_3D& pd = spline.getPathData();
    const int n = pd.n_points;
    double best_s = 0.0;
    double best_dist2 = std::numeric_limits<double>::max();
    for (int i = 0; i < n; ++i) {
        double dx = pd.X(i) - point.x();
        double dy = pd.Y(i) - point.y();
        double dz = pd.Z(i) - point.z();
        double d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < best_dist2) {
            best_dist2 = d2;
            best_s = pd.s(i);
        }
    }
    double ds = spline.getLength_3D() / (n - 1);
    double lo = std::max(0.0, best_s - ds);
    double hi = std::min(spline.getLength_3D(), best_s + ds);
    const double gr = 0.6180339887;
    for (int iter = 0; iter < 20; ++iter) {
        double s1 = hi - gr * (hi - lo);
        double s2 = lo + gr * (hi - lo);
        double d1 = (spline.getPosition_3D(s1) - point).squaredNorm();
        double d2 = (spline.getPosition_3D(s2) - point).squaredNorm();
        if (d1 < d2) hi = s2; else lo = s1;
    }
    return 0.5 * (lo + hi);
}

inline double arcLengthToTime(const std::vector<double>& cumulative_times,
                              double s, double spline_len) {
    int t_size = static_cast<int>(cumulative_times.size());
    if (t_size < 2 || spline_len <= 0.0) return 0.0;
    double idx_frac = s / spline_len * (t_size - 1);
    int idx_lo = std::max(0, std::min(t_size - 2, static_cast<int>(std::floor(idx_frac))));
    int idx_hi = idx_lo + 1;
    double alpha = idx_frac - idx_lo;
    return (1.0 - alpha) * cumulative_times[idx_lo] + alpha * cumulative_times[idx_hi];
}

inline double timeToArcLength(const std::vector<double>& result_t,
                              double t, double spline_len) {
    int t_size = static_cast<int>(result_t.size());
    if (t_size < 2 || spline_len <= 0.0) return 0.0;
    std::vector<double> cum(t_size, 0.0);
    for (int i = 1; i < t_size; ++i) cum[i] = cum[i-1] + result_t[i];
    double total_time = cum.back();
    if (t <= 0.0) return 0.0;
    if (t >= total_time) return spline_len;
    int lo = 0, hi = t_size - 1;
    while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        if (cum[mid] <= t) lo = mid;
        else hi = mid;
    }
    double alpha = (cum[hi] > cum[lo]) ? (t - cum[lo]) / (cum[hi] - cum[lo]) : 0.0;
    double idx_frac = lo + alpha;
    return idx_frac / (t_size - 1) * spline_len;
}

inline double velocityAtTime(const std::vector<double>& result_t,
                             double t, double spline_len) {
    int t_size = static_cast<int>(result_t.size());
    if (t_size < 2 || spline_len <= 0.0) return 0.0;
    std::vector<double> cum(t_size, 0.0);
    for (int i = 1; i < t_size; ++i) cum[i] = cum[i-1] + result_t[i];
    double total_time = cum.back();
    if (t >= total_time) return 0.0;
    double ds = spline_len / (t_size - 1);
    if (t <= 0.0) {
        return (result_t[1] > 1e-9) ? ds / result_t[1] : 0.0;
    }
    int lo = 0, hi = t_size - 1;
    while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        if (cum[mid] <= t) lo = mid;
        else hi = mid;
    }
    double dt_seg = result_t[hi];
    return (dt_seg > 1e-9) ? ds / dt_seg : 0.0;
}

void Pre4NMPC_Robust(InitData4NMPC& init_data_nmpc, const std::vector<double>& result_t, const int n_step, double dt, 
    const ArcLengthSpline& best_spline, corGen& corridor_gen, const OptimizationResult& coeffs, double t_cur, double spline_t_total,
    std::string frame) {

    int t_size = result_t.size();
    if (t_size < 2) {
        init_data_nmpc.spline_len = best_spline.getLength_3D();
        return;
    }
    std::vector<double> cumulative_times(t_size, 0.0);
    std::vector<double> target_times;

    double spline_len = best_spline.getLength_3D();

    std::vector<double> ref_s;
    std::vector<Eigen::Vector3d> ref_pos;
    std::vector<std::vector<double>> ref_coeffs;

    for (int i = 1; i < t_size; ++i) {
        cumulative_times[i] = cumulative_times[i - 1] + result_t[i];
    }

    for (int i = 0; i < n_step + 1; ++i) {
        double t = t_cur + i * dt;
        if ( t > spline_t_total ) t = spline_t_total;
        target_times.push_back(t);
    }

    for (double target_time : target_times) {
        int closest_index = 0;
        if (target_time == spline_t_total) {
            closest_index = t_size - 1;
        } else {
            double min_diff = std::abs(cumulative_times[0] - target_time);

            for (int j = 1; j < t_size; ++j) {
                double diff = std::abs(cumulative_times[j] - target_time);
                if (diff < min_diff) {
                    min_diff = diff;
                    closest_index = j;
                }
            }
        }

        double s_value = static_cast<double>(closest_index) / static_cast<double>(t_size - 1) * spline_len;
        ref_s.push_back(s_value);
        Eigen::Vector3d pos = best_spline.getPosition_3D(s_value);
        ref_pos.push_back(pos);
    }

    std::vector<Eigen::Matrix3d> ref_fsf(n_step + 1);
    const double smooth_alpha = 0.0;

    auto rotate_about = [](const Eigen::Vector3d& x,
                        const Eigen::Vector3d& axis_unit,
                        double ang) {
        return Eigen::AngleAxisd(ang, axis_unit) * x;
    };

    if (frame == "FSF") {
        FSFPrev prev;
        {
            Eigen::Matrix3d seed = corridor_gen.getFSFData().FSF[0];
            Eigen::Vector3d t = seed.col(0).normalized();
            Eigen::Vector3d n = (seed.col(1) - seed.col(1).dot(t) * t).normalized();
            Eigen::Vector3d b = t.cross(n);
            prev.t = t; prev.n = n; prev.b = b; prev.has_prev = true;
        }

        for (int i = 0; i < n_step + 1; ++i) {
            const double s = ref_s[i];
            const Eigen::Vector3d v = best_spline.getDerivative_3D(s);
            const Eigen::Vector3d a = best_spline.getSecondDerivative_3D(s);

            Eigen::Matrix3d F = corridor_gen.FSF_Function_Robust(
                v, a, &prev, 1e-10, 1e-10
            );

            if (F.col(0).dot(prev.t) < 0.0) {
                F = -F;
            }

            if (F.col(1).dot(prev.n) < 0.0) {
                F.col(1) = -F.col(1);
                F.col(2) = -F.col(2);
            }

            if (smooth_alpha > 0.0) {
                const Eigen::Vector3d& t = F.col(0);
                Eigen::Vector3d n = F.col(1), b = F.col(2);

                double sin_th = n.cross(prev.n).dot(t);
                double cos_th = n.dot(prev.n);
                double th = std::atan2(sin_th, cos_th);

                double dth = -smooth_alpha * th;
                if (std::abs(dth) > 1e-12) {
                    n = rotate_about(n, t, dth);
                    b = rotate_about(b, t, dth);
                }

                Eigen::Vector3d t_hat = F.col(0).normalized();
                Eigen::Vector3d n_hat = (n - n.dot(t_hat) * t_hat).normalized();
                Eigen::Vector3d b_hat = t_hat.cross(n_hat);
                F.col(0) = t_hat; F.col(1) = n_hat; F.col(2) = b_hat;
            } else {
                Eigen::Vector3d t_hat = F.col(0).normalized();
                Eigen::Vector3d n_hat = (F.col(1) - F.col(1).dot(t_hat) * t_hat).normalized();
                Eigen::Vector3d b_hat = t_hat.cross(n_hat);
                F.col(0) = t_hat; F.col(1) = n_hat; F.col(2) = b_hat;
            }
            if ((F.col(0).cross(F.col(1))).dot(F.col(2)) < 0.0) {
                F.col(1) = -F.col(1);
                F.col(2) = F.col(0).cross(F.col(1));
            }

            ref_fsf[i] = F;

            prev.t = F.col(0); prev.n = F.col(1); prev.b = F.col(2);
        }
    } else if (frame == "PTF") {

        Eigen::Matrix3d R_prev = corridor_gen.PTF_at_s(ref_s[0], best_spline);
        ref_fsf[0] = R_prev;

        for (int i = 1; i < n_step + 1; ++i) {
            const double s = ref_s[i];

            Eigen::Matrix3d R = corridor_gen.PTF_at_s(s, best_spline);

            if (R.col(1).dot(R_prev.col(1)) < 0.0) {
                R.col(1) = -R.col(1);
                R.col(2) = -R.col(2);
            }

            if (smooth_alpha > 0.0) {
                const Eigen::Vector3d& t = R.col(0);
                Eigen::Vector3d n = R.col(1), b = R.col(2);

                double sin_th = n.cross(R_prev.col(1)).dot(t);
                double cos_th = n.dot(R_prev.col(1));
                double th = std::atan2(sin_th, cos_th);

                double dth = -smooth_alpha * th;
                if (std::abs(dth) > 1e-12) {
                    n = rotate_about(n, t, dth);
                    b = rotate_about(b, t, dth);
                }
                Eigen::Vector3d t_hat = t.normalized();
                Eigen::Vector3d n_hat = (n - n.dot(t_hat)*t_hat).normalized();
                Eigen::Vector3d b_hat = t_hat.cross(n_hat);
                R.col(0)=t_hat; R.col(1)=n_hat; R.col(2)=b_hat;
            }

            if ((R.col(0).cross(R.col(1))).dot(R.col(2)) < 0.0) {
                R.col(1) = -R.col(1);
                R.col(2) =  R.col(0).cross(R.col(1));
            }

            ref_fsf[i] = R;
            R_prev = R;
        }        
    }

    for (int i = 0; i < ref_s.size(); ++i) {
        std::vector<double> ref_coeff = corridor_gen.polynomial_eval(ref_s[i], coeffs, corridor_gen.getPolyDegree());
        ref_coeffs.push_back(ref_coeff);
    }

    init_data_nmpc.cumulative_times = cumulative_times;
    init_data_nmpc.ref_data.ref_s = ref_s;
    init_data_nmpc.ref_data.ref_pos = ref_pos;
    init_data_nmpc.ref_data.ref_fsf = ref_fsf;
    init_data_nmpc.ref_data.ref_coeffs = ref_coeffs;
    init_data_nmpc.spline_len = spline_len;
}

#endif // _UTILITY_HPP_