#ifndef _NOMINAL_NMPC_COR_HPP_
#define _NOMINAL_NMPC_COR_HPP_

#include <nlopt.hpp>
#include "nominal_quad_dynamic.hpp"
#include <Eigen/Eigen>
#include <ros/ros.h>

struct RefData
{
    std::vector<double> ref_s;
    std::vector<Eigen::Vector3d> ref_pos;
    std::vector<Eigen::Matrix3d> ref_fsf;
    std::vector<std::vector<double>> ref_coeffs;
    Eigen::Vector3d TerminalPos; 
};


class NominalNMPC{
    public:
        static constexpr int n_step_ = 10;
        NominalQuadDynamic quad_dynamic_;

        double maxVel_;
        double real_maxVel_;  
        double contour_w_;
        double yaw_w_;
        double terminal_vel_w_;
        double terminal_radius_;

    private:
        nlopt::opt opt_;
        static constexpr int x_dim_ = NominalQuadDynamic::x_dim_;
        static constexpr int u_dim_ = NominalQuadDynamic::u_dim_;

        Eigen::Matrix<double, x_dim_, 1> state_[n_step_];
        Eigen::Matrix<double, x_dim_, n_step_ * u_dim_> state_g_[n_step_];
        Eigen::Matrix<double, 3, 1> acc_[n_step_];
        Eigen::Matrix<double, 3, n_step_ * u_dim_> acc_g_[n_step_];

        double dt_;

        Eigen::Matrix<double, x_dim_, 1> x0_;
        Eigen::Matrix<double, u_dim_, 1> u0_;
        Eigen::Matrix<double, 3 + u_dim_ + 4, 1> cost_w_; 

        void v_b_constraint_impl(unsigned m, double *result, unsigned n, const double *u, double *gradient);
        static void ellipsoid_constraint_cb(unsigned m, double* result, unsigned n, const double* x, double* grad, void* data);
        void ellipsoid_constraint_impl(unsigned m, double *result, unsigned n, const double *u, double *gradient);
        double cost_func_impl(const std::vector<double> &u, std::vector<double> &grad);
        static double cost_func_cb(const std::vector<double>& x, std::vector<double>& grad, void* data);
        static void v_b_constraint_cb(unsigned m, double* result, unsigned n, const double* x, double* grad, void* data);

        void rollout_and_sens(const double* u,
                            Eigen::Matrix<double, x_dim_, 1>* state,
                            Eigen::Matrix<double, x_dim_, n_step_ * u_dim_>* state_g,
                            Eigen::Matrix<double, 3, 1>* acc,
                            Eigen::Matrix<double, 3, n_step_ * u_dim_>* acc_g);        

    public:
        RefData ref_data_;
    
        NominalNMPC(double hover_ratio, std::string opt_algorithm, int maxeval, double dt, ros::NodeHandle& nh);
        void set_w(const Eigen::Matrix<double, 3 + u_dim_ + 4, 1> &cost_w);
        void set_ref_data(const RefData &ref_data);

        int solve(const Eigen::Matrix<double, x_dim_, 1> &state,
            Eigen::Matrix<double, u_dim_, 1> &u0,
            Eigen::Matrix<double, n_step_, u_dim_> &u,
            Eigen::Matrix<double, n_step_, x_dim_> &pred_x);

};



#endif // _NOMINAL_NMPC_HPP_