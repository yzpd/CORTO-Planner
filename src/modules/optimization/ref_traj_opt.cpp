#include "ref_traj_opt.hpp"
#include <ros/ros.h>

void RefTrajOpt::socp_interface(const int N, const double w1, const double wT, const ArcLengthSpline& spline, const double Vmax, const double Amax, 
                                const bool is_jerk, const double Jmax) {
    Model::t M  = new Model("ref_traj_opt"); auto _M = finally([&]() { M->dispose(); });

    double total_length = spline.getLength_3D();
    double h = total_length / (N-1);

    Variable::t t = M->variable("t", N, Domain::greaterThan(0.0));
    Variable::t w = M->variable("w", N, Domain::greaterThan(0.0));
    Variable::t x1 = M->variable("x1", N, Domain::greaterThan(0.0));
    Variable::t x2 = M->variable("x2", N, Domain::greaterThan(0.0));

    M->constraint("Constraints1_begin", w->index(0), Domain::equalsTo(w1));
    M->constraint("Constraints1_terminal", w->index(N - 1), Domain::equalsTo(wT));

    for(int i = 1; i < N - 1; ++i) {
        Expression::t th = Expr::vstack(t->index(i), Expr::constTerm(h * 0.5), x2->index(i));
        M->constraint("Constraints2_" + std::to_string(i), th, Domain::inRotatedQCone());

        Expression::t tw = Expr::vstack(t->index(i), Expr::mul(w->index(i), h * 0.5), x1->index(i));
        M->constraint("Constraints3_" + std::to_string(i), tw, Domain::inRotatedQCone());

        Expression::t x12 = Expr::vstack(x1->index(i), x2->index(i), Expr::constTerm(std::sqrt(2) * h));
        M->constraint("Constraints4_" + std::to_string(i), x12, Domain::inRotatedQCone());

        M->constraint("w_upper_" + std::to_string(i), w->index(i), Domain::lessThan((Vmax*Vmax)));

        double s = i * total_length / (N - 1);
        Eigen::Vector3d ddr_path = spline.getSecondDerivative_3D(s);
        double ddr = ddr_path.norm();

        Expression::t w_diff = Expr::sub(w->index(i), w->index(i-1));
        Expression::t dw = Expr::mul(w_diff, 1.0/h);

        Expression::t acc_expr = Expr::add(Expr::mul(ddr, w->index(i)), Expr::mul(dw, 0.5));

        M->constraint("acc_lower_" + std::to_string(i), acc_expr, Domain::greaterThan(-Amax));
        M->constraint("acc_upper_" + std::to_string(i), acc_expr, Domain::lessThan(Amax));     

        if (is_jerk) {
            Expression::t dw_diff = Expr::add(Expr::sub(w->index(i+1), Expr::mul(2.0, w->index(i))), w->index(i-1));
            Expression::t ddw = Expr::mul(dw_diff, 1.0/(h * Jmax));
            M->constraint("J_lower_" + std::to_string(i), Expr::add(ddw, t->index(i)), Domain::greaterThan(0.0));
            M->constraint("J_upper_" + std::to_string(i), Expr::sub(ddw, t->index(i)), Domain::lessThan(0.0));
        }
    }
    
    M->objective("obj", ObjectiveSense::Minimize, Expr::sum(t));
    M->solve();

    if (M->getPrimalSolutionStatus() == SolutionStatus::Optimal) {
        t_ = M->getVariable("t");
        auto t_val = t_->level();
        total_time_ = 0.0;
        vector<double> result_t;
        for(int i = 0; i < N; ++i) {
            total_time_ += (*t_val)[i];
            result_t.push_back((*t_val)[i]);
        }
        result_t_ = result_t;
        w_ = M->getVariable("w");
        x1_ = M->getVariable("x1");
        x2_ = M->getVariable("x2");
        solved_ = true;
    }
    else {
        solved_ = false;
        return;
    }
    return;
}