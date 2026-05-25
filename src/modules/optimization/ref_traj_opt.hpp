#ifndef _REF_TRAJ_OPT_HPP_
#define _REF_TRAJ_OPT_HPP_

#include "fusion.h"
#include <Eigen/Dense>
#include "cubic_spline/arc_length_spline.hpp"

using namespace mosek::fusion;
using namespace monty;
using namespace std;

class RefTrajOpt {
public:

    RefTrajOpt() {};
    ~RefTrajOpt() {};

    void socp_interface(const int N, const double w1, const double wT, const ArcLengthSpline& spline, const double Vmax, const double Amax,
                        const bool is_jerk, const double Jmax);

    double get_total_time() const { return total_time_; }

    const vector<double>& get_result_t() const { return result_t_; }

private:
    Variable::t t_;
    Variable::t w_;
    Variable::t x1_;
    Variable::t x2_;
    vector<double> result_t_;
    double total_time_ = 0.0;
    bool solved_ = false;
};

#endif // REF_TRAJ_OPT_HPP