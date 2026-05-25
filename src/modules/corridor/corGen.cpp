#include "corGen.hpp"
#include "corridor_lbfgs.hpp"
#include "lbfgs.hpp"

void corGen::init(ros::NodeHandle& nh) {
    nh.param("poly_degree", poly_degree_, 3);
    nh.param("is_LP", is_LP_, false);
    nh.param("MaxPointCloud", MaxPointCloud_, 1000);
    nh.param("n_sweep", n_sweep_, 10);
    n_angles_ = 18;
    ellipse_axis_max = 2.0;
    eps_ = 1e-1;
}

OptimizationResult corGen::corGenerator(const ArcLengthSpline& spline){
    updateSpline(spline);
    FSF_evaluate_Robust(spline);
    Projection_data pruned_points_local = processPointCloud("FSF");
    NLP_LBFGS(pruned_points_local);

    if (coeffs_.success) coeffs_.volume = computeVolume(coeffs_);
    return coeffs_;
}

void corGen::updateSpline(const ArcLengthSpline& spline) {
    spline_ = spline;
    n_eval_ = spline_.getPathData().s.size();
}

Eigen::Vector3d corGen::safe_normalize(const Eigen::Vector3d& x,
                                        const Eigen::Vector3d& fallback,
                                        double eps) {
    double n = x.norm();
    return (n > eps) ? (x / n) : fallback;
}

Eigen::Vector3d corGen::safe_cross_normalized(const Eigen::Vector3d& a,
                                                const Eigen::Vector3d& b,
                                                const Eigen::Vector3d& fallback,
                                                double eps) {
    Eigen::Vector3d c = a.cross(b);
    double n = c.norm();
    return (n > eps) ? (c / n) : fallback;
}

Eigen::Matrix3d corGen::FSF_Function_Robust(const Eigen::Vector3d& v,
                                        const Eigen::Vector3d& a,
                                        const FSFPrev* prev = nullptr,
                                        double eps_v,
                                        double eps_k) {
    Eigen::Vector3d t_fallback = (prev && prev->has_prev) ? prev->t : Eigen::Vector3d::UnitX();
    Eigen::Vector3d t = safe_normalize(v, t_fallback, eps_v);

    Eigen::Vector3d a_perp = a - a.dot(t) * t;
    bool curved = (a_perp.squaredNorm() > eps_k);

    Eigen::Vector3d up = (std::abs(t.z()) < 0.9) ? Eigen::Vector3d::UnitZ() : Eigen::Vector3d::UnitX();
    Eigen::Vector3d n_fallback = (prev && prev->has_prev)
        ? (prev->n - prev->n.dot(t) * t)
        : (up - up.dot(t) * t);

    Eigen::Vector3d n = curved
        ? safe_normalize(a_perp, n_fallback, eps_k)
        : safe_normalize(n_fallback, (up - up.dot(t) * t), eps_k);

    if (prev && prev->has_prev && n.dot(prev->n) < 0.0) n = -n;
    Eigen::Vector3d e3_fallback = (t.cross(n_fallback).norm() > eps_k)
        ? (t.cross(n_fallback)).normalized()
        : ((std::abs(t.x()) < 0.9) ? t.cross(Eigen::Vector3d::UnitX()).normalized()
                                   : t.cross(Eigen::Vector3d::UnitY()).normalized());

    Eigen::Vector3d b = safe_cross_normalized(t, n, e3_fallback, eps_k);

    n = safe_normalize(b.cross(t), n, eps_k);
    b = t.cross(n);

    Eigen::Matrix3d R;
    R.col(0) = t; R.col(1) = n; R.col(2) = b;

    if ((R.col(0).cross(R.col(1))).dot(R.col(2)) < 0.0) {
        R.col(1) = -R.col(1);
        R.col(2) = R.col(0).cross(R.col(1));
    }
    return R;
}

void corGen::FSF_evaluate_Robust(const ArcLengthSpline& spline) {
    FSF_data fsf_data;
    fsf_data.FSF.resize(n_eval_, Eigen::Matrix3d::Identity());

    FSFPrev prev; prev.has_prev = false;

    for (int i = 0; i < n_eval_; ++i) {
        double s = spline.getPathData().s[i];
        Eigen::Vector3d v = spline.getDerivative_3D(s);
        Eigen::Vector3d a = spline.getSecondDerivative_3D(s);

        Eigen::Matrix3d F = FSF_Function_Robust(v, a, prev.has_prev ? &prev : nullptr);

        if (prev.has_prev && F.col(0).dot(prev.t) < 0.0) {
            F.col(0) = -F.col(0);
            F.col(1) = -F.col(1);
        }

        if ((F.col(0).cross(F.col(1))).dot(F.col(2)) < 0.0) {
            F.col(1) = -F.col(1);
            F.col(2) = F.col(0).cross(F.col(1));
        }

        fsf_data.FSF[i] = F;

        prev.t = F.col(0); prev.n = F.col(1); prev.b = F.col(2);
        prev.has_prev = true;
    }

    fsf_data_ = fsf_data;
}

Projection_data corGen::processPointCloud(std::string frame) {
    if(!cloud_updated_) return {};
    if (!global_kdtree_built_ && !cloudMap_.points.empty()) {
        auto cloud_ptr = cloudMap_.makeShared();
        global_kdtree_.setInputCloud(cloud_ptr);
        global_kdtree_built_ = true;
    }

    CageParameters cage_params;
    size_t old_size = cloudMap_.points.size();
    get_cage(cage_params, frame);
    size_t cage_added = cloudMap_.points.size() - old_size;

    const double r_pre = std::min(ellipse_axis_max, 3.0) + 0.5;
    const double r_pre_sq = r_pre * r_pre;
    const int n_path_samples = std::min(n_eval_, 30);
    const int path_step = std::max(1, n_eval_ / n_path_samples);

    struct PathSample {
        Eigen::Vector3d pos;
        Eigen::Matrix3d R;
        double s_val;
    };
    std::vector<PathSample> path_samples;
    path_samples.reserve(n_path_samples + 2);

    for (int j = 0; j < n_eval_; j += path_step) {
        PathSample ps;
        ps.s_val = spline_.getPathData().s[j];
        ps.pos = spline_.getPosition_3D(ps.s_val);
        if (frame == "FSF") ps.R = fsf_data_.FSF[j];
        else                ps.R = ptf_data_.PTF[j];
        path_samples.push_back(ps);
    }
    if ((n_eval_ - 1) % path_step != 0) {
        int j = n_eval_ - 1;
        PathSample ps;
        ps.s_val = spline_.getPathData().s[j];
        ps.pos = spline_.getPosition_3D(ps.s_val);
        if (frame == "FSF") ps.R = fsf_data_.FSF[j];
        else                ps.R = ptf_data_.PTF[j];
        path_samples.push_back(ps);
    }
    int n_path = (int)path_samples.size();

    int n_segments = n_sweep_ - 1;
    double total_len = spline_.getLength_3D();
    std::vector<double> sweep_s(n_sweep_);
    for (int i = 0; i < n_sweep_; ++i)
        sweep_s[i] = i * total_len / std::max(1, n_sweep_ - 1);

    const int K_per_bin = 8;
    struct BinEntry { double yz_norm; Eigen::Vector3d local; double s_val; };
    std::vector<std::vector<BinEntry>> bins(n_segments);

    std::vector<bool> visited(old_size, false);
    const int query_step = std::max(1, n_path / 10);
    std::vector<int> radius_indices;
    std::vector<float> radius_dists;
    size_t n_local = 0;

    std::vector<int> local_indices;
    local_indices.reserve(old_size / 4);

    if (global_kdtree_built_) {
        for (int qi = 0; qi < n_path; qi += query_step) {
            pcl::PointXYZ search_pt;
            search_pt.x = path_samples[qi].pos.x();
            search_pt.y = path_samples[qi].pos.y();
            search_pt.z = path_samples[qi].pos.z();
            global_kdtree_.radiusSearch(search_pt, r_pre, radius_indices, radius_dists);
            for (int idx : radius_indices) {
                if (idx < (int)old_size && !visited[idx]) {
                    visited[idx] = true;
                    local_indices.push_back(idx);
                }
            }
        }
    }
    n_local = local_indices.size();

    pcl::PointCloud<pcl::PointXYZ>::Ptr path_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    path_cloud->points.resize(n_path);
    for (int pi = 0; pi < n_path; ++pi) {
        path_cloud->points[pi].x = path_samples[pi].pos.x();
        path_cloud->points[pi].y = path_samples[pi].pos.y();
        path_cloud->points[pi].z = path_samples[pi].pos.z();
    }
    pcl::KdTreeFLANN<pcl::PointXYZ> path_kdtree;
    path_kdtree.setInputCloud(path_cloud);

    std::vector<int> nn_idx(1);
    std::vector<float> nn_dist2(1);

    auto processPoint = [&](const pcl::PointXYZ& pt) {
        if (path_kdtree.nearestKSearch(pt, 1, nn_idx, nn_dist2) < 1) return;
        if (nn_dist2[0] > r_pre_sq) return;

        int best_pi = nn_idx[0];
        const PathSample& ps = path_samples[best_pi];
        Eigen::Vector3d diff(pt.x - ps.pos.x(), pt.y - ps.pos.y(), pt.z - ps.pos.z());
        Eigen::Vector3d local = ps.R.transpose() * diff;

        if (std::abs(local(0)) >= 0.01) return;
        double yz_norm = std::sqrt(local(1)*local(1) + local(2)*local(2));
        if (yz_norm < 0.05 || local.norm() > ellipse_axis_max) return;

        int seg = std::max(0, std::min(n_segments - 1,
            (int)(std::upper_bound(sweep_s.begin(), sweep_s.end(), ps.s_val) - sweep_s.begin()) - 1));
        bins[seg].push_back({yz_norm, local, ps.s_val});
    };

    for (int idx : local_indices) {
        processPoint(cloudMap_.points[idx]);
    }

    for (size_t ci = old_size; ci < old_size + cage_added; ++ci) {
        processPoint(cloudMap_.points[ci]);
    }

    if (cage_added > 0) {
        cloudMap_.points.resize(old_size);
        cloudMap_.width = old_size;
    }

    Projection_data pruned_data;
    std::vector<Eigen::Vector3d> pruned_points_local;
    std::vector<double> pruned_s_cloud;
    std::vector<Eigen::Vector3d> all_raw_pos;
    std::vector<double> all_raw_s;
    int total_raw = 0;

    for (auto& bin : bins) {
        total_raw += (int)bin.size();
        for (auto& e : bin) {
            all_raw_pos.push_back(e.local);
            all_raw_s.push_back(e.s_val);
        }

        if ((int)bin.size() <= max_per_bin_) {
            for (auto& e : bin) {
                pruned_points_local.push_back(e.local);
                pruned_s_cloud.push_back(e.s_val);
            }
        } else {
            std::vector<std::vector<const BinEntry*>> sectors(n_sectors_);
            for (auto& e : bin) {
                double angle = std::atan2(e.local(2), e.local(1));
                int si = (int)((angle + M_PI) / (2.0 * M_PI) * n_sectors_);
                si = std::max(0, std::min(n_sectors_ - 1, si));
                sectors[si].push_back(&e);
            }
            for (auto& sec : sectors) {
                std::sort(sec.begin(), sec.end(),
                    [](const BinEntry* a, const BinEntry* b){ return a->yz_norm < b->yz_norm; });
                int take = std::min(k_per_sector_, (int)sec.size());
                for (int k = 0; k < take; ++k) {
                    pruned_points_local.push_back(sec[k]->local);
                    pruned_s_cloud.push_back(sec[k]->s_val);
                }
            }
        }
    }

    pruned_data.pruned_cloud_pos = std::move(pruned_points_local);
    pruned_data.s_cloud = std::move(pruned_s_cloud);
    pruned_data.all_raw_pos = std::move(all_raw_pos);
    pruned_data.all_raw_s = std::move(all_raw_s);

    const int n_z_line_samples = 7;
    std::vector<Eigen::Vector2d> z_plane_pts;
    std::vector<double> z_plane_s_vals;
    const double z_margin = 0.05;

    std::vector<int> z_eval_indices;
    for (int sw = 0; sw < n_sweep_; ++sw) {
        double s_sw = sw * total_len / std::max(1, n_sweep_ - 1);
        int best_idx = 0;
        double best_diff = 1e9;
        for (int j = 0; j < n_eval_; ++j) {
            double diff = std::abs(spline_.getPathData().s[j] - s_sw);
            if (diff < best_diff) { best_diff = diff; best_idx = j; }
        }
        z_eval_indices.push_back(best_idx);
    }

    for (int si : z_eval_indices) {
        double s_val = spline_.getPathData().s[si];
        Eigen::Vector3d pos = spline_.getPosition_3D(s_val);
        Eigen::Matrix3d R;
        if (frame == "FSF") R = fsf_data_.FSF[si];
        else                R = ptf_data_.PTF[si];

        double e2z = R(2, 1);
        double e3z = R(2, 2);

        double z_targets[2] = { z_floor_ + z_margin, z_ceiling_ - z_margin };
        for (double z_target : z_targets) {
            double dz = z_target - pos.z();

            for (int j = 0; j < n_z_line_samples; ++j) {
                double t = -ellipse_axis_max
                         + j * 2.0 * ellipse_axis_max / (n_z_line_samples - 1);
                double local_y, local_z;

                if (std::abs(e3z) > 1e-6) {
                    local_y = t;
                    local_z = (dz - e2z * t) / e3z;
                } else if (std::abs(e2z) > 1e-6) {
                    local_z = t;
                    local_y = (dz - e3z * t) / e2z;
                } else {
                    continue;
                }

                if (Eigen::Vector2d(local_y, local_z).norm() > ellipse_axis_max)
                    continue;

                z_plane_pts.emplace_back(local_y, local_z);
                z_plane_s_vals.push_back(s_val);
            }
        }
    }
    pruned_data.z_plane_pts = z_plane_pts;
    pruned_data.z_plane_s   = z_plane_s_vals;

    return pruned_data;
}

double corGen::poly_basis(double s, int k, int degree) const{
    auto binomial_coeff = [](int n, int k) -> double {
        if (k > n || k < 0) return 0.0;
        double result = 1.0;
        for (int i = 0; i < k; ++i) {
            result = result * (n - i) / (i + 1);
        }
        return result;
    };
    double t_clamped = std::max(0.0, std::min(1.0, s));
    return binomial_coeff(degree, k) * std::pow(t_clamped, k) * 
            std::pow(1.0 - t_clamped, degree - k);
}

void corGen::NLP(const Projection_data& pruned_points_local){
    OptimizationResult result;
    result.success = false;

    int n_points_orig = pruned_points_local.pruned_cloud_pos.size();
    const auto& s_points = spline_.getPathData().s;
    int deg = poly_degree_;

    const int max_pc = MaxPointCloud_;
    const int K_per_bin = 5;

    std::vector<int> pc_indices;
    pc_indices.reserve(std::min(n_points_orig, max_pc));
    if (n_points_orig <= max_pc) {
        pc_indices.resize(n_points_orig);
        std::iota(pc_indices.begin(), pc_indices.end(), 0);
    } else {
        double s_total = spline_.getLength_3D();
        int n_bins = std::max(1, max_pc / K_per_bin);
        double bin_width = s_total / n_bins;

        std::vector<std::vector<std::pair<double, int>>> bins(n_bins);
        for (int i = 0; i < n_points_orig; ++i) {
            double sv = pruned_points_local.s_cloud[i];
            int bin = std::min(n_bins - 1, std::max(0, (int)(sv / bin_width)));
            double yz_norm = Eigen::Vector2d(
                pruned_points_local.pruned_cloud_pos[i](1),
                pruned_points_local.pruned_cloud_pos[i](2)).norm();
            bins[bin].emplace_back(yz_norm, i);
        }

        for (auto& bin : bins) {
            if ((int)bin.size() <= K_per_bin) {
                for (auto& p : bin) pc_indices.push_back(p.second);
            } else {
                std::partial_sort(bin.begin(), bin.begin() + K_per_bin, bin.end());
                for (int k = 0; k < K_per_bin; ++k) pc_indices.push_back(bin[k].second);
            }
        }
    }
    int n_points = static_cast<int>(pc_indices.size());

    int n_segments = n_sweep_ - 1;
    if (n_sweep_ == 1) n_segments = 1;
    std::vector<double> sweep_s(n_sweep_);
    for (int i = 0; i < n_sweep_; ++i) {
        sweep_s[i] = i * spline_.getLength_3D() / (n_sweep_ - 1);
    }
    int n_vars = n_segments * (poly_degree_ + 1);

    Model::t M = new Model("corridor_optimization"); auto _M = finally([&]() { M->dispose(); });

    const double var_bound_ab = 100.0;
    const double var_bound_c  = 50.0;
    const double var_bound_de = 20.0;
    Variable::t a = M->variable("a", n_vars, Domain::inRange(1e-6, var_bound_ab));
    Variable::t b = M->variable("b", n_vars, Domain::inRange(1e-6, var_bound_ab));
    Variable::t c = M->variable("c", n_vars, Domain::inRange(-var_bound_c, var_bound_c));
    Variable::t d = M->variable("d", n_vars, Domain::inRange(-var_bound_de, var_bound_de));
    Variable::t e = M->variable("e", n_vars, Domain::inRange(-var_bound_de, var_bound_de));
    const double delta_max = 0.8;
    Variable::t delta = M->variable("delta", n_vars, Domain::inRange(0.0, delta_max));

    int n_cost_pts = n_sweep_ + n_segments;
    auto arrCost = monty::new_array_ptr<double,2>(monty::shape(n_cost_pts, n_vars));
    for (int i = 0; i < n_cost_pts; ++i)
        for (int j = 0; j < n_vars; ++j)
            (*arrCost)(i, j) = 0.0;
    for (int i = 0; i < n_sweep_; ++i) {
        int seg = std::min(i, n_segments - 1);
        int offset = seg * (deg + 1);
        double denom = sweep_s[seg+1] - sweep_s[seg];
        double t = denom > 0 ? (sweep_s[i] - sweep_s[seg]) / denom : 0.0;
        for (int k = 0; k <= deg; ++k)
            (*arrCost)(i, offset + k) = poly_basis(t, k, deg);
    }
    for (int seg = 0; seg < n_segments; ++seg) {
        int row = n_sweep_ + seg;
        int offset = seg * (deg + 1);
        for (int k = 0; k <= deg; ++k)
            (*arrCost)(row, offset + k) = poly_basis(0.5, k, deg);
    }
    Matrix::t BasisCost = Matrix::dense(arrCost);
    Expression::t a_cost = Expr::mul(BasisCost, a);
    Expression::t b_cost = Expr::mul(BasisCost, b);
    Expression::t cost = Expr::sum(Expr::add(a_cost, b_cost));

    double lambda_smooth = 1e-2;
    auto arrS = monty::new_array_ptr<double,2>(monty::shape(n_sweep_, n_vars));
    for (int i = 0; i < n_sweep_; ++i)
        for (int j = 0; j < n_vars; ++j)
            (*arrS)(i, j) = 0.0;
    for (int i = 0; i < n_sweep_; ++i) {
        int seg = std::min(i, n_segments - 1);
        int offset = seg * (deg + 1);
        double denom = sweep_s[seg+1] - sweep_s[seg];
        double t = denom > 0 ? (sweep_s[i] - sweep_s[seg]) / denom : 0.0;
        for (int k = 0; k <= deg; ++k)
            (*arrS)(i, offset + k) = poly_basis(t, k, deg);
    }
    Matrix::t BasisS = Matrix::dense(arrS);

    if (n_sweep_ > 1) {
        int n_diff = n_sweep_ - 1;
        auto arrDiffB = monty::new_array_ptr<double,2>(monty::shape(n_diff, n_vars));
        for (int i = 0; i < n_diff; ++i)
            for (int j = 0; j < n_vars; ++j)
                (*arrDiffB)(i, j) = (*arrS)(i + 1, j) - (*arrS)(i, j);
        Matrix::t DiffB = Matrix::dense(arrDiffB);

        Expression::t da = Expr::mul(DiffB, a);
        Expression::t db = Expr::mul(DiffB, b);
        Expression::t dc = Expr::mul(DiffB, c);
        Expression::t dd = Expr::mul(DiffB, d);
        Expression::t de = Expr::mul(DiffB, e);
        Expression::t ddelta = Expr::mul(DiffB, delta);

        Expression::t all_diffs = Expr::vstack(
            Expr::vstack(Expr::vstack(da, db), Expr::vstack(dc, dd)),
            Expr::vstack(de, ddelta));
        int n_all = n_diff * 6;
        Variable::t ss = M->variable("ss", n_all, Domain::greaterThan(0.0));
        M->constraint(Expr::sub(ss, all_diffs), Domain::greaterThan(0.0));
        M->constraint(Expr::add(ss, all_diffs), Domain::greaterThan(0.0));
        cost = Expr::add(cost, Expr::mul(lambda_smooth, Expr::sum(ss)));
    }

    const double eps_min_eig = 0.1;
    const double lambda_delta = 1.0;

    for (int idx = 0; idx < n_vars; ++idx) {
        Expression::t a_k = a->index(idx);
        Expression::t b_k = b->index(idx);
        Expression::t c_k = c->index(idx);
        Expression::t d_k = d->index(idx);
        Expression::t e_k = e->index(idx);
        Expression::t delta_k = delta->index(idx);

        if (!is_LP_) {
            Expression::t Mk = Expr::vstack(
                Expr::hstack(Expr::hstack(Expr::sub(a_k, eps_min_eig), c_k), Expr::mul(0.5, d_k)),
                Expr::hstack(Expr::hstack(c_k, Expr::sub(b_k, eps_min_eig)), Expr::mul(0.5, e_k)),
                Expr::hstack(Expr::hstack(Expr::mul(0.5, d_k), Expr::mul(0.5, e_k)), delta_k));
            M->constraint(Mk, Domain::inPSDCone(3));
        } else {
            M->constraint(Expr::sub(Expr::sub(a_k, eps_min_eig), c_k), Domain::greaterThan(0.0));
            M->constraint(Expr::add(Expr::sub(a_k, eps_min_eig), c_k), Domain::greaterThan(0.0));
            M->constraint(Expr::sub(Expr::sub(b_k, eps_min_eig), c_k), Domain::greaterThan(0.0));
            M->constraint(Expr::add(Expr::sub(b_k, eps_min_eig), c_k), Domain::greaterThan(0.0));
            M->constraint(delta_k, Domain::greaterThan(0.0));
        }
    }
    cost = Expr::add(cost, Expr::mul(lambda_delta, Expr::sum(delta)));

    for (int bnd = 0; bnd < n_segments - 1; ++bnd) {
        int left_last   = bnd * (deg + 1) + deg;
        int right_first = (bnd + 1) * (deg + 1);

        M->constraint(Expr::sub(a->index(left_last), a->index(right_first)), Domain::equalsTo(0.0));
        M->constraint(Expr::sub(b->index(left_last), b->index(right_first)), Domain::equalsTo(0.0));
        M->constraint(Expr::sub(c->index(left_last), c->index(right_first)), Domain::equalsTo(0.0));
        M->constraint(Expr::sub(d->index(left_last), d->index(right_first)), Domain::equalsTo(0.0));
        M->constraint(Expr::sub(e->index(left_last), e->index(right_first)), Domain::equalsTo(0.0));

        int left_prev   = left_last - 1;
        int right_next  = right_first + 1;
        double ds_left  = sweep_s[bnd + 1] - sweep_s[bnd];
        double ds_right = sweep_s[bnd + 2] - sweep_s[bnd + 1];
        double inv_left  = 1.0 / std::max(1e-12, ds_left);
        double inv_right = 1.0 / std::max(1e-12, ds_right);

        auto c1_constraint = [&](Variable::t var) {
            Expression::t lhs = Expr::mul(inv_left, Expr::sub(var->index(left_last), var->index(left_prev)));
            Expression::t rhs = Expr::mul(inv_right, Expr::sub(var->index(right_next), var->index(right_first)));
            M->constraint(Expr::sub(lhs, rhs), Domain::equalsTo(0.0));
        };
        c1_constraint(a);
        c1_constraint(b);
        c1_constraint(c);
        c1_constraint(d);
        c1_constraint(e);
        M->constraint(Expr::sub(delta->index(left_last), delta->index(right_first)), Domain::equalsTo(0.0));
        c1_constraint(delta);
    }

    if (n_points > 0) {
        auto arrPC = monty::new_array_ptr<double,2>(monty::shape(n_points, n_vars));
        std::vector<int> s_index_pc(n_points, 0);
        for (int j = 0; j < n_points; ++j) {
            int orig_idx = pc_indices[j];
            double s_val_pc = pruned_points_local.s_cloud[orig_idx];
            int seg = std::max(0, std::min(n_segments - 1,
                int(std::upper_bound(sweep_s.begin(), sweep_s.end(), s_val_pc) - sweep_s.begin()) - 1));
            s_index_pc[j] = seg;
            int offset = seg * (deg + 1);
            double denom = sweep_s[seg+1] - sweep_s[seg];
            double t_pc = denom > 0 ? (s_val_pc - sweep_s[seg]) / denom : 0.0;
            for (int k = 0; k <= deg; ++k)
                (*arrPC)(j, offset + k) = poly_basis(t_pc, k, deg);
        }
        Matrix::t BasisPC = Matrix::dense(arrPC);

        Expression::t a_pc = Expr::mul(BasisPC, a);
        Expression::t b_pc = Expr::mul(BasisPC, b);
        Expression::t c_pc = Expr::mul(BasisPC, c); 
        Expression::t d_pc = Expr::mul(BasisPC, d);
        Expression::t e_pc = Expr::mul(BasisPC, e);

        auto Y   = monty::new_array_ptr<double,1>(monty::shape(n_points));
        auto Z   = monty::new_array_ptr<double,1>(monty::shape(n_points));
        auto Y2  = monty::new_array_ptr<double,1>(monty::shape(n_points));
        auto Z2  = monty::new_array_ptr<double,1>(monty::shape(n_points));
        auto YZ2 = monty::new_array_ptr<double,1>(monty::shape(n_points));
        auto ONE = monty::new_array_ptr<double,1>(monty::shape(n_points));
        for (int ii = 0; ii < n_points; ++ii) {
            int orig_idx = pc_indices[ii];
            const auto& point = pruned_points_local.pruned_cloud_pos[orig_idx];
            double y = point(1);
            double z = point(2);
            (*Y)(ii)   = y;
            (*Z)(ii)   = z;
            (*Y2)(ii)  = y * y;
            (*Z2)(ii)  = z * z;
            (*YZ2)(ii) = 2.0 * y * z;
            (*ONE)(ii) = 1.0;
        }    

        auto quad = Expr::add( Expr::add( Expr::mulElm(a_pc, Y2), Expr::mulElm(b_pc, Z2) ),
                                Expr::mulElm(c_pc, YZ2) );
        auto lin  = Expr::add( Expr::mulElm(d_pc, Y), Expr::mulElm(e_pc, Z) );
        auto gvec = Expr::add( quad, lin );
        M->constraint( Expr::sub(gvec, ONE), Domain::greaterThan(0.0) );
    }

    int n_z_plane = static_cast<int>(pruned_points_local.z_plane_pts.size());
    if (n_z_plane > 0) {
        auto arrZP = monty::new_array_ptr<double,2>(monty::shape(n_z_plane, n_vars));
        for (int j = 0; j < n_z_plane; ++j) {
            double s_val = pruned_points_local.z_plane_s[j];
            int seg = std::max(0, std::min(n_segments - 1,
                int(std::upper_bound(sweep_s.begin(), sweep_s.end(), s_val)
                    - sweep_s.begin()) - 1));
            int off = seg * (deg + 1);
            double den = sweep_s[seg + 1] - sweep_s[seg];
            double t_zp = den > 0 ? (s_val - sweep_s[seg]) / den : 0.0;
            for (int k = 0; k <= deg; ++k)
                (*arrZP)(j, off + k) = poly_basis(t_zp, k, deg);
        }
        Matrix::t BasisZP = Matrix::dense(arrZP);
        Expression::t a_zp = Expr::mul(BasisZP, a);
        Expression::t b_zp = Expr::mul(BasisZP, b);
        Expression::t c_zp = Expr::mul(BasisZP, c);
        Expression::t d_zp = Expr::mul(BasisZP, d);
        Expression::t e_zp = Expr::mul(BasisZP, e);

        auto Y_zp   = monty::new_array_ptr<double,1>(monty::shape(n_z_plane));
        auto Z_zp   = monty::new_array_ptr<double,1>(monty::shape(n_z_plane));
        auto Y2_zp  = monty::new_array_ptr<double,1>(monty::shape(n_z_plane));
        auto Z2_zp  = monty::new_array_ptr<double,1>(monty::shape(n_z_plane));
        auto YZ2_zp = monty::new_array_ptr<double,1>(monty::shape(n_z_plane));
        auto ONE_zp = monty::new_array_ptr<double,1>(monty::shape(n_z_plane));
        for (int ii = 0; ii < n_z_plane; ++ii) {
            double y = pruned_points_local.z_plane_pts[ii].x();
            double z = pruned_points_local.z_plane_pts[ii].y();
            (*Y_zp)(ii)   = y;
            (*Z_zp)(ii)   = z;
            (*Y2_zp)(ii)  = y * y;
            (*Z2_zp)(ii)  = z * z;
            (*YZ2_zp)(ii) = 2.0 * y * z;
            (*ONE_zp)(ii) = 1.0;
        }
        auto quad_zp = Expr::add(Expr::add(Expr::mulElm(a_zp, Y2_zp),
                                            Expr::mulElm(b_zp, Z2_zp)),
                                  Expr::mulElm(c_zp, YZ2_zp));
        auto lin_zp  = Expr::add(Expr::mulElm(d_zp, Y_zp), Expr::mulElm(e_zp, Z_zp));
        auto gvec_zp = Expr::add(quad_zp, lin_zp);
        M->constraint(Expr::sub(gvec_zp, ONE_zp), Domain::greaterThan(0.0));
    }

    M->objective("obj", ObjectiveSense::Minimize, cost);
    M->solve();

    auto sol_status = M->getPrimalSolutionStatus();
    if (sol_status == SolutionStatus::Optimal || sol_status == SolutionStatus::Feasible) {
        ndarray<double, 1> a_sol = *(a->level());
        ndarray<double, 1> b_sol = *(b->level());
        ndarray<double, 1> c_sol = *(c->level());
        ndarray<double, 1> d_sol = *(d->level());
        ndarray<double, 1> e_sol = *(e->level());

        result.coeffs_a = Eigen::MatrixXd::Zero(n_segments, poly_degree_ + 1);
        result.coeffs_b = Eigen::MatrixXd::Zero(n_segments, poly_degree_ + 1);
        result.coeffs_c = Eigen::MatrixXd::Zero(n_segments, poly_degree_ + 1);
        result.coeffs_d = Eigen::MatrixXd::Zero(n_segments, poly_degree_ + 1);
        result.coeffs_e = Eigen::MatrixXd::Zero(n_segments, poly_degree_ + 1);

        for (int i = 0; i < n_segments; ++i) {
            for (int k = 0; k < poly_degree_ + 1; ++k) {
                int idx = i * (poly_degree_ + 1) + k;
                result.coeffs_a(i, k) = a_sol[idx];
                result.coeffs_b(i, k) = b_sol[idx];
                result.coeffs_c(i, k) = c_sol[idx];
                result.coeffs_d(i, k) = d_sol[idx];
                result.coeffs_e(i, k) = e_sol[idx];
            }
        }
        result.success = true;
    } else{
        result.success = false;
    }

    coeffs_ = result;
}

// ===== L-BFGS based corridor solver (unconstrained reformulation) =====
static void extractCoeffs(const corridor_lbfgs::CorridorProblem& prob,
                          const Eigen::VectorXd& x, std::vector<double>& all_ctrl) {
    int cp_total = prob.ncp();
    int nfp = prob.n_free_per_param();
    all_ctrl.resize(6 * cp_total);
    for (int p = 0; p < 6; ++p)
        prob.expandCtrl(x.data() + p * nfp, all_ctrl.data() + p * cp_total);
}

static int checkViolations(const corridor_lbfgs::CorridorProblem& prob,
                           const std::vector<double>& all_ctrl,
                           const std::vector<Eigen::Vector3d>& pts,
                           const std::vector<double>& s_vals,
                           double& worst_g,
                           std::vector<int>* violated_indices = nullptr) {
    int n = (int)pts.size();
    int n_viol = 0;
    worst_g = 1e9;
    for (int i = 0; i < n; ++i) {
        double a_v, b_v, c_v, d_v, e_v;
        prob.evalEllipseAt(all_ctrl.data(), s_vals[i], a_v, b_v, c_v, d_v, e_v);
        double y = pts[i](1), z = pts[i](2);
        double g = a_v * y * y + b_v * z * z + 2.0 * c_v * y * z + d_v * y + e_v * z;
        if (g < 1.0) {
            n_viol++;
            worst_g = std::min(worst_g, g);
            if (violated_indices) violated_indices->push_back(i);
        }
    }
    return n_viol;
}

void corGen::NLP_LBFGS(const Projection_data& pruned_points_local) {
    OptimizationResult result;
    result.success = false;

    int n_segments = n_sweep_ - 1;
    if (n_sweep_ == 1) n_segments = 1;

    std::vector<double> sweep_s(n_sweep_);
    for (int i = 0; i < n_sweep_; ++i)
        sweep_s[i] = i * spline_.getLength_3D() / (n_sweep_ - 1);

    corridor_lbfgs::CorridorProblem prob;
    prob.n_segments = n_segments;
    prob.degree = poly_degree_;
    prob.sweep_s = sweep_s;
    prob.total_arc_length = spline_.getLength_3D();
    prob.eps_psd = eps_;
    prob.w_smooth = 1e-2;
    prob.w_delta = 1.0;

    std::vector<Eigen::Vector3d> active_obs_pos = pruned_points_local.pruned_cloud_pos;
    std::vector<double> active_obs_s = pruned_points_local.s_cloud;

    int n_zp = pruned_points_local.z_plane_pts.size();
    prob.zp.resize(n_zp, 3);
    for (int i = 0; i < n_zp; ++i) {
        prob.zp(i, 0) = pruned_points_local.z_plane_pts[i].x();
        prob.zp(i, 1) = pruned_points_local.z_plane_pts[i].y();
        prob.zp(i, 2) = pruned_points_local.z_plane_s[i];
    }

    int n_free = prob.n_free_total();
    int nfp = prob.n_free_per_param();
    Eigen::VectorXd x = Eigen::VectorXd::Zero(n_free);

    double init_diag = corridor_lbfgs::inv_softplus(3.0);
    double init_l33 = corridor_lbfgs::inv_softplus(0.5);
    for (int j = 0; j < nfp; ++j) {
        x[0 * nfp + j] = init_diag;
        x[2 * nfp + j] = init_diag;
        x[5 * nfp + j] = init_l33;
    }

    lbfgs::lbfgs_parameter_t params;
    params.mem_size = 8;
    params.g_epsilon = 0.0;
    params.past = 3;
    params.delta = 1e-3;
    params.max_linesearch = 20;

    double fx = 0.0;
    int ret = 0;
    int total_refine_rounds = 0;
    const int MAX_REFINE = 2;

    for (int refine = 0; refine <= MAX_REFINE; ++refine) {
        int n_obs = (int)active_obs_pos.size();
        prob.obs.resize(n_obs, 3);
        for (int i = 0; i < n_obs; ++i) {
            prob.obs(i, 0) = active_obs_pos[i](1);
            prob.obs(i, 1) = active_obs_pos[i](2);
            prob.obs(i, 2) = active_obs_s[i];
        }
        prob.precompute();

        double penalty_stages[] = {1e3, 1e4, 1e5};
        int iter_stages[] = {30, 40, 30};
        if (refine > 0) {
            penalty_stages[0] = 1e4;
            penalty_stages[1] = 1e5;
            penalty_stages[2] = 1e6;
            iter_stages[0] = 20; iter_stages[1] = 30; iter_stages[2] = 20;
        }

        for (int stage = 0; stage < 3; ++stage) {
            prob.w_obs = penalty_stages[stage];
            prob.w_zp  = penalty_stages[stage];
            params.max_iterations = iter_stages[stage];
            ret = lbfgs::lbfgs_optimize(x, fx, corridor_lbfgs::lbfgsCostCallback,
                                        nullptr, nullptr, &prob, params);
        }

        total_refine_rounds = refine;

        if (refine >= MAX_REFINE) break;
        std::vector<double> all_ctrl;
        extractCoeffs(prob, x, all_ctrl);

        std::vector<int> violated_idx;
        double worst_g;
        int n_viol = checkViolations(prob, all_ctrl,
                                     pruned_points_local.all_raw_pos,
                                     pruned_points_local.all_raw_s,
                                     worst_g, &violated_idx);

        if (n_viol == 0 || worst_g > 0.5) break;

        int added = 0;
        for (int vi : violated_idx) {
            active_obs_pos.push_back(pruned_points_local.all_raw_pos[vi]);
            active_obs_s.push_back(pruned_points_local.all_raw_s[vi]);
            added++;
        }
    }

    std::vector<double> all_ctrl;
    extractCoeffs(prob, x, all_ctrl);
    int cp_total = prob.ncp();

    result.coeffs_a = Eigen::MatrixXd::Zero(n_segments, poly_degree_ + 1);
    result.coeffs_b = Eigen::MatrixXd::Zero(n_segments, poly_degree_ + 1);
    result.coeffs_c = Eigen::MatrixXd::Zero(n_segments, poly_degree_ + 1);
    result.coeffs_d = Eigen::MatrixXd::Zero(n_segments, poly_degree_ + 1);
    result.coeffs_e = Eigen::MatrixXd::Zero(n_segments, poly_degree_ + 1);

    for (int idx = 0; idx < cp_total; ++idx) {
        double l11 = corridor_lbfgs::softplus(all_ctrl[0 * cp_total + idx]);
        double l21 = all_ctrl[1 * cp_total + idx];
        double l22 = corridor_lbfgs::softplus(all_ctrl[2 * cp_total + idx]);
        double l31 = all_ctrl[3 * cp_total + idx];
        double l32 = all_ctrl[4 * cp_total + idx];

        int seg = idx / (poly_degree_ + 1);
        int k   = idx % (poly_degree_ + 1);

        result.coeffs_a(seg, k) = prob.eps_psd + l11 * l11;
        result.coeffs_b(seg, k) = prob.eps_psd + l21 * l21 + l22 * l22;
        result.coeffs_c(seg, k) = l11 * l21;
        result.coeffs_d(seg, k) = 2.0 * l11 * l31;
        result.coeffs_e(seg, k) = 2.0 * (l21 * l31 + l22 * l32);
    }

    int n_all_raw = (int)pruned_points_local.all_raw_pos.size();
    double worst_g_all;
    int n_violated_all = checkViolations(prob, all_ctrl,
                                         pruned_points_local.all_raw_pos,
                                         pruned_points_local.all_raw_s,
                                         worst_g_all);
    int n_active = (int)active_obs_pos.size();
    double worst_g_active;
    int n_violated_active = checkViolations(prob, all_ctrl, active_obs_pos, active_obs_s,
                                            worst_g_active);

    result.success = (ret >= 0 || ret == lbfgs::LBFGSERR_MAXIMUMITERATION);
    if (n_violated_all > n_all_raw / 4) {
        result.success = false;
    }

    coeffs_ = result;
}

EllipseParameters corGen::gen_ellipse_params(const Eigen::Matrix2d& P_in,
                                             const Eigen::Vector2d& D_in) {
    EllipseParameters ep;

    Eigen::Matrix2d E = 0.5 * (P_in + P_in.transpose());

    const double jitter = 1e-9;
    Eigen::LLT<Eigen::Matrix2d> llt;
    Eigen::Matrix2d Espd = E;
    for (int t = 0; t < 3; ++t) {
        llt.compute(Espd);
        if (llt.info() == Eigen::Success) break;
        Espd = E + (10.0 * (t+1) * jitter) * Eigen::Matrix2d::Identity();
    }
    if (llt.info() != Eigen::Success) {
        ep.center.setZero(); ep.width = ep.height = 0.0; ep.angle = 0.0;
        return ep;
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(Espd);
    Eigen::Vector2d lam = es.eigenvalues().cwiseMax(1e-6);
    Eigen::Matrix2d V = es.eigenvectors();

    Eigen::Matrix2d Espd_safe = V * lam.asDiagonal() * V.transpose();
    Eigen::LLT<Eigen::Matrix2d> llt_safe(Espd_safe);
    Eigen::Vector2d EinvD = llt_safe.solve(D_in);
    Eigen::Vector2d center = -0.5 * EinvD;

    double r2 = 1.0 + 0.25 * D_in.dot(EinvD);
    if (r2 < 0.0) r2 = 0.0;

    double sa = std::sqrt(r2 / lam(0));
    double sb = std::sqrt(r2 / lam(1));
    int imax = (sa >= sb) ? 0 : 1;
    double angle = std::atan2(V(1, imax), V(0, imax));

    double center_bound = ellipse_axis_max;
    if (center.norm() > center_bound) {
        center = center.normalized() * center_bound;
    }

    ep.center = center;
    ep.width  = 2.0 * sa;
    ep.height = 2.0 * sb;
    ep.angle  = angle;

    if (!std::isfinite(ep.width) || !std::isfinite(ep.height) ||
        !std::isfinite(ep.angle) || !std::isfinite(ep.center.norm())) {
        ep.width = ep.height = 0.0; ep.angle = 0.0; ep.center.setZero();
    }
    return ep;
}


Eigen::Vector2d corGen::get_ellipse_points(const double width, const double height, const double angle, const double theta){
    double a = width / 2.0;
    double b = height / 2.0;

    if (a < 1e-12 || b < 1e-12) {
        return Eigen::Vector2d::Zero();
    }

    Eigen::Matrix2d rotation_matrix;
    rotation_matrix << std::cos(angle), -std::sin(angle),
                       std::sin(angle),  std::cos(angle);
    
    double dif_angle = theta - angle;
    double denom = std::sqrt(std::pow(a * std::sin(dif_angle), 2) + std::pow(b * std::cos(dif_angle), 2));
    if (denom < 1e-12) {
        return Eigen::Vector2d::Zero();
    }
    double r_ellipse = a * b / denom;
    Eigen::Vector2d rot = r_ellipse * Eigen::Vector2d(std::cos(dif_angle), std::sin(dif_angle));
    Eigen::Vector2d ellipse_point = rotation_matrix * rot;
    return ellipse_point;
}

std::vector<double> corGen::polynomial_eval(double s, const OptimizationResult& coeffs, int degree) const {
    int n_segments = n_sweep_ - 1;
    std::vector<double> s_segments(n_segments + 1);
    for (int i = 0; i <= n_segments; ++i) {
        s_segments[i] = i * spline_.getLength_3D() / n_segments;
    }
    int segment_idx = std::max(0, std::min(n_segments - 1, 
        int(std::upper_bound(s_segments.begin(), s_segments.end(), s) - s_segments.begin()) - 1));

    double a = 0.0, b = 0.0, c = 0.0, d = 0.0, e = 0.0;
    
    for (int k = 0; k < degree + 1; ++k) {
        double denom = s_segments[segment_idx + 1] - s_segments[segment_idx];
        double t = denom > 0 ? (s - s_segments[segment_idx]) / denom : 0.0;
        double basis_val = poly_basis(t, k, degree);
        a += coeffs.coeffs_a(segment_idx, k) * basis_val;
        b += coeffs.coeffs_b(segment_idx, k) * basis_val;
        c += coeffs.coeffs_c(segment_idx, k) * basis_val;
        d += coeffs.coeffs_d(segment_idx, k) * basis_val;
        e += coeffs.coeffs_e(segment_idx, k) * basis_val;
    }
    
    return {a, b, c, d, e};
}

CorridorVisualization corGen::cor_vis(const OptimizationResult& coeffs, std::string frame) {
    CorridorVisualization corridor_vis;
    const auto& s_eval = spline_.getPathData().s;
    Eigen::VectorXd angles = Eigen::VectorXd::LinSpaced(n_angles_, 0.0, 2.0 * M_PI);

    corridor_vis.ellipse_pts_world.resize(n_eval_, std::vector<Eigen::Vector3d>(n_angles_));

    std::vector<double> areas(n_eval_);


    double prev_w = -1.0, prev_h = -1.0;

    for (int i = 0; i < n_eval_; ++i) {
        double s = s_eval(i);
        std::vector<double> coeffs_eval = polynomial_eval(s, coeffs, poly_degree_);

        Eigen::Matrix2d P;
        P << coeffs_eval[0], coeffs_eval[2],
             coeffs_eval[2], coeffs_eval[1];

        Eigen::Vector2d pp(coeffs_eval[3], coeffs_eval[4]);

        EllipseParameters ellipse_params = gen_ellipse_params(P, pp);

        bool degenerate = (ellipse_params.width < 1e-6 || ellipse_params.height < 1e-6 ||
                           !std::isfinite(ellipse_params.width) || !std::isfinite(ellipse_params.height));

        double clamped_w = std::min(ellipse_params.width, 2.0 * ellipse_axis_max);
        double clamped_h = std::min(ellipse_params.height, 2.0 * ellipse_axis_max);

        areas[i] = degenerate ? 0.0 : M_PI * clamped_w * clamped_h / 4.0;

        if (!degenerate) { prev_w = clamped_w; prev_h = clamped_h; }

        Eigen::Vector3d pos = spline_.getPosition_3D(s);
        Eigen::Matrix3d frame_matrix;
        if (frame == "FSF") {
            frame_matrix = fsf_data_.FSF[i];
        } else if (frame == "PTF") {
            frame_matrix = ptf_data_.PTF[i];
        } else {
            throw std::runtime_error("Unknown frame type: " + frame);
        }

        for (int j = 0; j < n_angles_; ++j) {
            if (degenerate) {
                corridor_vis.ellipse_pts_world[i][j] = pos;
                continue;
            }
            double theta = angles(j);
            Eigen::Vector2d ellipse_point = get_ellipse_points(clamped_w, clamped_h, ellipse_params.angle, theta);
            Eigen::Vector2d center_basis = ellipse_point + ellipse_params.center;
            corridor_vis.ellipse_pts_world[i][j] = pos + center_basis(0) * frame_matrix.col(1) + center_basis(1) * frame_matrix.col(2);
        }

    }

    corridor_vis.parametric_volume = 0.0;
    for (int i = 0; i < n_eval_ - 1; ++i) {
        double dx  = s_eval(i + 1) - s_eval(i);
        double area = (areas[i] + areas[i + 1]) / 2.0; 
        corridor_vis.parametric_volume += area * dx;
    }

    return corridor_vis;
}

double corGen::computeVolume(const OptimizationResult& coeffs) {
    const auto& s_eval = spline_.getPathData().s;
    double vol = 0.0;
    for (int i = 0; i < n_eval_ - 1; ++i) {
        double ds = s_eval(i + 1) - s_eval(i);
        auto getArea = [&](int idx) -> double {
            double s = s_eval(idx);
            std::vector<double> ce = polynomial_eval(s, coeffs, poly_degree_);
            Eigen::Matrix2d P;
            P << ce[0], ce[2], ce[2], ce[1];
            Eigen::Vector2d pp(ce[3], ce[4]);
            EllipseParameters ep = gen_ellipse_params(P, pp);
            if (ep.width < 1e-6 || ep.height < 1e-6 ||
                !std::isfinite(ep.width) || !std::isfinite(ep.height)) return 0.0;
            double w = std::min(ep.width, 2.0 * ellipse_axis_max);
            double h = std::min(ep.height, 2.0 * ellipse_axis_max);
            return M_PI * w * h / 4.0;
        };
        vol += (getArea(i) + getArea(i + 1)) / 2.0 * ds;
    }
    return vol;
}

void corGen::get_cage(const CageParameters& params, std::string frame) {
    std::vector<double> s_wrap;
    int n_cage_points = 30;
    for (int i = 0; i < n_cage_points; ++i) s_wrap.push_back(0.0 + i * 1.0 / (n_cage_points - 1));

    Eigen::Vector3d horizontal_vec(1, 0, 0);

    double total_length = spline_.getLength_3D();

    int initial_size = cloudMap_.points.size();

    for(int i = 0; i < s_wrap.size(); ++i) {
        double s = s_wrap[i];
        double s_val = s * total_length;
        
        const auto& s_points = spline_.getPathData().s;
        int ind_i = 0;
        double min_diff = std::abs(s_points[0] - s_val);
        for(int j = 1; j < s_points.size(); ++j) {
            double diff = std::abs(s_points[j] - s_val);
            if(diff < min_diff) {
                min_diff = diff;
                ind_i = j;
            }
        }    

        Eigen::Vector3d p_i = spline_.getPosition_3D(s_points[ind_i]);
        Eigen::Matrix3d frame_matrix;
        if (frame == "FSF") {
            frame_matrix = fsf_data_.FSF[ind_i];
        } else if (frame == "PTF") {
            frame_matrix = ptf_data_.PTF[ind_i];
        } else {
            throw std::runtime_error("Unknown frame type: " + frame);
        }
        Eigen::Vector3d e1_i = frame_matrix.col(0);

        Eigen::Vector3d z_axis(0, 0, 1);
        Eigen::Vector3d i_horizontal = e1_i.cross(z_axis);
        if (i_horizontal.norm() < 1e-6) {
            Eigen::Vector3d y_axis(0, 1, 0);
            i_horizontal = e1_i.cross(y_axis);
        }
        i_horizontal.normalize();
        Eigen::Vector3d i_vertical = i_horizontal.cross(e1_i);
        i_vertical.normalize();

        double local_h_min, local_h_max;
        if (std::abs(i_vertical.z()) > 1e-6) {
            double floor_v = (z_floor_ - p_i.z()) / i_vertical.z();
            double ceil_v  = (z_ceiling_ - p_i.z()) / i_vertical.z();
            local_h_min = std::min(floor_v, ceil_v);
            local_h_max = std::max(floor_v, ceil_v);
            local_h_min = std::max(local_h_min, -2.0 * ellipse_axis_max);
            local_h_max = std::min(local_h_max,  2.0 * ellipse_axis_max);
        } else {
            local_h_min = params.h_min;
            local_h_max = params.h_min + params.h;
        }

        for(int j = 0; j < params.n_topbottom; ++j) {
            double h_offset = -params.l/2.0 + j * params.l / (params.n_topbottom - 1);
            
            Eigen::Vector3d bottom_point = p_i + local_h_min * i_vertical + h_offset * i_horizontal;
            pcl::PointXYZ bottom_pcl;
            bottom_pcl.x = bottom_point(0);
            bottom_pcl.y = bottom_point(1);
            bottom_pcl.z = bottom_point(2);
            cloudMap_.points.push_back(bottom_pcl);
            
            Eigen::Vector3d top_point = p_i + local_h_max * i_vertical + h_offset * i_horizontal;
            pcl::PointXYZ top_pcl;
            top_pcl.x = top_point(0);
            top_pcl.y = top_point(1);
            top_pcl.z = top_point(2);
            cloudMap_.points.push_back(top_pcl);
        }

        for(int j = 0; j < params.n_sides; ++j) {
            double v_offset = local_h_min + j * (local_h_max - local_h_min) / (params.n_sides - 1);
            
            Eigen::Vector3d right_point = p_i + (params.l/2.0) * i_horizontal + v_offset * i_vertical;
            pcl::PointXYZ right_pcl;
            right_pcl.x = right_point(0);
            right_pcl.y = right_point(1);
            right_pcl.z = right_point(2);
            cloudMap_.points.push_back(right_pcl);
            
            Eigen::Vector3d left_point = p_i + (-params.l/2.0) * i_horizontal + v_offset * i_vertical;
            pcl::PointXYZ left_pcl;
            left_pcl.x = left_point(0);
            left_pcl.y = left_point(1);
            left_pcl.z = left_point(2);
            cloudMap_.points.push_back(left_pcl);
        }
        
        if(params.covers && (i == 0 || i == (int)s_wrap.size() - 1)) {
            double h_range = local_h_max - local_h_min;
            std::vector<double> cover_heights = {local_h_min, 
                                               local_h_min + h_range/3.0,
                                               local_h_min + 2*h_range/3.0, 
                                               local_h_max};
            
            for(double hh : cover_heights) {
                for(int j = 0; j < params.n_topbottom; ++j) {
                    double h_offset = -params.l/2.0 + j * params.l / (params.n_topbottom - 1);
                    
                    Eigen::Vector3d cover_point = p_i + hh * i_vertical + h_offset * i_horizontal;
                    pcl::PointXYZ cover_pcl;
                    cover_pcl.x = cover_point(0);
                    cover_pcl.y = cover_point(1);
                    cover_pcl.z = cover_point(2);
                    cloudMap_.points.push_back(cover_pcl);
                }
            }
        }        
    }

    cloudMap_.width = cloudMap_.points.size();
    cloudMap_.height = 1;
    cloudMap_.is_dense = true;

}

Eigen::Matrix3d corGen::initial_frame(const Eigen::Vector3d& e10, const Eigen::Vector3d& e3_des) {
    Eigen::Vector3d e30 = closest_to_A_perpendicular_to_B(e3_des, e10);
    if (e30.squaredNorm() < 1e-12) {
        throw std::runtime_error("e30 is too close to zero, cannot normalize.");
    }
    e30.normalize();
    
    Eigen::Vector3d e20 = e30.cross(e10);
    e20.normalize();
    
    Eigen::Matrix3d PTF0;
    PTF0.col(0) = e10;
    PTF0.col(1) = e20;
    PTF0.col(2) = e30;
    
    return PTF0;
}

void corGen::PTF_evaluate(const ArcLengthSpline& spline) {
    ptf_data_.PTF.assign(n_eval_, Eigen::Matrix3d::Identity());

    std::vector<double> sgrid(n_eval_);
    const double smax = spline.getLength_3D();
    for (int i=0;i<n_eval_;++i) sgrid[i] = smax * double(i)/double(n_eval_-1);

    Eigen::Vector3d p0 = spline.getPosition_3D(sgrid[0]);
    Eigen::Vector3d v0 = spline.getDerivative_3D(sgrid[0]);
    if (v0.norm() < 1e-9) {
        Eigen::Vector3d p1 = spline.getPosition_3D(sgrid[1]);
        v0 = (p1 - p0) / std::max(sgrid[1]-sgrid[0],1e-9);
    }
    Eigen::Matrix3d R = initial_frame(v0.normalized(), Eigen::Vector3d::UnitZ());
    ptf_data_.PTF[0] = R;

    for (int i=0;i<n_eval_-1;++i){
        Eigen::Vector3d t0 = R.col(0);
        Eigen::Vector3d v1 = spline.getDerivative_3D(sgrid[i+1]);
        Eigen::Vector3d t1 = v1.normalized();
        Eigen::Vector3d k = t0.cross(t1);
        double s = k.norm(), c = t0.dot(t1);
        Eigen::Matrix3d Rmin = Eigen::Matrix3d::Identity();
        if (s > 1e-9) {
            Eigen::Vector3d ax = k / s;
            double th = std::atan2(s, c);
            Eigen::Matrix3d K; K << 0,-ax.z(),ax.y(), ax.z(),0,-ax.x(), -ax.y(),ax.x(),0;
            Rmin = Eigen::Matrix3d::Identity() + std::sin(th)*K + (1-std::cos(th))*(K*K);
        } else if (c < 0.0) {
            Eigen::Vector3d ax = t0.unitOrthogonal();
            Eigen::Matrix3d K; K << 0,-ax.z(),ax.y(), ax.z(),0,-ax.x(), -ax.y(),ax.x(),0;
            Rmin = Eigen::Matrix3d::Identity() + 2.0*(K*K);
        }

        Eigen::Vector3d e2 = (Rmin * R.col(1)).normalized();
        Eigen::Vector3d e3 = t1.cross(e2).normalized();
        R.col(0) = t1;  R.col(1) = e2;  R.col(2) = e3;

        if (R.col(1).dot(ptf_data_.PTF[i].col(1)) < 0) { R.col(1) = -R.col(1); R.col(2) = t1.cross(R.col(1)); }

        Eigen::JacobiSVD<Eigen::Matrix3d> svd(R, Eigen::ComputeFullU|Eigen::ComputeFullV);
        R = svd.matrixU()*svd.matrixV().transpose();

        ptf_data_.PTF[i+1] = R;
    }
}


OptimizationResult corGen::corGeneratorPTF(const ArcLengthSpline& spline) {
    updateSpline(spline);
    PTF_evaluate(spline);
    Projection_data pruned_points_local = processPointCloud("PTF");
    NLP_LBFGS(pruned_points_local);
    if (coeffs_.success) coeffs_.volume = computeVolume(coeffs_);
    const bool coeff_finite = coeffs_.coeffs_a.allFinite() && coeffs_.coeffs_b.allFinite()
                           && coeffs_.coeffs_c.allFinite() && coeffs_.coeffs_d.allFinite()
                           && coeffs_.coeffs_e.allFinite();
    return coeffs_;
}

Eigen::Matrix3d corGen::PTF_at_s(double s, const ArcLengthSpline& spline) const {
    const double smax = spline.getLength_3D();
    const int N = n_eval_;
    const double ds = smax / std::max(1, N-1);

    int i = std::max(0, std::min(int(std::floor(s/ds)), N-2));
    double s0 = i * ds, s1 = (i+1) * ds;
    double alpha = (s1 > s0) ? (s - s0) / (s1 - s0) : 0.0;

    Eigen::Matrix3d R0 = ptf_data_.PTF[i];
    Eigen::Matrix3d R1 = ptf_data_.PTF[i+1];
    Eigen::Quaterniond q0(R0), q1(R1);
    if (q0.dot(q1) < 0.0) q1.coeffs() *= -1.0;
    Eigen::Quaterniond qi = q0.slerp(alpha, q1);
    Eigen::Matrix3d R = qi.toRotationMatrix();

    Eigen::Vector3d t_ref = spline.getDerivative_3D(s).normalized();
    Eigen::Vector3d t_cur = R.col(0);
    Eigen::Vector3d k = t_cur.cross(t_ref);
    double sn = k.norm(), cs = t_cur.dot(t_ref);
    if (sn > 1e-9) {
        Eigen::Vector3d ax = k / sn;
        double th = std::atan2(sn, cs);
        Eigen::AngleAxisd aa(th, ax);
        R = aa.toRotationMatrix() * R;
    } else if (cs < 0.0) {
        Eigen::Vector3d ax = t_cur.unitOrthogonal();
        R = (Eigen::AngleAxisd(M_PI, ax)).toRotationMatrix() * R;
    }

    Eigen::Vector3d t = R.col(0).normalized();
    Eigen::Vector3d n = (R.col(1) - R.col(1).dot(t)*t).normalized();
    Eigen::Vector3d b = t.cross(n);
    R.col(0)=t; R.col(1)=n; R.col(2)=b;
    if ((R.col(0).cross(R.col(1))).dot(R.col(2)) < 0.0) {
        R.col(1) = -R.col(1);
        R.col(2) =  R.col(0).cross(R.col(1));
    }
    return R;
}