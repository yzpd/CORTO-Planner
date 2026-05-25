#include "map/map.hpp"
#include "ros_interface/ros_interface.hpp"
#include "path_searching/topo_prm.hpp"
#include "cubic_spline/arc_length_spline.hpp"
#include "optimization/ref_traj_opt.hpp"
#include <thread>
#include <mutex>
#include <atomic>
#include <memory>
#include "corridor/corGen.hpp"
#include "nmpc/nominal_nmpc_cor.hpp"
#include "utility/utility.hpp"
#include <chrono>

enum FSMState { WAIT_TARGET, PLAN_LOCAL, EXEC_TRAJ, GOAL_REACHED };

struct PlanResult {
    ArcLengthSpline spline;
    OptimizationResult coeffs;
    RefTrajOpt ref_traj_opt;
    double spline_t_total = 0.0;
    int pool_id = 0;
    int corridor_idx = 0;
    Eigen::Vector3d stitch_point = Eigen::Vector3d::Zero();
    double replan_ms = 0.0;
    std::vector<std::vector<Eigen::Vector3d>> viz_paths;
    bool valid = false;
    bool topo_failed = false;
};

static double initProgressTime(const RefTrajOpt& ref_traj, const ArcLengthSpline& spline,
                               const Eigen::Vector3d& pos, double spline_t_total) {
    const double spline_len = spline.getLength_3D();
    if (spline_len <= 0.0) return 0.0;
    const auto& result_t = ref_traj.get_result_t();
    const int t_size = static_cast<int>(result_t.size());
    if (t_size < 2) return 0.0;
    std::vector<double> cum(t_size, 0.0);
    for (int i = 1; i < t_size; ++i) cum[i] = cum[i - 1] + result_t[i];
    const double s_proj = projectOnSpline(spline, pos);
    double t_init = arcLengthToTime(cum, s_proj, spline_len);
    return std::max(0.0, std::min(t_init, spline_t_total));
}

int main(int argc, char **argv){
    ros::init(argc, argv, "planner");
    ros::NodeHandle nh("~");

    ros::Duration(2.0).sleep();

    std::string map_path;
    nh.getParam("map_file", map_path);
    double MAX_VEL, MAX_ACC;
    nh.param("ref/max_velocity", MAX_VEL, 3.0);
    nh.param("ref/max_acceleration", MAX_ACC, 3.0);
    double MAX_JERK = 100;
    bool ref_use_jerk = true;
    int NV = 100;

    // Replanning parameters
    double planning_horizon, replan_thresh, emergency_time, goal_tolerance;
    planning_horizon = 8.0;
    replan_thresh = 1.0;
    emergency_time = 0.8;
    goal_tolerance = 0.2;
    double goal_z_offset = 0.0;

    double obstacle_inflation = 0.0;
    nh.param("obstacle_inflation", obstacle_inflation, 0.0);
    double ellipse_axis_max_val = 2.0;
    nh.param("ellipse_axis_max", ellipse_axis_max_val, 2.0);
    double sensing_radius = std::min(ellipse_axis_max_val, 3.0) + 0.5;

    GridMap gridmap;
    gridmap.read_json_file(map_path);
    ROS_INFO("Loaded JSON map: %s", map_path.c_str());
    gridmap.update_grid_map(obstacle_inflation, false);

    int MaxPathN = 5;
    std::vector<corGen> corridor_gen(MaxPathN);
    for (auto& gen : corridor_gen) {
        gen.init(nh);
    }
    std::vector<corGen> corridor_gen_bg(MaxPathN);
    for (auto& gen : corridor_gen_bg) {
        gen.init(nh);
    }
    std::vector<corGen>* cor_pools[2] = {&corridor_gen, &corridor_gen_bg};
    int active_cor_pool = 0;

    Vector3d map_size = gridmap.size();
    RosInterface ros_inte(map_size);
    ros::Duration(2.0).sleep();

    ros_inte.publish_map_surface(gridmap);
    pcl::PointCloud<pcl::PointXYZ> global_cloud = ros_inte.grid_to_cloud(gridmap);
    for (auto& gen : corridor_gen) gen.setCloudMap(global_cloud);
    for (auto& gen : corridor_gen_bg) gen.setCloudMap(global_cloud);
    ros::spinOnce();

    unique_ptr<TopologyPRM> topo_prm(new TopologyPRM());
    topo_prm->setGridMap(&gridmap);
    topo_prm->init(nh);

    double start_pos_x, start_pos_y, start_pos_z;
    nh.getParam("start_pos_x", start_pos_x);
    nh.getParam("start_pos_y", start_pos_y);
    nh.getParam("start_pos_z", start_pos_z);
    Vector3d start_pos(start_pos_x, start_pos_y, start_pos_z);
    const Vector3d map_origin(-map_size(0) / 2.0, -map_size(1) / 2.0, -0.01);
    start_pos += map_origin;

    double goal_pos_x, goal_pos_y, goal_pos_z;
    nh.param("goal_pos_x", goal_pos_x, 13.0);
    nh.param("goal_pos_y", goal_pos_y, -3.0);
    nh.param("goal_pos_z", goal_pos_z, 1.0);
    Vector3d launch_goal(goal_pos_x, goal_pos_y, goal_pos_z);
    launch_goal += map_origin;

    geometry_msgs::PoseStamped::ConstPtr nav_goal_msg;
    ros::Subscriber goal_sub = nh.subscribe<geometry_msgs::PoseStamped>(
        "/move_base_simple/goal", 1,
        [&](const geometry_msgs::PoseStamped::ConstPtr& msg) {
            nav_goal_msg = msg;
        }
    );

    ros::spinOnce();

    ROS_INFO("Setting up NMPC ...");
    double dt = 0.05;
    ros::Rate rate(1.0 / dt);
    double hover_ratio = 0.6;
    NominalNMPC nmpc(hover_ratio, "LD_SLSQP", 200, dt, nh);
    Eigen::Matrix<double, 3 + NominalQuadDynamic::u_dim_ + 4, 1> cost_w;
    double w_theta = 5.0;
    nh.param("nmpc/w_theta", w_theta, 5.0);
    cost_w << 0.0, 0.0, 0.005,
                0.2, 0.2, 0.2, 0.2,
                60, 100, 10.0,
                w_theta;
    nmpc.set_w(cost_w);

    FSMState fsm_state = WAIT_TARGET;
    Vector3d final_goal;
    bool first_plan = true;

    ArcLengthSpline best_spline;
    int best_corridor_idx = -1;
    OptimizationResult coeffs4NMPC;
    RefTrajOpt best_ref_traj_opt;

    double t_cur = 0.0;
    double total_length = 0.0;
    double spline_t_total = 0.0;
    double kp_progress = 0.5;
    nh.param("nmpc/kp_progress", kp_progress, 0.5);

    std::mutex pending_mutex;
    PlanResult pending_result;
    std::atomic<bool> replan_running{false};
    std::atomic<bool> new_plan_ready{false};

    double replan_latency_ema = 0.05;
    double stitch_tolerance = 0.5;

    Eigen::Matrix<double, NominalQuadDynamic::x_dim_, 1> x0;
    x0 << start_pos(0), start_pos(1), start_pos(2),
          0.0, 0.0, 0.0,
          0.0, 0.0, 0.0, 1.0;
    Eigen::Matrix<double, NominalQuadDynamic::u_dim_, 1> u0;
    u0 << 0.0, 0.0, 0.0, hover_ratio;
    Eigen::Matrix<double, NominalNMPC::n_step_, NominalQuadDynamic::u_dim_> U;
    U.setZero();
    for (int k = 0; k < NominalNMPC::n_step_; ++k) U(k, 3) = hover_ratio;
    Eigen::Matrix<double, NominalNMPC::n_step_, NominalQuadDynamic::x_dim_> last_pred_x;
    bool last_pred_x_valid = false;
    bool last_nmpc_ok = false;

    ros::Duration(1.0).sleep();

    bool is_goal_rviz = false;
    ROS_INFO("Start: [%.2f, %.2f, %.2f]", start_pos.x(), start_pos.y(), start_pos.z());
    ROS_INFO("Goal: [%.2f, %.2f, %.2f]", launch_goal.x(), launch_goal.y(), launch_goal.z());
    string frame = "PTF";
    nh.getParam("frame", frame);

    int replan_count = 0;
    int consecutive_fails = 0;
    const int MAX_CONSECUTIVE_FAILS = 50;

    while (ros::ok()) {

        switch (fsm_state) {

        case WAIT_TARGET: {
            bool start_mission = false;
            if (nav_goal_msg) {
                if (is_goal_rviz) {
                    final_goal = Vector3d(nav_goal_msg->pose.position.x,
                                          nav_goal_msg->pose.position.y,
                                          nav_goal_msg->pose.position.z + goal_z_offset);
                } else {
                    final_goal = launch_goal;
                    final_goal.z() += goal_z_offset;
                }
                nav_goal_msg.reset();
                start_mission = true;
            } else if (!is_goal_rviz && first_plan) {
                final_goal = launch_goal;
                final_goal.z() += goal_z_offset;
                start_mission = true;
            }

            if (start_mission) {
                first_plan = true;
                total_length = 0.0;
                replan_count = 0;

                x0 << start_pos(0), start_pos(1), start_pos(2),
                        0.0, 0.0, 0.0,
                        0.0, 0.0, 0.0, 1.0;
                
                u0 << 0.0, 0.0, 0.0, hover_ratio;
                U.setZero();
                for (int k = 0; k < NominalNMPC::n_step_; ++k) U(k, 3) = hover_ratio;
                last_pred_x_valid = false;
                last_nmpc_ok = false;

                fsm_state = PLAN_LOCAL;
            }
            break;
        }

        case PLAN_LOCAL: {
            Vector3d plan_start = x0.head<3>();
            Vector3d dir_to_goal = final_goal - plan_start;
            double dist_to_goal = dir_to_goal.norm();
            bool is_last_segment = (dist_to_goal <= planning_horizon * 1.2);
            Vector3d local_goal = is_last_segment ? final_goal
                                 : (plan_start + planning_horizon * dir_to_goal.normalized());

            double v_start_val = first_plan ? 0.0 : Eigen::Vector3d(x0(VX), x0(VY), x0(VZ)).norm();

            // ---- Topo PRM ----
            vector<Eigen::Vector3d> empty_start_pts, empty_end_pts;
            list<GraphNode::Ptr> graph;
            vector<vector<Eigen::Vector3d>> raw_paths, filtered_paths;

            topo_prm->findTopoPaths(plan_start, local_goal,
                                    empty_start_pts, empty_end_pts,
                                    graph, raw_paths, filtered_paths);

            consecutive_fails = 0;
            ros_inte.visualizeGraphNodes(graph);

            // ---- Spline fitting ----
            vector<vector<Eigen::Vector3d>> spline_topo_paths;
            vector<ArcLengthSpline> splines;

            for (const auto& group : filtered_paths) {
                if (group.size() < 2) continue;

                vector<Eigen::Vector3d> dense_path;
                int segments_per_edge = 30;

                for (size_t i = 0; i < group.size() - 1; ++i) {
                    Eigen::Vector3d s = group[i];
                    Eigen::Vector3d e = group[i + 1];
                    for (int j = 0; j < segments_per_edge; ++j) {
                        double t = (double)j / segments_per_edge;
                        dense_path.push_back(s + t * (e - s));
                    }
                }
                dense_path.push_back(group.back());

                ArcLengthSpline spline;
                Eigen::VectorXd X(dense_path.size()), Y(dense_path.size()), Z(dense_path.size());
                for (size_t i = 0; i < dense_path.size(); ++i) {
                    X(i) = dense_path[i].x();
                    Y(i) = dense_path[i].y();
                    Z(i) = dense_path[i].z();
                }
                spline.gen3DSpline(X, Y, Z);
                splines.push_back(spline);

                double total_arc_length = spline.getLength_3D();
                Eigen::VectorXd s_vec;
                s_vec.setLinSpaced(NV, 0, total_arc_length);
                vector<Eigen::Vector3d> spline_path;
                for (int i = 0; i < NV; ++i) {
                    spline_path.push_back(spline.getPosition_3D(s_vec(i)));
                }
                spline_topo_paths.push_back(spline_path);
            }


            if (splines.empty()) {
                if (first_plan) {ros_inte.publish_stop(); fsm_state = WAIT_TARGET; }
                else fsm_state = EXEC_TRAJ;
                break;
            }

            vector<RefTrajOpt> ref_traj_opts(splines.size());
            vector<vector<Eigen::Vector3d>> filtered_spline_paths;
            std::vector<int> valid_indices;
            vector<std::thread> opt_threads;
            for (size_t i = 0; i < splines.size(); ++i) {
                opt_threads.push_back(std::thread([&ref_traj_opts, i, &splines, MAX_VEL, MAX_ACC, MAX_JERK, NV, v_start_val, ref_use_jerk]() {
                    RefTrajOpt ref_traj_opt;
                    ref_traj_opt.socp_interface(NV, v_start_val, 0.0, splines[i], MAX_VEL, MAX_ACC, ref_use_jerk, MAX_JERK);
                    ref_traj_opts[i] = ref_traj_opt;
                }));
            }
            for (auto& t : opt_threads) {
                if (t.joinable()) t.join();
            }

            vector<double> total_times(ref_traj_opts.size(), 0.0);
            for (size_t i = 0; i < ref_traj_opts.size(); ++i) {
                total_times[i] = ref_traj_opts[i].get_total_time();
            }
            double time_threshold = 0.2;
            double min_time = 1e5;
            for (size_t i = 0; i < total_times.size(); ++i) {
                if (total_times[i] > 1e-6 && ref_traj_opts[i].get_result_t().size() >= 2)
                    min_time = std::min(min_time, total_times[i]);
            }
            if (min_time == 1e5) {
                if (first_plan) {ros_inte.publish_stop(); fsm_state = WAIT_TARGET; }
                else fsm_state = EXEC_TRAJ;
                break;
            }
            for (size_t i = 0; i < total_times.size(); ++i) {
                if (total_times[i] > 1e-6 && ref_traj_opts[i].get_result_t().size() >= 2
                    && total_times[i] < min_time + time_threshold) {
                    valid_indices.push_back(i);
                    filtered_spline_paths.push_back(spline_topo_paths[i]);
                }
            }
            
            vector<std::thread> corridor_threads;
            vector<OptimizationResult> all_coeffs(valid_indices.size());
            vector<bool> corridor_success(valid_indices.size(), false);

            if ((int)valid_indices.size() > MaxPathN) {
                if (first_plan) {ros_inte.publish_stop(); fsm_state = WAIT_TARGET; }
                else fsm_state = EXEC_TRAJ;
                break;
            }

            if (frame == "FSF") {
                for (size_t i = 0; i < valid_indices.size(); ++i) {
                    corridor_threads.push_back(std::thread([&, i]() {
                        OptimizationResult coeffs = corridor_gen[i].corGenerator(splines[valid_indices[i]]);
                        all_coeffs[i] = coeffs;
                        corridor_success[i] = coeffs.success;
                    }));
                }
            } else if (frame == "PTF") {
                for (size_t i = 0; i < valid_indices.size(); ++i) {
                    corridor_threads.push_back(std::thread([&, i]() {
                        OptimizationResult coeffs = corridor_gen[i].corGeneratorPTF(splines[valid_indices[i]]);
                        all_coeffs[i] = coeffs;
                        corridor_success[i] = coeffs.success;
                    }));
                }
            }

            for (auto& t : corridor_threads) {
                if (t.joinable()) t.join();
            }
            int successful_corridors = 0;
            int local_best_idx = -1;
            double best_volume = 0.0;

            for (size_t i = 0; i < corridor_success.size(); ++i) {
                if (corridor_success[i]) {
                    successful_corridors++;
                    if (all_coeffs[i].volume > best_volume) {
                        best_volume = all_coeffs[i].volume;
                        local_best_idx = i;
                    }
                }
            }

            if (!std::isfinite(best_volume) || successful_corridors == 0) {
                if (first_plan) {ros_inte.publish_stop(); fsm_state = WAIT_TARGET; }
                else fsm_state = EXEC_TRAJ;
                break;
            }

            best_corridor_idx = local_best_idx;
            best_spline = splines[valid_indices[best_corridor_idx]];
            coeffs4NMPC = all_coeffs[best_corridor_idx];
            best_ref_traj_opt = ref_traj_opts[valid_indices[best_corridor_idx]];
            spline_t_total = best_ref_traj_opt.get_total_time();
            t_cur = initProgressTime(best_ref_traj_opt, best_spline, x0.head<3>(), spline_t_total);
            active_cor_pool = 0;
            last_pred_x_valid = false;
            last_nmpc_ok = false;

            first_plan = false;
            replan_count++;

            ros_inte.drawFilterPaths(filtered_spline_paths, 0.1, 0, 1);
            CorridorVisualization corridor_vis = corridor_gen[best_corridor_idx].cor_vis(coeffs4NMPC, frame);
            ros_inte.visualizeCorridorMarkers(corridor_vis);
            fsm_state = EXEC_TRAJ;
            ros::spinOnce();
            break;
        }

        case EXEC_TRAJ: {
            if (new_plan_ready.load()) {
                PlanResult ready_plan;
                {
                    std::lock_guard<std::mutex> lk(pending_mutex);
                    ready_plan = std::move(pending_result);
                    new_plan_ready.store(false);
                }

                if (ready_plan.valid) {
                    const double stitch_err = (x0.head<3>() - ready_plan.stitch_point).norm();
                    replan_latency_ema = 0.9 * replan_latency_ema + 0.1 * (ready_plan.replan_ms / 1000.0);
                    replan_latency_ema = std::max(dt, replan_latency_ema);

                    if (stitch_err <= stitch_tolerance) {
                        best_spline = std::move(ready_plan.spline);
                        coeffs4NMPC = std::move(ready_plan.coeffs);
                        best_ref_traj_opt = std::move(ready_plan.ref_traj_opt);
                        spline_t_total = ready_plan.spline_t_total;
                        active_cor_pool = ready_plan.pool_id;
                        best_corridor_idx = ready_plan.corridor_idx;
                        t_cur = initProgressTime(best_ref_traj_opt, best_spline, x0.head<3>(), spline_t_total);
                        replan_count++;
                        consecutive_fails = 0;

                        U.setZero();
                        for (int k = 0; k < NominalNMPC::n_step_; ++k) U(k, 3) = hover_ratio;
                        u0 << 0.0, 0.0, 0.0, hover_ratio;

                        ros_inte.drawFilterPaths(ready_plan.viz_paths, 0.1, 0, 1);
                        corGen& vis_cor = (*cor_pools[active_cor_pool])[best_corridor_idx];
                        CorridorVisualization corridor_vis = vis_cor.cor_vis(coeffs4NMPC, frame);
                        ros_inte.visualizeCorridorMarkers(corridor_vis);
                    }
                } else if (ready_plan.topo_failed) {
                    consecutive_fails++;
                    if (consecutive_fails >= MAX_CONSECUTIVE_FAILS) {
                        fsm_state = GOAL_REACHED;
                        break;
                    }
                }
            }

            double dist_to_goal = (final_goal - x0.head<3>()).norm();
            if (dist_to_goal < goal_tolerance) {
                ROS_INFO("[FSM] GOAL_REACHED!");
                fsm_state = GOAL_REACHED;
                break;
            }

            double no_replan_dist = std::max(2.0, goal_tolerance * 4.0);
            bool near_goal = (dist_to_goal < no_replan_dist);

            bool need_replan = false;
            if (!near_goal) {
                if (t_cur > replan_thresh) {
                    need_replan = true;
                }
                if (t_cur > spline_t_total - emergency_time) {
                    need_replan = true;
                }
            }

            if (need_replan && !replan_running.load()) {
                replan_running.store(true);
                int bg_pool_id = 1 - active_cor_pool;

                Vector3d plan_start_pos;
                double snap_v;
                if (last_pred_x_valid && last_nmpc_ok) {
                    const int k_pred = std::min(
                        NominalNMPC::n_step_ - 1,
                        std::max(0, static_cast<int>(std::ceil(replan_latency_ema / dt)) - 1));
                    plan_start_pos = last_pred_x.row(k_pred).head<3>().transpose();
                    snap_v = Eigen::Vector3d(
                        last_pred_x(k_pred, VX), last_pred_x(k_pred, VY), last_pred_x(k_pred, VZ)).norm();
                } else {
                    const Eigen::Vector3d vel(x0(VX), x0(VY), x0(VZ));
                    plan_start_pos = x0.head<3>() + vel * replan_latency_ema;
                    snap_v = vel.norm();
                }

                std::thread([&, plan_start_pos, snap_v, bg_pool_id]() {
                    const auto replan_start = std::chrono::steady_clock::now();
                    Vector3d dir = final_goal - plan_start_pos;
                    double d2g = dir.norm();
                    bool last_seg = (d2g <= planning_horizon * 1.2);
                    Vector3d local_goal = last_seg ? final_goal
                                        : (plan_start_pos + planning_horizon * dir.normalized());

                    vector<Eigen::Vector3d> es, ee;
                    list<GraphNode::Ptr> g;
                    vector<vector<Eigen::Vector3d>> rp, fp;
                    topo_prm->findTopoPaths(plan_start_pos, local_goal, es, ee, g, rp, fp);

                    vector<vector<Eigen::Vector3d>> sp_paths;
                    vector<ArcLengthSpline> sp_splines;
                    for (const auto& group : fp) {
                        if (group.size() < 2) continue;
                        vector<Eigen::Vector3d> dense;
                        int seg = 30;
                        for (size_t i = 0; i < group.size() - 1; ++i) {
                            Eigen::Vector3d s = group[i], e = group[i+1];
                            for (int j = 0; j < seg; ++j) {
                                double t = (double)j / seg;
                                dense.push_back(s + t * (e - s));
                            }
                        }
                        dense.push_back(group.back());
                        ArcLengthSpline sp;
                        Eigen::VectorXd X(dense.size()), Y(dense.size()), Z(dense.size());
                        for (size_t i = 0; i < dense.size(); ++i) {
                            X(i) = dense[i].x(); Y(i) = dense[i].y(); Z(i) = dense[i].z();
                        }
                        sp.gen3DSpline(X, Y, Z);
                        sp_splines.push_back(sp);
                        double arc = sp.getLength_3D();
                        Eigen::VectorXd sv; sv.setLinSpaced(NV, 0, arc);
                        vector<Eigen::Vector3d> sp_path;
                        for (int i = 0; i < NV; ++i) sp_path.push_back(sp.getPosition_3D(sv(i)));
                        sp_paths.push_back(sp_path);
                    }

                    if (sp_splines.empty()) {
                        PlanResult res; res.valid = false;
                        { std::lock_guard<std::mutex> lk(pending_mutex); pending_result = std::move(res); }
                        new_plan_ready.store(true);
                        replan_running.store(false);
                        return;
                    }

                    vector<RefTrajOpt> rtos(sp_splines.size());
                    vector<vector<Eigen::Vector3d>> filt_paths;
                    std::vector<int> vidx;

                    vector<std::thread> ot;
                    for (size_t i = 0; i < sp_splines.size(); ++i) {
                        ot.push_back(std::thread([&rtos, i, &sp_splines, MAX_VEL, MAX_ACC, MAX_JERK, NV, snap_v, ref_use_jerk]() {
                            RefTrajOpt r; r.socp_interface(NV, snap_v, 0.0, sp_splines[i], MAX_VEL, MAX_ACC, ref_use_jerk, MAX_JERK);
                            rtos[i] = r;
                        }));
                    }
                    for (auto& th : ot) if (th.joinable()) th.join();

                    vector<double> ttimes(rtos.size(), 0.0);
                    for (size_t i = 0; i < rtos.size(); ++i) ttimes[i] = rtos[i].get_total_time();
                    double thr = 0.2;
                    double mint = 1e5;
                    for (size_t i = 0; i < ttimes.size(); ++i) {
                        if (ttimes[i] > 1e-6 && rtos[i].get_result_t().size() >= 2)
                            mint = std::min(mint, ttimes[i]);
                    }
                    if (mint == 1e5) {
                        PlanResult res; res.valid = false;
                        { std::lock_guard<std::mutex> lk(pending_mutex); pending_result = std::move(res); }
                        new_plan_ready.store(true);
                        replan_running.store(false);
                        return;
                    }
                    for (size_t i = 0; i < ttimes.size(); ++i) {
                        if (ttimes[i] > 1e-6 && rtos[i].get_result_t().size() >= 2
                            && ttimes[i] < mint + thr) {
                            vidx.push_back(i);
                            filt_paths.push_back(sp_paths[i]);
                        }
                    }

                    if ((int)vidx.size() > MaxPathN) {
                        PlanResult res; res.valid = false;
                        { std::lock_guard<std::mutex> lk(pending_mutex); pending_result = std::move(res); }
                        new_plan_ready.store(true);
                        replan_running.store(false);
                        return;
                    }

                    auto& bg_cor = *cor_pools[bg_pool_id];
                    vector<std::thread> ct;
                    vector<OptimizationResult> ac(vidx.size());
                    vector<bool> cs(vidx.size(), false);
                    if (frame == "FSF") {
                        for (size_t i = 0; i < vidx.size(); ++i) {
                            ct.push_back(std::thread([&bg_cor, &ac, &cs, &sp_splines, &vidx, i]() {
                                ac[i] = bg_cor[i].corGenerator(sp_splines[vidx[i]]);
                                cs[i] = ac[i].success;
                            }));
                        }
                    } else if (frame == "PTF") {
                        for (size_t i = 0; i < vidx.size(); ++i) {
                            ct.push_back(std::thread([&bg_cor, &ac, &cs, &sp_splines, &vidx, i]() {
                                ac[i] = bg_cor[i].corGeneratorPTF(sp_splines[vidx[i]]);
                                cs[i] = ac[i].success;
                            }));
                        }
                    }
                    for (auto& th : ct) if (th.joinable()) th.join();

                    int best_idx = -1;
                    double best_vol = 0.0;
                    int n_ok = 0;
                    for (size_t i = 0; i < cs.size(); ++i) {
                        if (cs[i]) { n_ok++; if (ac[i].volume > best_vol) { best_vol = ac[i].volume; best_idx = i; } }
                    }

                    if (!std::isfinite(best_vol) || n_ok == 0) {
                        PlanResult res; res.valid = false;
                        { std::lock_guard<std::mutex> lk(pending_mutex); pending_result = std::move(res); }
                        new_plan_ready.store(true);
                        replan_running.store(false);
                        return;
                    }

                    PlanResult res;
                    res.valid = true;
                    res.spline = sp_splines[vidx[best_idx]];
                    res.coeffs = ac[best_idx];
                    res.ref_traj_opt = rtos[vidx[best_idx]];
                    res.spline_t_total = res.ref_traj_opt.get_total_time();
                    res.pool_id = bg_pool_id;
                    res.corridor_idx = best_idx;
                    res.stitch_point = plan_start_pos;
                    res.replan_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - replan_start).count();
                    res.viz_paths = std::move(filt_paths);

                    {
                        std::lock_guard<std::mutex> lk(pending_mutex);
                        pending_result = std::move(res);
                    }
                    new_plan_ready.store(true);
                    replan_running.store(false);
                }).detach();
            }

            corGen& active_corridor = (*cor_pools[active_cor_pool])[best_corridor_idx];
            Eigen::Matrix<double, NominalNMPC::n_step_, NominalQuadDynamic::x_dim_> pred_x;

            InitData4NMPC Init_data_nmpc;
            Pre4NMPC_Robust(Init_data_nmpc, best_ref_traj_opt.get_result_t(), NominalNMPC::n_step_, dt,
                            best_spline, active_corridor, coeffs4NMPC, t_cur, spline_t_total, frame);

            if (Init_data_nmpc.ref_data.ref_pos.empty()) {
                Eigen::Matrix<double, NominalQuadDynamic::x_dim_, 1> sim_x1;
                sim_x1.setZero();
                nmpc.quad_dynamic_.rk4_func(x0, u0, dt, sim_x1);
                total_length += (sim_x1.head<3>() - x0.head<3>()).norm();
                x0 = sim_x1;
                Eigen::Vector4d q(x0(QX), x0(QY), x0(QZ), x0(QW));
                ros_inte.publish_pose(x0.head<3>(), q);
                ros_inte.publish_odom(x0.head<3>(), q, Eigen::Vector3d(x0(VX), x0(VY), x0(VZ)));
                ros::spinOnce();
                break;
            }

            Init_data_nmpc.ref_data.TerminalPos = final_goal;
            nmpc.set_ref_data(Init_data_nmpc.ref_data);

            {
                int n_clamp = 0;
                for (int i = 0; i <= NominalNMPC::n_step_; ++i) {
                    if (t_cur + i * dt >= spline_t_total - 1e-6) n_clamp++;
                }
                static double last_ctx_log_t = -1.0;
                const bool repl = replan_running.load();
                const bool need_ctx = (n_clamp > 0 || repl || t_cur >= replan_thresh - 0.15);
                if (need_ctx && std::abs(t_cur - last_ctx_log_t) > 0.08) {
                    last_ctx_log_t = t_cur;
                }
            }

            int nmpc_result = nmpc.solve(x0, u0, U, pred_x);
            last_nmpc_ok = (nmpc_result > 0);
            if (last_nmpc_ok) {
                last_pred_x = pred_x;
                last_pred_x_valid = true;
            }

            if (nmpc_result > 0) {
                vector<Eigen::Vector3d> nmpc_traj;
                for (int i = 0; i < nmpc.n_step_; ++i) {
                    nmpc_traj.push_back(Eigen::Vector3d(pred_x(i, 0), pred_x(i, 1), pred_x(i, 2)));
                }
                ros_inte.publish_predict_traj(nmpc_traj);

                if (NominalNMPC::n_step_ > 1) {
                    U.topRows(NominalNMPC::n_step_ - 1) = U.bottomRows(NominalNMPC::n_step_ - 1);
                    U.row(NominalNMPC::n_step_ - 1) = U.row(NominalNMPC::n_step_ - 2);
                }
                u0 = U.row(0).transpose();
            } else {
                U.setZero();
                for (int k = 0; k < NominalNMPC::n_step_; ++k) U(k, 3) = hover_ratio;
                u0 << 0.0, 0.0, 0.0, hover_ratio;
            }

            Eigen::Matrix<double, NominalQuadDynamic::x_dim_, 1> sim_x1;
            sim_x1.setZero();
            nmpc.quad_dynamic_.rk4_func(x0, u0, dt, sim_x1);
            total_length += (sim_x1.head<3>() - x0.head<3>()).norm();
            x0 = sim_x1;

            double s_actual = projectOnSpline(best_spline, x0.head<3>());
            double t_actual = arcLengthToTime(Init_data_nmpc.cumulative_times,
                                                s_actual, Init_data_nmpc.spline_len);
            double progress_err_t = t_actual - t_cur;
            double t_cur_new = t_cur + dt + kp_progress * progress_err_t;
            t_cur = std::max(t_cur, std::min(t_cur_new, spline_t_total));
            

            Eigen::Vector4d q(x0(QX), x0(QY), x0(QZ), x0(QW));
            ros_inte.publish_pose(x0.head<3>(), q);
            Eigen::Vector3d vel(x0(VX), x0(VY), x0(VZ));
            ros_inte.publish_vel(vel, q);
            ros_inte.publish_odom(x0.head<3>(), q, vel);
            ros_inte.publish_control(u0);

            // Publish colored local point cloud within sensing range
            {
                corGen& ac = (*cor_pools[active_cor_pool])[0];
                if (ac.global_kdtree_built_ && !ac.cloudMap_.empty()) {
                    pcl::PointXYZ search_pt;
                    search_pt.x = x0(PX); search_pt.y = x0(PY); search_pt.z = x0(PZ);
                    std::vector<int> nn_idx;
                    std::vector<float> nn_dist;
                    ac.global_kdtree_.radiusSearch(search_pt, sensing_radius, nn_idx, nn_dist);
                    if (!nn_idx.empty()) {
                        ros_inte.publish_local_cloud(ac.cloudMap_, nn_idx, 0.0, gridmap.size()(2));
                    }
                }
            }

            ros::spinOnce();

            break;
        }

        case GOAL_REACHED: {
            ros_inte.publish_stop();
            fsm_state = WAIT_TARGET;
            break;
        }

        }

        ros::spinOnce();
        rate.sleep();
    }

    return 0;
}
