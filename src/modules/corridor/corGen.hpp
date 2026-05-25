#ifndef _COR_GEN_HPP_
#define _COR_GEN_HPP_

#include <ros/ros.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/PointCloud2.h>
#include "cubic_spline/arc_length_spline.hpp"
#include <Eigen/Eigen>
#include "fusion.h"

using namespace mosek::fusion;
using namespace monty;

struct FSF_data{
    std::vector<Eigen::Matrix3d> FSF;
};

struct PTF_data {
    std::vector<Eigen::Matrix3d> PTF;  
};

struct Projection_data{
    std::vector<Eigen::Vector3d> pruned_cloud_pos;
    std::vector<double> s_cloud;
    std::vector<bool> ind_proj;
    std::vector<Eigen::Vector2d> z_plane_pts;
    std::vector<double>          z_plane_s;
    std::vector<Eigen::Vector3d> all_raw_pos;
    std::vector<double>          all_raw_s;
};

struct OptimizationResult {
    double volume = 0.0;
    Eigen::MatrixXd coeffs_a;
    Eigen::MatrixXd coeffs_b;
    Eigen::MatrixXd coeffs_c;
    Eigen::MatrixXd coeffs_d;
    Eigen::MatrixXd coeffs_e;
    bool success;
};

struct EllipseParameters {
    double width;
    double height;
    double angle;
    Eigen::Vector2d center;
};

struct CorridorVisualization {
    std::vector<std::vector<Eigen::Vector3d>> ellipse_pts_world;
    double parametric_volume;
};

struct CageParameters {
    double l = 4.0;
    double h = 3.0;
    double h_min = -1.0;
    int n_topbottom = 15;
    int n_sides = 15;
    bool covers = true;
};

struct FSFPrev {
    Eigen::Vector3d t, n, b;
    bool has_prev = false;
};

class corGen
{
private:
    // Parameters
    int poly_degree_;
    bool is_LP_;
    double ellipse_axis_max;
    double eps_;
    int n_eval_;
    int MaxPointCloud_;
    double z_floor_ = 0.0;
    double z_ceiling_ = 3.0;
    int n_sectors_ = 18;
    int k_per_sector_ = 2;
    int max_per_bin_ = 32;
    ArcLengthSpline spline_;
    int n_sweep_ = 10;
    
    FSF_data fsf_data_;
    PTF_data ptf_data_;
    Projection_data projection_data_;
    int n_angles_;

    void updateSpline(const ArcLengthSpline& spline);
    void FSF_evaluate_Robust(const ArcLengthSpline& spline);
    
    void get_cage(const CageParameters& params, std::string frame);
    void NLP(const Projection_data& pruned_points_local);
    void NLP_LBFGS(const Projection_data& pruned_points_local);
    double poly_basis(double s, int k, int degree) const;
    Projection_data processPointCloud(std::string frame);

    EllipseParameters gen_ellipse_params(const Eigen::Matrix2d& P, const Eigen::Vector2d& pp);
    Eigen::Vector2d get_ellipse_points(const double width, const double height, const double angle, const double theta);

    Eigen::Vector3d safe_normalize(const Eigen::Vector3d& x,
                                const Eigen::Vector3d& fallback,
                                double eps = 1e-12);
    Eigen::Vector3d safe_cross_normalized(const Eigen::Vector3d& a,
                                            const Eigen::Vector3d& b,
                                            const Eigen::Vector3d& fallback,
                                            double eps = 1e-12);

    Eigen::Vector3d closest_to_A_perpendicular_to_B(const Eigen::Vector3d& A, const Eigen::Vector3d& B) {
        Eigen::Vector3d B_normalized = B.normalized();
        return A - B_normalized * B_normalized.dot(A);
    }

    Eigen::Matrix3d initial_frame(const Eigen::Vector3d& e10, const Eigen::Vector3d& e3_des);
    void PTF_evaluate(const ArcLengthSpline& spline);

public:
    corGen(){};
    ~corGen(){};

    OptimizationResult coeffs_;
    pcl::PointCloud<pcl::PointXYZ> cloudMap_;
    pcl::KdTreeFLANN<pcl::PointXYZ> global_kdtree_;
    bool global_kdtree_built_ = false;

    bool cloud_updated_ = false;

    void init(ros::NodeHandle& nh);

    void setCloudMap(const pcl::PointCloud<pcl::PointXYZ>& cloud) {
        cloudMap_ = cloud;
        cloud_updated_ = true;
        global_kdtree_built_ = false;
    }

    OptimizationResult corGenerator(const ArcLengthSpline& spline);
    OptimizationResult corGeneratorPTF(const ArcLengthSpline& spline);

    CorridorVisualization cor_vis(const OptimizationResult& coeffs, std::string frame);
    double computeVolume(const OptimizationResult& coeffs);

    FSF_data getFSFData() const { return fsf_data_; }

    std::vector<double> polynomial_eval(double s, const OptimizationResult& coeffs, int degree) const;

    int getPolyDegree() const { return poly_degree_; }

    Eigen::Matrix3d FSF_Function_Robust(const Eigen::Vector3d& v,
                                        const Eigen::Vector3d& a,
                                        const FSFPrev* prev,
                                        double eps_v = 1e-10,
                                        double eps_k = 1e-10);

    Eigen::Matrix3d PTF_at_s(double s, const ArcLengthSpline& spline) const;
};




#endif // _COR_GEN_HPP_
