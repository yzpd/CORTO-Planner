#pragma once

#include <ros/ros.h>
#include <tf/transform_broadcaster.h>

#include "map/map.hpp"
#include "path_searching/topo_prm.hpp"
#include "corridor/corGen.hpp"
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <std_msgs/Bool.h>
#include <nav_msgs/Odometry.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

class RosInterface {
private:
    ros::Publisher predict_traj_pub_;
    ros::Publisher filter_path_pub_;
    ros::Publisher graph_nodes_pub_;
    ros::Publisher corridor_marker_pub_;
    ros::Publisher pose_pub_;
    ros::Publisher vel_pub_;
    ros::Publisher control_pub_;
    ros::Publisher stop_pub_;
    ros::Publisher local_cloud_pub_;
    ros::Publisher odom_pub_;
    tf::TransformBroadcaster broadcaster;

    int last_spline_path_count_ = 0;

public:
    ros::Publisher grid_map_pub_;

    Vector3d map_size_;

    RosInterface(Vector3d map_size);
    pcl::PointCloud<pcl::PointXYZ> grid_to_cloud(GridMap &map);
    void publish_map_surface(const GridMap &map, double vis_res = 0.05);

    void publish_predict_traj(vector<Vector3d> &traj);
    void publish_pose(Vector3d pos, Vector4d quat);
    void publish_odom(Vector3d pos, Vector4d quat, Vector3d vel);
    void publish_vel(Eigen::Vector3d vel, Eigen::Vector4d quat);
    void publish_control(Eigen::Vector4d u);
    void publish_stop();
    void drawFilterPaths(const vector<vector<Eigen::Vector3d>>& filter_paths, 
                                          double line_width, int id, double alpha);
    void displayLineList_filter(const vector<Eigen::Vector3d>& list1,
                                    const vector<Eigen::Vector3d>& list2, 
                                    double line_width,
                                    const Eigen::Vector4d& color, 
                                    int id, int pub_id,
                                    double alpha);                                

    void visualizeGraphNodes(const list<GraphNode::Ptr>& graph);

    void visualizeCorridorMarkers(const CorridorVisualization& corridor_vis);

    void publish_local_cloud(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                             const std::vector<int>& indices,
                             double z_min, double z_max);

    Vector3d quaternion_to_rpy(const Quaterniond& q);
    Quaterniond rpy_to_quaternion(Vector3d rpy);
};
