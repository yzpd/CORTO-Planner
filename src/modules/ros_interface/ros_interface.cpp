#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/LinearMath/Quaternion.h>
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/WrenchStamped.h>
#include "ros_interface.hpp"

#define COLOR(color, R, G, B, A) color.a = A / 255.0, color.r = R / 255.0, color.g = G / 255.0, color.b = B / 255.0

RosInterface::RosInterface(Vector3d map_size)
    : map_size_(map_size) {
    ros::NodeHandle nh;
    grid_map_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/map", 1);
    predict_traj_pub_ = nh.advertise<visualization_msgs::Marker>("/predict_traj", 1);
    filter_path_pub_ = nh.advertise<visualization_msgs::Marker>("/filter_path", 1);
    graph_nodes_pub_ = nh.advertise<visualization_msgs::Marker>("/graph_nodes", 1);
    corridor_marker_pub_ = nh.advertise<visualization_msgs::MarkerArray>("corridor_markers", 10);
    vel_pub_ = nh.advertise<geometry_msgs::TwistStamped>("/vel", 1);
    pose_pub_ = nh.advertise<geometry_msgs::PoseStamped>("/pose", 1);
    control_pub_ = nh.advertise<geometry_msgs::WrenchStamped>("/control", 1);
    stop_pub_ = nh.advertise<std_msgs::Bool>("/stop", 1);
    local_cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/local_cloud", 1);
    odom_pub_ = nh.advertise<nav_msgs::Odometry>("/drone_0_visual_slam/odom", 1);
}

void RosInterface::publish_stop() {
    std_msgs::Bool stop_msg;
    stop_msg.data = true;
    stop_pub_.publish(stop_msg);
}

pcl::PointCloud<pcl::PointXYZ> RosInterface::grid_to_cloud(GridMap &map) {
    pcl::PointCloud<pcl::PointXYZ> cloudMap;
    double resolution = map.resolution();
    double z_margin = resolution * 2.5;
    for (double z = resolution / 2.0; z < map_size_(2); z += resolution) {
        if (z < z_margin || z > map_size_(2) - z_margin) continue;
        for (double y = resolution / 2.0; y < map_size_(1); y += resolution) {
            for (double x = resolution / 2.0; x < map_size_(0); x += resolution) {
                if (map(Vector3d(x, y ,z))) {
                    pcl::PointXYZ pt;
                    pt.x = x - map_size_(0) / 2.0, pt.y = y - map_size_(1) / 2.0, pt.z = z;
                    cloudMap.points.push_back(pt);
                }
            }
        }
    }
    cloudMap.width = cloudMap.points.size();
    cloudMap.height = 1;
    cloudMap.is_dense = true;
    return cloudMap;
}

void RosInterface::publish_map_surface(const GridMap &map, double vis_res) {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    double half_x = map.size()(0) / 2.0;
    double half_y = map.size()(1) / 2.0;

    for (const auto &cyl : map.cylinders()) {
        double r = cyl.radius_;
        double h = cyl.high_;
        int n_angle = std::max(12, (int)(2.0 * M_PI * r / vis_res));
        int n_h = std::max(3, (int)(h / vis_res));
        for (int ai = 0; ai < n_angle; ai++) {
            double theta = 2.0 * M_PI * ai / n_angle;
            double px = cyl.pos_(0) + r * cos(theta) - half_x;
            double py = cyl.pos_(1) + r * sin(theta) - half_y;
            for (int hi = 0; hi <= n_h; hi++) {
                double pz = cyl.pos_(2) + h * hi / n_h;
                cloud.points.push_back(pcl::PointXYZ(px, py, pz));
            }
        }
    }

    for (const auto &circ : map.circles()) {
        double r1 = circ.radius1_, r2 = circ.radius2_;
        double th = circ.theta_, thick = circ.thick_;
        double cos_t = cos(th), sin_t = sin(th);
        double r_max = std::max(r1, r2);
        int n_ring = std::max(36, (int)(2.0 * M_PI * r_max / vis_res));
        int n_tube = std::max(6, (int)(2.0 * M_PI * thick / vis_res));
        for (int ai = 0; ai < n_ring; ai++) {
            double alpha = 2.0 * M_PI * ai / n_ring;
            double cy_local = r1 * cos(alpha);
            double cz_local = r2 * sin(alpha);
            for (int ti = 0; ti < n_tube; ti++) {
                double beta = 2.0 * M_PI * ti / n_tube;
                double dy = thick * cos(beta);
                double dz = thick * sin(beta);
                double ly = cy_local + dy;
                double lz = cz_local + dz;
                double px = -sin_t * ly + circ.pos_(0) - half_x;
                double py =  cos_t * ly + circ.pos_(1) - half_y;
                double pz = lz + circ.pos_(2);
                cloud.points.push_back(pcl::PointXYZ(px, py, pz));
            }
        }
    }

    cloud.width = cloud.points.size();
    cloud.height = 1;
    cloud.is_dense = true;

    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(cloud, msg);
    msg.header.frame_id = "world";
    grid_map_pub_.publish(msg);
}

void RosInterface::publish_local_cloud(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                                       const std::vector<int>& indices,
                                       double z_min, double z_max) {
    pcl::PointCloud<pcl::PointXYZRGB> colored;
    colored.reserve(indices.size());
    double z_margin = 0.25;
    double z_lo = z_min + z_margin;
    double z_hi = z_max - z_margin;
    double z_range = std::max(z_hi - z_lo, 0.1);

    for (int idx : indices) {
        const auto& pt = cloud.points[idx];
        if (pt.z < z_lo || pt.z > z_hi) continue;
        pcl::PointXYZRGB cp;
        cp.x = pt.x; cp.y = pt.y; cp.z = pt.z;

        float ratio = std::max(0.0, std::min((double)(pt.z - z_lo) / z_range, 1.0));
        float h = (1.0f - ratio) * 240.0f;
        float s = 1.0f, v = 1.0f;
        float c = v * s;
        float x_hsv = c * (1.0f - std::fabs(std::fmod(h / 60.0f, 2.0f) - 1.0f));
        float m = v - c;
        float r, g, b;
        if      (h < 60)  { r = c; g = x_hsv; b = 0; }
        else if (h < 120) { r = x_hsv; g = c; b = 0; }
        else if (h < 180) { r = 0; g = c; b = x_hsv; }
        else if (h < 240) { r = 0; g = x_hsv; b = c; }
        else if (h < 300) { r = x_hsv; g = 0; b = c; }
        else              { r = c; g = 0; b = x_hsv; }
        cp.r = (uint8_t)((r + m) * 255);
        cp.g = (uint8_t)((g + m) * 255);
        cp.b = (uint8_t)((b + m) * 255);
        colored.push_back(cp);
    }
    colored.width = colored.size();
    colored.height = 1;
    colored.is_dense = true;

    sensor_msgs::PointCloud2 msg;
    pcl::toROSMsg(colored, msg);
    msg.header.frame_id = "world";
    msg.header.stamp = ros::Time::now();
    local_cloud_pub_.publish(msg);
}

void RosInterface::publish_predict_traj(vector<Eigen::Vector3d> &traj) {
    visualization_msgs::Marker tr;
    tr.header.frame_id = "world";
    tr.header.stamp = ros::Time::now(); 
    tr.ns = "predict trajectory";
    tr.action = visualization_msgs::Marker::ADD;
    tr.pose.orientation.w = 1.0;
    tr.pose.orientation.x = 0.0;
    tr.pose.orientation.y = 0.0;
    tr.pose.orientation.z = 0.0;
    tr.id    = 0;
    tr.type    = visualization_msgs::Marker::LINE_STRIP;
    tr.scale.x = 0.05 * 1.5;
    tr.scale.y = 0.05 * 1.5;
    tr.scale.z = 0.05 * 1.5;
    COLOR(tr.color, 255, 255, 0, 255);
    for (const auto &tp : traj) {
        geometry_msgs::Point p;
        p.x = tp.x(), p.y = tp.y(), p.z = tp.z();
        tr.points.push_back(p);
    }
    predict_traj_pub_.publish(tr);
}

void RosInterface::publish_vel(Eigen::Vector3d vel, Eigen::Vector4d quat) {
    geometry_msgs::TwistStamped twist;
    Vector3d ang = quaternion_to_rpy(Quaterniond(quat.w(), quat.x(), quat.y(), quat.z()));
    twist.header.frame_id = "world";
    twist.header.stamp = ros::Time::now();
    twist.twist.linear.x = vel.x();
    twist.twist.linear.y = vel.y();
    twist.twist.linear.z = vel.z();
    twist.twist.angular.x = ang.x();
    twist.twist.angular.y = ang.y();
    twist.twist.angular.z = ang.z();

    vel_pub_.publish(twist);
}

void RosInterface::publish_control(Eigen::Vector4d u) {
    geometry_msgs::WrenchStamped control_msg;
    control_msg.header.frame_id = "world";
    control_msg.header.stamp = ros::Time::now();
    control_msg.wrench.force.x = u(0);
    control_msg.wrench.force.y = u(1);
    control_msg.wrench.force.z = u(2);
    control_msg.wrench.torque.x = u(3);
    control_msg.wrench.torque.y = 0;
    control_msg.wrench.torque.z = 0;

    control_pub_.publish(control_msg);
}

void RosInterface::publish_pose(Vector3d pos, Vector4d quat) {
    geometry_msgs::PoseStamped pose;
    pose.header.frame_id = "world";
    pose.header.stamp = ros::Time::now();
    pose.pose.position.x = pos.x();
    pose.pose.position.y = pos.y();
    pose.pose.position.z = pos.z();
    pose.pose.orientation.x = quat.x();
    pose.pose.orientation.y = quat.y();
    pose.pose.orientation.z = quat.z();
    pose.pose.orientation.w = quat.w();

    tf::Transform transform;
    transform.setOrigin(tf::Vector3(pos.x(), pos.y(), pos.z()));
    Vector3d ang = quaternion_to_rpy(Quaterniond(quat.w(), quat.x(), quat.y(), quat.z()));
    ang(1) = ang(0) = 0;
    Quaterniond q = rpy_to_quaternion(ang);
    transform.setRotation(tf::Quaternion(quat.x(), quat.y(), quat.z(), quat.w()));
    broadcaster.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "quad"));
}

void RosInterface::publish_odom(Vector3d pos, Vector4d quat, Vector3d vel) {
    nav_msgs::Odometry odom;
    odom.header.frame_id = "world";
    odom.header.stamp = ros::Time::now();
    odom.child_frame_id = "body";
    odom.pose.pose.position.x = pos.x();
    odom.pose.pose.position.y = pos.y();
    odom.pose.pose.position.z = pos.z();
    odom.pose.pose.orientation.x = quat.x();
    odom.pose.pose.orientation.y = quat.y();
    odom.pose.pose.orientation.z = quat.z();
    odom.pose.pose.orientation.w = quat.w();
    odom.twist.twist.linear.x = vel.x();
    odom.twist.twist.linear.y = vel.y();
    odom.twist.twist.linear.z = vel.z();
    odom_pub_.publish(odom);
}

Vector3d RosInterface::quaternion_to_rpy(const Quaterniond& q) {
    const double &qw = q.w();
    const double &qx = q.x();
    const double &qy = q.y();
    const double &qz = q.z();

    Vector3d rpy;
    rpy.x() = std::atan2(2. * (qw*qx + qy*qz), 1. - 2. * (qx*qx + qy*qy));
    double sin_pitch = 2. * (qw*qy - qz*qx);
    rpy.y() = std::asin(sin_pitch);
    rpy.z() = std::atan2(2. * (qw*qz + qx*qy), 1. - 2. * (qy*qy + qz*qz));
    return rpy;    
}


Quaterniond RosInterface::rpy_to_quaternion(Vector3d rpy)
{
    double cy = cos(rpy[2] * 0.5);
    double sy = sin(rpy[2] * 0.5);
    double cp = cos(rpy[1] * 0.5);
    double sp = sin(rpy[1] * 0.5);
    double cr = cos(rpy[0] * 0.5);
    double sr = sin(rpy[0] * 0.5);
 
    Quaterniond q;
    q.w() = cy * cp * cr + sy * sp * sr;
    q.x() = cy * cp * sr - sy * sp * cr;
    q.y() = sy * cp * sr + cy * sp * cr;
    q.z() = sy * cp * cr - cy * sp * sr;
 
    return q;
}

void RosInterface::drawFilterPaths(const vector<vector<Eigen::Vector3d>>& filter_paths, 
                                          double line_width, int id, double alpha) {
  vector<Eigen::Vector4d> colors = {
    Eigen::Vector4d(1.0, 0.0, 0.0, 0.8),
    Eigen::Vector4d(0.0, 1.0, 0.0, 0.8),
    Eigen::Vector4d(0.0, 0.0, 1.0, 0.8),
    Eigen::Vector4d(1.0, 1.0, 0.0, 0.8),
    Eigen::Vector4d(1.0, 0.0, 1.0, 0.8),
    Eigen::Vector4d(0.0, 1.0, 1.0, 0.8),
    Eigen::Vector4d(1.0, 0.5, 0.0, 0.8),
    Eigen::Vector4d(0.5, 0.0, 1.0, 0.8),
  };

  vector<Eigen::Vector3d> empty;
  for (int i = 0; i < last_spline_path_count_; ++i) {
    displayLineList_filter(empty, empty, line_width, colors[0], id + i, 1, 1.0);
  }

  int current_path_count = filter_paths.size();
  last_spline_path_count_ = current_path_count;

  for (int path_id = 0; path_id < current_path_count; ++path_id) {
    const auto& path = filter_paths[path_id];
    
    Eigen::Vector4d color = colors[path_id % colors.size()];
    
    vector<Eigen::Vector3d> line_pt1, line_pt2;
    
    if (path.size() < 2) continue;
    
    for (int i = 0; i < path.size() - 1; ++i) {
      line_pt1.push_back(path[i]);
      line_pt2.push_back(path[i + 1]);
    }
    
    if (!line_pt1.empty()) {
      displayLineList_filter(line_pt1, line_pt2, line_width, color, id + path_id, 1, alpha);
    }
  }
}

void RosInterface::displayLineList_filter(const vector<Eigen::Vector3d>& list1,
                                  const vector<Eigen::Vector3d>& list2, 
                                  double line_width,
                                  const Eigen::Vector4d& color, 
                                  int id, int pub_id,
                                  double alpha) {
    visualization_msgs::Marker mk;
    mk.header.frame_id = "world";
    mk.header.stamp = ros::Time::now();
    mk.type = visualization_msgs::Marker::LINE_LIST;
    mk.action = visualization_msgs::Marker::DELETE;
    mk.id = id;
    filter_path_pub_.publish(mk);

    mk.action = visualization_msgs::Marker::ADD;
    mk.pose.orientation.x = 0.0;
    mk.pose.orientation.y = 0.0;
    mk.pose.orientation.z = 0.0;
    mk.pose.orientation.w = 1.0;

    mk.color.r = color(0);
    mk.color.g = color(1);
    mk.color.b = color(2);
    mk.color.a = alpha;
    mk.scale.x = line_width;

    geometry_msgs::Point pt;
    for (int i = 0; i < int(list1.size()); ++i) {
        pt.x = list1[i](0);
        pt.y = list1[i](1);
        pt.z = list1[i](2);
        mk.points.push_back(pt);

        pt.x = list2[i](0);
        pt.y = list2[i](1);
        pt.z = list2[i](2);
        mk.points.push_back(pt);
    }
    filter_path_pub_.publish(mk);
    ros::Duration(0.001).sleep();
}

void RosInterface::visualizeGraphNodes(const list<GraphNode::Ptr>& graph) {
    visualization_msgs::Marker clear_marker;
    clear_marker.header.frame_id = "world";
    clear_marker.header.stamp = ros::Time::now();
    clear_marker.action = visualization_msgs::Marker::DELETEALL;
    graph_nodes_pub_.publish(clear_marker);
    ros::Duration(0.01).sleep();

    visualization_msgs::Marker guard_marker;
    guard_marker.header.frame_id = "world";
    guard_marker.header.stamp = ros::Time::now();
    guard_marker.ns = "guard_nodes";
    guard_marker.id = 0;
    guard_marker.type = visualization_msgs::Marker::SPHERE_LIST;
    guard_marker.action = visualization_msgs::Marker::ADD;
    guard_marker.pose.orientation.w = 1.0;
    guard_marker.scale.x = 0.15;
    guard_marker.scale.y = 0.15;
    guard_marker.scale.z = 0.15;
    COLOR(guard_marker.color, 0, 255, 0, 200);

    visualization_msgs::Marker connector_marker;
    connector_marker.header.frame_id = "world";
    connector_marker.header.stamp = ros::Time::now();
    connector_marker.ns = "connector_nodes";
    connector_marker.id = 1;
    connector_marker.type = visualization_msgs::Marker::SPHERE_LIST;
    connector_marker.action = visualization_msgs::Marker::ADD;
    connector_marker.pose.orientation.w = 1.0;
    connector_marker.scale.x = 0.1;
    connector_marker.scale.y = 0.1;
    connector_marker.scale.z = 0.1;
    COLOR(connector_marker.color, 255, 0, 0, 200);

    visualization_msgs::Marker special_marker;
    special_marker.header.frame_id = "world";
    special_marker.header.stamp = ros::Time::now();
    special_marker.ns = "special_nodes";
    special_marker.id = 2;
    special_marker.type = visualization_msgs::Marker::SPHERE_LIST;
    special_marker.action = visualization_msgs::Marker::ADD;
    special_marker.pose.orientation.w = 1.0;
    special_marker.scale.x = 0.25;
    special_marker.scale.y = 0.25;
    special_marker.scale.z = 0.25;

    int guard_count = 0, connector_count = 0, special_count = 0;

    for (auto iter = graph.begin(); iter != graph.end(); ++iter) {
        geometry_msgs::Point pt;
        pt.x = (*iter)->pos_.x();
        pt.y = (*iter)->pos_.y();
        pt.z = (*iter)->pos_.z();

        if ((*iter)->id_ == 0 || (*iter)->id_ == 1) {
            special_marker.points.push_back(pt);
            
            std_msgs::ColorRGBA color;
            if ((*iter)->id_ == 0) {
                COLOR(color, 0, 0, 255, 255);
            } else {
                COLOR(color, 255, 255, 0, 255);
            }
            special_marker.colors.push_back(color);
            special_count++;
        } else if ((*iter)->type_ == GraphNode::Guard) {
            guard_marker.points.push_back(pt);
            guard_count++;
        } else if ((*iter)->type_ == GraphNode::Connector) {
            connector_marker.points.push_back(pt);
            connector_count++;
        }
    }

    if (guard_count > 0) {
        graph_nodes_pub_.publish(guard_marker);
        ros::Duration(0.001).sleep();
    }
    if (connector_count > 0) {
        graph_nodes_pub_.publish(connector_marker);
        ros::Duration(0.001).sleep();
    }
    if (special_count > 0) {
        graph_nodes_pub_.publish(special_marker);
        ros::Duration(0.001).sleep();
    }
}

void RosInterface::visualizeCorridorMarkers(const CorridorVisualization& corridor_vis) {
    visualization_msgs::MarkerArray marker_array;
    
    int n_eval = corridor_vis.ellipse_pts_world.size();
    int n_angles = corridor_vis.ellipse_pts_world[0].size();

    for (int j = 0; j < n_angles; ++j) {
        visualization_msgs::Marker marker;
        marker.header.frame_id = "world";
        marker.header.stamp = ros::Time::now();
        marker.ns = "corridor_markers";
        marker.id = j;
        marker.type = visualization_msgs::Marker::LINE_STRIP;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.01;
        marker.color.r = 0.7;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;

        for (int i = 1; i < n_eval - 1; ++i) {
            geometry_msgs::Point pt;
            pt.x = corridor_vis.ellipse_pts_world[i][j](0);
            pt.y = corridor_vis.ellipse_pts_world[i][j](1);
            pt.z = corridor_vis.ellipse_pts_world[i][j](2);
            marker.points.push_back(pt);
        }

        marker_array.markers.push_back(marker);
    }

    for (int i = 1; i < n_eval - 1; i += 2) {
        visualization_msgs::Marker circle_marker;
        circle_marker.header.frame_id = "world";
        circle_marker.header.stamp = ros::Time::now();
        circle_marker.ns = "corridor_cross_sections";
        circle_marker.id = i + n_angles;
        circle_marker.type = visualization_msgs::Marker::LINE_STRIP;
        circle_marker.action = visualization_msgs::Marker::ADD;
        
        circle_marker.scale.x = 0.01;
        circle_marker.color.r = 0.0;
        circle_marker.color.g = 0.5;
        circle_marker.color.b = 0.0;
        circle_marker.color.a = 0.7;
        
        for (int j = 0; j < n_angles; ++j) {
            geometry_msgs::Point pt;
            pt.x = corridor_vis.ellipse_pts_world[i][j](0);
            pt.y = corridor_vis.ellipse_pts_world[i][j](1);
            pt.z = corridor_vis.ellipse_pts_world[i][j](2);
            circle_marker.points.push_back(pt);
        }
        
        if (!circle_marker.points.empty()) {
            circle_marker.points.push_back(circle_marker.points[0]);
        }
        
        marker_array.markers.push_back(circle_marker);
    }

    corridor_marker_pub_.publish(marker_array);
}