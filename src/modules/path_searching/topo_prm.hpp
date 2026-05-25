#ifndef _TOPO_PRM_H
#define _TOPO_PRM_H

#include <random>
#include <ros/ros.h>
#include <Eigen/Eigen>
#include "../map/map.hpp"
#include "../raycast/raycast.hpp"

using namespace std;

class GraphNode {
private:
  /* data */

public:
  enum NODE_TYPE { Guard = 1, Connector = 2 };

  enum NODE_STATE { NEW = 1, CLOSE = 2, OPEN = 3 };

  GraphNode(/* args */) {
  }
  GraphNode(Eigen::Vector3d pos, NODE_TYPE type, int id) {
    pos_ = pos;
    type_ = type;
    state_ = NEW;
    id_ = id;
  }
  ~GraphNode() {
  }

  vector<shared_ptr<GraphNode>> neighbors_;
  Eigen::Vector3d pos_;
  NODE_TYPE type_;
  NODE_STATE state_;
  int id_;

  typedef shared_ptr<GraphNode> Ptr;
};


class TopologyPRM{
  private:
    const GridMap* grid_map_{nullptr};
    bool use_grid_map_{false};
    Eigen::Vector3d grid_map_origin_{Eigen::Vector3d::Zero()};

    random_device rd_;
    default_random_engine eng_;
    uniform_real_distribution<double> rand_pos_;

    Eigen::Vector3d sample_r_;
    Eigen::Vector3d translation_;
    Eigen::Matrix3d rotation_;    

    double resolution_;
    Eigen::Vector3d sample_inflate_;
    double max_sample_time_;
    int max_sample_num_;
    int max_raw_path_;

    list<GraphNode::Ptr> graph_;
    vector<vector<Eigen::Vector3d>> raw_paths_;
    vector<vector<Eigen::Vector3d>> short_paths_;
    vector<vector<Eigen::Vector3d>> final_paths_;    
    vector<Eigen::Vector3d> start_pts_, end_pts_;   

    vector<RayCaster> casters_;
    Eigen::Vector3d offset_;

    inline Eigen::Vector3d getSample();

    vector<GraphNode::Ptr> findVisibGuard(Eigen::Vector3d pt);
    bool lineVisib(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, double thresh,
                    Eigen::Vector3d& pc, int caster_id = 0);
    bool lineVisibCoarse(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
                          double thresh, Eigen::Vector3d& pc);
    bool needConnection(GraphNode::Ptr g1, GraphNode::Ptr g2, Eigen::Vector3d pt);
    bool sameTopoPath(const vector<Eigen::Vector3d>& path1,
                               const vector<Eigen::Vector3d>& path2, double thresh);  

    vector<Eigen::Vector3d> discretizePath(const vector<Eigen::Vector3d>& path, int pt_num);
    vector<Eigen::Vector3d> discretizeLine(Eigen::Vector3d p1, Eigen::Vector3d p2);

    void pruneGraph();

    void shortcutPathsVis();
    vector<Eigen::Vector3d> shortcutPathVis(const vector<Eigen::Vector3d>& path);

    void depthFirstSearch(vector<GraphNode::Ptr>& vis);

  public:
    double clearance_;

    TopologyPRM(/* args */);
    ~TopologyPRM();

    void init(ros::NodeHandle& nh);
    void setGridMap(const GridMap* grid_map);

    void findTopoPaths(Eigen::Vector3d start, Eigen::Vector3d end,
                       vector<Eigen::Vector3d> start_pts, vector<Eigen::Vector3d> end_pts,
                       list<GraphNode::Ptr>& graph, vector<vector<Eigen::Vector3d>>& raw_paths,
                       vector<vector<Eigen::Vector3d>>& filtered_paths);

    list<GraphNode::Ptr> createGraph(Eigen::Vector3d start, Eigen::Vector3d end);
    vector<vector<Eigen::Vector3d>> searchPaths();
    double pathLength(const vector<Eigen::Vector3d>& path);
    vector<vector<Eigen::Vector3d>> pruneEquivalent(vector<vector<Eigen::Vector3d>>& paths);

};

#endif