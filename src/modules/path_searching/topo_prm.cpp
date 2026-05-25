#include "topo_prm.hpp"
#include <thread>

using namespace std;

TopologyPRM::TopologyPRM() {}

TopologyPRM::~TopologyPRM() {}

void TopologyPRM::init(ros::NodeHandle& nh){
    graph_.clear();
    eng_ = default_random_engine(rd_());
    rand_pos_ = uniform_real_distribution<double>(-1.0, 1.0);

    resolution_ = grid_map_->resolution();

    offset_ = Eigen::Vector3d(0.5, 0.5, 0.5);

    clearance_ = 0.1;
    sample_inflate_(0) = 1.0;
    sample_inflate_(1) = 2.0;
    sample_inflate_(2) = 0.5;
    max_sample_num_ = 80;
    max_sample_time_ = 0.1;
    max_raw_path_ = 6;

    for (int i = 0; i < max_raw_path_; ++i) {
        casters_.push_back(RayCaster());
    }
}

void TopologyPRM::setGridMap(const GridMap* grid_map) {
    grid_map_ = grid_map;
    use_grid_map_ = (grid_map_ != nullptr);
    if (use_grid_map_) {
        const Eigen::Vector3d map_size = grid_map_->size();
        grid_map_origin_ = Eigen::Vector3d(-map_size(0) / 2.0, -map_size(1) / 2.0, -0.01);
    }
}

void TopologyPRM::findTopoPaths(Eigen::Vector3d start, Eigen::Vector3d end,
                                vector<Eigen::Vector3d> start_pts, vector<Eigen::Vector3d> end_pts,
                                list<GraphNode::Ptr>& graph, vector<vector<Eigen::Vector3d>>& raw_paths,
                                vector<vector<Eigen::Vector3d>>& filtered_paths) {
  start_pts_ = start_pts;
  end_pts_ = end_pts;
  graph = createGraph(start, end);
  raw_paths = searchPaths();
  shortcutPathsVis();
  filtered_paths = pruneEquivalent(short_paths_);
  final_paths_ = filtered_paths;
}

list<GraphNode::Ptr> TopologyPRM::createGraph(Eigen::Vector3d start, Eigen::Vector3d end) {
    graph_.clear();
    GraphNode::Ptr start_node = GraphNode::Ptr(new GraphNode(start, GraphNode::Guard, 0));
    GraphNode::Ptr end_node = GraphNode::Ptr(new GraphNode(end, GraphNode::Guard, 1));

    graph_.push_back(start_node);
    graph_.push_back(end_node);

    sample_r_(0) = 0.5 * (end - start).norm() + sample_inflate_(0);
    sample_r_(1) = sample_inflate_(1);
    sample_r_(2) = sample_inflate_(2);
    translation_ = 0.5 * (start + end);

    Eigen::Vector3d xtf, ytf, ztf, downward(0, 0, -1);
    xtf = (end - translation_).normalized();
    ytf = xtf.cross(downward).normalized();
    ztf = xtf.cross(ytf);

    rotation_.col(0) = xtf;
    rotation_.col(1) = ytf;
    rotation_.col(2) = ztf;

    int node_id = 1;

    int sample_num = 0;
    double sample_time = 0.0;
    int valid_samples = 0;
    int collision_samples = 0;
    int added_guards = 0;
    int added_connectors = 0;
    
    Eigen::Vector3d pt;
    while (sample_time < max_sample_time_ && sample_num < max_sample_num_) {

        pt = getSample();
        ++sample_num;
        double dist;
        Eigen::Vector3d grad;
        const Eigen::Vector3d pt_map = pt - grid_map_origin_;
        if (!grid_map_->check_in_map(pt_map) || (*grid_map_)(pt_map)) {
            dist = 0.0;
        } else {
            dist = std::numeric_limits<double>::infinity();
        }

        if (dist <= clearance_) {
            collision_samples++;
            continue;
        }

        valid_samples++;

        vector<GraphNode::Ptr> visib_guards = findVisibGuard(pt);
        if (visib_guards.size() == 0) {
            GraphNode::Ptr guard = GraphNode::Ptr(new GraphNode(pt, GraphNode::Guard, ++node_id));
            graph_.push_back(guard);
            added_guards++;
        } else if (visib_guards.size() == 2) {
            bool need_connect = needConnection(visib_guards[0], visib_guards[1], pt);
            if (!need_connect) {
                continue;
            }
            GraphNode::Ptr connector = GraphNode::Ptr(new GraphNode(pt, GraphNode::Connector, ++node_id));
            graph_.push_back(connector);
            visib_guards[0]->neighbors_.push_back(connector);
            visib_guards[1]->neighbors_.push_back(connector);

            connector->neighbors_.push_back(visib_guards[0]);
            connector->neighbors_.push_back(visib_guards[1]);
            added_connectors++;
        }
    }

    pruneGraph();

    return graph_;
}

vector<vector<Eigen::Vector3d>> TopologyPRM::pruneEquivalent(vector<vector<Eigen::Vector3d>>& paths) {
  vector<vector<Eigen::Vector3d>> pruned_paths;
  if (paths.size() < 1) return pruned_paths;

  vector<int> exist_paths_id;
  exist_paths_id.push_back(0);

  for (int i = 1; i < paths.size(); ++i) {
    bool new_path = true;

    for (int j = 0; j < exist_paths_id.size(); ++j) {
      bool same_topo = sameTopoPath(paths[i], paths[exist_paths_id[j]], 0.0);

      if (same_topo) {
        new_path = false;
        break;
      }
    }

    if (new_path) {
      exist_paths_id.push_back(i);
    }
  }

  for (int i = 0; i < exist_paths_id.size(); ++i) {
    pruned_paths.push_back(paths[exist_paths_id[i]]);
  }

  return pruned_paths;
}

Eigen::Vector3d TopologyPRM::getSample() {
Eigen::Vector3d pt;
pt(0) = rand_pos_(eng_) * sample_r_(0);
pt(1) = rand_pos_(eng_) * sample_r_(1);
pt(2) = rand_pos_(eng_) * sample_r_(2);

pt = rotation_ * pt + translation_;

return pt;
}

vector<GraphNode::Ptr> TopologyPRM::findVisibGuard(Eigen::Vector3d pt) {
    vector<GraphNode::Ptr> visib_guards;
    Eigen::Vector3d pc;

    int visib_num = 0;

    for (list<GraphNode::Ptr>::iterator iter = graph_.begin(); iter != graph_.end(); ++iter) {
        if ((*iter)->type_ == GraphNode::Connector) continue;

        if (lineVisibCoarse(pt, (*iter)->pos_, resolution_, pc)) {
            visib_guards.push_back((*iter));
            ++visib_num;
            if (visib_num > 2) break;
        }
    }

    return visib_guards;
}

bool TopologyPRM::lineVisibCoarse(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
                                   double thresh, Eigen::Vector3d& pc) {
    Eigen::Vector3d dir = p2 - p1;
    double len = dir.norm();
    if (len < 1e-6) return true;
    dir /= len;

    double step = 3.0 * resolution_;
    int n_steps = (int)(len / step);

    for (int i = 0; i <= n_steps; ++i) {
        Eigen::Vector3d pt = p1 + std::min((double)i * step, len) * dir;
        const Eigen::Vector3d pt_map = pt - grid_map_origin_;
        if (!grid_map_->check_in_map(pt_map) || (*grid_map_)(pt_map)) {
            pc = pt;
            return false;
        }
    }
    return true;
}

bool TopologyPRM::lineVisib(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2, double thresh,
                            Eigen::Vector3d& pc, int caster_id) {
    Eigen::Vector3d ray_pt;
    Eigen::Vector3d pt;
    double dist;

    bool setInput_result = casters_[caster_id].setInput(p1 / resolution_, p2 / resolution_);
    while (casters_[caster_id].step(ray_pt)) {
        pt(0) = (ray_pt(0) + offset_(0)) * resolution_;
        pt(1) = (ray_pt(1) + offset_(1)) * resolution_;
        pt(2) = (ray_pt(2) + offset_(2)) * resolution_;

        const Eigen::Vector3d pt_map = pt - grid_map_origin_;
        if (!grid_map_->check_in_map(pt_map) || (*grid_map_)(pt_map)) {
            dist = 0.0;
        } else {
            dist = std::numeric_limits<double>::infinity();
        }

        if (dist <= thresh) {
            pc = pt;
            return false;
        }
    }
    return true;
}

bool TopologyPRM::needConnection(GraphNode::Ptr g1, GraphNode::Ptr g2, Eigen::Vector3d pt) {
    vector<Eigen::Vector3d> path1(3), path2(3);
    path1[0] = g1->pos_;
    path1[1] = pt;
    path1[2] = g2->pos_;

    path2[0] = g1->pos_;
    path2[2] = g2->pos_;

    vector<Eigen::Vector3d> connect_pts;
    bool has_connect = false;
    for (int i = 0; i < g1->neighbors_.size(); ++i) {
        for (int j = 0; j < g2->neighbors_.size(); ++j) {
        if (g1->neighbors_[i]->id_ == g2->neighbors_[j]->id_) {
            path2[1] = g1->neighbors_[i]->pos_;
            bool same_topo = sameTopoPath(path1, path2, 0.0);
            if (same_topo) {
            if (pathLength(path1) < pathLength(path2)) {
                g1->neighbors_[i]->pos_ = pt;
            }
            return false;
            }
        }
        }
    }
    return true;
}

bool TopologyPRM::sameTopoPath(const vector<Eigen::Vector3d>& path1,
                                const vector<Eigen::Vector3d>& path2, double thresh) {
    double len1 = pathLength(path1);
    double len2 = pathLength(path2);

    double max_len = max(len1, len2);

    double topo_step = 3.0 * resolution_;
    int pt_num = std::max(3, (int)ceil(max_len / topo_step));

    vector<Eigen::Vector3d> pts1 = discretizePath(path1, pt_num);
    vector<Eigen::Vector3d> pts2 = discretizePath(path2, pt_num);

    Eigen::Vector3d pc;
    for (int i = 0; i < pt_num; ++i) {
        if (!lineVisib(pts1[i], pts2[i], thresh, pc)) {
        return false;
        }
    }

    return true;
}

double TopologyPRM::pathLength(const vector<Eigen::Vector3d>& path) {
    double length = 0.0;
    if (path.size() < 2) return length;

    for (int i = 0; i < path.size() - 1; ++i) {
        length += (path[i + 1] - path[i]).norm();
    }
    return length;
}

vector<Eigen::Vector3d> TopologyPRM::discretizePath(const vector<Eigen::Vector3d>& path, int pt_num) {
    vector<double> len_list;
    len_list.push_back(0.0);

    for (int i = 0; i < path.size() - 1; ++i) {
        double inc_l = (path[i + 1] - path[i]).norm();
        len_list.push_back(inc_l + len_list[i]);
    }

    double len_total = len_list.back();
    double dl = len_total / double(pt_num - 1);
    double cur_l;

    vector<Eigen::Vector3d> dis_path;
    for (int i = 0; i < pt_num; ++i) {
        cur_l = double(i) * dl;

        int idx = -1;
        for (int j = 0; j < len_list.size() - 1; ++j) {
        if (cur_l >= len_list[j] - 1e-4 && cur_l <= len_list[j + 1] + 1e-4) {
            idx = j;
            break;
        }
        }

        double lambda = (cur_l - len_list[idx]) / (len_list[idx + 1] - len_list[idx]);
        Eigen::Vector3d inter_pt = (1 - lambda) * path[idx] + lambda * path[idx + 1];
        dis_path.push_back(inter_pt);
    }

    return dis_path;
    }

void TopologyPRM::pruneGraph() {
    if (graph_.size() > 2) {
        for (list<GraphNode::Ptr>::iterator iter1 = graph_.begin();
            iter1 != graph_.end() && graph_.size() > 2; ++iter1) {
        if ((*iter1)->id_ <= 1) continue;

        if ((*iter1)->neighbors_.size() <= 1) {
            for (list<GraphNode::Ptr>::iterator iter2 = graph_.begin(); iter2 != graph_.end(); ++iter2) {
                for (vector<GraphNode::Ptr>::iterator it_nb = (*iter2)->neighbors_.begin();
                    it_nb != (*iter2)->neighbors_.end(); ++it_nb) {
                        if ((*it_nb)->id_ == (*iter1)->id_) {
                            (*iter2)->neighbors_.erase(it_nb);
                            break;
                        }
                }
            }

            graph_.erase(iter1);
            iter1 = graph_.begin();
        }
        }
    }
}

void TopologyPRM::shortcutPathsVis() {
    short_paths_.resize(raw_paths_.size());
    for (size_t i = 0; i < raw_paths_.size(); ++i) {
        short_paths_[i] = shortcutPathVis(raw_paths_[i]);
    }
}

vector<Eigen::Vector3d> TopologyPRM::shortcutPathVis(const vector<Eigen::Vector3d>& path) {
    if (path.size() <= 2) return path;
    vector<Eigen::Vector3d> result;
    result.push_back(path.front());
    size_t cur = 0;

    while (cur < path.size() - 1) {
        size_t farthest = cur + 1;
        Eigen::Vector3d pc;
        for (size_t j = path.size() - 1; j > cur + 1; --j) {
            if (lineVisibCoarse(path[cur], path[j], resolution_, pc)) {
                farthest = j;
                break;
            }
        }
        result.push_back(path[farthest]);
        cur = farthest;
    }

    if (result.size() > 2) {
        Eigen::Vector3d axis = (result.back() - result.front());
        double axis_len = axis.norm();
        if (axis_len > 1e-6) {
            Eigen::Vector3d axis_n = axis / axis_len;
            vector<Eigen::Vector3d> cleaned;
            cleaned.push_back(result.front());
            double prev_proj = 0.0;
            for (size_t i = 1; i < result.size() - 1; ++i) {
                double proj = (result[i] - result.front()).dot(axis_n);
                if (proj > prev_proj) {
                    cleaned.push_back(result[i]);
                    prev_proj = proj;
                }
            }
            cleaned.push_back(result.back());
            result = cleaned;
        }
    }

    return result;
}

vector<Eigen::Vector3d> TopologyPRM::discretizeLine(Eigen::Vector3d p1, Eigen::Vector3d p2) {
    Eigen::Vector3d dir = p2 - p1;
    double len = dir.norm();
    int seg_num = ceil(len / resolution_);

    vector<Eigen::Vector3d> line_pts;
    if (seg_num <= 0) {
        return line_pts;
    }

    for (int i = 0; i <= seg_num; ++i) line_pts.push_back(p1 + dir * double(i) / double(seg_num));

    return line_pts;
}

vector<vector<Eigen::Vector3d>> TopologyPRM::searchPaths() {
    raw_paths_.clear();

    vector<GraphNode::Ptr> visited;
    visited.push_back(graph_.front());

    depthFirstSearch(visited);
    return raw_paths_;
    }

    void TopologyPRM::depthFirstSearch(vector<GraphNode::Ptr>& vis) {
    GraphNode::Ptr cur = vis.back();

    for (int i = 0; i < cur->neighbors_.size(); ++i) {
        if (cur->neighbors_[i]->id_ == 1) {
        vector<Eigen::Vector3d> path;
        for (int j = 0; j < vis.size(); ++j) {
            path.push_back(vis[j]->pos_);
        }
        path.push_back(cur->neighbors_[i]->pos_);

        raw_paths_.push_back(path);
        if (raw_paths_.size() >= max_raw_path_) return;

        break;
        }
    }

    for (int i = 0; i < cur->neighbors_.size(); ++i) {
        if (cur->neighbors_[i]->id_ == 1) continue;

        bool revisit = false;
        for (int j = 0; j < vis.size(); ++j) {
        if (cur->neighbors_[i]->id_ == vis[j]->id_) {
            revisit = true;
            break;
        }
        }
        if (revisit) continue;

        vis.push_back(cur->neighbors_[i]);
        depthFirstSearch(vis);
        if (raw_paths_.size() >= max_raw_path_) return;

        vis.pop_back();
    }
}