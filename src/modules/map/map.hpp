#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include "json.hpp"

using namespace std;
using namespace Eigen;

class ObsCylinder {
public:
    Vector3d pos_;
    double high_;
    double radius_;

    ObsCylinder(Vector3d pos, double high, double radius) 
        : pos_(pos), high_(high), radius_(radius) {}
};

class ObsCircle {
public:
    Vector3d pos_;
    double radius1_;
    double radius2_;
    double theta_;
    double thick_;

    ObsCircle(Vector3d pos, double r1, double r2, double theta, double thick)
        : pos_(pos), radius1_(r1), radius2_(r2), theta_(theta), thick_(thick) {}
};


class GridMap {
private:
    vector<bool> grid_map_;
    double resolution_;
    Vector3d size_;
    Vector3i isize_;

    vector<ObsCylinder> cylinders_;
    vector<ObsCircle> circles_;

public:
    GridMap() {
        resolution_ = 0.0;
        size_ = Vector3d(0., 0., 0.);
        isize_ = Vector3i(0., 0., 0.);
    }
    GridMap(double resolution, Vector3d size) : resolution_(resolution), size_(size) {
        isize_ = Vector3i(round(size_(0) / resolution_), round(size_(1) / resolution_), round(size_(2) / resolution_));
        grid_map_.resize(isize_(0) * isize_(1) * isize_(2));
        for (int i = 0; i < grid_map_.size(); i++) {
            grid_map_[i] = false;
        }
    }
    void update_grid_map(double inflation = 0.0, bool add_boundary = false, bool surface_only = false);
    bool read_json_file(const std::string& path);
    Vector3i pos2idx(const Vector3d pos) const {return Vector3i(pos.x() / resolution_, pos.y() / resolution_, pos.z() / resolution_);}
    bool check_in_map(Vector3i idx) const {
        if (idx.x() >= isize_(0) || idx.y() >= isize_(1) || idx.z() >= isize_(2) 
            || idx.x() < 0 || idx.y() < 0 || idx.z() < 0) {
            return false;
        }
        return true;
    }
    bool check_in_map(Vector3d pos) const {
        Vector3i idx = pos2idx(pos);
        if (idx.x() >= isize_(0) || idx.y() >= isize_(1) || idx.z() >= isize_(2) 
            || idx.x() < 0 || idx.y() < 0 || idx.z() < 0) {
            return false;
        }
        return true;
    }
    bool operator()(Vector3i idx) const {
        if (check_in_map(idx)) {
            return grid_map_[idx.z() * isize_(1) * isize_(0) + idx.y() * isize_(0) + idx.x()];
        } else {
            return true;
        }
    }
    bool operator()(int x, int y, int z) const {
        Vector3i idx(x, y, z);
        if (check_in_map(idx)) {
            return grid_map_[idx.z() * isize_(1) * isize_(0) + idx.y() * isize_(0) + idx.x()];
        } else {
            return true;
        }
    }
    bool operator()(Vector3d pos) const {
        Vector3i idx = pos2idx(pos);
        if (check_in_map(idx)) {
            return grid_map_[idx.z() * isize_(1) * isize_(0) + idx.y() * isize_(0) + idx.x()];
        } else {
            return true;
        }
    }
    bool operator()(double x, double y, double z) const {
        Vector3i idx = pos2idx(Vector3d(x, y, z));
        if (check_in_map(idx)) {
            return grid_map_[idx.z() * isize_(1) * isize_(0) + idx.y() * isize_(0) + idx.x()];
        } else {
            return true;
        }
    }
    void set_grid(Vector3i idx, bool obs) {
        if (check_in_map(idx)) {
            grid_map_[idx.z() * isize_(1) * isize_(0) + idx.y() * isize_(0) + idx.x()] = obs;
        }
    } 
    void set_grid(Vector3d pos, bool obs) {
        Vector3i idx = pos2idx(pos);
        set_grid(idx, obs);
    }
    const double resolution() const {return resolution_;}
    const Vector3d size() const {return size_;}
    const Vector3i isize() const {return isize_;}

    const vector<ObsCylinder>& cylinders() const { return cylinders_; }
    const vector<ObsCircle>& circles() const { return circles_; }
};