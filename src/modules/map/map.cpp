#include <fstream>

#include "map.hpp"

bool GridMap::read_json_file(const std::string& path) {
    using json = nlohmann::json;
    std::ifstream file(path);
    if (!file) {
        cerr << "Fail to open JSON map file: " << path << endl;
        throw "Fail to open JSON map file";
        return false;
    }
    json j;
    file >> j;

    resolution_ = j.at("resolution").get<double>();
    auto ms = j.at("map_size");
    size_ = Vector3d(ms[0].get<double>(), ms[1].get<double>(), ms[2].get<double>());
    isize_ = Vector3i(round(size_(0) / resolution_), round(size_(1) / resolution_), round(size_(2) / resolution_));
    grid_map_.resize(isize_(0) * isize_(1) * isize_(2));
    std::fill(grid_map_.begin(), grid_map_.end(), false);

    cylinders_.clear();

    double half_x = size_(0) / 2.0;
    double half_y = size_(1) / 2.0;

    for (auto& obs : j.at("obstacles")) {
        std::string type = obs.at("type").get<std::string>();
        double wx = obs.at("x").get<double>();
        double wy = obs.at("y").get<double>();
        double wz = obs.at("z").get<double>();
        double mx = wx + half_x;
        double my = wy + half_y;

        if (type == "cylinder") {
            double radius = obs.at("radius").get<double>();
            double height = obs.at("height").get<double>();
            cylinders_.push_back(ObsCylinder(Vector3d(mx, my, wz), height, radius));
        } else if (type == "circle") {
            double r1 = obs.at("radius1").get<double>();
            double r2 = obs.at("radius2").get<double>();
            double theta = obs.at("theta").get<double>();
            double thick = obs.at("thick").get<double>();
            circles_.push_back(ObsCircle(Vector3d(mx, my, wz), r1, r2, theta, thick));
        } else {
            cerr << "Unknown obstacle type in JSON: " << type << endl;
        }
    }

    return true;
}

void GridMap::update_grid_map(double inflation, bool add_boundary, bool surface_only) {
    grid_map_.resize(isize_(0) * isize_(1) * isize_(2));
    std::fill(grid_map_.begin(), grid_map_.end(), false);

    for (const auto &cyl : cylinders_) {
        double cx = floor(cyl.pos_(0) / resolution_) * resolution_ + resolution_ / 2.0;
        double cy = floor(cyl.pos_(1) / resolution_) * resolution_ + resolution_ / 2.0;
        int widNum = (int)ceil(cyl.radius_ * 2.0 / resolution_);
        int heiNum = (int)ceil(cyl.high_ / resolution_);
        for (int r = -widNum / 2; r < widNum / 2; r++) {
            for (int s = -widNum / 2; s < widNum / 2; s++) {
                double vx = cx + (r + 0.5) * resolution_;
                double vy = cy + (s + 0.5) * resolution_;
                double dist = (Vector2d(vx, vy) - Vector2d(cx, cy)).norm();
                if (dist > cyl.radius_)
                    continue;
                if (surface_only && dist < cyl.radius_ - resolution_)
                    continue;
                for (int t = 0; t < heiNum; t++) {
                    double vz = cyl.pos_(2) + (t + 0.5) * resolution_;
                    set_grid(Vector3d(vx, vy, vz), true);
                }
            }
        }
    }

    for (const auto &circ : circles_) {
        double cos_t = cos(circ.theta_);
        double sin_t = sin(circ.theta_);
        for (double angle = 0.0; angle < M_PI * 2; angle += resolution_ / 2.0) {
            Vector3d local(0.0, circ.radius1_ * cos(angle), circ.radius2_ * sin(angle));
            Vector3d world;
            world(0) = cos_t * local(0) - sin_t * local(1) + circ.pos_(0);
            world(1) = sin_t * local(0) + cos_t * local(1) + circ.pos_(1);
            world(2) = local(2) + circ.pos_(2);
            set_grid(world, true);
        }
    }

    if (add_boundary) {
        for (int z = 0; z < isize_(2); z++) {
            for (int x = 0; x < isize_(0); x++) {
                for (int y = 0; y < isize_(1); y++) {
                    if (x == 0 || x == isize_(0) - 1 || y == 0 || y == isize_(1) - 1 || z == 0 || z == isize_(2) - 1) {
                        set_grid(Vector3i(x, y, z), true);
                    }
                }
            }
        }
    }

    if (inflation > 1e-6) {
        int inf_cells = std::max(1, (int)std::round(inflation / resolution_));
        std::vector<bool> inflated = grid_map_;
        for (int z = 0; z < isize_(2); z++) {
            for (int y = 0; y < isize_(1); y++) {
                for (int x = 0; x < isize_(0); x++) {
                    if (!grid_map_[z * isize_(1) * isize_(0) + y * isize_(0) + x]) continue;
                    for (int dz = -inf_cells; dz <= inf_cells; dz++) {
                        int nz = z + dz;
                        if (nz < 0 || nz >= isize_(2)) continue;
                        for (int dy = -inf_cells; dy <= inf_cells; dy++) {
                            int ny = y + dy;
                            if (ny < 0 || ny >= isize_(1)) continue;
                            for (int dx = -inf_cells; dx <= inf_cells; dx++) {
                                int nx = x + dx;
                                if (nx < 0 || nx >= isize_(0)) continue;
                                inflated[nz * isize_(1) * isize_(0) + ny * isize_(0) + nx] = true;
                            }
                        }
                    }
                }
            }
        }
        grid_map_ = inflated;
    }
}