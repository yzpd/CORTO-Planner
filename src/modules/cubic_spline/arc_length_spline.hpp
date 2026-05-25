// Copyright 2019 Alexander Liniger

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

#ifndef _ARC_LENGTH_SPLINE_H_
#define _ARC_LENGTH_SPLINE_H_

#include "cubic_spline.hpp"
// #include "types.hpp"
// #include "Params/params.hpp"
#include <map>

//return value
struct RawPath{
    Eigen::VectorXd X;
    Eigen::VectorXd Y;
};

struct RawPath_3D{
    Eigen::VectorXd X;
    Eigen::VectorXd Y;
    Eigen::VectorXd Z;
};

// data struct
struct PathData{
    Eigen::VectorXd X;
    Eigen::VectorXd Y;
    Eigen::VectorXd s;
    int n_points;
};

struct PathData_3D{
    Eigen::VectorXd X;
    Eigen::VectorXd Y;
    Eigen::VectorXd Z;
    Eigen::VectorXd s;
    int n_points;
};

class ArcLengthSpline {
public:
    static constexpr int N_SPLINE = 200;

    // X and Y spline used for final spline fit
    void gen2DSpline(const Eigen::VectorXd &X,const Eigen::VectorXd &Y);
    void gen3DSpline(const Eigen::VectorXd &X,const Eigen::VectorXd &Y,const Eigen::VectorXd &Z);

    Eigen::Vector2d getPosition(double) const;
    Eigen::Vector3d getPosition_3D(const double s) const;

    Eigen::Vector2d getDerivative(double) const;
    Eigen::Vector3d getDerivative_3D(const double s) const;

    Eigen::Vector2d getSecondDerivative(double) const;
    Eigen::Vector3d getSecondDerivative_3D(const double s) const;

    double getLength() const;
    double getLength_3D() const;

    // double porjectOnSpline(const State &x) const;

    ArcLengthSpline();
    // ArcLengthSpline(const PathToJson &path);
    // void setParam(const Param &param) { param_ = param; };

    CubicSpline getSplineX() const { return spline_x_; }
    CubicSpline getSplineY() const { return spline_y_; }
    CubicSpline getSplineZ() const { return spline_z_; }
    PathData_3D getPathData() const { return path_data_3D_; }
    

private:
    void setData(const Eigen::VectorXd &X_in,const Eigen::VectorXd &Y_in);
    void setData_3D(const Eigen::VectorXd &X_in,const Eigen::VectorXd &Y_in,const Eigen::VectorXd &Z_in);

    void setRegularData(const Eigen::VectorXd &X_in,const Eigen::VectorXd &Y_in,const Eigen::VectorXd &s_in);
    void setRegularData_3D(const Eigen::VectorXd &X_in,const Eigen::VectorXd &Y_in,const Eigen::VectorXd &Z_in,const Eigen::VectorXd &s_in);

    Eigen::VectorXd compArcLength(const Eigen::VectorXd &X_in,const Eigen::VectorXd &Y_in) const;
    Eigen::VectorXd compArcLength_3D(const Eigen::VectorXd &X_in,const Eigen::VectorXd &Y_in,const Eigen::VectorXd &Z_in) const;

    PathData resamplePath(const CubicSpline &initial_spline_x,const CubicSpline &initial_spline_y,double total_arc_length) const;
    PathData_3D resamplePath_3D(const CubicSpline &initial_spline_x,const CubicSpline &initial_spline_y,const CubicSpline &initial_spline_z,const double total_arc_length) const;

    RawPath outlierRemoval(const Eigen::VectorXd &X_original,const Eigen::VectorXd &Y_original) const;
    RawPath_3D outlierRemoval_3D(const Eigen::VectorXd &X_original,const Eigen::VectorXd &Y_original,const Eigen::VectorXd &Z_original) const;

    void fitSpline(const Eigen::VectorXd &X,const Eigen::VectorXd &Y);
    void fitSpline_3D(const Eigen::VectorXd &X,const Eigen::VectorXd &Y,const Eigen::VectorXd &Z);

    double unwrapInput(double x) const;

    PathData path_data_;      // initial data and data used for successive fitting
    PathData_3D path_data_3D_; // initial data and data used for successive fitting

//    PathData pathDataFinal; // final data
    CubicSpline spline_x_;
    CubicSpline spline_y_;
    CubicSpline spline_z_;
    // Param param_;
};

#endif