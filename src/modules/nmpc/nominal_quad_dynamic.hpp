#ifndef _NOMINAL_QUAD_DYNAMIC_HPP_
#define _NOMINAL_QUAD_DYNAMIC_HPP_

#include <Eigen/Eigen>
#include <vector>

#define PX 0
#define PY 1
#define PZ 2
#define VX 3
#define VY 4
#define VZ 5
#define QW 6
#define QX 7
#define QY 8
#define QZ 9
#define RX 0
#define RY 1
#define RZ 2
#define TH 3

class NominalQuadDynamic {
    public:
        static const int x_dim_ = 10; // [px, py, pz, vx, vy, vz, qw, qx, qy, qz]
        static const int u_dim_ = 4; // [rx, ry, rz, th]

        const double hover_ratio_;

        Eigen::Matrix<double, x_dim_, 1> aux;
        Eigen::Matrix<double, x_dim_, 1> xdot1;
        Eigen::Matrix<double, x_dim_, 1> xdot2;
        Eigen::Matrix<double, x_dim_, 1> xdot3;
        Eigen::Matrix<double, x_dim_, 1> xdot4;
        Eigen::Matrix<double, x_dim_, x_dim_> xd1gx0;
        Eigen::Matrix<double, x_dim_, x_dim_> xd2gx1;
        Eigen::Matrix<double, x_dim_, x_dim_> xd3gx2;
        Eigen::Matrix<double, x_dim_, x_dim_> xd4gx3;
        Eigen::Matrix<double, x_dim_, u_dim_> xd1gu;
        Eigen::Matrix<double, x_dim_, u_dim_> xd2gu;
        Eigen::Matrix<double, x_dim_, u_dim_> xd3gu;
        Eigen::Matrix<double, x_dim_, u_dim_> xd4gu;
        Eigen::Matrix<double, x_dim_, x_dim_> x1gx0;
        Eigen::Matrix<double, x_dim_, x_dim_> x2gx0;
        Eigen::Matrix<double, x_dim_, x_dim_> x3gx0;
        Eigen::Matrix<double, x_dim_, u_dim_> x1gu;
        Eigen::Matrix<double, x_dim_, u_dim_> x2gu;
        Eigen::Matrix<double, x_dim_, u_dim_> x3gu;
        Eigen::Matrix<double, x_dim_, x_dim_> iden;

        NominalQuadDynamic(double hover_ratio) : 
            hover_ratio_(hover_ratio) {
            aux.setZero();
            xdot1.setZero();
            xdot2.setZero();
            xdot3.setZero();
            xdot4.setZero();
            xd1gx0.setZero();
            xd2gx1.setZero();
            xd3gx2.setZero();
            xd4gx3.setZero();
            xd1gu.setZero();
            xd2gu.setZero();
            xd3gu.setZero();
            xd4gu.setZero();
            x1gx0.setZero();
            x2gx0.setZero();
            x3gx0.setZero();
            x1gu.setZero();
            x2gu.setZero();
            x3gu.setZero();
            iden = Eigen::Matrix<double, x_dim_, x_dim_>::Identity();
        }        
        
        void xdot_func(const Eigen::Matrix<double, x_dim_, 1>& x,
                        const Eigen::VectorXd& u,
                        Eigen::Matrix<double, x_dim_, 1>& xdot,
                        Eigen::Matrix<double, x_dim_, x_dim_>& gx,
                        Eigen::Matrix<double, x_dim_, u_dim_>& gu) {
                    
                        const double& px = x(PX);
                        const double& py = x(PY);
                        const double& pz = x(PZ);
                        const double& vx = x(VX);
                        const double& vy = x(VY);
                        const double& vz = x(VZ);
                        const double& qw = x(QW);
                        const double& qx = x(QX);
                        const double& qy = x(QY);
                        const double& qz = x(QZ);
                        const double& rx = u(RX);
                        const double& ry = u(RY);
                        const double& rz = u(RZ);
                        const double& th = u(TH) * 9.81 / hover_ratio_;

                        xdot << vx, vy, vz,
                                2 * (qx * qz + qw * qy) * th, 2 * (qy * qz - qw * qx) * th, (1 - 2 * (qx * qx + qy * qy)) * th - 9.81,
                                0.5 * (-rx * qx - ry * qy - rz * qz), 0.5 * (rx * qw + rz * qy - ry * qz), 0.5 * (ry * qw - rz * qx + rx * qz), 0.5 * (rz * qw + ry * qx - rx * qy);
                        
                        gx(PX, VX) = 1.0;
                        gx(PY, VY) = 1.0;
                        gx(PZ, VZ) = 1.0;
                        gx(VX, QX) = 2 * qz * th; gx(VX, QY) = 2 * qw * th; gx(VX, QZ) = 2 * qx * th; gx(VX, QW) = 2 * qy * th;
                        gx(VY, QX) = -2 * qw * th; gx(VY, QY) = 2 * qz * th; gx(VY, QZ) = 2 * qy * th; gx(VY, QW) = -2 * qx * th;
                        gx(VZ, QX) = -4 * qx * th; gx(VZ, QY) = -4 * qy * th; 
                        gx(QW, QX) = -0.5 * rx; gx(QW, QY) = -0.5 * ry; gx(QW, QZ) = -0.5 * rz;
                        gx(QX, QW) = 0.5 * rx; gx(QX, QY) = 0.5 * rz; gx(QX, QZ) = -0.5 * ry;
                        gx(QY, QW) = 0.5 * ry; gx(QY, QX) = -0.5 * rz; gx(QY, QZ) = 0.5 * rx;
                        gx(QZ, QW) = 0.5 * rz; gx(QZ, QX) = 0.5 * ry; gx(QZ, QY) = -0.5 * rx;

                        gu(VX, TH) = 2 * (qx * qz + qw * qy) * 9.81 / hover_ratio_;
                        gu(VY, TH) = 2 * (qy * qz - qw * qx) * 9.81 / hover_ratio_;
                        gu(VZ, TH) = (1 - 2 * (qx * qx + qy * qy)) * 9.81 / hover_ratio_;
                        gu(QW, RX) = -0.5 * qx; gu(QW, RY) = -0.5 * qy; gu(QW, RZ) = -0.5 * qz;
                        gu(QX, RX) = 0.5 * qw; gu(QX, RY) = -0.5 * qz; gu(QX, RZ) = 0.5 * qy;
                        gu(QY, RX) = 0.5 * qz; gu(QY, RY) = 0.5 * qw; gu(QY, RZ) = -0.5 * qx;
                        gu(QZ, RX) = -0.5 * qy; gu(QZ, RY) = 0.5 * qx; gu(QZ, RZ) = 0.5 * qw;
                    }
        
        void xdot_func(const Eigen::Matrix<double, x_dim_, 1>& x,
                        const Eigen::VectorXd& u,
                        Eigen::Matrix<double, x_dim_, 1>& xdot){

                        const double& px = x(PX);
                        const double& py = x(PY);
                        const double& pz = x(PZ);
                        const double& vx = x(VX);
                        const double& vy = x(VY);
                        const double& vz = x(VZ);
                        const double& qw = x(QW);
                        const double& qx = x(QX);
                        const double& qy = x(QY);
                        const double& qz = x(QZ);
                        const double& rx = u(RX);
                        const double& ry = u(RY);
                        const double& rz = u(RZ);
                        const double& th = u(TH) * 9.81 / hover_ratio_;

                        xdot << vx, vy, vz,
                                2 * (qx * qz + qw * qy) * th, 2 * (qy * qz - qw * qx) * th, (1 - 2 * (qx * qx + qy * qy)) * th - 9.81,
                                0.5 * (-rx * qx - ry * qy - rz * qz), 0.5 * (rx * qw + rz * qy - ry * qz), 0.5 * (ry * qw - rz * qx + rx * qz), 0.5 * (rz * qw + ry * qx - rx * qy);                        
                       }

        void rk4_func(const Eigen::Matrix<double, x_dim_, 1> &x0, const Eigen::VectorXd &u,
            const double &dt, Eigen::Matrix<double, x_dim_, 1> &x1,
            Eigen::Matrix<double, x_dim_, x_dim_> &gx0, Eigen::Matrix<double, x_dim_, u_dim_> &gu,
            Eigen::Matrix<double, 3, 1> &acc, Eigen::Matrix<double, 3, x_dim_> &accdotx0, Eigen::Matrix<double, 3, u_dim_> &accdotu) {
            xdot_func(x0, u, xdot1, xd1gx0, xd1gu);
            acc = xdot1.block(3, 0, 3, 1);
            accdotx0 = xd1gx0.block(3, 0, 3, x_dim_);
            accdotu = xd1gu.block(3, 0, 3, u_dim_);
            aux = x0 + xdot1 * dt / 2.0;
            x1gx0 = iden + dt * 1 / 2.0 * xd1gx0;
            x1gu = dt / 2.0 * xd1gu;
            xdot_func(aux, u, xdot2, xd2gx1, xd2gu);
            aux = x0 + xdot2 * dt / 2.0;
            x2gx0 = iden + dt * 1 / 2.0 * xd2gx1 * x1gx0;
            x2gu = dt / 2.0 * (xd2gx1 * x1gu + xd2gu);
            xdot_func(aux, u, xdot3, xd3gx2, xd3gu);
            aux = x0 + xdot3 * dt;
            x3gx0 = iden + dt * xd3gx2 * x2gx0;
            x3gu = dt * (xd3gx2 * x2gu + xd3gu);
            xdot_func(aux, u, xdot4, xd4gx3, xd4gu);
            x1 = x0 + dt * (1 / 6. * xdot1 + 1 / 3. * xdot2 + 1 / 3. * xdot3 + 1 / 6. * xdot4);
            
            gx0 = iden + 
                dt * (1 / 6.0 * xd1gx0
                + 1 / 3.0 * xd2gx1 * x1gx0
                + 1 / 3.0 * xd3gx2 * x2gx0
                + 1 / 6.0 * xd4gx3 * x3gx0
                );
            gu = dt * (1 / 6.0 * xd1gu + 1 / 3.0 * (xd2gu + xd2gx1 * x1gu) + 1 / 3.0 * (xd3gu + xd3gx2 * x2gu) + 1 / 6.0 * (xd4gu + xd4gx3 * x3gu));
        }

        void rk4_func(const Eigen::Matrix<double, x_dim_, 1> &x0, const Eigen::VectorXd &u,
            const double &dt, Eigen::Matrix<double, x_dim_, 1> &x1) {
            xdot_func(x0, u, xdot1);
            aux = x0 + xdot1 * (dt / 2.0);
            xdot_func(aux, u, xdot2);
            aux = x0 + xdot2 * (dt / 2.0);
            xdot_func(aux, u, xdot3);
            aux = x0 + xdot3 * dt;
            xdot_func(aux, u, xdot4);
            x1 = x0 + (dt / 6.0) * (xdot1 + 2 * xdot2 + 2 * xdot3 + xdot4);
        }        


};





#endif // _NOMINAL_QUAD_DYNAMIC_HPP_