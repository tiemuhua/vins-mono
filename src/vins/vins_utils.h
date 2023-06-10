//
// Created by gjt on 5/14/23.
//

#ifndef VINS_UTILS_H
#define VINS_UTILS_H
#include <cmath>
#include <Eigen/Eigen>

namespace vins {

    typedef const Eigen::Matrix3d & ConstMat3dRef;
    typedef const Eigen::Vector3d & ConstVec3dRef;
    typedef const Eigen::Quaterniond & ConstQuatRef;
    typedef Eigen::Matrix3d & Mat3dRef;
    typedef Eigen::Vector3d & Vec3dRef;
    typedef Eigen::Quaterniond & QuatRef;

    namespace utils {
        /**
         * C数组的向量运算 todo tiemuhua 使用map<Eigen::Vector>运算，比较eigen和原生C的速度
         * */
        inline void arrayMinus(const double * first, const double * second, double * target, int length) {
            Eigen::Map<const Eigen::VectorXd> first_vec(first, length);
            Eigen::Map<const Eigen::VectorXd> second_vec(second, length);
            Eigen::Map<const Eigen::VectorXd> target_vec(target, length);
            target_vec = first_vec - second_vec;
        }
        inline void arrayPlus(const double * first, const double * second, double * target, int length) {
            for (int i = 0; i < length; ++i) {
                target[i] = first[i] + second[i];
            }
        }
        inline void arrayMultiply(const double* first, double * target, const double k, int length) {
            for (int i = 0; i < length; ++i) {
                target[i] = first[i] * k;
            }
        }

        /**
         * 角度运算
         * */
        template<typename T>
        T normalizeAngle180(const T &angle_degrees) {
            if (angle_degrees > T(180.0))
                return angle_degrees - T(360.0);
            else if (angle_degrees < T(-180.0))
                return angle_degrees + T(360.0);
            else
                return angle_degrees;
        };
        template<typename T>
        T normalizeAnglePi(const T &angle_degrees) {
            constexpr T pi = 3.1415926575;
            if (angle_degrees > pi)
                return angle_degrees - 2*pi;
            else if (angle_degrees < -pi)
                return angle_degrees + 2*pi;
            else
                return angle_degrees;
        };

        template<typename Derived>
        static void reduceVector(vector<Derived> &v, const vector<uchar> &status) {
            int j = 0;
            for (int i = 0; i < int(v.size()); i++)
                if (status[i])
                    v[j++] = v[i];
            v.resize(j);
        }

        /**
         * 矩阵运算
         * */
        template<typename Derived>
        static Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> &theta) {
            typedef typename Derived::Scalar Scalar_t;

            Eigen::Quaternion<Scalar_t> dq;
            Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
            half_theta /= static_cast<Scalar_t>(2.0);
            dq.w() = static_cast<Scalar_t>(1.0);
            dq.x() = half_theta.x();
            dq.y() = half_theta.y();
            dq.z() = half_theta.z();
            return dq;
        }

        template<typename Derived>
        static Eigen::Matrix<typename Derived::Scalar, 3, 3> skewSymmetric(const Eigen::MatrixBase<Derived> &q) {
            Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
            ans << typename Derived::Scalar(0), -q(2), q(1),
                    q(2), typename Derived::Scalar(0), -q(0),
                    -q(1), q(0), typename Derived::Scalar(0);
            return ans;
        }

        template<typename Derived>
        static Eigen::Quaternion<typename Derived::Scalar> positify(const Eigen::QuaternionBase<Derived> &q) {
            return q;
        }

        template<typename Derived>
        static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qleft(const Eigen::QuaternionBase<Derived> &q) {
            Eigen::Quaternion<typename Derived::Scalar> qq = positify(q);
            Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
            ans(0, 0) = qq.w(), ans.template block<1, 3>(0, 1) = -qq.vec().transpose();
            ans.template block<3, 1>(1, 0) = qq.vec(), ans.template block<3, 3>(1, 1) =
                    qq.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() + skewSymmetric(qq.vec());
            return ans;
        }

        template<typename Derived>
        static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qright(const Eigen::QuaternionBase<Derived> &p) {
            Eigen::Quaternion<typename Derived::Scalar> pp = positify(p);
            Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
            ans(0, 0) = pp.w(), ans.template block<1, 3>(0, 1) = -pp.vec().transpose();
            ans.template block<3, 1>(1, 0) = pp.vec(), ans.template block<3, 3>(1, 1) =
                    pp.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() - skewSymmetric(pp.vec());
            return ans;
        }

        // todo tiemuhuaguo 感觉顺序搞错了
        inline Eigen::Vector3d rot2rpy(ConstMat3dRef R) {
            return R.eulerAngles(2,1,0);
        }
        inline Eigen::Vector3d rot2ypr(ConstMat3dRef R) {
            return R.eulerAngles(0,1,2);
        }
        inline Eigen::Matrix3d ypr2rot(ConstVec3dRef ypr) {
            return  Eigen::AngleAxisd(ypr.z(), Eigen::Vector3d::UnitZ()).toRotationMatrix() *
                    Eigen::AngleAxisd(ypr.y(), Eigen::Vector3d::UnitY()).toRotationMatrix() *
                    Eigen::AngleAxisd(ypr.x(), Eigen::Vector3d::UnitX()).toRotationMatrix();
        }

        template<size_t N>
        struct uint_ {
        };

        template<size_t N, typename Lambda, typename IterT>
        void unroller(const Lambda &f, const IterT &iter, uint_<N>) {
            unroller(f, iter, uint_<N - 1>());
            f(iter + N);
        }

        template<typename Lambda, typename IterT>
        void unroller(const Lambda &f, const IterT &iter, uint_<0>) {
            f(iter);
        }

        inline void quat2array(const Eigen::Quaterniond &q, double *arr) {
            arr[0] = q.w();
            arr[1] = q.x();
            arr[2] = q.y();
            arr[3] = q.z();
        }

        inline Eigen::Quaterniond array2quat(const double *arr) {
            Eigen::Quaterniond q;
            q.w() = arr[0];
            q.x() = arr[1];
            q.y() = arr[2];
            q.z() = arr[3];
        }

        inline void vec3d2array(const Eigen::Vector3d &vec, double *arr) {
            arr[0] = vec(0);
            arr[1] = vec(1);
            arr[2] = vec(2);
        }

        inline Eigen::Vector3d array2vec3d(const double *arr) {
            return {arr[0], arr[1], arr[2]};
        }

        inline cv::Point3f eigenVec2CVVec(const Eigen::Vector3d &vec) {
            return cv::Point3f(vec(0), vec(1), vec(2));
        }
    };
}


#endif //VINS_UTILS_H
