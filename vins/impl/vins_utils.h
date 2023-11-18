//
// Created by gjt on 5/14/23.
//

#ifndef VINS_UTILS_H
#define VINS_UTILS_H

#include <sys/time.h>
#include <cmath>
#include <sstream>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include "vins_model.h"

#define Synchronized(mutex_)  for(ScopedLocker locker(mutex_); locker.cnt < 1; locker.cnt++)

class ScopedLocker {
public:
    explicit ScopedLocker(std::mutex &mutex) : guard(mutex) {}

    std::lock_guard<std::mutex> guard;
    int cnt = 0;
};

#define PRINT_FUNCTION_TIME_COST vins::utils::FunctionTimer function_timer_##__FUNCTION__(__FUNCTION__);

namespace vins {

    namespace utils {

        class Timer {
        public:
            Timer() {
                gettimeofday(&tv, nullptr);
            }
            int getCostUs() {
                timeval tv1{};
                gettimeofday(&tv1, nullptr);
                // 括号顺序保证了不会溢出
                return ((tv1.tv_sec - tv.tv_sec) * 1000000 + tv1.tv_usec) - tv.tv_usec;
            }
            timeval tv{};
        };
        class FunctionTimer {
        public:
            FunctionTimer(std::string function_name)
            : function_name_(std::move(function_name)){}
            ~FunctionTimer() {
                LOG(INFO) << function_name_ << " cost time ms:" << timer_.getCostUs() / 1e3;
            }

        private:
            std::string function_name_;
            Timer timer_;
        };

        constexpr double pi = 3.1415926;

        template<typename T, typename F>
        void erase_if_wrapper(std::vector<T> &vec, const F& func) {
            auto it = std::remove_if(vec.begin(), vec.end(), func);
            vec.erase(it, vec.end());
        }
        template<typename T, typename F>
        int count_if_wrapper(const std::vector<T> &vec, const F& func) {
            return std::count_if(vec.begin(), vec.end(), func);
        }
        template<typename T>
        void insert_move_wrapper(std::vector<T> &vec1, std::vector<T> &vec2) {
            vec1.insert(vec1.end(), std::make_move_iterator(vec2.begin()), std::make_move_iterator(vec2.end()));
        }

        template<typename T>
        std::string eigen2string(T mat) {
            std::stringstream sstream;
            sstream << mat;
            std::string str;
            sstream >> str;
            return str;
        }

        inline cv::Point3f cvPoint2fToCvPoint3f(const cv::Point2f &p2d, double depth) {
            cv::Point3f cv_p3;
            cv_p3.x = p2d.x * depth;
            cv_p3.y = p2d.y * depth;
            cv_p3.z = 1.0 * depth;
            return cv_p3;
        }

        template<typename Derived>
        void reduceVector(std::vector<Derived> &v, const std::vector<uint8_t> &status) {
            int j = 0;
            for (int i = 0; i < int(v.size()); i++)
                if (status[i])
                    v[j++] = v[i];
            v.resize(j);
        }

        /****************** 向量运算 ******************/
        //todo tiemuhua 使用map<Eigen::Vector>运算，比较eigen和原生C的速度
        template<typename T>
        inline void arrayMinus(const T *first, const T *second, T *target, int length) {
            for (int i = 0; i < length; ++i) {
                target[i] = first[i] - second[i];
            }
        }

        template<typename T>
        inline void arrayPlus(const T *first, const T *second, T *target, int length) {
            for (int i = 0; i < length; ++i) {
                target[i] = first[i] + second[i];
            }
        }

        template<typename T>
        inline void arrayMultiply(const T *first, T *target, const T k, int length) {
            for (int i = 0; i < length; ++i) {
                target[i] = first[i] * k;
            }
        }

        /****************** 角度运算 ******************/
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
            if (angle_degrees > pi)
                return angle_degrees - 2 * pi;
            else if (angle_degrees < -pi)
                return angle_degrees + 2 * pi;
            else
                return angle_degrees;
        }

        /****************** 矩阵运算 ******************/
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
            ans.template block<3, 1>(1, 0) = qq.vec();
            auto identity = Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity();
            ans.template block<3, 3>(1, 1) = qq.w() * identity + skewSymmetric(qq.vec());
            return ans;
        }

        template<typename Derived>
        static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qright(const Eigen::QuaternionBase<Derived> &p) {
            Eigen::Quaternion<typename Derived::Scalar> pp = positify(p);
            Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
            ans(0, 0) = pp.w(), ans.template block<1, 3>(0, 1) = -pp.vec().transpose();
            ans.template block<3, 1>(1, 0) = pp.vec(), ans.template block<3, 3>(1, 1) =
                                                               pp.w() *
                                                               Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() -
                                                               skewSymmetric(pp.vec());
            return ans;
        }

        // todo tiemuhuaguo 感觉顺序搞错了
        // T: ceres::Jet<double, 8>
        template<class T>
        inline Eigen::Matrix<T, 3, 1> rot2rpy(const Eigen::Matrix<T, 3, 3> &R) {
            return R.eulerAngles(2, 1, 0);
        }

        inline Eigen::Vector3d rot2rpy(const Eigen::Matrix3d& R) {
            return R.eulerAngles(2, 1, 0);
        }

        template<class T>
        inline Eigen::Matrix<T, 3, 1> rot2ypr(const Eigen::Matrix<T, 3, 3> &R) {
            return R.eulerAngles(0, 1, 2);
        }

        inline Eigen::Vector3d rot2ypr(const Eigen::Matrix3d& R) {
            return R.eulerAngles(0, 1, 2);
        }

        inline Eigen::Matrix3d ypr2rot(const Eigen::Vector3d& ypr) {
            return Eigen::AngleAxisd(ypr.z(), Eigen::Vector3d::UnitZ()).toRotationMatrix() *
                   Eigen::AngleAxisd(ypr.y(), Eigen::Vector3d::UnitY()).toRotationMatrix() *
                   Eigen::AngleAxisd(ypr.x(), Eigen::Vector3d::UnitX()).toRotationMatrix();
        }

        template<class T>
        inline Eigen::Matrix<T, 3, 3> ypr2rot(const Eigen::Matrix<T, 3, 1> &ypr) {
            return Eigen::AngleAxis<T>(ypr.z(), Eigen::Matrix<T, 3, 1>::UnitZ()).toRotationMatrix() *
                   Eigen::AngleAxis<T>(ypr.y(), Eigen::Matrix<T, 3, 1>::UnitY()).toRotationMatrix() *
                   Eigen::AngleAxis<T>(ypr.x(), Eigen::Matrix<T, 3, 1>::UnitX()).toRotationMatrix();
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

        /****************** C数组和eigen互转 ******************/
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
            return q;
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
            return {static_cast<float>(vec(0)), static_cast<float>(vec(1)), static_cast<float>(vec(2))};
        }
    };
}


#endif //VINS_UTILS_H
