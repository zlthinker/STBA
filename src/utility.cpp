#include "utility.h"
#include <chrono>
#include <vector>
#include <cassert>
#include <fstream>

Mat3 AngleAxis2Matrix(Vec3 const & angle_axis)
{
    double angle = std::max(angle_axis.norm(), EPSILON);
    Vec3 axis = angle_axis / angle;
    AxisAngle rotation(angle, axis);
    return rotation.toRotationMatrix();
}

Vec3 Quaternion2AngleAxis(Vec4 const & quaternion)
{
    double qw = quaternion[0];
    double qx = quaternion[1];
    double qy = quaternion[2];
    double qz = quaternion[3];
    double angle = 2 * acos(qw);
    double sq = std::max(EPSILON, 1 - qw * qw);
    double x = qx / sqrt(sq);
    double y = qy / sqrt(sq);
    double z = qz / sqrt(sq);
    return Vec3(angle * x, angle * y, angle * z);
}

Vec3 RotatePoint(Vec3 const & angle_axis, Vec3 const & point)
{
    double angle = std::max(angle_axis.norm(), EPSILON);
    Vec3 axis = angle_axis / angle;
    double cos = std::cos(angle);
    double sin = std::sin(angle);
    Vec3 cross_prod = axis.cross(point);
    double dot_prod = axis.dot(point);
    return cos * point + sin * cross_prod + (1-cos) * dot_prod * axis;
}

Mat93  RotationMat2AngleAxisJacobian(Vec3 const & angle_axis)
{
    MatX jacobian_mat_angleaxis = MatX::Zero(9, 3);    // jacobian of rotation matrix w.r.t 3D angle-axis representation
    double angle = std::max(angle_axis.norm(), EPSILON);
    double x = angle_axis(0);
    double y = angle_axis(1);
    double z = angle_axis(2);
    double cos_angle = std::cos(angle);
    double sin_angle = std::sin(angle);
    double r = sin_angle / angle;
    double s = (1 - cos_angle) / (angle * angle);
    double d_r_angle = (cos_angle * angle - sin_angle) / (angle * angle);   // derivative of r w.r.t angle
    double d_s_angle = (sin_angle * angle + cos_angle * 2 - 2) / (std::pow(angle, 3));   // derivative of s w.r.t angle
    double d_angle_x = x / angle;        // derivative of angle w.r.t x
    double d_angle_y = y / angle;        // derivative of angle w.r.t y
    double d_angle_z = z / angle;        // derivative of angle w.r.t z
    double d_r_x = d_r_angle * d_angle_x;
    double d_r_y = d_r_angle * d_angle_y;
    double d_r_z = d_r_angle * d_angle_z;
    double d_s_x = d_s_angle * d_angle_x;
    double d_s_y = d_s_angle * d_angle_y;
    double d_s_z = d_s_angle * d_angle_z;
    double x2_y2 = x*x + y*y;
    double x2_z2 = x*x + z*z;
    double y2_z2 = y*y + z*z;

    jacobian_mat_angleaxis(0, 0) = -y2_z2 * d_s_x;
    jacobian_mat_angleaxis(0, 1) = -(2 * y * s + y2_z2 * d_s_y);
    jacobian_mat_angleaxis(0, 2) = -(2 * z * s + y2_z2 * d_s_z);
    jacobian_mat_angleaxis(1, 0) = -z * d_r_x + y * (s + x * d_s_x);
    jacobian_mat_angleaxis(1, 1) = -z * d_r_y + x * (s + y * d_s_y);
    jacobian_mat_angleaxis(1, 2) = -(r + z * d_r_z) + x * y * d_s_z;
    jacobian_mat_angleaxis(2, 0) = y * d_r_x + z * (s + x * d_s_x);
    jacobian_mat_angleaxis(2, 1) = (r + y * d_r_y) + x * z * d_s_y;
    jacobian_mat_angleaxis(2, 2) = y * d_r_z + x * (s + z * d_s_z);
    jacobian_mat_angleaxis(3, 0) = z * d_r_x + y * (s + x * d_s_x);
    jacobian_mat_angleaxis(3, 1) = z * d_r_y + x * (s + y * d_s_y);
    jacobian_mat_angleaxis(3, 2) = (r + z * d_r_z) + x * y * d_s_z;
    jacobian_mat_angleaxis(4, 0) = -(2 * x * s + x2_z2 * d_s_x);
    jacobian_mat_angleaxis(4, 1) = -x2_z2 * d_s_y;
    jacobian_mat_angleaxis(4, 2) = -(2 * z * s + x2_z2 * d_s_z);
    jacobian_mat_angleaxis(5, 0) = -(r + x * d_r_x) + y * z * d_s_x;
    jacobian_mat_angleaxis(5, 1) = -x * d_r_y + z * (s + y * d_s_y);
    jacobian_mat_angleaxis(5, 2) = -x * d_r_z + y * (s + z * d_s_z);
    jacobian_mat_angleaxis(6, 0) = -y * d_r_x + z * (s + x * d_s_x);
    jacobian_mat_angleaxis(6, 1) = -(r + y * d_r_y) + x * z * d_s_y;
    jacobian_mat_angleaxis(6, 2) = -y * d_r_z + x * (s + z * d_s_z);
    jacobian_mat_angleaxis(7, 0) = (r + x * d_r_x) + y * z * d_s_x;
    jacobian_mat_angleaxis(7, 1) = x * d_r_y + z * (s + y * d_s_y);
    jacobian_mat_angleaxis(7, 2) = x * d_r_z + y * (s + z * d_s_z);
    jacobian_mat_angleaxis(8, 0) = -(2 * x * s + x2_y2 * d_s_x);
    jacobian_mat_angleaxis(8, 1) = -(2 * y * s + x2_y2 * d_s_y);
    jacobian_mat_angleaxis(8, 2) = -x2_y2 * d_s_z;

    return jacobian_mat_angleaxis;
}

/*!
 * @brief Compute derivative of local point w.r.t axis-angle parameters enbodying the rotation transformation
 * @brief Reference: https://math.stackexchange.com/questions/64253/jacobian-matrix-of-the-rodrigues-formula-exponential-map
 * @param angle_axis: The angle-axis representation of a 3D rotation
 * @param input: The input 3D vector
 */
Mat3 LocalPoint2RotationJacobian(Vec3 const & angle_axis, Vec3 const & input)
{
    MatX jacobian_out_mat = MatX::Zero(3, 9);    // jacobian of output vector w.r.t rotation matrix
    //    std::cout << jacobian_out_mat << "\n";
    jacobian_out_mat(0, 0) = input(0);
    jacobian_out_mat(0, 1) = input(1);
    jacobian_out_mat(0, 2) = input(2);
    jacobian_out_mat(1, 3) = input(0);
    jacobian_out_mat(1, 4) = input(1);
    jacobian_out_mat(1, 5) = input(2);
    jacobian_out_mat(2, 6) = input(0);
    jacobian_out_mat(2, 7) = input(1);
    jacobian_out_mat(2, 8) = input(2);

    Mat93 jacobian_mat_angleaxis = RotationMat2AngleAxisJacobian(angle_axis);
    Mat3 jacobian = jacobian_out_mat * jacobian_mat_angleaxis;
    return jacobian;
}

/*!
 * @brief Compute derivative of local point w.r.t translation
 */
Mat3 LocalPoint2TranslationJacobian()
{
    return Mat3::Identity();
}

Mat3 LocalPoint2GlobalPointJacobian(Vec3 const & angle_axis)
{
    return AngleAxis2Matrix(angle_axis);
}

/*!
 * @brief The jacobian of projection in pixel space w.r.t distorted projetion in camera space
 * u = fx' + u0, v = fy' + v0
 */
Mat2 Projection2DistProjectionJacobian(double const focal)
{
    Mat2 jacobian = Mat2::Zero();
    jacobian(0, 0) = focal;
    jacobian(1, 1) = focal;
    return jacobian;
}

/*!
 * @brief The jacobian of distorted projection w.r.t the undistorted projection, both in camera space
 * x' = x (1 + k1 r^2 + k2 r^4 + k3 r^6), y' = y (1 + k1 r^2 + k2 r^4 + k3 r^6), r^2 = x^2 + y^2
 */
Mat2 DistProjection2UndistProjectionJacobian(Vec2 const & undist_proj,
                                             Vec3 const & radial_distortion)
{
    double r2 = undist_proj(0) * undist_proj(0) + undist_proj(1) * undist_proj(1);
    double k1 = radial_distortion(0);
    double k2 = radial_distortion(1);
    double k3 = radial_distortion(2);
    double coeff = 1 + k1 * r2 + k2 * r2 *r2 + k3 * r2 * r2 * r2;
    double r2_x = 2 * undist_proj(0);
    double r2_y = 2 * undist_proj(1);
    Mat2 jacobian;
    jacobian(0, 0) = coeff + undist_proj(0) * (k1 + 2*k2*r2 + 3*k3*r2*r2) * r2_x;
    jacobian(0, 1) = undist_proj(0) * (k1 + 2*k2*r2 + 3*k3*r2*r2) * r2_y;
    jacobian(1, 0) = undist_proj(1) * (k1 + 2*k2*r2 + 3*k3*r2*r2) * r2_x;
    jacobian(1, 1) = coeff + undist_proj(1) * (k1 + 2*k2*r2 + 3*k3*r2*r2) * r2_y;
    return jacobian;
}

/*!
 * @brief The jacobian of undistorted projection w.r.t local point, both in camera space
 * x = X/Z, y = Y/Z
 */
Mat23 UndistProjection2LocalPointJacobian(Vec3 const & local_point)
{
    MatX jacobian = MatX::Zero(2, 3);
    double x = local_point(0);
    double y = local_point(1);
    double z = std::max(local_point(2), EPSILON);
    jacobian(0, 0) = 1 / z;
    jacobian(0, 1) = 0;
    jacobian(0, 2) = -x / (z * z);
    jacobian(1, 0) = 0;
    jacobian(1, 1) = 1 / z;
    jacobian(1, 2) = -y / (z * z);
    return jacobian;
}

Mat26 Projection2IntrinsicJacobian(double const focal,
                                   Vec3 const & radial_distortion,
                                   Vec3 const & angle_axis,
                                   Vec3 const & translation,
                                   Vec3 const & global_point)
{
    Vec3 local_point = RotatePoint(angle_axis, global_point) + translation;
    double depth = std::max(local_point(2), MIN_DEPTH);
    double x = local_point(0) / depth;
    double y = local_point(1) / depth;
    double r2 = x * x + y * y;
    double r4 = r2 * r2;
    double r6 = r2 * r4;
    double k1 = radial_distortion(0);
    double k2 = radial_distortion(1);
    double k3 = radial_distortion(2);
    Mat26 jacobian_intrinsic = Mat26::Zero();
    jacobian_intrinsic(0, 0) = x * (1 + k1 * r2 + k2 * r4 + k3 * r6);
    jacobian_intrinsic(1, 0) = y * (1 + k1 * r2 + k2 * r4 + k3 * r6);
    jacobian_intrinsic(0, 1) = 1;
    jacobian_intrinsic(1, 2) = 1;
    jacobian_intrinsic(0, 3) = x * r2 * focal;
    jacobian_intrinsic(0, 4) = x * r4 * focal;
    jacobian_intrinsic(0, 5) = x * r6 * focal;
    jacobian_intrinsic(1, 3) = y * r2 * focal;
    jacobian_intrinsic(1, 4) = y * r4 * focal;
    jacobian_intrinsic(1, 5) = y * r6 * focal;
    return jacobian_intrinsic;
}

Mat23 Projection2RotationJacobian(double const focal,
                                  Vec3 const & radial_distortion,
                                  Vec3 const & angle_axis,
                                  Vec3 const & translation,
                                  Vec3 const & global_point)
{
    Vec3 local_point = RotatePoint(angle_axis, global_point) + translation;
    Vec2 undist_proj(local_point(0) / std::max(local_point(2), MIN_DEPTH), local_point(1) / std::max(local_point(2), MIN_DEPTH));
    Mat2 jacobian_proj_distproj = Projection2DistProjectionJacobian(focal);
    Mat2 jacobian_distproj_undistproj = DistProjection2UndistProjectionJacobian(undist_proj, radial_distortion);
    Mat23 jacobian_undistproj_local = UndistProjection2LocalPointJacobian(local_point);
    Mat3 jacobian_local_rotation = LocalPoint2RotationJacobian(angle_axis, global_point);
    Mat23 jacobian_proj_rotation = jacobian_proj_distproj * jacobian_distproj_undistproj *
            jacobian_undistproj_local * jacobian_local_rotation;
    return jacobian_proj_rotation;
}

Mat23 Projection2TranslationJacobian(double const focal,
                                     Vec3 const & radial_distortion,
                                     Vec3 const & angle_axis,
                                     Vec3 const & translation,
                                     Vec3 const & global_point)
{
    Vec3 local_point = RotatePoint(angle_axis, global_point) + translation;
    Vec2 undist_proj(local_point(0) / std::max(local_point(2), MIN_DEPTH), local_point(1) / std::max(local_point(2), MIN_DEPTH));
    Mat2 jacobian_proj_distproj = Projection2DistProjectionJacobian(focal);
    Mat2 jacobian_distproj_undistproj = DistProjection2UndistProjectionJacobian(undist_proj, radial_distortion);
    Mat23 jacobian_undistproj_local = UndistProjection2LocalPointJacobian(local_point);
    Mat3 jacobian_local_translation = LocalPoint2TranslationJacobian();
    Mat23 jacobian_proj_translation = jacobian_proj_distproj * jacobian_distproj_undistproj *
            jacobian_undistproj_local * jacobian_local_translation;
    return jacobian_proj_translation;
}

Mat23 Projection2GlobalPointJacobian(double const focal,
                                     Vec3 const & radial_distortion,
                                     Vec3 const & angle_axis,
                                     Vec3 const & translation,
                                     Vec3 const & global_point)
{
    Vec3 local_point = RotatePoint(angle_axis, global_point) + translation;
    Vec2 undist_proj(local_point(0) / std::max(local_point(2), MIN_DEPTH), local_point(1) / std::max(local_point(2), MIN_DEPTH));
    Mat2 jacobian_proj_distproj = Projection2DistProjectionJacobian(focal);
    Mat2 jacobian_distproj_undistproj = DistProjection2UndistProjectionJacobian(undist_proj, radial_distortion);
    Mat23 jacobian_undistproj_local = UndistProjection2LocalPointJacobian(local_point);
    Mat3 jacobian_local_global = LocalPoint2GlobalPointJacobian(angle_axis);
    Mat23 jacobian_proj_global = jacobian_proj_distproj * jacobian_distproj_undistproj *
            jacobian_undistproj_local * jacobian_local_global;
    return jacobian_proj_global;
}

void ProjectAndGradient(Vec3 const & rotation, Vec3 const & translation, Vec3 const & point,
                        double const focal, double const u, double const v,
                        Vec3 const & radial_distortion, Vec2 const & projection,
                        Mat23 & rotation_jacobian,
                        Mat23 & translation_jacobian,
                        Mat23 & point_jacobian,
                        Mat26 & intrinsic_jacobian)
{
    rotation_jacobian = Projection2RotationJacobian(focal, radial_distortion, rotation, translation, point);
    translation_jacobian = Projection2TranslationJacobian(focal, radial_distortion, rotation, translation, point);
    point_jacobian = Projection2GlobalPointJacobian(focal, radial_distortion, rotation, translation, point);
    intrinsic_jacobian = Projection2IntrinsicJacobian(focal, radial_distortion, rotation, translation, point);
}

bool Project(double const focal, double const u, double const v,
             Vec3 const & angle_axis, Vec3 const & translation,
             Vec3 const & global_point,
             Vec3 const & radial_distortion,
             Vec2 & reprojection)
{
    Vec3 local_point = RotatePoint(angle_axis, global_point) + translation;
    if (local_point[2] < MIN_DEPTH)
    {
        return false;
    }
    local_point[2] = std::max(local_point[2], MIN_DEPTH);
    DT x = local_point[0] / local_point[2];
    DT y = local_point[1] / local_point[2];
    DT r2 = x * x + y * y;
    DT r4 = r2 * r2;
    DT r6 = r4 * r2;
    DT coeff = 1 + radial_distortion(0) * r2 + radial_distortion(1) * r4 + radial_distortion(2) * r6;
//    if (coeff > 1e5)
//    {
//        std::cout << "[Project] Too large coeff: " << coeff << "\n";
//        return false;
//    }
    x *= coeff;
    y *= coeff;
    x = x * focal + u;
    y = y * focal + v;
    reprojection = Vec2(x, y);
    return IsNumericalValid(reprojection);
}

double RandomNoise(double min, double max)
{
    assert(min < max && "[RandomNoise] Min < Max");
    double length = max - min;
    return min + length * std::rand() / double(RAND_MAX);
}

double GaussianNoise(double mean, double stddev)
{
    assert(stddev > 0 && "[GaussianNoise] stddev > 0");
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::normal_distribution<double> dist(mean, stddev);
    return dist(generator);
}

bool ReadLinesFromFile(std::string const & file_path, std::vector<std::string> & lines)
{
    std::ifstream stream(file_path.c_str());

    if (!stream.is_open())
        return false;

    std::string line;
    while (stream.good())
    {
        std::getline(stream, line);
        if (!line.empty())
            lines.push_back(line);
    }

    if (!stream.eof())
        return false;

    return true;
}

bool ReadCameraGroup(std::string const & camera_group_file,
                     std::unordered_map<size_t, size_t> & camera_group_map)
{
    camera_group_map.clear();

    if (camera_group_file.empty())
        return false;

    std::vector<std::string> lines;
    if(!ReadLinesFromFile(camera_group_file, lines))
        return false;

    size_t line_num = lines.size();
    for(size_t i = 0; i < line_num; i++)
    {
        size_t camera_index, group_index;
        std::stringstream camera_group_stream;
        camera_group_stream << lines[i];
        camera_group_stream >> camera_index;
        camera_group_stream >> group_index;
        camera_group_map[camera_index] = group_index;
    }
    return true;
}
