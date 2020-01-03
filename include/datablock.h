#ifndef DATABLOCK_H
#define DATABLOCK_H

#include "utility.h"
#include <unordered_set>
#include <unordered_map>
#include <vector>
#include <cassert>
#include <cstdlib>
#include <iostream>

class PoseBlock
{
public:
    PoseBlock() : poses_(NULL), delta_poses_(NULL), temp_delta_poses_(NULL), pose_num_(0) {}
    PoseBlock(size_t pose_num) : pose_num_(pose_num)
    {
        poses_ = new DT[pose_num * 6];
        delta_poses_ = new DT[pose_num * 6];
        temp_delta_poses_ = new DT[pose_num * 6];
        std::fill(delta_poses_, delta_poses_ + pose_num * 6, 0.0);
    }
    ~PoseBlock()
    {
        if (poses_ != NULL)             delete [] poses_;
        if (delta_poses_ != NULL)       delete [] delta_poses_;
        if (temp_delta_poses_ != NULL)  delete [] temp_delta_poses_;
    }

    void Create(size_t pose_num)
    {
        if (poses_ != NULL)         delete [] poses_;
        if (delta_poses_ != NULL)   delete [] delta_poses_;
        if (temp_delta_poses_ != NULL)  delete [] temp_delta_poses_;
        poses_ = new DT[pose_num * 6];
        delta_poses_ = new DT[pose_num * 6];
        temp_delta_poses_ = new DT[pose_num * 6];
        std::fill(delta_poses_, delta_poses_ + pose_num * 6, 0.0);
        pose_num_ = pose_num;
    }

    inline size_t PoseNum() const { return pose_num_; }
    inline void SetPose(size_t idx, Vec3 const & angle_axis, Vec3 const & translation)
    {
        assert(idx < pose_num_ && "Pose index out of range");
        DT* ptr = poses_ + idx * 6;
        Eigen::Map<Vec6> pose(ptr);
        pose.head<3>() = angle_axis;
        pose.tail<3>() = translation;
    }
    inline void GetPose(size_t idx, Vec3 & angle_axis, Vec3 & translation) const
    {
        assert(idx < pose_num_ && "Pose index out of range");
        DT* ptr = poses_ + idx * 6;
        angle_axis = Vec3(ptr);        // first rotation, second translation
        translation = Vec3(ptr + 3);
    }
    inline void GetPose(size_t idx, Vec6 & pose) const
    {
        assert(idx < pose_num_ && "Pose index out of range");
        DT* ptr = poses_ + idx * 6;
        pose = Vec6(ptr);
    }
    inline void SetDeltaPose(size_t idx, Vec6 const & dy)
    {
        assert(idx < pose_num_ && "Pose index out of range");
        DT* ptr = delta_poses_ + idx * 6;
        Eigen::Map<Vec6> pose(ptr);
        pose = dy;
    }
    inline void SetDeltaPose(size_t idx, Vec3 const & angle_axis, Vec3 const & translation)
    {
        assert(idx < pose_num_ && "Pose index out of range");
        DT* ptr = delta_poses_ + idx * 6;
        Eigen::Map<Vec6> pose(ptr);
        pose.head<3>() = angle_axis;
        pose.tail<3>() = translation;
    }
    inline void IncreDeltaPose(size_t idx, Vec3 const & angle_axis, Vec3 const & translation)
    {
        assert(idx < pose_num_ && "Pose index out of range");
        DT* ptr = delta_poses_ + idx * 6;
        Eigen::Map<Vec6> pose(ptr);
        pose.head<3>() += angle_axis;
        pose.tail<3>() += translation;
    }
    inline void GetDeltaPose(size_t idx, Vec3 & angle_axis, Vec3 & translation) const
    {
        assert(idx < pose_num_ && "Pose index out of range");
        DT* ptr = delta_poses_ + idx * 6;
        angle_axis = Vec3(ptr);        // first rotation, second translation
        translation = Vec3(ptr + 3);
    }
    inline void GetDeltaPose(size_t idx, Vec6 & dy) const
    {
        assert(idx < pose_num_ && "Pose index out of range");
        DT* ptr = delta_poses_ + idx * 6;
        dy = Vec6(ptr);
    }
    inline void SetTempDeltaPose(size_t idx, Vec3 const & angle_axis, Vec3 const & translation)
    {
        assert(idx < pose_num_ && "Pose index out of range");
        DT* ptr = temp_delta_poses_ + idx * 6;
        Eigen::Map<Vec6> pose(ptr);
        pose.head<3>() = angle_axis;
        pose.tail<3>() = translation;
    }
    inline void GetTempDeltaPose(size_t idx, Vec3 & angle_axis, Vec3 & translation) const
    {
        assert(idx < pose_num_ && "Pose index out of range");
        DT* ptr = temp_delta_poses_ + idx * 6;
        angle_axis = Vec3(ptr);        // first rotation, second translation
        translation = Vec3(ptr + 3);
    }
    inline void GetTempDeltaPose(size_t idx, Vec6 & dy) const
    {
        assert(idx < pose_num_ && "Pose index out of range");
        DT* ptr = temp_delta_poses_ + idx * 6;
        dy = Vec6(ptr);
    }
    inline void AverageDeltaPose(size_t const batch_size)
    {
        for (size_t i = 0; i < pose_num_ * 6; i++)
        {
            delta_poses_[i] /= DT(batch_size);
            temp_delta_poses_[i] = delta_poses_[i];
        }
    }

    void UpdatePose()
    {
        for (size_t i = 0; i < pose_num_ * 6; i++)
            poses_[i] += delta_poses_[i];
    }
    void ClearUpdate()
    {
        std::fill(delta_poses_, delta_poses_ + pose_num_ * 6, 0.0);
    }

private:
    DT* poses_;
    DT* delta_poses_;
    DT* temp_delta_poses_;
    size_t pose_num_;
};

class PointBlock
{
public:
    PointBlock() : points_(NULL), delta_points_(NULL), point_num_(0) {}
    PointBlock(size_t point_num) : point_num_(point_num)
    {
        points_ = new DT[point_num * 3];
        delta_points_ = new DT[point_num * 3];
        colors_ = new DT[point_num * 3];
        std::fill(delta_points_, delta_points_ + point_num * 3, 0.0);
    }
    ~PointBlock()
    {
        if (points_ != NULL) delete [] points_;
        if (delta_points_ != NULL) delete [] delta_points_;
        if (colors_ != NULL) delete [] colors_;
    }

    void Create(size_t point_num)
    {
        if (points_ != NULL) delete [] points_;
        if (delta_points_ != NULL) delete [] delta_points_;
        points_ = new DT[point_num * 3];
        delta_points_ = new DT[point_num * 3];
        std::fill(delta_points_, delta_points_ + point_num * 3, 0.0);
        colors_ = new DT[point_num * 3];
        point_num_ = point_num;
    }
    inline size_t PointNum() const { return point_num_; }
    inline void SetPoint(size_t idx, Vec3 const & pt)
    {
        assert(idx < point_num_ && "Point index out of range");
        DT* ptr = points_ + idx * 3;
        Eigen::Map<Vec3> point(ptr);
        point = pt;
    }
    inline void GetPoint(size_t idx, Vec3 & pt) const
    {
        DT* ptr = points_ + idx * 3;
        pt = Vec3(ptr);
    }
    inline void SetDeltaPoint(size_t idx, Vec3 const & pt)
    {
        assert(idx < point_num_ && "Point index out of range");
        DT* ptr = delta_points_ + idx * 3;
        Eigen::Map<Vec3> point(ptr);
        point = pt;
    }
    inline void IncreDeltaPoint(size_t idx, Vec3 const & pt)
    {
        assert(idx < point_num_ && "Point index out of range");
        DT* ptr = delta_points_ + idx * 3;
        Eigen::Map<Vec3> point(ptr);
        point += pt;
    }
    inline void GetDeltaPoint(size_t idx, Vec3 & pt) const
    {
        DT* ptr = delta_points_ + idx * 3;
        pt = Vec3(ptr);
    }
    inline void AverageDeltaPoint(size_t const batch_size)
    {
        for (size_t i = 0; i < point_num_ * 3; i++)
            delta_points_[i] /= DT(batch_size);
    }
    void UpdatePoint()
    {
        for (size_t i = 0; i < point_num_ * 3; i++)
            points_[i] += delta_points_[i];
    }
    void ClearUpdate()
    {
        std::fill(delta_points_, delta_points_ + point_num_ * 3, 0.0);
    }
    inline void SetColor(size_t idx, Vec3 const & color)
    {
        assert(idx < point_num_ && "Point index out of range");
        DT* ptr = colors_ + idx * 3;
        Eigen::Map<Vec3> clr(ptr);
        clr = color;
    }
    inline void GetColor(size_t idx, Vec3 & color) const
    {
        DT* ptr = colors_ + idx * 3;
        color = Vec3(ptr);
    }

private:
    DT* points_;
    DT* delta_points_;
    DT* colors_;
    size_t point_num_;
};

class IntrinsicBlock
{
public:
    IntrinsicBlock() : intrinsics_(NULL), delta_intrinsics_(NULL), group_num_(0) {}
    IntrinsicBlock(size_t group_num) : group_num_(group_num)
    {
        intrinsics_ = new DT[group_num * 6];    // focal, u, v, radial_distortion
        delta_intrinsics_ = new DT[group_num * 6];
        std::fill(delta_intrinsics_, delta_intrinsics_ + group_num * 6, 0.0);
    }
    ~IntrinsicBlock()
    {
        if (intrinsics_ != NULL)             delete intrinsics_;
        if (delta_intrinsics_ != NULL)       delete delta_intrinsics_;
    }
    void Create(size_t group_num)
    {
        if (intrinsics_ != NULL)             delete intrinsics_;
        if (delta_intrinsics_ != NULL)       delete delta_intrinsics_;
        intrinsics_ = new DT[group_num * 6];
        delta_intrinsics_ = new DT[group_num * 6];
        std::fill(delta_intrinsics_, delta_intrinsics_ + group_num * 6, 0.0);
        group_num_ = group_num;
    }
    inline size_t GroupNum() const { return group_num_; }
    inline void SetIntrinsic(size_t idx, Vec6 const & intr)
    {
        assert(idx < group_num_ && "Group index out of range");
        DT* ptr = intrinsics_ + idx * 6;
        Eigen::Map<Vec6> intrinsic(ptr);
        intrinsic = intr;
    }
    inline void GetIntrinsic(size_t idx, Vec6 & intr) const
    {
        assert(idx < group_num_ && "Group index out of range");
        DT* ptr = intrinsics_ + idx * 6;
        intr = Vec6(ptr);
    }
    inline void SetDeltaIntrinsic(size_t idx, Vec6 const & intr)
    {
        assert(idx < group_num_ && "Group index out of range");
        DT* ptr = delta_intrinsics_ + idx * 6;
        Eigen::Map<Vec6> intrinsic(ptr);
        intrinsic = intr;
    }
    inline void GetDeltaIntrinsic(size_t idx, Vec6 & intr) const
    {
        assert(idx < group_num_ && "Group index out of range");
        DT* ptr = delta_intrinsics_ + idx * 6;
        intr = Vec6(ptr);
    }
    void UpdateIntrinsics()
    {
        for (size_t i = 0; i < group_num_ * 6; i++)
            intrinsics_[i] += delta_intrinsics_[i];
    }
    void ClearUpdate()
    {
        std::fill(delta_intrinsics_, delta_intrinsics_ + group_num_ * 6, 0.0);
    }

private:
    DT * intrinsics_;
    DT * delta_intrinsics_;
    size_t group_num_;
};

class ProjectionBlock
{
public:
    ProjectionBlock() : projections_(NULL), indexes_(NULL), projection_num_(0) {}
    ProjectionBlock(size_t proj_num) : projection_num_(proj_num)
    {
        projections_ = new DT[proj_num * 2];
        indexes_ = new size_t[proj_num * 2];
    }
    ~ProjectionBlock()
    {
        if (projections_ != NULL) delete [] projections_;
        if (indexes_ != NULL) delete [] indexes_;
    }

    void Create(size_t proj_num)
    {
        if (projections_ != NULL) delete [] projections_;
        if (indexes_ != NULL) delete [] indexes_;
        projections_ = new DT[proj_num * 2];
        indexes_ = new size_t[proj_num * 2];
        projection_num_ = proj_num;
    }
    inline size_t ProjectionNum() const { return projection_num_; }
    inline void SetProjection(size_t idx, size_t camera_index, size_t point_index, Vec2 const & proj)
    {
        assert(idx < projection_num_ && "Projection index out of range");
        indexes_[2 * idx] = camera_index;
        indexes_[2 * idx + 1] = point_index;
        projections_[2 * idx] = proj(0);
        projections_[2 * idx + 1] = proj(1);
    }
    inline void GetProjection(size_t idx, Vec2 & proj) const
    {
        DT* ptr = projections_ + idx * 2;
        proj = Vec2(ptr);
    }
    inline size_t PoseIndex(size_t idx) const
    {
        assert(idx < projection_num_ && "Projection index out of range");
        return indexes_[2 * idx];
    }
    inline size_t PointIndex(size_t idx) const
    {
        assert(idx < projection_num_ && "Projection index out of range");
        return indexes_[2 * idx + 1];
    }

private:
    DT * projections_;
    size_t * indexes_;
    size_t projection_num_;
};

class PointMeta
{
public:
    PointMeta() : cluster_num_(0), data_(NULL) {}
    PointMeta(size_t cluster_num) : cluster_num_(cluster_num), data_(NULL)
    {
        data_ = new DT[cluster_num * 15];
        std::fill(data_, data_ + cluster_num * 15, 0.0);
    }
    ~PointMeta()
    {
        if (data_ != NULL)  delete data_;
    }
    void GetJpJp(size_t cluster_index, Mat3 & JpJp) const
    {
        assert(cluster_index < cluster_num_ && "[MPointMeta::GetJpJp] Cluster index out of range");
        DT * ptr = data_ + cluster_index * 15;
        JpJp = Mat3(ptr);
    }
    void SetJpJp(size_t cluster_index, Mat3 const & JpJp)
    {
        assert(cluster_index < cluster_num_ && "[MPointMeta::SetJpJp] Cluster index out of range");
        for (size_t i = 0; i < 3; i++)
            for (size_t j = 0; j < 3; j++)
                data_[cluster_index * 15 + i * 3 + j] = JpJp(i, j);
    }
    void GetDiagonal(size_t cluster_index, Vec3 & diagonal) const
    {
        assert(cluster_index < cluster_num_ && "[MPointMeta::GetDiagonal] Cluster index out of range");
        for (size_t i = 0; i < 3; i++)
            diagonal(i) = data_[cluster_index * 15 + i * 4];
    }
    void SetDiagonal(size_t cluster_index, Vec3 const & diagonal)
    {
        assert(cluster_index < cluster_num_ && "[MPointMeta::SetJpJp] Cluster index out of range");
        for (size_t i = 0; i < 3; i++)
            data_[cluster_index * 15 + i * 4] = diagonal(i);
    }
    void AddDiagonal(size_t cluster_index, Vec3 const & diagonal)
    {
        assert(cluster_index < cluster_num_ && "[MPointMeta::SetJpJp] Cluster index out of range");
        for (size_t i = 0; i < 3; i++)
            data_[cluster_index * 15 + i * 4] += diagonal(i);
    }
    void GetJpe(size_t cluster_index, Vec3 & Jpe) const
    {
        assert(cluster_index < cluster_num_ && "[MPointMeta::GetJpe] Cluster index out of range");
        DT * ptr = data_ + cluster_index * 15 + 9;
        Jpe = Vec3(ptr);
    }
    void SetJpe(size_t cluster_index, Vec3 const & Jpe)
    {
        assert(cluster_index < cluster_num_ && "[MPointMeta::GetJpe] Cluster index out of range");
        for (size_t i = 0; i < 3; i++)
            data_[cluster_index * 15 + 9 + i] = Jpe(i);
    }
    void GetDeltaPoint(size_t cluster_index, Vec3 & dz) const
    {
        assert(cluster_index < cluster_num_ && "[MPointMeta::GetDeltaPoint] Cluster index out of range");
        DT * ptr = data_ + cluster_index * 15 + 12;
        dz = Vec3(ptr);
    }
    void SetDeltaPoint(size_t cluster_index, Vec3 const & dz)
    {
        assert(cluster_index < cluster_num_ && "[MPointMeta::GetDeltaPoint] Cluster index out of range");
        for (size_t i = 0; i < 3; i++)
            data_[cluster_index * 15 + 12 + i] = dz(i);
    }

private:
    size_t cluster_num_;
    DT * data_; // JpJp, Jpe, dz
};

class BundleBlock
{
public:
    struct DGroup
    {
        DGroup() {}
        DGroup(size_t i, Vec6 const & intri) : id (i), intrinsic(intri) {}
        DGroup(size_t i, Vec6 const & intri, int w, int h) : id (i), intrinsic(intri), width(w), height(h) {}

        void Print() const
        {
            std::cout << "[Group] " << id << "\n" << "intrinsic: " << intrinsic << "\n";
        }

        size_t id;
        Vec6 intrinsic; // focal, u0, v0, radial_distortion[3]
        int width;
        int height;
    };

    struct DCamera
    {
        DCamera() {}
        DCamera(size_t i, size_t g_id, Vec3 const & aa, Vec3 const & trans) :
            id(i), group_id(g_id), axis_angle(aa), translation(trans) {}
        DCamera(size_t i, size_t g_id, Vec3 const & aa, Vec3 const & trans, std::string const & path) :
            id(i), group_id(g_id), axis_angle(aa), translation(trans), image_path(path) {}


        void Print() const
        {
            std::cout << "[Camera] " << id << "\n"
                      << "axis angle: " << axis_angle << "\n" << "translation: " << translation << "\n";
        }

        size_t id;
        size_t group_id;
        std::unordered_set<size_t> linked_projections;
        std::unordered_set<size_t> linked_cameras;
        Vec3 axis_angle;
        Vec3 translation;

        std::string image_path;
    };
    struct DTrack
    {
        DTrack() {}
        DTrack(size_t i) : id(i) {}
        DTrack(size_t i, Vec3 const & pos, Vec3 const & c) :
            id(i), position(pos), color(c) {}

        void Print() const
        {
            std::cout << "[Track] " << id << "\n" << "position: " << position << "\n";
        }

        size_t id;
        std::unordered_set<size_t> linked_projections;
        Vec3 position;
        Vec3 color;
    };
    struct DProjection
    {
        DProjection() {}
        DProjection(size_t i, size_t ci, size_t ti, Vec2 const & proj) :
            id(i), camera_id(ci), track_id(ti), projection(proj) {}

        void Print() const
        {
            std::cout << "[Projection] " << id << "\n" << "camera id: " << camera_id << "\n"
                      << "track id:" << track_id << "\n" << "projection: " << projection << "\n";
        }

        size_t id;
        size_t camera_id;
        size_t track_id;
        Vec2 projection;
    };

public:
    std::vector<size_t> GroupIndexes() const;
    std::vector<size_t> CameraIndexes() const;
    std::vector<size_t> TrackIndexes() const;
    std::vector<size_t> ProjectionIndexes() const;

    DGroup const & GetGroup(size_t id) const;
    DGroup & GetGroup(size_t id);
    DCamera const & GetCamera(size_t id) const;
    DCamera & GetCamera(size_t id);
    DTrack const & GetTrack(size_t id) const;
    DTrack & GetTrack(size_t id);
    DProjection const & GetProjection(size_t id) const;
    DProjection & GetProjection(size_t id);

    void GetCommonPoints(std::unordered_map<size_t, std::unordered_map<size_t, std::vector<size_t> > > & common_point_map) const;

    bool LoadColmapTxt(std::string const & cameras_path, std::string const & images_path, std::string const & points_path);

    void SaveColmapTxt(std::string const & cameras_path, std::string const & images_path, std::string const & points_path) const;

    void Print() const;

    void AddGaussianNoiseToTrack(DT mean, DT sigma);

    void AddGaussianNoiseToCameraTranslation(DT mean, DT sigma);

private:
    std::unordered_map<size_t, DGroup> groups_;
    std::unordered_map<size_t, DCamera> cameras_;
    std::unordered_map<size_t, DTrack> tracks_;
    std::unordered_map<size_t, DProjection> projections_;
};

#endif // DATABLOCK_H

