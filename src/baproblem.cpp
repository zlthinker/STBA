#include "STBA/baproblem.h"

#include <fstream>
#include <Eigen/IterativeLinearSolvers>

BAProblem::BAProblem() :
    pose_block_(), point_block_(), intrinsic_block_(), projection_block_(),
    loss_function_(NULL),
    residual_(NULL),
    pose_jacobian_(NULL),
    point_jacobian_(NULL),
    intrinsic_jacobian_(NULL),
    fix_intrinsic_(false),
    max_degree_(1000),
    thread_num_(1),
    linear_solver_type_(ADAPTIVE),
    tp_(NULL),
    Tcp_(NULL),
    Tip_(NULL)
{
    loss_function_ = new HuberLoss();
}

BAProblem::BAProblem(LossType loss_type) :
    pose_block_(), point_block_(), projection_block_(),
    loss_function_(NULL),
    residual_(NULL),
    pose_jacobian_(NULL),
    point_jacobian_(NULL),
    intrinsic_jacobian_(NULL),
    fix_intrinsic_(false),
    max_degree_(1000),
    thread_num_(1),
    linear_solver_type_(ADAPTIVE),
    tp_(NULL),
    Tcp_(NULL),
    Tip_(NULL)
{
    switch(loss_type)
    {
    case NULLLossType:
        loss_function_ = new NULLLoss();
        break;
    case CauchyLossType:
        loss_function_ = new CauchyLoss();
        break;
    default:
        loss_function_ = new HuberLoss();
    }
}

BAProblem::BAProblem(size_t pose_num, size_t group_num, size_t point_num, size_t proj_num) :
    pose_block_(pose_num), point_block_(point_num), projection_block_(proj_num),
    loss_function_(NULL), fix_intrinsic_(false), max_degree_(1000), linear_solver_type_(ADAPTIVE)
{
    Create(pose_num, group_num, point_num, proj_num);
    loss_function_ = new HuberLoss();
}

BAProblem::~BAProblem()
{
    Delete();
    if (loss_function_ != NULL)                 delete loss_function_;
}

bool BAProblem::Create(size_t pose_num, size_t group_num, size_t point_num, size_t proj_num)
{
    DT memory = pose_num * 12 + group_num * 12 + point_num * 6 + proj_num * 2 +
            proj_num * 2 + proj_num * 12 + proj_num * 6 + point_num * 3 + proj_num * 18 +
            proj_num * 12 + proj_num * 18 + std::pow((pose_num + group_num) * 6, 2);

    memory = memory * 4 / (1024 * 1024 * 1024);  // kB
    std::cout << "[BAProblem::Create] A memory of " << memory << " GB is going to be allocated.\n";

    if (!pose_block_.Create(pose_num))          return false;
    if (!intrinsic_block_.Create(group_num))    return false;
    if (!point_block_.Create(point_num))         return false;
    if (!projection_block_.Create(proj_num))   return false;

    Delete();
    try
    {
        residual_ = new DT[2 * proj_num];
        pose_jacobian_ = new DT[2 * proj_num * 6];   // The i-th item stores the jacobian of i-th projection w.r.t corresponding camera
        point_jacobian_ = new DT[2 * proj_num * 3];  // The i-th item stores the jacobian of i-th projection w.r.t corresponding point

        tp_ = new DT[point_num * 3];
        Tcp_ = new DT[proj_num * 6 * 3];
        if (!fix_intrinsic_)
        {
            intrinsic_jacobian_ = new DT[2 * proj_num * 6];
            Tip_ = new DT[proj_num * 6 * 3];
        }
    }
    catch (std::bad_alloc & e)
    {
        std::cout << "[BAProblem::Create] Catching bad_alloc: " << e.what() << std::endl;
        return false;
    }
    return true;
}

bool BAProblem::Initialize(BundleBlock const & bundle_block)
{
    std::vector<size_t> group_indexes = bundle_block.GroupIndexes();
    std::vector<size_t> camera_indexes = bundle_block.CameraIndexes();
    std::vector<size_t> point_indexes = bundle_block.TrackIndexes();
    std::vector<size_t> projection_indexes = bundle_block.ProjectionIndexes();
    std::sort(group_indexes.begin(), group_indexes.end());
    std::sort(camera_indexes.begin(), camera_indexes.end());
    std::sort(point_indexes.begin(), point_indexes.end());
    std::sort(projection_indexes.begin(), projection_indexes.end());
    size_t group_num = group_indexes.size();
    size_t pose_num = camera_indexes.size();
    size_t point_num = point_indexes.size();
    size_t projection_num = projection_indexes.size();

    if (!Create(pose_num, group_num, point_num, projection_num))    return false;

    std::unordered_map<size_t, size_t> group_map;
    for (size_t i = 0; i < group_indexes.size(); i++)
    {
        size_t index = group_indexes[i];
        group_map[index] = i;
        group_index_map_[i] = index;
    }

    max_degree_ = 0;
    std::unordered_map<size_t, size_t> pose_map;
    for (size_t i = 0; i < pose_num; i++)
    {
        size_t index = camera_indexes[i];
        pose_map[index] = i;
        pose_index_map_[i] = index;

        BundleBlock::DCamera const & camera = bundle_block.GetCamera(index);
        BundleBlock::DGroup const & group = bundle_block.GetGroup(camera.group_id);
        SetPose(i, camera.axis_angle, camera.translation);
        assert(group_map.find(group.id) != group_map.end() && "Camera has no group");
        size_t group_index = group_map[group.id];
        SetIntrinsic(group_index, i, group.intrinsic);
        pose_projection_map_[i] = std::unordered_map<size_t, size_t>();

        max_degree_ = std::max(max_degree_, camera.linked_cameras.size());
    }

    std::unordered_map<size_t, size_t> point_map;
    for (size_t i = 0; i < point_num; i++)
    {
        size_t index = point_indexes[i];
        point_map[index] = i;
        point_index_map_[i] = index;
        BundleBlock::DTrack const & track = bundle_block.GetTrack(index);
        SetPoint(i, track.position);
        SetColor(i, track.color);
        point_projection_map_[i] = std::unordered_map<size_t, size_t>();
    }

    for (size_t i = 0; i < projection_indexes.size(); i++)
    {
        size_t index = projection_indexes[i];
        BundleBlock::DProjection const & projection = bundle_block.GetProjection(index);
        size_t camera_index = projection.camera_id;
        size_t track_index = projection.track_id;
        assert(pose_map.find(camera_index) != pose_map.end() && "Pose index not found");
        assert(point_map.find(track_index) != point_map.end() && "Point index not found");
        size_t pose_index = pose_map[camera_index];
        size_t point_index = point_map[track_index];
        SetProjection(i, pose_index, point_index, projection.projection);
    }

    std::unordered_map<size_t, std::unordered_map<size_t, std::vector<size_t> > > common_track_map;
    bundle_block.GetCommonPoints(common_track_map);
    std::unordered_map<size_t, std::unordered_map<size_t, std::vector<size_t> > >::const_iterator it1;
    for (it1 = common_track_map.begin(); it1 != common_track_map.end(); it1++)
    {
        size_t camera_index1 = it1->first;
        assert(pose_map.find(camera_index1) != pose_map.end());
        size_t pose_index1 = pose_map[camera_index1];
        std::unordered_map<size_t, std::vector<size_t> > const & local_map = it1->second;
        std::unordered_map<size_t, std::vector<size_t> >::const_iterator it2;
        for (it2 = local_map.begin(); it2 != local_map.end(); it2++)
        {
            size_t camera_index2 = it2->first;
            assert(pose_map.find(camera_index2) != pose_map.end());
            size_t pose_index2 = pose_map[camera_index2];
            if (pose_index1 > pose_index2)  continue;
            std::vector<size_t> const & track_indexes = it2->second;

            std::unordered_set<size_t> point_index_set;
            for (size_t i = 0; i < track_indexes.size(); i++)
            {
                size_t track_index = track_indexes[i];
                assert(point_map.find(track_index) != point_map.end());
                size_t point_index = point_map[track_index];
                point_index_set.insert(point_index);
            }
            std::vector<size_t> point_indexes(point_index_set.begin(), point_index_set.end());
            SetCommonPoints(pose_index1, pose_index2, point_indexes);
        }
    }

    std::stringstream local_stream;
    local_stream << "[Initialize] \n"
                 << "# poses: " << PoseNum() << "\n"
                 << "# groups: " << GroupNum() << "\n"
                 << "# points: " << PointNum() << "\n"
                 << "# projections: " << ProjectionNum() << "\n"
                 << "max degree: " << max_degree_ << "\n";
    std::cout << local_stream.str();
    stream_ << local_stream.str();

    return true;
}

void BAProblem::Update(BundleBlock & bundle_block) const
{
    size_t group_num = GroupNum();
    size_t pose_num = PoseNum();
    size_t point_num = PointNum();

    for (size_t i = 0; i < group_num; i++)
    {
        size_t group_index = group_index_map_.find(i)->second;
        BundleBlock::DGroup & group = bundle_block.GetGroup(group_index);
        GetIntrinsic(i, group.intrinsic);
    }

    for (size_t i = 0; i < pose_num; i++)
    {
        size_t pose_index = pose_index_map_.find(i)->second;
        BundleBlock::DCamera & camera = bundle_block.GetCamera(pose_index);
        GetPose(i, camera.axis_angle, camera.translation);
    }

    for (size_t i = 0; i < point_num; i++)
    {
        size_t point_index = point_index_map_.find(i)->second;
        BundleBlock::DTrack & track = bundle_block.GetTrack(point_index);
        GetPoint(i, track.position);
    }
}

void BAProblem::SaveReport(std::string const & report_path) const
{
    std::ofstream fout(report_path);
    fout << stream_.str();
    fout.close();
}

void BAProblem::SetIntrinsic(size_t idx, size_t camera_index, Vec6 const & intrinsic)
{
    assert(camera_index < PoseNum() && "Pose index of projection out of range");
    intrinsic_block_.SetIntrinsic(idx, intrinsic);
    pose_group_map_[camera_index] = idx;
    std::unordered_map<size_t, std::vector<size_t> >::iterator it = group_pose_map_.find(idx);
    if (it == group_pose_map_.end())
    {
        std::vector<size_t> pose_indexes = {camera_index};
        group_pose_map_[idx] = pose_indexes;
    }
    else
    {
        std::vector<size_t> & pose_indexes = it->second;
        pose_indexes.push_back(camera_index);
    }
}

void BAProblem::SetProjection(size_t idx,size_t camera_index, size_t point_index, Vec2 const & proj)
{
    assert(camera_index < pose_block_.PoseNum() && "Pose index of projection out of range");
    assert(point_index < point_block_.PointNum() && "Point index of projection out of range");
    projection_block_.SetProjection(idx, camera_index, point_index, proj);
    std::unordered_map<size_t, std::unordered_map<size_t, size_t> >::iterator it = pose_projection_map_.find(camera_index);
    if (it != pose_projection_map_.end())
    {
        std::unordered_map<size_t, size_t> & map = it->second;
        map[point_index] = idx;
    }
    else
    {
        std::unordered_map<size_t, size_t> map;
        map[point_index] = idx;
        pose_projection_map_[camera_index] = map;
    }

    it = point_projection_map_.find(point_index);
    if (it != point_projection_map_.end())
    {
        std::unordered_map<size_t, size_t> & map = it->second;
        map[camera_index] = idx;
    }
    else
    {
        std::unordered_map<size_t, size_t> map;
        map[camera_index] = idx;
        point_projection_map_[point_index] = map;
    }
}

void BAProblem::SetCommonPoints(size_t pose_index1, size_t pose_index2, std::vector<size_t> const & points)
{
    assert(pose_index1 < pose_block_.PoseNum() && "[SetCommonPoints] Pose index out of range");
    assert(pose_index2 < pose_block_.PoseNum() && "[SetCommonPoints] Pose index out of range");
    size_t index1 = std::min(pose_index1, pose_index2);
    size_t index2 = std::max(pose_index1, pose_index2);
    std::unordered_map<size_t, std::unordered_map<size_t, std::vector<size_t> > >::iterator it1 = common_point_map_.find(index1);
    if (it1 != common_point_map_.end())
    {
        std::unordered_map<size_t, std::vector<size_t> > & map = it1->second;
        map[index2] = points;
    }
    else
    {
        std::unordered_map<size_t, std::vector<size_t> > map;
        map[index2] = points;
        common_point_map_[index1] = map;
    }
}

size_t BAProblem::GetProjectionIndex(size_t pose_index, size_t point_index) const
{
    std::unordered_map<size_t, std::unordered_map<size_t, size_t> >::const_iterator it1 = pose_projection_map_.find(pose_index);
    assert(it1 != pose_projection_map_.end() && "[GetProjectionIndex] Pose index not found");
    std::unordered_map<size_t, size_t> const & map = it1->second;
    std::unordered_map<size_t, size_t>::const_iterator it2 = map.find(point_index);
    assert(it2 != map.end() && "[GetProjectionIndex] Point index not found");
    return it2->second;
}

void BAProblem::GetCommonPoints(size_t pose_index1, size_t pose_index2, std::vector<size_t> & points) const
{
    points.clear();
    size_t index1 = std::min(pose_index1, pose_index2);
    size_t index2 = std::max(pose_index1, pose_index2);
    std::unordered_map<size_t, std::unordered_map<size_t, std::vector<size_t> > >::const_iterator it1 = common_point_map_.find(index1);
    if (it1 == common_point_map_.end()) return;
    std::unordered_map<size_t, std::vector<size_t> > const & map = it1->second;
    std::unordered_map<size_t, std::vector<size_t> >::const_iterator it2 = map.find(index2);
    if (it2 == map.end()) return;
    points = it2->second;
}

void BAProblem::GetResidual(size_t proj_index, Vec2 & residual) const
{
    assert(proj_index < projection_block_.ProjectionNum() && "[GetResidual] Projection index out of range");
    DT * ptr = residual_ + proj_index * 2;
    residual = Vec2(ptr);
}

void BAProblem::SetResidual(size_t proj_index, Vec2 const & residual)
{
    assert(proj_index < projection_block_.ProjectionNum() && "[SetResidual] Projection index out of range");
    residual_[proj_index * 2] = residual(0);
    residual_[proj_index * 2 + 1] = residual(1);
}

void BAProblem::GetPoseJacobian(size_t proj_index, Mat26 & jacobian) const
{
    assert(proj_index < projection_block_.ProjectionNum() && "[GetPoseJacobian] Projection index out of range");
    DT * ptr = pose_jacobian_ + proj_index * 12;
    jacobian = Mat26(ptr);
}

void BAProblem::SetPoseJacobian(size_t proj_index, Mat23 const & jacobian_rotation, Mat23 const & jacobian_translation)
{
    assert(proj_index < projection_block_.ProjectionNum() && "[SetPoseJacobian] Projection index out of range");
    pose_jacobian_[proj_index * 12] = jacobian_rotation(0, 0);           // store in row major
    pose_jacobian_[proj_index * 12 + 1] = jacobian_rotation(0, 1);
    pose_jacobian_[proj_index * 12 + 2] = jacobian_rotation(0, 2);
    pose_jacobian_[proj_index * 12 + 3] = jacobian_translation(0, 0);
    pose_jacobian_[proj_index * 12 + 4] = jacobian_translation(0, 1);
    pose_jacobian_[proj_index * 12 + 5] = jacobian_translation(0, 2);
    pose_jacobian_[proj_index * 12 + 6] = jacobian_rotation(1, 0);
    pose_jacobian_[proj_index * 12 + 7] = jacobian_rotation(1, 1);
    pose_jacobian_[proj_index * 12 + 8] = jacobian_rotation(1, 2);
    pose_jacobian_[proj_index * 12 + 9] = jacobian_translation(1, 0);
    pose_jacobian_[proj_index * 12 + 10] = jacobian_translation(1, 1);
    pose_jacobian_[proj_index * 12 + 11] = jacobian_translation(1, 2);
}

void BAProblem::GetIntrinsicJacobian(size_t proj_index, Mat26 & jacobian) const
{
    assert(proj_index < projection_block_.ProjectionNum() && "[GetIntrinsicJacobian] Projection index out of range");
    DT * ptr = intrinsic_jacobian_ + proj_index * 12;
    jacobian = Mat26(ptr);
}

void BAProblem::SetIntrinsicJacobian(size_t proj_index, Mat26 const & jacobian)
{
    assert(proj_index < projection_block_.ProjectionNum() && "[GetIntrinsicJacobian] Projection index out of range");
    DT * ptr = intrinsic_jacobian_ + proj_index * 12;
    for (size_t i = 0; i < 6; i++)
    {
        ptr[i] = jacobian(0, i);
        ptr[i + 6] = jacobian(1, i);
    }
}

void BAProblem::GetPointJacobian(size_t proj_index, Mat23 & jacobian) const
{
    assert(proj_index < projection_block_.ProjectionNum() && "[GetPointJacobian] Projection index out of range");
    DT * ptr = point_jacobian_ + proj_index * 6;
    jacobian = Mat23(ptr);
}

void BAProblem::SetPointJacobian(size_t proj_index, Mat23 const & jacobian_point)
{
    assert(proj_index < projection_block_.ProjectionNum() && "[SetPointJacobian] Projection index out of range");
    point_jacobian_[proj_index * 6] = jacobian_point(0, 0);
    point_jacobian_[proj_index * 6 + 1] = jacobian_point(0, 1);
    point_jacobian_[proj_index * 6 + 2] = jacobian_point(0, 2);
    point_jacobian_[proj_index * 6 + 3] = jacobian_point(1, 0);
    point_jacobian_[proj_index * 6 + 4] = jacobian_point(1, 1);
    point_jacobian_[proj_index * 6 + 5] = jacobian_point(1, 2);
}

void BAProblem::GetPose(VecX & poses) const
{
    size_t pose_num = PoseNum();
    poses.resize(pose_num * 6);
    for (size_t i = 0; i < pose_num; i++)
    {
        Vec6 local_pose;
        pose_block_.GetPose(i, local_pose);
        poses.segment(6 * i, 6) = local_pose;
    }
}

void BAProblem::GetIntrinsic(VecX & intrinsics) const
{
    size_t group_num = GroupNum();
    intrinsics.resize(group_num * 6);
    for (size_t i = 0; i < group_num; i++)
    {
        Vec6 local_intrinsic;
        intrinsic_block_.GetIntrinsic(i, local_intrinsic);
        intrinsics.segment(6 * i, 6) = local_intrinsic;
    }
}

void BAProblem::GetPoint(VecX & points) const
{
    size_t point_num = PointNum();
    points.resize(point_num * 3);
    for (size_t i = 0; i < point_num; i++)
    {
        Vec3 local_point;
        point_block_.GetPoint(i, local_point);
        points.segment(3 * i, 3) = local_point;
    }
}

void BAProblem::GetPoseUpdate(VecX & update) const
{
    size_t pose_num = PoseNum();
    update.resize(pose_num * 6);
    for (size_t i = 0; i < pose_num; i++)
    {
        Vec6 local_update;
        pose_block_.GetDeltaPose(i, local_update);
        update.segment(6 * i, 6) = local_update;
    }
}

void BAProblem::GetIntrinsicUpdate(VecX & update) const
{
    if (fix_intrinsic_) return;
    size_t group_num = GroupNum();
    update.resize(group_num * 6);
    for (size_t i = 0; i < group_num; i++)
    {
        Vec6 local_update;
        intrinsic_block_.GetDeltaIntrinsic(i, local_update);
        update.segment(6 * i, 6) = local_update;
    }
}

void BAProblem::GetPointUpdate(VecX & update) const
{
    size_t point_num = PointNum();
    update.resize(point_num * 3);
    for (size_t i = 0; i < point_num; i++)
    {
        Vec3 local_update;
        point_block_.GetDeltaPoint(i, local_update);
        update.segment(3 * i, 3) = local_update;
    }
}

size_t BAProblem::GetPoseGroup(size_t pose_index) const
{
    std::unordered_map<size_t, size_t>::const_iterator it = pose_group_map_.find(pose_index);
    assert(it != pose_group_map_.end() && "[GetPoseGroup] Pose index not found");
    return it->second;
}

void BAProblem::GetTp(size_t point_index, Vec3 & tp) const
{
    assert(point_index < PointNum() && "[GetTp] Point index out of range");
    for (size_t i = 0; i < 3; i++)
        tp(i) = tp_[point_index * 3 + i];
}
void BAProblem::SetTp(size_t point_index, Vec3 const & tp)
{
    assert(point_index < PointNum() && "[SetTp] Point index out of range");
    for (size_t i = 0; i < 3; i++)
        tp_[point_index * 3 + i] = tp(i);
}

void BAProblem::GetTcp(size_t proj_index, Mat63 & Tcp) const
{
    assert(proj_index < ProjectionNum() && "[GetTcp] Projection index out of range");
    DT * ptr = Tcp_ + proj_index * 6 * 3;
    Tcp = Mat63(ptr);
}
void BAProblem::SetTcp(size_t proj_index, Mat63 const & Tcp)
{
    assert(proj_index < ProjectionNum() && "[SetTcp] Projection index out of range");
    DT * ptr = Tcp_ + proj_index * 6 * 3;

    for (size_t i = 0; i < 6; i++)
        for (size_t j = 0; j < 3; j++)
            ptr[i * 3 + j] = Tcp(i, j);
}

void BAProblem::GetTip(size_t proj_index, Mat63 & Tip) const
{
    assert(proj_index < ProjectionNum() && "[GetTip] Projection index out of range");
    DT * ptr = Tip_ + proj_index * 6 * 3;
    Tip = Mat63(ptr);
}
void BAProblem::SetTip(size_t proj_index, Mat63 const & Tip)
{
    assert(proj_index < ProjectionNum() && "[SetTip] Projection index out of range");
    DT * ptr = Tip_ + proj_index * 6 * 3;

    for (size_t i = 0; i < 6; i++)
        for (size_t j = 0; j < 3; j++)
            ptr[i * 3 + j] = Tip(i, j);
}

void BAProblem::EvaluateResidual()
{
    ClearResidual();
    size_t proj_num = projection_block_.ProjectionNum();

#pragma omp parallel for
    for (size_t i = 0; i < proj_num; i++)
    {
        size_t pose_index = projection_block_.PoseIndex(i);
        size_t point_index = projection_block_.PointIndex(i);
        Vec3 angle_axis, translation, point;
        Vec6 intrinsic;
        pose_block_.GetPose(pose_index, angle_axis, translation);
        point_block_.GetPoint(point_index, point);
        GetPoseIntrinsic(pose_index, intrinsic);
        Vec2 reprojection_error, reprojection;
        if (!Project(intrinsic(0), intrinsic(1), intrinsic(2), angle_axis, translation, point, intrinsic.tail<3>(), reprojection))
            continue;
        Vec2 projection;
        projection_block_.GetProjection(i, projection);
        reprojection_error = reprojection - projection;
        loss_function_->CorrectResiduals(reprojection_error);
        SetResidual(i, reprojection_error);
    }
}

double BAProblem::EvaluateSquareResidual(bool const update) const
{
    double error = 0.0;
    size_t proj_num = projection_block_.ProjectionNum();
    for (size_t i = 0; i < proj_num; i++)
    {
        size_t pose_index = projection_block_.PoseIndex(i);
        size_t group_index = GetPoseGroup(pose_index);
        size_t point_index = projection_block_.PointIndex(i);
        Vec3 angle_axis, translation, point;
        Vec6 intrinsic;
        pose_block_.GetPose(pose_index, angle_axis, translation);
        point_block_.GetPoint(point_index, point);
        GetPoseIntrinsic(pose_index, intrinsic);
        if (update)
        {
            Vec3 delta_angle_axis, delta_translation, delta_point;
            pose_block_.GetDeltaPose(pose_index, delta_angle_axis, delta_translation);
            point_block_.GetDeltaPoint(point_index, delta_point);
            angle_axis += delta_angle_axis;
            translation += delta_translation;
            point += delta_point;
            if (!fix_intrinsic_)
            {
                Vec6 delta_intrinsic;
                intrinsic_block_.GetDeltaIntrinsic(group_index, delta_intrinsic);
                intrinsic += delta_intrinsic;
            }
        }
        Vec2 reprojection;
        if (!Project(intrinsic(0), intrinsic(1), intrinsic(2), angle_axis, translation, point, intrinsic.tail<3>(), reprojection))
            continue;
        Vec2 projection;
        projection_block_.GetProjection(i, projection);
        Vec2 reprojection_error = reprojection - projection;
        loss_function_->CorrectResiduals(reprojection_error);
        double residual_square = reprojection_error.squaredNorm();
        error += residual_square;
    }
    return error;
}

void BAProblem::ReprojectionError(double & mean, double & median, double & max, bool const update) const
{
    size_t proj_num = projection_block_.ProjectionNum();
    assert(proj_num > 0 && "[ReprojectionError] Empty projection");
    std::vector<double> errors(proj_num, 0.0);

#pragma omp parallel for
    for (size_t i = 0; i < proj_num; i++)
    {
        Vec2 projection;
        projection_block_.GetProjection(i, projection);
        size_t pose_index = projection_block_.PoseIndex(i);
        size_t group_index = GetPoseGroup(pose_index);
        size_t point_index = projection_block_.PointIndex(i);
        Vec3 angle_axis, translation, point;
        Vec6 intrinsic;
        pose_block_.GetPose(pose_index, angle_axis, translation);
        GetPoseIntrinsic(pose_index, intrinsic);
        point_block_.GetPoint(point_index, point);
        if (update)
        {
            Vec3 delta_angle_axis, delta_translation, delta_point;
            pose_block_.GetDeltaPose(pose_index, delta_angle_axis, delta_translation);
            point_block_.GetDeltaPoint(point_index, delta_point);
            angle_axis += delta_angle_axis;
            translation += delta_translation;
            point += delta_point;
            if (!fix_intrinsic_)
            {
                Vec6 delta_intrinsic;
                intrinsic_block_.GetDeltaIntrinsic(group_index, delta_intrinsic);
                intrinsic += delta_intrinsic;
            }
        }
        Vec2 reprojection;
        if (!Project(intrinsic(0), intrinsic(1), intrinsic(2), angle_axis, translation, point, intrinsic.tail<3>(), reprojection))
            continue;
        double reprojection_error = (reprojection - projection).norm();
        if (reprojection_error > MAX_REPROJ_ERROR) continue;
        errors[i] = reprojection_error;
    }
    double sum_error = 0.0, max_error = 0.0;
    for (size_t i = 0; i < proj_num; i++)
    {
        sum_error += errors[i];
        max_error = std::max(max_error, errors[i]);
    }
    std::nth_element(errors.begin(), errors.begin() + errors.size() / 2, errors.end());
    mean = sum_error / double(errors.size());
    median = errors[errors.size() / 2];
    max = max_error;
}

/*!
 * @Depend EvaluateResidual
 */
double BAProblem::EvaluateSquareError(bool const update) const
{
    size_t proj_num = projection_block_.ProjectionNum();
    assert(proj_num > 0 && "[EvaluateSquareError] Empty projection");
    double error = 0;

#pragma omp parallel for reduction(+:error)
    for (size_t i = 0; i < proj_num; i++)
    {
        Vec2 projection;
        projection_block_.GetProjection(i, projection);
        size_t pose_index = projection_block_.PoseIndex(i);
        size_t group_index = GetPoseGroup(pose_index);
        size_t point_index = projection_block_.PointIndex(i);
        Vec3 angle_axis, translation, point;
        Vec6 intrinsic;
        pose_block_.GetPose(pose_index, angle_axis, translation);
        GetPoseIntrinsic(pose_index, intrinsic);
        point_block_.GetPoint(point_index, point);
        if (update)
        {
            Vec3 delta_angle_axis, delta_translation, delta_point;
            pose_block_.GetDeltaPose(pose_index, delta_angle_axis, delta_translation);
            point_block_.GetDeltaPoint(point_index, delta_point);
            angle_axis += delta_angle_axis;
            translation += delta_translation;
            point += delta_point;
            if (!fix_intrinsic_)
            {
                Vec6 delta_intrinsic;
                intrinsic_block_.GetDeltaIntrinsic(group_index, delta_intrinsic);
                intrinsic += delta_intrinsic;
            }
        }
        Vec2 reprojection;
        if (!Project(intrinsic(0), intrinsic(1), intrinsic(2), angle_axis, translation, point, intrinsic.tail<3>(), reprojection))
            continue;
        Vec2 reprojection_error = reprojection - projection;
        double robust_error = loss_function_->Loss(reprojection_error.squaredNorm());
        error += robust_error;
    }
    return error * 0.5;
}


/*!
 * @Depend EvaluateResidual
 */
void BAProblem::EvaluateJacobian()
{
    ClearPoseJacobian();
    ClearPointJacobian();
    size_t proj_num = projection_block_.ProjectionNum();

#pragma omp parallel for
    for (size_t j = 0; j < proj_num; j++)
    {
        // Multiply a factor for robustfication
        size_t pose_index = projection_block_.PoseIndex(j);
        size_t point_index = projection_block_.PointIndex(j);
        Vec3 angle_axis, translation, point;
        Vec6 intrinsic;
        pose_block_.GetPose(pose_index, angle_axis, translation);
        GetPoseIntrinsic(pose_index, intrinsic);
        point_block_.GetPoint(point_index, point);
        Vec2 reprojection;
        if (!Project(intrinsic(0), intrinsic(1), intrinsic(2), angle_axis, translation, point, intrinsic.tail<3>(), reprojection))
            continue;
        Vec2 projection;
        projection_block_.GetProjection(j, projection);
        Vec2 reprojection_error = reprojection - projection;

        Mat23 jacobian_rotation;
        Mat23 jacobian_translation;
        Mat23 jacobian_point;
        Mat26 jacobian_intrinsic;

        ProjectAndGradient(angle_axis, translation, point, intrinsic(0), intrinsic(1), intrinsic(2), intrinsic.tail<3>(),
                           projection, jacobian_rotation, jacobian_translation, jacobian_point, jacobian_intrinsic);

        // Correct Jacobian due to robust function
        loss_function_->CorrectJacobian<2, 3>(reprojection_error, jacobian_rotation);
        loss_function_->CorrectJacobian<2, 3>(reprojection_error, jacobian_translation);
        loss_function_->CorrectJacobian<2, 3>(reprojection_error, jacobian_point);
        loss_function_->CorrectJacobian<2, 6>(reprojection_error, jacobian_intrinsic);

        SetPoseJacobian(j, jacobian_rotation, jacobian_translation);
        SetPointJacobian(j, jacobian_point);
        if (!fix_intrinsic_)
            SetIntrinsicJacobian(j, jacobian_intrinsic);
    }
}

bool BAProblem::EvaluateCamera(DT const lambda)
{
    size_t const track_num = PointNum();
    size_t const pose_num = PoseNum();
    size_t const group_num = fix_intrinsic_ ? 0 : GroupNum();
    MatX A = MatX::Zero(6 * (pose_num + group_num), 6 * (pose_num + group_num));
    VecX intercept = VecX::Zero(6 * (pose_num + group_num));

    for (size_t tidx = 0; tidx < track_num; tidx++)
    {
        size_t track_index = tidx;
        Mat3 Hpp = Mat3::Zero();
        Vec3 bp = Vec3::Zero();
        std::vector<std::pair<size_t, size_t> > projection_pairs = GetProjectionsInTrack(tidx);
        std::vector<Mat63> Hcp, Hip;
        Hcp.reserve(projection_pairs.size());
        Hip.reserve(projection_pairs.size());
        for (size_t pidx = 0; pidx < projection_pairs.size(); pidx++)
        {
            size_t pose_index = projection_pairs[pidx].first;
            size_t projection_index = projection_pairs[pidx].second;
            Mat26 pose_jacobian;
            Mat23 point_jacobian;
            Vec2 residual;
            GetPoseJacobian(projection_index, pose_jacobian);
            GetPointJacobian(projection_index, point_jacobian);
            GetResidual(projection_index, residual);
            Hpp += point_jacobian.transpose() * point_jacobian;
            bp += -point_jacobian.transpose() * residual;

            Mat6 Hcc = pose_jacobian.transpose() * pose_jacobian;
            for (size_t i = 0; i < 6; i++)  Hcc(i, i) += lambda * Hcc(i, i);
            A.block(pose_index * 6, pose_index * 6, 6, 6) += Hcc;
            intercept.segment(pose_index * 6, 6) += -pose_jacobian.transpose() * residual;
            Hcp.push_back(pose_jacobian.transpose() * point_jacobian);

            if (!fix_intrinsic_)
            {
                size_t group_index = GetPoseGroup(pose_index);
                Mat26 intrinsic_jacobian;
                GetIntrinsicJacobian(projection_index, intrinsic_jacobian);
                Mat6 Hii = intrinsic_jacobian.transpose() * intrinsic_jacobian;
                for (size_t i = 0; i < 6; i++)  Hii(i, i) += lambda * Hii(i, i);
                A.block((pose_num + group_index) * 6, (pose_num + group_index) * 6, 6, 6) += Hii;
                intercept.segment((pose_num + group_index) * 6, 6) += -intrinsic_jacobian.transpose() * residual;
                Hip.push_back(intrinsic_jacobian.transpose() * point_jacobian);

                Mat6 Hci = pose_jacobian.transpose() * intrinsic_jacobian;
                A.block(pose_index * 6, (pose_num + group_index) * 6, 6, 6) += Hci;
                A.block((pose_num + group_index) * 6, pose_index * 6, 6, 6) += Hci.transpose();
            }
        }
        // augment the diagonal of Hpp
        for (size_t i = 0; i < 3; i++)
            Hpp(i, i) += lambda * Hpp(i, i);
        Mat3 Hpp_inv = Mat3::Zero();
        if (std::abs(Determinant(Hpp)) > EPSILON)
            Hpp_inv = Hpp.inverse();
        Vec3 tp = Hpp_inv * bp;
        SetTp(track_index, tp);

        for (size_t pidx = 0; pidx < projection_pairs.size(); pidx++)
        {
            size_t pose_index = projection_pairs[pidx].first;
            size_t projection_index = projection_pairs[pidx].second;
            size_t group_index = GetPoseGroup(pose_index);
            intercept.segment(pose_index * 6, 6) -= Hcp[pidx] * tp;
            Mat63 Tcp = Hcp[pidx] * Hpp_inv;
            SetTcp(projection_index, Tcp);
            Mat63 Tip;

            if (!fix_intrinsic_)
            {
                size_t group_index = GetPoseGroup(pose_index);
                intercept.segment((pose_num + group_index) * 6, 6) -= Hip[pidx] * tp;
                Tip = Hip[pidx] * Hpp_inv;
                SetTip(projection_index, Tip);
            }

            for (size_t pidx2 = 0; pidx2 < projection_pairs.size(); pidx2++)
            {
                size_t pose_index2 = projection_pairs[pidx2].first;
                size_t group_index2 = GetPoseGroup(pose_index2);
                if (pose_index <= pose_index2)
                {
                    Mat6 Hcc2 = Tcp * Hcp[pidx2].transpose();
                    A.block(pose_index * 6, pose_index2 * 6, 6, 6) -= Hcc2;
                    if (pose_index != pose_index2)
                        A.block(pose_index2 * 6, pose_index * 6, 6, 6) -= Hcc2.transpose();
                }
                if (!fix_intrinsic_ )
                {
                    if (group_index <= group_index2)
                    {
                        Mat6 Hii2 = Tip * Hip[pidx2].transpose();
                        A.block((pose_num + group_index) * 6, (pose_num + group_index2) * 6, 6, 6) -= Hii2;
                        if (group_index != group_index2)
                            A.block((pose_num + group_index2) * 6, (pose_num + group_index) * 6, 6, 6) -= Hii2.transpose();
                    }

                    Mat6 Hci = Tcp * Hip[pidx2].transpose();
                    A.block(pose_index * 6, (pose_num + group_index2) * 6, 6, 6) -= Hci;
                    A.block((pose_num + group_index2) * 6, pose_index * 6, 6, 6) -= Hci.transpose();
                }
            }
        }
    }

    VecX delta_camera;
    if (!SolveLinearSystem(A, intercept, delta_camera))
        return false;
    for (size_t i = 0; i < pose_num; i++)
    {
        pose_block_.SetDeltaPose(i, delta_camera.segment(i * 6, 6));
    }
    for (size_t i = 0; i < group_num; i++)
    {
        intrinsic_block_.SetDeltaIntrinsic(i, delta_camera.segment((i + pose_num) * 6, 6));
    }
//    std::cout << "A:\n" << A << "\n"
//    << "intercept:\n" << intercept << "\n"
//    << "delta camera:\n" << delta_camera << "\n";
    return true;
}

void BAProblem::EvaluatePoint()
{
    size_t const track_num = PointNum();

#pragma omp parallel for
    for (size_t tidx = 0; tidx < track_num; tidx++)
    {
        size_t track_index = tidx;
        Vec3 tp;
        GetTp(track_index, tp);

        std::vector<std::pair<size_t, size_t> > projection_pairs = GetProjectionsInTrack(tidx);
        for (size_t pidx = 0; pidx < projection_pairs.size(); pidx++)
        {
            size_t pose_index = projection_pairs[pidx].first;
            size_t projection_index = projection_pairs[pidx].second;

            Vec6 delta_pose;
            pose_block_.GetDeltaPose(pose_index, delta_pose);
            Mat63 Tcp;
            GetTcp(projection_index, Tcp);
            tp -= Tcp.transpose() * delta_pose;

            if (!fix_intrinsic_)
            {
                size_t group_index = GetPoseGroup(pose_index);
                Vec6 delta_intrinsic;
                intrinsic_block_.GetDeltaIntrinsic(group_index, delta_intrinsic);
                Mat63 Tip;
                GetTip(projection_index, Tip);
                tp -= Tip.transpose() * delta_intrinsic;
            }
        }
        point_block_.SetDeltaPoint(track_index, tp);
    }
}

void BAProblem::UpdateParam()
{
    pose_block_.UpdatePose();
    point_block_.UpdatePoint();
    if (!fix_intrinsic_)    intrinsic_block_.UpdateIntrinsics();
}

void BAProblem::ClearUpdate()
{
    pose_block_.ClearUpdate();
    point_block_.ClearUpdate();
    intrinsic_block_.ClearUpdate();
}

void BAProblem::ClearResidual()
{
    std::fill(residual_, residual_ + 2 * ProjectionNum(), 0.0);
}

void BAProblem::ClearPoseJacobian()
{
    std::fill(pose_jacobian_, pose_jacobian_ + 2 * ProjectionNum() * 6, 0.0);
}

void BAProblem::ClearIntrinsicJacobian()
{
    if (intrinsic_jacobian_)
        std::fill(intrinsic_jacobian_, intrinsic_jacobian_ + 2 * ProjectionNum() * 6, 0.0);
}

void BAProblem::ClearPointJacobian()
{
    std::fill(point_jacobian_, point_jacobian_ + 2 * ProjectionNum() * 3, 0.0);
}


void BAProblem::Delete()
{
    if (residual_ != NULL)                              delete [] residual_;
    if (pose_jacobian_ != NULL)                         delete [] pose_jacobian_;
    if (point_jacobian_ != NULL)                        delete [] point_jacobian_;
    if (intrinsic_jacobian_ != NULL)                    delete [] intrinsic_jacobian_;
}

/*!
 * @brief Ax = b
 * @return If the solution contains NaN values, e.g. due to singular A, return false.
 */
bool BAProblem::SolveLinearSystem(MatX const & A, VecX const & b, VecX & x) const
{
    bool ret = false;

    switch (linear_solver_type_)
    {
    case SPARSE:
        ret = SolveLinearSystemSparse(A.sparseView(), b, x);
        break;
    case DENSE:
        ret = SolveLinearSystemDense(A, b, x);
        break;
    case ITERATIVE:
        ret = SolveLinearSystemIterative(A.sparseView(), b, x);
        break;
    case ADAPTIVE:
    {
        size_t dimension = b.rows() / 6;
        if (dimension < 300)
        {
            ret = SolveLinearSystemDense(A, b, x);
        }
        else
        {
            ret = SolveLinearSystemIterative(A.sparseView(), b, x);
        }
        break;
    }
    default:
    {
        std::cout << "No linear solver stype specified.\n";
        exit(0);
    }
    }
    return ret;
}

bool BAProblem::SolveLinearSystem(SMat const & A, VecX const & b, VecX & x) const
{
    bool ret = false;

    switch (linear_solver_type_)
    {
    case SPARSE:
    {
        ret = SolveLinearSystemSparse(A, b, x);
        break;
    }
    case DENSE:
    {
        MatX A_dense = MatX(A);
        ret = SolveLinearSystemDense(A_dense, b, x);
        break;
    }
    case ITERATIVE:
    {
        ret = SolveLinearSystemIterative(A, b, x);
        break;
    }
    case ADAPTIVE:
    {
        size_t dimension = b.rows() / 6;
        if (dimension < 300)
        {
            MatX A_dense = MatX(A);
            ret = SolveLinearSystemDense(A_dense, b, x);
        }
        else
        {
            ret = SolveLinearSystemIterative(A, b, x);
        }
        break;
    }
    default:
    {
        std::cout << "No linear solver stype specified.\n";
        exit(0);
    }
    }

    return ret;
}

/*!
 * @brief Ax = b
 */
bool BAProblem::SolveLinearSystemDense(MatX const & A, VecX const & b, VecX & x) const
{
    // QR is more accurate than LDLT, but slower in efficiency.
    x = A.ldlt().solve(b);
    //    x = A.colPivHouseholderQr().solve(b);
    return IsNumericalValid(x);
}

bool BAProblem::SolveLinearSystemSparse(SMat const & A, VecX const & b, VecX & x) const
{
    SimplicialLLT<SparseMatrix<DT> > solver;
    x = solver.compute(A).solve(b);
    return IsNumericalValid(x);
}

bool BAProblem::SolveLinearSystemIterative(SMat const & A, VecX const & b, VecX & x) const
{
    ConjugateGradient<SparseMatrix<DT>, Upper> cg;
    cg.setMaxIterations(500);
    cg.setTolerance(1e-6);
    x = cg.compute(A).solve(b);
    return IsNumericalValid(x);
}

std::vector<std::pair<size_t, size_t> > BAProblem::GetProjectionsInTrack(size_t const track_id) const
{
    std::vector<std::pair<size_t, size_t> > projection_pairs;
    std::unordered_map<size_t, std::unordered_map<size_t, size_t> >::const_iterator it = point_projection_map_.find(track_id);
    if (it != point_projection_map_.end())
    {
        std::unordered_map<size_t, size_t> const & projection_map = it->second;
        projection_pairs.reserve(projection_map.size());
        std::unordered_map<size_t, size_t>::const_iterator it2 = projection_map.begin();
        for (; it2 != projection_map.end(); it2++)
        {
            projection_pairs.push_back(std::make_pair(it2->first, it2->second));
        }
    }
    return std::move(projection_pairs);
}

double BAProblem::Step() const
{
    VecX poses, points, delta_pose, delta_point;
    GetPose(poses);
    GetPoint(points);
    GetPoseUpdate(delta_pose);
    GetPointUpdate(delta_point);
    double step_norm = delta_pose.squaredNorm() + delta_point.squaredNorm();
    double param_norm = poses.squaredNorm() + points.squaredNorm();
    if (!fix_intrinsic_)
    {
        VecX intrinsics, delta_intrinsics;
        GetIntrinsic(intrinsics);
        GetIntrinsicUpdate(delta_intrinsics);
        step_norm += delta_intrinsics.squaredNorm();
        param_norm += intrinsics.squaredNorm();
    }
    double relative_step = std::sqrt(step_norm / param_norm);
    return relative_step;
}


