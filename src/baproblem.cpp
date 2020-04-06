#include "STBA/baproblem.h"

#include <fstream>
#include <numeric>
#include <Eigen/IterativeLinearSolvers>

BAProblem::BAProblem() :
    pose_block_(), point_block_(), intrinsic_block_(), projection_block_(),
    residual_(NULL),
    pose_jacobian_(NULL),
    point_jacobian_(NULL),
    pose_jacobian_square_(NULL),
    point_jacobian_square_(NULL),
    pose_point_jacobian_product_(NULL),
    pose_gradient_(NULL),
    point_gradient_(NULL),
    Ec_Cinv_w_(NULL),
    loss_function_(NULL),
    thread_num_(1),
    max_degree_(1000),
    linear_solver_type_(ADAPTIVE)
{
    loss_function_ = new HuberLoss();
}

BAProblem::BAProblem(LossType loss_type) :
    pose_block_(), point_block_(), projection_block_(),
    residual_(NULL),
    pose_jacobian_(NULL),
    point_jacobian_(NULL),
    pose_jacobian_square_(NULL),
    point_jacobian_square_(NULL),
    pose_point_jacobian_product_(NULL),
    pose_gradient_(NULL),
    point_gradient_(NULL),
    Ec_Cinv_w_(NULL),
    loss_function_(NULL),
    thread_num_(1),
    max_degree_(1000),
    linear_solver_type_(ADAPTIVE)
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
    loss_function_(NULL), max_degree_(1000), linear_solver_type_(ADAPTIVE)
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
            proj_num * 2 + proj_num * 12 + proj_num * 6 + pose_num * 36 + point_num * 9 +
            proj_num * 18 + pose_num * 6 + point_num * 3 + pose_num * 6;
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
        pose_jacobian_square_ = new DT[pose_num * 6 * 6];
        point_jacobian_square_ = new DT[point_num * 3 * 3];
        pose_point_jacobian_product_ = new DT[proj_num * 6 * 3];
        pose_gradient_ = new DT[pose_num * 6];
        point_gradient_ = new DT[point_num * 3];
        Ec_Cinv_w_ = new DT[pose_num * 6];
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
                point_indexes.push_back(point_index);
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
    size_t pose_num = PoseNum();
    size_t point_num = PointNum();

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

void BAProblem::GetJcJc(size_t pose_index, Mat6 & JcJc) const
{
    assert(pose_index < pose_block_.PoseNum() && "[GetJcJc] Pose index out of range");
    DT * ptr = pose_jacobian_square_ + pose_index * 6 * 6;
    JcJc = Mat6(ptr);
}

void BAProblem::GetJcJc(std::vector<size_t> const & pose_indexes, MatX & JcJc) const
{
    size_t pose_num = pose_indexes.size();
    JcJc = MatX::Zero(pose_num * 6, pose_num * 6);

    for (size_t i = 0; i < pose_num; i++)
    {
        size_t pose_index = pose_indexes[i];
        Mat6 local_JcJc;
        GetJcJc(pose_index, local_JcJc);
        JcJc.block(6 * i, 6 * i, 6, 6) = local_JcJc;
    }
}

void BAProblem::GetJcJc(std::vector<size_t> const & pose_indexes, SMat & JcJc) const
{
    size_t pose_num = pose_indexes.size();
    JcJc.resize(pose_num * 6, pose_num * 6);
    JcJc.reserve(Eigen::VectorXi::Constant(pose_num * 6, 6));

    for (size_t i = 0; i < pose_num; i++)
    {
        size_t pose_index = pose_indexes[i];
        Mat6 local_JcJc;
        GetJcJc(pose_index, local_JcJc);
        for (size_t j = 0; j < 6; j++)
            for (size_t k = 0; k < 6; k++)
                JcJc.insert(6 * i + j, 6 * i + k) = local_JcJc(j, k);
    }
}

void BAProblem::GetJcJc(MatX & JcJc) const
{
    size_t pose_num = PoseNum();
    JcJc = MatX::Zero(pose_num * 6, pose_num * 6);

    for (size_t i = 0; i < pose_num; i++)
    {
        Mat6 local_JcJc;
        GetJcJc(i, local_JcJc);
        JcJc.block(6 * i, 6 * i, 6, 6) = local_JcJc;
    }
}

void BAProblem::SetJcJc(size_t pose_index, Mat6 const & JcJc)
{
    assert(pose_index < pose_block_.PoseNum() && "[SetJcJc] Pose index out of range");
    for (size_t i = 0; i < 6; i++)
        for (size_t j = 0; j < 6; j++)
            pose_jacobian_square_[pose_index * 6 * 6 + i * 6 + j] = JcJc(i, j);
}

void BAProblem::IncreJcJc(size_t pose_index, Mat6 const & JcJc)
{
    assert(pose_index < pose_block_.PoseNum() && "[IncreJcJc] Pose index out of range");
    for (size_t i = 0; i < 6; i++)
        for (size_t j = 0; j < 6; j++)
            pose_jacobian_square_[pose_index * 6 * 6 + i * 6 + j] += JcJc(i, j);
}

void BAProblem::GetJpJp(size_t point_index, Mat3 & JpJp) const
{
    assert(point_index < point_block_.PointNum() && "[GetJpJp] Point index out of range");
    DT * ptr = point_jacobian_square_ + point_index * 3 * 3;
    JpJp = Mat3(ptr);
}

void BAProblem::SetJpJp(size_t point_index, Mat3 const & JpJp)
{
    assert(point_index < point_block_.PointNum() && "[SetJpJp] Point index out of range");
    for (size_t i = 0; i < 3; i++)
        for (size_t j = 0; j < 3; j++)
            point_jacobian_square_[point_index * 3 * 3 + i * 3 + j] = JpJp(i, j);
}

void BAProblem::IncreJpJp(size_t point_index, Mat3 const & JpJp)
{
    assert(point_index < point_block_.PointNum() && "[IncreJpJp] Point index out of range");
    for (size_t i = 0; i < 3; i++)
        for (size_t j = 0; j < 3; j++)
            point_jacobian_square_[point_index * 3 * 3 + i * 3 + j] += JpJp(i, j);
}

void BAProblem::GetJcJp(size_t proj_index, Mat63 & JcJp) const
{
    assert(proj_index < projection_block_.ProjectionNum() && "[GetJcJp] Projection index out of range");
    DT * ptr = pose_point_jacobian_product_ + proj_index * 6 * 3;
    JcJp = Mat63(ptr);
}

void BAProblem::GetJcJp(size_t pose_index, size_t point_index, Mat63 & JcJp) const
{
    size_t proj_index = GetProjectionIndex(pose_index, point_index);
    GetJcJp(proj_index, JcJp);
}

void BAProblem::SetJcJp(size_t proj_index, Mat63 const & JcJp)
{
    assert(proj_index < projection_block_.ProjectionNum() && "[SetJcJp] Projection index out of range");
    for (size_t i = 0; i < 6; i++)
        for (size_t j = 0; j < 3; j++)
            pose_point_jacobian_product_[proj_index * 6 * 3 + i * 3 + j] = JcJp(i, j);
}

void BAProblem::SetJcJp(size_t pose_index, size_t point_index, Mat63 const & JcJp)
{
    size_t proj_index = GetProjectionIndex(pose_index, point_index);
    SetJcJp(proj_index, JcJp);
}

void BAProblem::GetJce(size_t pose_index, Vec6 & Jce) const
{
    assert(pose_index < pose_block_.PoseNum() && "[GetJce] Pose index out of range");
    DT * ptr = pose_gradient_ + pose_index * 6;
    Jce = Vec6(ptr);
}

void BAProblem::GetJce(std::vector<size_t> const & pose_indexes, VecX & Jce) const
{
    size_t pose_num = pose_indexes.size();
    Jce.resize(pose_num * 6);
    for (size_t i = 0; i < pose_indexes.size(); i++)
    {
        size_t pose_index = pose_indexes[i];
        Vec6 local_Jce;
        GetJce(pose_index, local_Jce);
        Jce.segment(i * 6, 6) = local_Jce;
    }
}

void BAProblem::GetJce(VecX & Jce) const
{
    size_t pose_num = pose_block_.PoseNum();
    Jce.resize(pose_num * 6);
    for (size_t i = 0; i < pose_num; i++)
    {
        Vec6 local_Jce;
        GetJce(i, local_Jce);
        Jce.segment(i * 6, 6) = local_Jce;
    }
}

void BAProblem::SetJce(size_t pose_index, Vec6 const & Jce)
{
    assert(pose_index < pose_block_.PoseNum() && "[GetJce] Pose index out of range");
    for (size_t i = 0; i < 6; i++)
        pose_gradient_[pose_index * 6 + i] = Jce(i);
}

void BAProblem::IncreJce(size_t pose_index, Vec6 const & Jce)
{
    assert(pose_index < pose_block_.PoseNum() && "[IncreJce] Pose index out of range");
    for (size_t i = 0; i < 6; i++)
        pose_gradient_[pose_index * 6 + i] += Jce(i);
}

void BAProblem::GetJpe(size_t point_index, Vec3 & Jpe) const
{
    assert(point_index < point_block_.PointNum() && "[GetJpe] Point index out of range");
    DT * ptr = point_gradient_ + point_index * 3;
    Jpe = Vec3(ptr);
}

void BAProblem::GetJpe(VecX & Jpe) const
{
    size_t point_num = point_block_.PointNum();
    Jpe.resize(point_num * 3);
    for (size_t i = 0; i < point_num; i++)
    {
        Vec3 local_Jpe;
        GetJpe(i, local_Jpe);
        Jpe.segment(3 * i, 3) = local_Jpe;
    }
}

void BAProblem::SetJpe(size_t point_index, Vec3 const & Jpe)
{
    assert(point_index < point_block_.PointNum() && "[SetJpe] Point index out of range");
    for (size_t i = 0; i < 3; i++)
        point_gradient_[point_index * 3 + i] = Jpe(i);
}

void BAProblem::IncreJpe(size_t point_index, Vec3 const & Jpe)
{
    assert(point_index < point_block_.PointNum() && "[SetJpe] Point index out of range");
    for (size_t i = 0; i < 3; i++)
        point_gradient_[point_index * 3 + i] += Jpe(i);
}

void BAProblem::GetEcw(size_t pose_index, Vec6 & ECw) const
{
    assert(pose_index < pose_block_.PoseNum() && "[GetECw] Pose index out of range");
    DT * ptr = Ec_Cinv_w_ + pose_index * 6;
    ECw = Vec6(ptr);
}

void BAProblem::GetEcw(std::vector<size_t> const & pose_indexes, VecX & ECw) const
{
    size_t pose_num = pose_indexes.size();
    ECw = VecX::Zero(6 * pose_num);
    for (size_t i = 0; i < pose_num; i++)
    {
        size_t pose_index = pose_indexes[i];
        Vec6 local_ECw;
        GetEcw(pose_index, local_ECw);
        ECw.segment(6 * i, 6) = local_ECw;
    }
}

void BAProblem::GetEcw(VecX & Ecw) const
{
    size_t pose_num = PoseNum();
    Ecw = VecX::Zero(6 * pose_num);
    for (size_t i = 0; i < pose_num; i++)
    {
        Vec6 local_ECw;
        GetEcw(i, local_ECw);
        Ecw.segment(6 * i, 6) = local_ECw;
    }
}

void BAProblem::SetECw(size_t pose_index, Vec6 const & ECw)
{
    assert(pose_index < pose_block_.PoseNum() && "[SetECw] Pose index out of range");
    for (size_t i = 0; i < 6; i++)
        Ec_Cinv_w_[pose_index * 6 + i] = ECw(i);
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
    }
}

/*!
 * @brief Evaluate the Jacobian square of a single camera, which is the sum of
 * Jacobian square of all the projections in this camera.
 */
void BAProblem::EvaluateJcJc(size_t pose_index, Mat6 & JcJc) const
{
    JcJc = Mat6::Zero();
    std::unordered_map<size_t, std::unordered_map<size_t, size_t> >::const_iterator it1 = pose_projection_map_.find(pose_index);
    assert(it1 != pose_projection_map_.end() && "[EvaluateJcJc] Pose index not found");
    std::unordered_map<size_t, size_t> const & map = it1->second;
    std::unordered_map<size_t, size_t>::const_iterator it2 = map.begin();
    for (; it2 != map.end(); it2++)
    {
        size_t proj_index = it2->second;
        Mat26 jacobian;
        GetPoseJacobian(proj_index, jacobian);
        JcJc += jacobian.transpose() * jacobian;
    }
}

/*!
 * @brief Jc^TJc is a block diagonal matrix, with each block of size 6x6,
 * since each projection (residual term) only corresponds to a unique camera.
 */
void BAProblem::EvaluateJcJc()
{
    ClearJcJc();
    size_t pose_num = pose_block_.PoseNum();
#pragma omp parallel for
    for (size_t i = 0; i < pose_num; i++)
    {
        Mat6 jcjc;
        EvaluateJcJc(i, jcjc);
        SetJcJc(i, jcjc);
    }
}

/*!
 * @brief Evaluate the Jacobian square of a single point, which is the sum of
 * Jacobian square of all the projections of this point.
 */
void BAProblem::EvaluateJpJp(size_t point_index, Mat3 & JpJp) const
{
    JpJp = Mat3::Zero();
    std::unordered_map<size_t, std::unordered_map<size_t, size_t> >::const_iterator it1 = point_projection_map_.find(point_index);
    assert(it1 != point_projection_map_.end() && "[EvaluateJpJp] Point index not found");
    std::unordered_map<size_t, size_t> const & map = it1->second;
    std::unordered_map<size_t, size_t>::const_iterator it2 = map.begin();
    for (; it2 != map.end(); it2++)
    {
        size_t proj_index = it2->second;
        Mat23 jacobian;
        GetPointJacobian(proj_index, jacobian);
        JpJp += jacobian.transpose() * jacobian;
    }
    if (!IsNumericalValid(JpJp))
        JpJp = Mat3::Zero();
}

/*!
 * @brief Jp^TJp is a block diagonal matrix, with each block of size 3x3,
 * since each projection (residual term) only corresponds to a unique point.
 */
void BAProblem::EvaluateJpJp()
{
    ClearJpJp();
    size_t point_num = point_block_.PointNum();
#pragma omp parallel for
    for (size_t i = 0; i < point_num; i++)
    {
        Mat3 jpjp;
        EvaluateJpJp(i, jpjp);
        SetJpJp(i, jpjp);
    }
}

void BAProblem::EvaluateJcJp(size_t proj_index, Mat63 & JcJp) const
{
    Mat26 pose_jacobian;
    GetPoseJacobian(proj_index, pose_jacobian);
    Mat23 point_jacobian;
    GetPointJacobian(proj_index, point_jacobian);
    JcJp = pose_jacobian.transpose() * point_jacobian;
}

void BAProblem::EvaluateJcJp(size_t pose_index, size_t point_index, Mat63 & JcJp) const
{
    size_t proj_index = GetProjectionIndex(pose_index, point_index);
    EvaluateJcJp(proj_index, JcJp);
}

/*!
 * @brief Jc^TJp is a sparse matrix, whose nonzero element number is equal
 * to the number of projections.
 */
void BAProblem::EvaluateJcJp()
{
    ClearJcJp();
    size_t proj_num = projection_block_.ProjectionNum();

#pragma omp parallel for
    for (size_t i = 0; i < proj_num; i++)
    {
        Mat63 JcJp;
        EvaluateJcJp(i, JcJp);
        SetJcJp(i, JcJp);
    }
}

/*!
 * @brief Jc^Te - It denotes the gradient of the sum-of-square cost w.r.t the pose
 */
void BAProblem::EvaluateJce(size_t pose_index, Vec6 & Je) const
{
    Je = Vec6::Zero();
    std::unordered_map<size_t, std::unordered_map<size_t, size_t> >::const_iterator it1 = pose_projection_map_.find(pose_index);
    assert(it1 != pose_projection_map_.end() && "[EvaluateJce] Pose index not found");
    std::unordered_map<size_t, size_t> const & map = it1->second;
    std::unordered_map<size_t, size_t>::const_iterator it2 = map.begin();
    for (; it2 != map.end(); it2++)
    {
        size_t proj_index = it2->second;
        Mat26 pose_jacobian;
        GetPoseJacobian(proj_index, pose_jacobian);

        Vec2 residual;
        GetResidual(proj_index, residual);
        Je += pose_jacobian.transpose() * residual;
    }
}

void BAProblem::EvaluateJce(std::vector<size_t> const & pose_indexes, VecX & Je) const
{
    size_t pose_num = pose_indexes.size();
    Je.resize(pose_num * 6);
    for (size_t i = 0; i < pose_num; i++)
    {
        size_t pose_index = pose_indexes[i];
        Vec6 local_Je;
        EvaluateJce(pose_index, local_Je);
        Je.segment(i * 6, 6) = local_Je;
    }
}

void BAProblem::EvaluateJce()
{
    ClearJce();
    size_t pose_num = pose_block_.PoseNum();

#pragma omp parallel for
    for (size_t i = 0; i < pose_num; i++)
    {
        Vec6 local_Jce;
        EvaluateJce(i, local_Jce);
        SetJce(i, local_Jce);
    }
}

/*!
 * @brief Jp^Te - It denotes the gradient of the sum-of-square cost w.r.t the point
 */
void BAProblem::EvaluateJpe(size_t point_index, Vec3 & Je) const
{
    Je = Vec3::Zero();
    std::unordered_map<size_t, std::unordered_map<size_t, size_t> >::const_iterator it1 = point_projection_map_.find(point_index);
    assert(it1 != point_projection_map_.end() && "[EvaluateJpe] Point index not found");
    std::unordered_map<size_t, size_t> const & map = it1->second;
    std::unordered_map<size_t, size_t>::const_iterator it2 = map.begin();
    for (; it2 != map.end(); it2++)
    {
        size_t proj_index = it2->second;
        Mat23 point_jacobian;
        GetPointJacobian(proj_index, point_jacobian);

        Vec2 residual;
        GetResidual(proj_index, residual);
        Je += point_jacobian.transpose() * residual;
    }
}

void BAProblem::EvaluateJpe(std::vector<size_t> const & point_indexes, VecX & Jpe) const
{
    size_t point_num = point_indexes.size();
    Jpe.resize(point_num * 3);
    for (size_t i = 0; i < point_num; i++)
    {
        size_t point_index = point_indexes[i];
        Vec3 local_Jpe;
        EvaluateJpe(point_index, local_Jpe);
        Jpe.segment(i * 3, 3) = local_Jpe;
    }
}

void BAProblem::EvaluateJpe()
{
    ClearJpe();
    size_t point_num = point_block_.PointNum();
#pragma omp parallel for
    for (size_t i = 0; i < point_num; i++)
    {
        Vec3 local_Jpe;
        EvaluateJpe(i, local_Jpe);
        SetJpe(i, local_Jpe);
    }
}

bool BAProblem::EvaluateEcEc(size_t pose_index1, size_t pose_index2, Mat6 & EcEc) const
{
    EcEc.setZero();
    std::vector<size_t> points;
    GetCommonPoints(pose_index1, pose_index2, points);
    if (points.empty()) return false;

    for (size_t i = 0; i < points.size(); i++)
    {
        size_t point_index = points[i];
        Mat63 Jc1Jp, Jc2Jp;
        Mat3 JpJp;
        GetJcJp(pose_index1, point_index, Jc1Jp);
        GetJcJp(pose_index2, point_index, Jc2Jp);
        GetJpJp(point_index, JpJp);
        if (std::abs(Determinant(JpJp)) > EPSILON)
        {
            Mat3 JpJp_inv = JpJp.inverse();
            EcEc += Jc1Jp * JpJp_inv * Jc2Jp.transpose();
            assert(IsNumericalValid(EcEc));
        }
    }
    return true;
}

/*!
 * @brief A 6x6 EC_-1E^T block w.r.t cameras is occupied iff two cameras share common points.
 */
void BAProblem::EvaluateEcEc(std::vector<size_t> const & pose_indexes, MatX & EcEc) const
{
    size_t pose_num = pose_indexes.size();
    EcEc = MatX::Zero(pose_num * 6, pose_num * 6);

    std::vector<std::pair<size_t, size_t> > pose_pairs;
    for (size_t i = 0; i < pose_num; i++)
    {
        for (size_t j = i; j < pose_num; j++)
        {
            pose_pairs.push_back(std::make_pair(i, j));
        }
    }

    for (size_t i = 0; i < pose_pairs.size(); i++)
    {
        size_t index1 = pose_pairs[i].first;
        size_t index2 = pose_pairs[i].second;
        size_t pose_index1 = pose_indexes[index1];
        size_t pose_index2 = pose_indexes[index2];
        Mat6 local_EcEc;
        bool ret = EvaluateEcEc(pose_index1, pose_index2, local_EcEc);
        if (ret)
        {
            EcEc.block(6 * index1, 6 * index2, 6, 6) = local_EcEc;
            if (pose_index1 != pose_index2)
                EcEc.block(6 * index2, 6 * index1, 6, 6) = local_EcEc.transpose();
        }
    }
}

void BAProblem::EvaluateEcEc(std::vector<size_t> const & pose_indexes, SMat & EcEc) const
{
    size_t pose_num = pose_indexes.size();
    EcEc.resize(pose_num * 6, pose_num * 6);
    EcEc.reserve(Eigen::VectorXi::Constant(pose_num * 6, max_degree_ * 6));

    std::vector<std::pair<size_t, size_t> > pose_pairs;
    for (size_t i = 0; i < pose_num; i++)
    {
        for (size_t j = i; j < pose_num; j++)
        {
            pose_pairs.push_back(std::make_pair(i, j));
        }
    }

    for (size_t i = 0; i < pose_pairs.size(); i++)
    {
        size_t index1 = pose_pairs[i].first;
        size_t index2 = pose_pairs[i].second;
        size_t pose_index1 = pose_indexes[index1];
        size_t pose_index2 = pose_indexes[index2];
        Mat6 local_EcEc;
        bool ret = EvaluateEcEc(pose_index1, pose_index2, local_EcEc);
        if (ret)
        {
            for (size_t j = 0; j < 6; j++)
                for (size_t k = 0; k < 6; k++)
                {
                    EcEc.insert(6 * index1 + j, 6 * index2 + k) = local_EcEc(j, k);
                    if (pose_index1 != pose_index2)
                        EcEc.insert(6 * index2 + j, 6 * index1 + k) = local_EcEc(k, j);
                }
        }
    }
}

void BAProblem::EvaluateEcEc(MatX & EcEc) const
{
    size_t pose_num = PoseNum();
    EcEc = MatX::Zero(pose_num * 6, pose_num * 6);

    size_t point_num = PointNum();

    #pragma omp parallel for
    for (size_t i = 0; i < point_num; i++)
    {
        std::unordered_map<size_t, std::unordered_map<size_t, size_t> >::const_iterator it1 = point_projection_map_.find(i);
        assert(it1 != point_projection_map_.end() && "[EvaluateEcEc] Point index not found");
        std::unordered_map<size_t, size_t> const & map = it1->second;
        std::unordered_map<size_t, size_t>::const_iterator it2 = map.begin();
        std::vector<size_t> pose_indexes, proj_indexes;
        for (; it2 != map.end(); it2++)
        {
            size_t pose_index = it2->first;
            size_t proj_index = it2->second;
            pose_indexes.push_back(pose_index);
            proj_indexes.push_back(proj_index);
        }
        Mat3 JpJp;
        GetJpJp(i, JpJp);
        if (std::abs(Determinant(JpJp)) > EPSILON)
        {
            Mat3 JpJp_inv = JpJp.inverse();
            for (size_t j = 0; j < pose_indexes.size(); j++)
            {
                size_t pose_index1 = pose_indexes[j];
                size_t proj_index1 = proj_indexes[j];
                Mat63 Jc1Jp;
                GetJcJp(proj_index1, Jc1Jp);
                for (size_t k = j; k < pose_indexes.size(); k++)
                {
                    size_t pose_index2 = pose_indexes[k];
                    size_t proj_index2 = proj_indexes[k];
                    Mat63 Jc2Jp;
                    GetJcJp(proj_index2, Jc2Jp);
                    Mat6 ece = Jc1Jp * JpJp_inv * Jc2Jp.transpose();
                    EcEc.block(pose_index1 * 6, pose_index2 * 6, 6, 6) += ece;
                    if (pose_index1 != pose_index2)
                    {
                        EcEc.block(pose_index2 * 6, pose_index1 * 6, 6, 6) += ece.transpose();
                    }
                }
            }
        }
    }
}

void BAProblem::EvaluateEcEc(SMat & EcEc) const
{
    size_t pose_num = PoseNum();
    EcEc.resize(pose_num * 6, pose_num * 6);
    EcEc.reserve(Eigen::VectorXi::Constant(pose_num * 6, max_degree_ * 6));

    std::vector<std::pair<size_t, size_t> > pose_pairs;
    for (size_t i = 0; i < pose_num; i++)
    {
        for (size_t j = i; j < pose_num; j++)
        {
            pose_pairs.push_back(std::make_pair(i, j));
        }
    }

    for (size_t i = 0; i < pose_pairs.size(); i++)
    {
        size_t index1 = pose_pairs[i].first;
        size_t index2 = pose_pairs[i].second;
        size_t pose_index1 = index1;
        size_t pose_index2 = index2;
        Mat6 local_EcEc;
        bool ret = BAProblem::EvaluateEcEc(pose_index1, pose_index2, local_EcEc);
        if (ret)
        {
            for (size_t j = 0; j < 6; j++)
                for (size_t k = 0; k < 6; k++)
                {
                    EcEc.insert(6 * index1 + j, 6 * index2 + k) = local_EcEc(j, k);
                    if (pose_index1 != pose_index2)
                        EcEc.insert(6 * index2 + j, 6 * index1 + k) = local_EcEc(k, j);
                }
        }
    }
}

/*!
 * @brief EcC^-1w, w = -Jp^Te
 */
void BAProblem::EvaluateEcw(size_t pose_index, Vec6 & Ecw) const
{
    Ecw = Vec6::Zero();
    std::unordered_map<size_t, std::unordered_map<size_t, size_t> >::const_iterator it1 = pose_projection_map_.find(pose_index);
    assert(it1 != pose_projection_map_.end() && "[EvaluateECw] Pose index not found");
    std::unordered_map<size_t, size_t> const & map = it1->second;
    std::unordered_map<size_t, size_t>::const_iterator it2 = map.begin();
    for (; it2 != map.end(); it2++)
    {
        size_t point_index = it2->first;
        size_t proj_index = it2->second;
        Mat63 JcJp;
        Mat3 JpJp;
        Vec3 Jpe;
        GetJcJp(proj_index, JcJp);
        GetJpJp(point_index, JpJp);
        GetJpe(point_index, Jpe);

        if (std::abs(Determinant(JpJp)) > EPSILON)
        {
            Mat3 JpJp_inv = JpJp.inverse();
            Ecw += JcJp * JpJp_inv * (-Jpe);
        }
    }
}

void BAProblem::EvaluateEcw()
{
    ClearECw();

    size_t pose_num = PoseNum();
#pragma omp parallel for
    for (size_t i = 0; i < pose_num; i++)
    {
        Vec6 local_Ecw;
        EvaluateEcw(i, local_Ecw);
        SetECw(i, local_Ecw);
    }
}


/*!
 * @brief Read Schur Complement Trick in https://zlthinker.github.io/optimization-for-least-square-problem#schur-complement-trick
 */
void BAProblem::EvaluateB(MatX & B) const
{
    GetJcJc(B);
    return;
}

/*!
 * @brief S = B - EC^-1E^T, B = Jc^TJc, C = Jp^TJp, E = Jc^TJp, omitting intrinsic blocks here
 */
void BAProblem::EvaluateSchurComplement(std::vector<size_t> const & pose_indexes, MatX & S) const
{
    MatX JcJc, ECE;
    GetJcJc(pose_indexes, JcJc);
    EvaluateEcEc(pose_indexes, ECE);
    S = JcJc - ECE;
}

void BAProblem::EvaluateSchurComplement(std::vector<size_t> const & pose_indexes, SMat & S) const
{
    SMat JcJc, ECE;
    size_t pose_num = PoseNum();
    S.resize(pose_num * 6, pose_num * 6);
    S.reserve(Eigen::VectorXi::Constant(pose_num * 6, max_degree_ * 6));

    GetJcJc(pose_indexes, JcJc);
    EvaluateEcEc(pose_indexes, ECE);
    S = JcJc - ECE;
}

/*!
 * @brief Schur complement including pose blocks and intrinsic blocks
 */
void BAProblem::EvaluateSchurComplement(MatX & S) const
{
    MatX B, EE;
    EvaluateB(B);
    EvaluateEcEc(EE);
    S = B - EE;
}

void BAProblem::EvaluateSchurComplement(SMat & S) const
{
    SMat B, EE;

    size_t pose_num = PoseNum();
    std::vector<size_t> pose_indexes(pose_num);
    std::iota(pose_indexes.begin(), pose_indexes.end(), 0);
    GetJcJc(pose_indexes, B);
    EvaluateEcEc(EE);
    S = B - EE;
}

/*!
 * @brief S dy = b = -Jc^Te - EC^-1w, omitting intrinsic blocks here
 */
bool BAProblem::EvaluateDeltaPose(std::vector<size_t> const & pose_indexes, VecX & dy) const
{
    bool ret;
    if (pose_indexes.size() < 5000)
    {
        MatX S;
        VecX Jce, ECw, b;
        EvaluateSchurComplement(pose_indexes, S);
        GetJce(pose_indexes, Jce);
        GetEcw(pose_indexes, ECw);
        b = -Jce - ECw;
        ret = SolveLinearSystem(S, b, dy);
    }
    else
    {
        SMat S;
        VecX Jce, ECw, b;
        EvaluateSchurComplement(pose_indexes, S);
        GetJce(pose_indexes, Jce);
        GetEcw(pose_indexes, ECw);
        b = -Jce - ECw;
        ret = SolveLinearSystem(S, b, dy);
    }

    return ret;
}

bool BAProblem::EvaluateDeltaPose(std::vector<size_t> const & pose_indexes)
{
    VecX dy;
    if (!EvaluateDeltaPose(pose_indexes, dy))
    {
        std::cout << "[EvaluateDeltaPose] Fail in solver linear system.\n";
        return false;
    }
    for (size_t i = 0; i < pose_indexes.size(); i++)
    {
        size_t pose_index = pose_indexes[i];
        Vec3 delta_angle_axis = dy.segment(i * 6, 3);
        Vec3 delta_translation = dy.segment(i * 6 + 3, 3);
        pose_block_.SetDeltaPose(pose_index, delta_angle_axis, delta_translation);
    }
    return true;
}

bool BAProblem::EvaluateDeltaPose()
{
    std::vector<size_t> pose_indexes(PoseNum());
    std::iota(pose_indexes.begin(), pose_indexes.end(), 0);
    return EvaluateDeltaPose(pose_indexes);
}

/*!
 * @brief E^T dy
 */
void BAProblem::EvaluateEDeltaPose(size_t point_index, Vec3 & Edy) const
{
    Edy = Vec3::Zero();
    std::unordered_map<size_t, std::unordered_map<size_t, size_t> >::const_iterator it1 = point_projection_map_.find(point_index);
    assert(it1 != point_projection_map_.end() && "[EvaluateEDeltaPose] Point index not found");
    std::unordered_map<size_t, size_t> const & map = it1->second;
    std::unordered_map<size_t, size_t>::const_iterator it2 = map.begin();
    for (; it2 != map.end(); it2++)
    {
        size_t pose_index = it2->first;
        size_t proj_index = it2->second;
        Mat63 JcJp;
        Vec6 dy;
        GetJcJp(proj_index, JcJp);
        pose_block_.GetDeltaPose(pose_index, dy);
        Edy += JcJp.transpose() * dy;
    }
}

void BAProblem::EvaluateEDelta(size_t point_index, Vec3 & Edy) const
{
    Edy = Vec3::Zero();

    EvaluateEDeltaPose(point_index, Edy);
}

/*!
 * @brief dz = C^-1 (-Jp^Te -E^T dy), C = Jp^TJp
 */
void BAProblem::EvaluateDeltaPoint(size_t point_index, Vec3 & dz)
{
    Mat3 JpJp;
    Vec3 Jpe, Edy;
    GetJpJp(point_index, JpJp);
    GetJpe(point_index, Jpe);
    EvaluateEDelta(point_index, Edy);
    if (std::abs(Determinant(JpJp)) > EPSILON)
    {
        Mat3 JpJp_inv = JpJp.inverse();
        dz = JpJp_inv * (-Jpe - Edy);
        assert(IsNumericalValid(dz));
    }
    else
    {
        dz = Vec3::Zero();
    }
}

void BAProblem::EvaluateDeltaPoint()
{
    size_t point_num = PointNum();
    for (size_t i = 0; i < point_num; i++)
    {
        Vec3 dz;
        EvaluateDeltaPoint(i, dz);
        point_block_.SetDeltaPoint(i, dz);
    }
}

void BAProblem::UpdateParam()
{
    pose_block_.UpdatePose();
    point_block_.UpdatePoint();
}

void BAProblem::ClearUpdate()
{
    pose_block_.ClearUpdate();
    point_block_.ClearUpdate();
}

void BAProblem::ClearResidual()
{
    std::fill(residual_, residual_ + 2 * ProjectionNum(), 0.0);
}

void BAProblem::ClearPoseJacobian()
{
    std::fill(pose_jacobian_, pose_jacobian_ + 2 * ProjectionNum() * 6, 0.0);
}

void BAProblem::ClearPointJacobian()
{
    std::fill(point_jacobian_, point_jacobian_ + 2 * ProjectionNum() * 3, 0.0);
}

void BAProblem::ClearJcJc()
{
    std::fill(pose_jacobian_square_, pose_jacobian_square_ + PoseNum() * 6 * 6, 0.0);
}

void BAProblem::ClearJpJp()
{
    std::fill(point_jacobian_square_, point_jacobian_square_ + PointNum() * 3 * 3, 0.0);
}

void BAProblem::ClearJcJp()
{
    std::fill(pose_point_jacobian_product_, pose_point_jacobian_product_ + ProjectionNum() * 6 * 3, 0.0);
}

void BAProblem::ClearJce()
{
    std::fill(pose_gradient_, pose_gradient_ + PoseNum() * 6, 0.0);
}

void BAProblem::ClearJpe()
{
    std::fill(point_gradient_, point_gradient_ + PointNum() * 3, 0.0);
}

void BAProblem::ClearECw()
{
    std::fill(Ec_Cinv_w_, Ec_Cinv_w_ + PoseNum() * 6, 0.0);
}

/*!
 * @brief In LM algorithm, the "Hessian" matrix's diagonal is augmented as (H + lambda I)x = b
 */
void BAProblem::GetDiagonal(VecX & diagonal) const
{
    size_t pose_num = PoseNum();
    size_t point_num = PointNum();
    diagonal.resize(6 * pose_num + 3 * point_num);
    for (size_t i = 0; i < pose_num; i++)
    {
        diagonal(6 * i) = pose_jacobian_square_[6 * 6 * i];
        diagonal(6 * i + 1) = pose_jacobian_square_[6 * 6 * i + 7];
        diagonal(6 * i + 2) = pose_jacobian_square_[6 * 6 * i + 14];
        diagonal(6 * i + 3) = pose_jacobian_square_[6 * 6 * i + 21];
        diagonal(6 * i + 4) = pose_jacobian_square_[6 * 6 * i + 28];
        diagonal(6 * i + 5) = pose_jacobian_square_[6 * 6 * i + 35];
    }
    for (size_t i = 0; i < point_num; i++)
    {
        diagonal(6 * pose_num + 3 * i) = point_jacobian_square_[3 * 3 * i];
        diagonal(6 * pose_num + 3 * i + 1) = point_jacobian_square_[3 * 3 * i + 4];
        diagonal(6 * pose_num + 3 * i + 2) = point_jacobian_square_[3 * 3 * i + 8];
    }
}

void BAProblem::SetDiagonal(VecX const & diagonal)
{
    size_t pose_num = PoseNum();
    size_t point_num = PointNum();
    assert(pose_num * 6 + point_num * 3 == diagonal.size() && "[SetDiagonal] Size disagrees");
    for (size_t i = 0; i < pose_num; i++)
    {
        pose_jacobian_square_[6 * 6 * i] = diagonal(6 * i);
        pose_jacobian_square_[6 * 6 * i + 7] = diagonal(6 * i + 1);
        pose_jacobian_square_[6 * 6 * i + 14] = diagonal(6 * i + 2);
        pose_jacobian_square_[6 * 6 * i + 21] = diagonal(6 * i + 3);
        pose_jacobian_square_[6 * 6 * i + 28] = diagonal(6 * i + 4);
        pose_jacobian_square_[6 * 6 * i + 35] = diagonal(6 * i + 5);
    }
    for (size_t i = 0; i < point_num; i++)
    {
        point_jacobian_square_[3 * 3 * i] = diagonal(6 * pose_num + 3 * i);
        point_jacobian_square_[3 * 3 * i + 4] = diagonal(6 * pose_num + 3 * i + 1);
        point_jacobian_square_[3 * 3 * i + 8] = diagonal(6 * pose_num + 3 * i + 2);
    }
}

void BAProblem::GetPoseDiagonal(VecX & diagonal) const
{
    size_t pose_num = PoseNum();
    diagonal.resize(6 * pose_num);
    for (size_t i = 0; i < pose_num; i++)
    {
        diagonal(6 * i) = pose_jacobian_square_[6 * 6 * i];
        diagonal(6 * i + 1) = pose_jacobian_square_[6 * 6 * i + 7];
        diagonal(6 * i + 2) = pose_jacobian_square_[6 * 6 * i + 14];
        diagonal(6 * i + 3) = pose_jacobian_square_[6 * 6 * i + 21];
        diagonal(6 * i + 4) = pose_jacobian_square_[6 * 6 * i + 28];
        diagonal(6 * i + 5) = pose_jacobian_square_[6 * 6 * i + 35];
    }
}

void BAProblem::SetPoseDiagonal(VecX const & diagonal)
{
    size_t pose_num = PoseNum();
    assert(pose_num * 6 == diagonal.size() && "[SetPoseDiagonal] Size disagrees");
    for (size_t i = 0; i < pose_num; i++)
    {
        pose_jacobian_square_[6 * 6 * i] = diagonal(6 * i);
        pose_jacobian_square_[6 * 6 * i + 7] = diagonal(6 * i + 1);
        pose_jacobian_square_[6 * 6 * i + 14] = diagonal(6 * i + 2);
        pose_jacobian_square_[6 * 6 * i + 21] = diagonal(6 * i + 3);
        pose_jacobian_square_[6 * 6 * i + 28] = diagonal(6 * i + 4);
        pose_jacobian_square_[6 * 6 * i + 35] = diagonal(6 * i + 5);
    }
}

void BAProblem::GetPointDiagonal(VecX & diagonal) const
{
    size_t point_num = PointNum();
    diagonal.resize(3 * point_num);
    for (size_t i = 0; i < point_num; i++)
    {
        diagonal(3 * i) = point_jacobian_square_[3 * 3 * i];
        diagonal(3 * i + 1) = point_jacobian_square_[3 * 3 * i + 4];
        diagonal(3 * i + 2) = point_jacobian_square_[3 * 3 * i + 8];
    }
}

void BAProblem::SetPointDiagonal(VecX const & diagonal)
{
    size_t point_num = PointNum();
    assert(point_num * 3 == diagonal.size() && "[SetPointDiagonal] Size disagrees");
    for (size_t i = 0; i < point_num; i++)
    {
        point_jacobian_square_[3 * 3 * i] = diagonal(3 * i);
        point_jacobian_square_[3 * 3 * i + 4] = diagonal(3 * i + 1);
        point_jacobian_square_[3 * 3 * i + 8] = diagonal(3 * i + 2);
    }
}

void BAProblem::Delete()
{
    if (residual_ != NULL)                              delete [] residual_;
    if (pose_jacobian_ != NULL)                         delete [] pose_jacobian_;
    if (point_jacobian_ != NULL)                        delete [] point_jacobian_;
    if (pose_jacobian_square_ != NULL)                  delete [] pose_jacobian_square_;
    if (point_jacobian_square_ != NULL)                 delete [] point_jacobian_square_;
    if (pose_point_jacobian_product_ != NULL)           delete [] pose_point_jacobian_product_;
    if (pose_gradient_ != NULL)                         delete [] pose_gradient_;
    if (point_gradient_ != NULL)                        delete [] point_gradient_;
    if (Ec_Cinv_w_ != NULL)                              delete [] Ec_Cinv_w_;
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
    ConjugateGradient<SparseMatrix<DT>, Lower|Upper> cg;
    cg.setMaxIterations(500);
    cg.setTolerance(1e-6);
    x = cg.compute(A).solve(b);
    return IsNumericalValid(x);
}


