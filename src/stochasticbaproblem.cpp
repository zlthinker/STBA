#include "STBA/stochasticbaproblem.h"

#include <fstream>

StochasticBAProblem::StochasticBAProblem() : LMBAProblem(), cluster_(NULL), complementary_clustering_(true)
{
    cluster_ = new Louvain();
    cluster_->SetMaxCommunity(100);
    cluster_->SetTemperature(10);
    SetIntrinsicFixed(true);
}

StochasticBAProblem::StochasticBAProblem(size_t max_iter,
                                         double radius,
                                         LossType loss_type,
                                         size_t max_community)
    : LMBAProblem(max_iter, radius, loss_type), cluster_(NULL), complementary_clustering_(true)
{
    cluster_ = new Louvain();
    cluster_->SetMaxCommunity(max_community);
    cluster_->SetTemperature(10);
    SetIntrinsicFixed(true);
}

StochasticBAProblem::StochasticBAProblem(size_t max_iter,
                                         double radius,
                                         LossType loss_type,
                                         size_t max_community,
                                         double temperature,
                                         size_t batch_size,
                                         bool complementary_clustering)
    : LMBAProblem(max_iter, radius, loss_type), cluster_(NULL), complementary_clustering_(complementary_clustering)
{
    cluster_ = new Louvain();
    cluster_->SetMaxCommunity(max_community);
    cluster_->SetTemperature(temperature);
    SetIntrinsicFixed(true);
}

StochasticBAProblem::StochasticBAProblem(size_t pose_num, size_t group_num, size_t point_num, size_t proj_num) : LMBAProblem(pose_num, group_num, point_num, proj_num), cluster_(NULL)
{
    cluster_ = new Louvain();
    cluster_->SetMaxCommunity(100);
    SetIntrinsicFixed(true);
}

StochasticBAProblem::~StochasticBAProblem()
{
    if (cluster_ != NULL)
        delete cluster_;
}

void StochasticBAProblem::Solve()
{
    std::cout << "[Solve] max_community = " << cluster_->GetMaxCommunity() << "\n"
              << "[Solve] temperature = " << cluster_->GetTemperature() << "\n"
              << "[Solve] complementary clustering = " << complementary_clustering_ << "\n"
              << "[Solve] thread number = " << thread_num_ << "\n";

    last_square_error_ = EvaluateSquareError(false);
    double mean_error, median_error, max_error;
    ReprojectionError(mean_error, median_error, max_error, false);
    std::cout << "[Solve] Before: mean / median / max reprojection error = "
              << mean_error << " / " << median_error << " / " << max_error << "\n";
    evaluate_ = true;
    cluster_->Cluster();

    time_ = std::chrono::system_clock::now();
    for (iter_ = 0; iter_ < max_iteration_; iter_++)
    {
        SamplingControl();
        ClearUpdate();
        if (evaluate_)
        {
            RunCluster();
            EvaluateResidual();
            EvaluateJacobian();
        }

        if (EvaluateCamera(1.0 / mu_))
        {
            EvaluatePoint();
            square_error_ = EvaluateSquareError(true);
            if (StopCriterionGradient() || StopCriterionUpdate() || StopCriterionRadius() || StopCriterionRelativeCostChange())
                break;
            step_accept_ = StepAccept();
        }
        else
        {
            std::cout << "Fail in EvaluateDeltaPose.\n";
            step_accept_ = false;
        }

        if (step_accept_) // accept, descrease lambda
        {
            Print();
            IncreaseRadius();
            last_square_error_ = square_error_;
            UpdateParam();
            evaluate_ = true;
        }
        else // reject, increase lambda
        {
            Print();
            DecreaseRadius();
            evaluate_ = false;
        }
    }

    std::cout << "[Solve] Before: mean / median / max reprojection error = "
              << mean_error << " / " << median_error << " / " << max_error << "\n";
    stream_ << "[Solve] Before: mean / median / max reprojection error = "
            << mean_error << " / " << median_error << " / " << max_error << "\n";
    ReprojectionError(mean_error, median_error, max_error, false);
    std::cout << "[Solve] After: mean / median / max reprojection error = "
              << mean_error << " / " << median_error << " / " << max_error << "\n";
    stream_ << "[Solve] After: mean / median / max reprojection error = "
            << mean_error << " / " << median_error << " / " << max_error << "\n";
    stream_ << "[Setting] max_community = " << cluster_->GetMaxCommunity() << "\n"
            << "[Setting] temperature = " << cluster_->GetTemperature() << "\n"
            << "[Setting] complementary clustering = " << complementary_clustering_ << "\n"
            << "[Setting] thread number = " << thread_num_ << "\n"
            << "[Setting] STBA\n";
}

size_t StochasticBAProblem::GetPoseCluster(size_t pose_index) const
{
    assert(pose_index < PoseNum() && "[GetPoseCluster] Pose index not found");
    return pose_cluster_map_[pose_index];
}

bool StochasticBAProblem::StepAccept()
{
    return last_square_error_ > square_error_;
}

bool StochasticBAProblem::Initialize(BundleBlock const &bundle_block)
{
    if (!BAProblem::Initialize(bundle_block))
        return false;
    InitializeCluster();
    return true;
}

void StochasticBAProblem::InitializeCluster()
{
    std::vector<size_t> nodes(PoseNum());
    std::unordered_map<size_t, std::unordered_map<size_t, double>> edges;
    std::iota(nodes.begin(), nodes.end(), 0);

    std::unordered_map<size_t, std::unordered_map<size_t, std::vector<size_t>>>::const_iterator it1 = common_point_map_.begin();
    for (; it1 != common_point_map_.end(); it1++)
    {
        size_t pose_index1 = it1->first;
        size_t point_num1 = pose_projection_map_.find(pose_index1)->second.size();
        std::unordered_map<size_t, double> edge_map;
        std::unordered_map<size_t, std::vector<size_t>> const &map = it1->second;
        std::unordered_map<size_t, std::vector<size_t>>::const_iterator it2 = map.begin();
        for (; it2 != map.end(); it2++)
        {
            size_t pose_index2 = it2->first;
            if (pose_index1 == pose_index2)
                continue;
            std::vector<size_t> const &points = it2->second;
            size_t point_num2 = pose_projection_map_.find(pose_index2)->second.size();
            edge_map[pose_index2] = double(points.size()) / (point_num1 + point_num2 - points.size());
        }
        edges[pose_index1] = edge_map;
    }
    cluster_->Initialize(nodes, edges);
}

void StochasticBAProblem::RunCluster()
{
    std::vector<std::pair<size_t, size_t>> initial_pairs;
    cluster_->GetEdgesAcrossClusters(initial_pairs);
    cluster_->Reinitialize();
    if (complementary_clustering_)
        cluster_->StochasticCluster(initial_pairs);
    else
        cluster_->StochasticCluster();
    double broken_edge_weight = 0.0;
    for (size_t i = 0; i < initial_pairs.size(); i++)
    {
        size_t index1 = initial_pairs[i].first;
        size_t index2 = initial_pairs[i].second;
        broken_edge_weight += cluster_->EdgeWeight(index1, index2);
    }
    //    connectivity_sample_ratio_ = 1.0 - broken_edge_weight / cluster_->SumEdgeWeight();
    connectivity_sample_ratio_ = 1.0 - initial_pairs.size() / double(cluster_->EdgeNum());

    std::vector<std::vector<size_t>> cluster_poses;
    cluster_->GetClusters(cluster_poses);
    size_t cluster_num = cluster_poses.size();
    pose_cluster_map_.resize(PoseNum(), 0);
    for (size_t i = 0; i < cluster_num; i++)
    {
        std::vector<size_t> const &pose_cluster = cluster_poses[i];
        for (size_t j = 0; j < pose_cluster.size(); j++)
        {
            size_t pose_index = pose_cluster[j];
            assert(pose_index < PoseNum() && "[RunCluster] Pose index out of range");
            pose_cluster_map_[pose_index] = i;
        }
    }
}

void StochasticBAProblem::Print()
{
    double delta_loss = last_square_error_ - square_error_;
    double max_gradient = MaxGradient();
    double step = Step();
    double modualarity = cluster_->Modularity();
    std::vector<std::vector<size_t>> clusters;
    cluster_->GetClusters(clusters);
    double mean_error, median_error, max_error;
    ReprojectionError(mean_error, median_error, max_error, true);
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    std::chrono::duration<double> elapse = now - time_;
    double duration = elapse.count();

    size_t width = 9;
    std::string status = step_accept_ ? std::string("[Update] ") : std::string("[Reject] ");
    std::stringstream local_stream;
    local_stream << std::setprecision(3) << std::scientific
                 << status << std::left << std::setw(3) << iter_ << ", "
                 << "d: " << std::setw(width + 1) << delta_loss << ", "
                 << "F0: " << std::setw(width) << last_square_error_ << ", "
                 << "F1: " << std::setw(width) << square_error_ << ", "
                 << "g: " << std::setw(width) << max_gradient << ", "
                 << "mu: " << std::setw(width) << mu_ << ", "
                 << "h: " << std::setw(width) << step << ", "
                 << std::setprecision(3) << std::fixed
                 << "me: " << std::setw(6) << median_error << ", "
                 << "ae: " << std::setw(6) << mean_error << ", "
                 << "In: " << std::setw(1) << use_inner_step_ << ", "
                 << "Corr: " << std::setw(1) << use_correction_ << ", "
                 << "#C: " << std::setw(4) << clusters.size() << ", "
                 << "Q: " << std::setw(5) << modualarity << ", "
                 << "s: " << std::setw(5) << connectivity_sample_ratio_ << ", "
                 << std::setprecision(1) << std::fixed
                 << "t: " << std::setw(5) << duration << "\n";
    std::cout << local_stream.str();
    stream_ << local_stream.str();
}

void StochasticBAProblem::SaveCameraCluster(std::string const &save_path)
{
    size_t camera_num = PoseNum();
    std::vector<std::vector<size_t>> clusters;
    cluster_->GetClusters(clusters);
    size_t cluster_num = clusters.size();

    std::ofstream fout(save_path);
    fout << camera_num << "\t" << cluster_num << "\n";

    for (size_t i = 0; i < cluster_num; i++)
    {
        size_t cluster_index = i;
        std::vector<size_t> const &camera_indexes = clusters[i];
        for (size_t j = 0; j < camera_indexes.size(); j++)
        {
            size_t camera_index = camera_indexes[j];
            size_t group_index = GetPoseGroup(camera_index);
            Vec6 intrinsic;
            GetIntrinsic(group_index, intrinsic);
            double focal = intrinsic(0);
            double u = intrinsic(1);
            double v = intrinsic(2);
            fout << camera_index << "\t" << cluster_index << "\t" << focal << "\t" << u << "\t" << v << "\n";
            Vec3 angle_axis, translation;
            pose_block_.GetPose(camera_index, angle_axis, translation);
            Mat3 rotation = AngleAxis2Matrix(angle_axis);
            Vec3 center = -rotation.transpose() * translation;
            fout << rotation << "\n"
                 << center(0) << " " << center(1) << " " << center(2) << "\n";
        }
    }
    fout.close();
}

void StochasticBAProblem::SamplingControl()
{
    //    if (iter_ % 30 == 0 && iter_ != 0)
    //        batch_size_ *= 2;
}

bool StochasticBAProblem::EvaluateCamera(DT const lambda)
{
    use_inner_step_ = false;
    use_correction_ = (lambda <= 1.0);

    size_t const track_num = PointNum();
    size_t const pose_num = PoseNum();

    std::vector<MatX> A_mats;
    std::vector<VecX> intercept_vecs;
    MatX full_A;
    VecX full_intercept;
    std::vector<Mat6> pose_diagonals;
    if (use_inner_step_)
    {
        full_A = MatX::Zero(6 * pose_num, 6 * pose_num);
        full_intercept = VecX::Zero(6 * pose_num);
        pose_diagonals.resize(pose_num, Mat6::Zero());
    }
    std::vector<std::vector<size_t>> clusters;
    cluster_->GetClusters(clusters);
    size_t cluster_num = clusters.size();
    std::unordered_map<size_t, size_t> pose_local_map;   // <camera id, local index in a cluster>
    std::unordered_map<size_t, size_t> pose_cluster_map; // <camera id, cluster id>
    for (size_t i = 0; i < cluster_num; i++)
    {
        std::vector<size_t> const &indexes = clusters[i];
        size_t pose_num = indexes.size();
        A_mats.push_back(MatX::Zero(6 * pose_num, 6 * pose_num));
        intercept_vecs.push_back(VecX::Zero(6 * pose_num));
        for (size_t j = 0; j < pose_num; j++)
        {
            size_t pose_index = indexes[j];
            pose_local_map[pose_index] = j; // pose (pose_index) is the j-th pose in i-th cluster
            pose_cluster_map[pose_index] = i;
        }
    }

    for (size_t tidx = 0; tidx < track_num; tidx++)
    {
        size_t track_index = tidx;
        Mat3 Hpp = Mat3::Zero();
        Vec3 bp = Vec3::Zero();
        std::unordered_map<size_t, Mat3> Hpp_map; // point jacobian square for each cluster
        std::unordered_map<size_t, Vec3> bp_map;  // point gradient for each cluster
        std::vector<std::pair<size_t, size_t>> projection_pairs = GetProjectionsInTrack(tidx);
        std::vector<Mat63> Hcp;
        Hcp.reserve(projection_pairs.size());

        for (size_t pidx = 0; pidx < projection_pairs.size(); pidx++)
        {
            size_t pose_index = projection_pairs[pidx].first;
            size_t cluster_index = pose_cluster_map[pose_index];
            size_t pose_local_index = pose_local_map[pose_index];
            size_t projection_index = projection_pairs[pidx].second;
            Mat26 pose_jacobian;
            Mat23 point_jacobian;
            Vec2 residual;
            GetPoseJacobian(projection_index, pose_jacobian);
            GetPointJacobian(projection_index, point_jacobian);
            GetResidual(projection_index, residual);
            Mat3 point_jacobian_square = point_jacobian.transpose() * point_jacobian;
            Vec3 point_gradient = -point_jacobian.transpose() * residual;
            Hpp += point_jacobian_square;
            bp += point_gradient;
            std::unordered_map<size_t, Mat3>::iterator it = Hpp_map.find(cluster_index);
            if (it != Hpp_map.end())
            {
                Mat3 &Hpp_c = it->second;
                Hpp_c += point_jacobian_square;
            }
            else
            {
                Hpp_map[cluster_index] = point_jacobian_square;
            }
            std::unordered_map<size_t, Vec3>::iterator it2 = bp_map.find(cluster_index);
            if (it2 != bp_map.end())
            {
                Vec3 &bp_c = it2->second;
                bp_c += point_gradient;
            }
            else
            {
                bp_map[cluster_index] = point_gradient;
            }

            Mat6 Hcc = pose_jacobian.transpose() * pose_jacobian;
            Vec6 pose_gradient = -pose_jacobian.transpose() * residual;

            if (use_inner_step_)
            {
                full_A.block(pose_index * 6, pose_index * 6, 6, 6) += Hcc;
                full_intercept.segment(pose_index * 6, 6) += pose_gradient;
                pose_diagonals[pose_index] += Hcc;
            }
            for (size_t i = 0; i < 6; i++)
                Hcc(i, i) += lambda * Hcc(i, i);
            Hcp.push_back(pose_jacobian.transpose() * point_jacobian);

            VecX &intercept = intercept_vecs[cluster_index];
            intercept.segment(pose_local_index * 6, 6) += pose_gradient;
            MatX &A = A_mats[cluster_index];
            A.block(pose_local_index * 6, pose_local_index * 6, 6, 6) += Hcc;
        }
        // augment the diagonal of Hpp
        for (size_t i = 0; i < 3; i++)
            Hpp(i, i) += lambda * Hpp(i, i);
        Mat3 Hpp_inv = InverseMat(Hpp);
        Vec3 tp = Hpp_inv * bp;
        SetTp(track_index, tp);

        // augment the diagonal of Hpp_map
        std::unordered_map<size_t, Mat3> Hpp_map_inv;
        std::unordered_map<size_t, Mat3>::iterator it = Hpp_map.begin();
        for (; it != Hpp_map.end(); it++)
        {
            size_t cluster_index = it->first;
            Mat3 & Hpp_c = it->second;
            for (size_t i = 0; i < 3; i++)
                Hpp_c(i, i) += lambda * Hpp_c(i, i);
            Hpp_map_inv.insert(std::make_pair(cluster_index, InverseMat(Hpp_c)));
        }

        if (use_correction_)
            SteepestDescentCorrection(Hpp_map, bp_map);

        for (size_t pidx = 0; pidx < projection_pairs.size(); pidx++)
        {
            size_t pose_index = projection_pairs[pidx].first;
            size_t cluster_index = pose_cluster_map[pose_index];
            size_t pose_local_index = pose_local_map[pose_index];
            size_t projection_index = projection_pairs[pidx].second;
            if (use_inner_step_)
            {
                full_intercept.segment(pose_index * 6, 6) -= Hcp[pidx] * tp;
            }

            Mat63 Tcp = Hcp[pidx] * Hpp_inv;
            SetTcp(projection_index, Tcp);

            VecX &intercept = intercept_vecs[cluster_index];
            Mat3 const &Hpp_c_inv = Hpp_map_inv[cluster_index];
            Vec3 const &bp_c = bp_map[cluster_index];
            intercept.segment(pose_local_index * 6, 6) -= Hcp[pidx] * Hpp_c_inv * bp_c;

            MatX &A = A_mats[cluster_index];
            for (size_t pidx2 = 0; pidx2 < projection_pairs.size(); pidx2++)
            {
                size_t pose_index2 = projection_pairs[pidx2].first;
                size_t cluster_index2 = pose_cluster_map[pose_index2];

                if (use_inner_step_ && pose_index <= pose_index2)
                {
                    Mat6 Hcc2 = Tcp * Hcp[pidx2].transpose();
                    full_A.block(pose_index * 6, pose_index2 * 6, 6, 6) -= Hcc2;
                    if (pose_index != pose_index2)
                        full_A.block(pose_index2 * 6, pose_index * 6, 6, 6) -= Hcc2.transpose();
                }

                if (cluster_index != cluster_index2)
                    continue; // skip camera connections across different clusters
                size_t pose_local_index2 = pose_local_map[pose_index2];
                if (pose_local_index <= pose_local_index2)
                {
                    Mat6 Hcc2 = Hcp[pidx] * Hpp_c_inv * Hcp[pidx2].transpose();
                    A.block(pose_local_index * 6, pose_local_index2 * 6, 6, 6) -= Hcc2;
                    if (pose_local_index != pose_local_index2)
                        A.block(pose_local_index2 * 6, pose_local_index * 6, 6, 6) -= Hcc2.transpose();
                }
            }
        }
    }

    size_t sum_broken = 0;
#pragma omp parallel for reduction(+: sum_broken)
    for (size_t i = 0; i < cluster_num; i++)
    {
        MatX const &A = A_mats[i];
        VecX const &intercept = intercept_vecs[i];
        VecX delta_camera;
        if (!SolveLinearSystem(A, intercept, delta_camera))
            sum_broken++;
        std::vector<size_t> const &indexes = clusters[i];
        size_t pose_num = indexes.size();
        for (size_t j = 0; j < pose_num; j++)
        {
            size_t pose_index = indexes[j];
            pose_block_.SetDeltaPose(pose_index, delta_camera.segment(j * 6, 6));
        }
    }
    if (sum_broken != 0)
        return false;

    if (use_inner_step_)
    {
        std::vector<Mat6> pose_diagonals_inv;
        pose_diagonals_inv.resize(pose_num);
        #pragma omp parallel for
        for (size_t pidx = 0; pidx < pose_num; pidx++)
        {
            Mat6 pose_diag_inv = pose_diagonals[pidx].inverse();
            if (!IsNumericalValid(pose_diag_inv))
                pose_diag_inv = Mat6::Zero();
            pose_diagonals_inv[pidx] = pose_diag_inv;
        }
        VecX delta_pose;
        GetPoseUpdate(delta_pose);
        for (size_t i = 0; i < 4; i++)
        {
            VecX delta_intercept = full_intercept - full_A * delta_pose; 
            for (size_t pidx = 0; pidx < pose_num; pidx++)
            {
                delta_pose.segment(6 * pidx, 6) += pose_diagonals_inv[pidx] * delta_intercept.segment(6 * pidx, 6);
            }
        }
        #pragma omp parallel for
        for (size_t pidx = 0; pidx < pose_num; pidx++)
        {
            pose_block_.SetDeltaPose(pidx, delta_pose.segment(6 * pidx, 6));
        }
    }
    return true;
}

void StochasticBAProblem::SteepestDescentCorrection(std::unordered_map<size_t, Mat3> const &Hpp_map,
                                                    std::unordered_map<size_t, Vec3> &bp_map)
{
    size_t point_cluster_num = bp_map.size();
    if (point_cluster_num <= 1)
        return;
    MatX A = MatX::Zero(3 * (point_cluster_num - 1), 3 * point_cluster_num);
    for (size_t i = 0; i < point_cluster_num - 1; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            A(i * 3 + j, j) = 1;
            A(i * 3 + j, (i + 1) * 3 + j) = -1;
        }
    }

    MatX Hpp_inv = MatX::Identity(3 * point_cluster_num, 3 * point_cluster_num);
    VecX Hpp_inv_g = VecX::Zero(3 * point_cluster_num);
    std::unordered_map<size_t, Vec3>::const_iterator it1 = bp_map.begin();
    std::unordered_map<size_t, Mat3>::const_iterator it2 = Hpp_map.begin();
    size_t i = 0;
    for (; it1 != bp_map.end(); it1++, it2++, i++)
    {
        Vec3 const & bp_c = it1->second;
        Mat3 const & Hpp_c = it2->second;

        for (size_t j = 0; j < 3; j++)
        {
            if (Hpp_c(j, j) < EPSILON)   return;
            Hpp_inv(3 * i + j, 3 * i + j) = 1.0 / Hpp_c(j, j);
            Hpp_inv_g(3 * i + j) = bp_c(j) / Hpp_c(j, j);
        }
    }
    MatX AHA = A * Hpp_inv * A.transpose();
    MatX AHA_inv = AHA.inverse();
    VecX correction = A.transpose() * (AHA_inv * (A * Hpp_inv_g));
    if (!IsNumericalValid(correction))
        return;

    i = 0;
    std::unordered_map<size_t, Vec3>::iterator it3 = bp_map.begin();
    for (; it3 != bp_map.end(); it3++, i++)
    {
        Vec3 &bp_c = it3->second;
        bp_c -= correction.segment(i * 3, 3);
    }
}