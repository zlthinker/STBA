//#include "STBA/stochasticbaproblem.h"

//#include <fstream>

//StochasticBAProblem::StochasticBAProblem() : LMBAProblem(), cluster_(NULL), batch_size_(1), inner_step_(4), complementary_clustering_(true)
//{
//    cluster_ = new Louvain();
//    cluster_->SetMaxCommunity(100);
//    cluster_->SetTemperature(10);
//    SetIntrinsicFixed(true);
//}

//StochasticBAProblem::StochasticBAProblem(size_t max_iter,
//                                         double radius,
//                                         LossType loss_type,
//                                         size_t max_community,
//                                         size_t inner_step)
//    :LMBAProblem(max_iter, radius, loss_type), cluster_(NULL), batch_size_(1), inner_step_(inner_step), complementary_clustering_(true)
//{
//    cluster_ = new Louvain();
//    cluster_->SetMaxCommunity(max_community);
//    cluster_->SetTemperature(10);
//    SetIntrinsicFixed(true);
//}

//StochasticBAProblem::StochasticBAProblem(size_t max_iter,
//                                         double radius,
//                                         LossType loss_type,
//                                         size_t max_community,
//                                         double temperature,
//                                         size_t batch_size,
//                                         size_t inner_step,
//                                         bool complementary_clustering)
//    :LMBAProblem(max_iter, radius, loss_type), cluster_(NULL), batch_size_(batch_size), inner_step_(inner_step), complementary_clustering_(complementary_clustering)
//{
//    cluster_ = new Louvain();
//    cluster_->SetMaxCommunity(max_community);
//    cluster_->SetTemperature(temperature);
//    SetIntrinsicFixed(true);
//}

//StochasticBAProblem::StochasticBAProblem(size_t pose_num, size_t group_num, size_t point_num, size_t proj_num) :
//    LMBAProblem(pose_num, group_num, point_num, proj_num), cluster_(NULL), batch_size_(1), inner_step_(0)
//{
//    cluster_ = new Louvain();
//    cluster_->SetMaxCommunity(100);
//    SetIntrinsicFixed(true);
//}

//StochasticBAProblem::~StochasticBAProblem()
//{
//    if (cluster_ != NULL)   delete cluster_;
//}

//void StochasticBAProblem::Solve()
//{
//    /*
//    std::cout << "[Solve] max_community = " << cluster_->GetMaxCommunity() << "\n"
//              << "[Solve] temperature = " << cluster_->GetTemperature() << "\n"
//              << "[Solve] batch size = " << batch_size_ << "\n"
//              << "[Solve] inner step = " << inner_step_ << "\n"
//              << "[Solve] complementary clustering = " << complementary_clustering_ << "\n"
//              << "[Solve] thread number = " << thread_num_ << "\n";

//    last_square_error_ = EvaluateSquareError(false);
//    double mean_error, median_error, max_error;
//    ReprojectionError(mean_error, median_error, max_error, false);
//    std::cout << "[Solve] Before: mean / median / max reprojection error = "
//              << mean_error << " / " << median_error << " / " << max_error << "\n";
//    evaluate_ = true;
//    cluster_->Cluster();

//    time_ = std::chrono::system_clock::now();
//    for (iter_ = 0; iter_ < max_iteration_; iter_++)
//    {
//        SamplingControl();
//        ClearUpdate();
//        if (evaluate_)
//        {
//            EvaluateResidual();
//            EvaluateJacobian();
//            EvaluateJcJc();
//            EvaluateJcJp();
//            EvaluateJce();
//            BAProblem::EvaluateJpJp();
//            BAProblem::EvaluateJpe();
//        }

//        LMBAProblem::AugmentPoseDiagonal();
//        LMBAProblem::AugmentPointDiagonal();

//        if (EvaluateDeltaPose())
//        {
//            EvaluateDeltaPoint();
//            square_error_ = EvaluateSquareError(true);
//            if (StopCriterionGradient() || StopCriterionUpdate()
//                    || StopCriterionRadius() || StopCriterionRelativeCostChange())
//                break;
//            step_accept_ = StepAccept();
//        }
//        else
//        {
//            std::cout << "Fail in EvaluateDeltaPose.\n";
//            step_accept_ = false;
//        }

//        if (step_accept_)      // accept, descrease lambda
//        {
//            Print();
//            IncreaseRadius();
//            last_square_error_ = square_error_;
//            UpdateParam();
//            evaluate_ = true;
//        }
//        else                                                      // reject, increase lambda
//        {
//            Print();
//            DecreaseRadius();
//            LMBAProblem::ResetPoseDiagonal();
//            LMBAProblem::ResetPointDiagonal();
//            evaluate_ = false;
//        }
//    }

//    std::cout << "[Solve] Before: mean / median / max reprojection error = "
//              << mean_error << " / " << median_error << " / " << max_error << "\n";
//    stream_ << "[Solve] Before: mean / median / max reprojection error = "
//            << mean_error << " / " << median_error << " / " << max_error << "\n";
//    ReprojectionError(mean_error, median_error, max_error, false);
//    std::cout << "[Solve] After: mean / median / max reprojection error = "
//              << mean_error << " / " << median_error << " / " << max_error << "\n";
//    stream_ << "[Solve] After: mean / median / max reprojection error = "
//            << mean_error << " / " << median_error << " / " << max_error << "\n";
//    stream_ << "[Setting] max_community = " << cluster_->GetMaxCommunity() << "\n"
//            << "[Setting] temperature = " << cluster_->GetTemperature() << "\n"
//            << "[Setting] batch size = " << batch_size_ << "\n"
//            << "[Setting] inner step = " << inner_step_ << "\n"
//            << "[Setting] complementary clustering = " << complementary_clustering_ << "\n"
//            << "[Setting] thread number = " << thread_num_ << "\n"
//            << "[Setting] STBA\n";
//            */
//}

//size_t StochasticBAProblem::GetPoseCluster(size_t pose_index) const
//{
//    assert(pose_index < PoseNum() && "[GetPoseCluster] Pose index not found");
//    return pose_cluster_map_[pose_index];
//}

//size_t StochasticBAProblem::GetPointLocalCluster(size_t point_index, size_t cluster_index) const
//{
//    size_t local_point_cluster;
//    bool found = false;
//    std::vector<size_t> const & point_clusters = point_cluster_map_[point_index];
//    for (size_t j = 0; j < point_clusters.size(); j++)
//    {
//        if (point_clusters[j] == cluster_index)
//        {
//            local_point_cluster = j;
//            found = true;
//            break;
//        }
//    }
//    assert(found && "[GetPointLocalCluster] Point cluster not found");
//    return local_point_cluster;
//}

//size_t StochasticBAProblem::GetPointClusters(size_t point_index, std::vector<size_t> & cluster_indexes) const
//{
//    std::unordered_set<size_t> cluster_set;
//    std::unordered_map<size_t, std::unordered_map<size_t, size_t> >::const_iterator it1 = point_projection_map_.find(point_index);
//    assert(it1 != point_projection_map_.end() && "[GetPointClusters] Point index not found");
//    std::unordered_map<size_t, size_t> const & map = it1->second;
//    std::unordered_map<size_t, size_t>::const_iterator it2 = map.begin();
//    for (; it2 != map.end(); it2++)
//    {
//        size_t pose_index = it2->first;
//        size_t cluster_index = GetPoseCluster(pose_index);
//        cluster_set.insert(cluster_index);
//    }
//    cluster_indexes = std::vector<size_t>(cluster_set.begin(), cluster_set.end());
//    return cluster_indexes.size();
//}

//void StochasticBAProblem::AugmentPointDiagonal()
//{
//    std::vector<std::vector<Vec3> >  aug_point_diagonal;
//    GetPointDiagonal(cluster_point_diagonal_);
//    aug_point_diagonal = cluster_point_diagonal_;
//    GetPointAugDiagonal(aug_point_diagonal);
//    AddPointDiagonal(aug_point_diagonal);
//}

//void StochasticBAProblem::ResetPointDiagonal()
//{
//    SetPointDiagonal(cluster_point_diagonal_);
//}

//void StochasticBAProblem::GetJpJp(size_t point_index, size_t local_cluster_index, Mat3 & JpJp) const
//{
//    assert(point_index < PointNum() && "[GetJpJp] Point index out of range");
//    PointMeta const * point_ = point_meta_[point_index];
//    point_->GetJpJp(local_cluster_index, JpJp);
//}

//void StochasticBAProblem::SetJpJp(size_t point_index, size_t local_cluster_index, Mat3 const & JpJp)
//{
//    assert(point_index < PointNum() && "[SetJpJp] Point index out of range");
//    PointMeta * point_ = point_meta_[point_index];
//    point_->SetJpJp(local_cluster_index, JpJp);
//}

//void StochasticBAProblem::GetJpe(size_t point_index, size_t local_cluster_index, Vec3 & Jpe) const
//{
//    assert(point_index < PointNum() && "[GetJpe] Point index out of range");
//    PointMeta const * point_ = point_meta_[point_index];
//    point_->GetJpe(local_cluster_index, Jpe);
//}

//void StochasticBAProblem::SetJpe(size_t point_index, size_t local_cluster_index, Vec3 const & Jpe)
//{
//    assert(point_index < PointNum() && "[SetJpe] Point index out of range");
//    PointMeta * point_ = point_meta_[point_index];
//    point_->SetJpe(local_cluster_index, Jpe);
//}

//void StochasticBAProblem::GetDeltaPoint(size_t point_index, size_t local_cluster_index, Vec3 & dz) const
//{
//    assert(point_index < PointNum() && "[Getdz] Point index out of range");
//    PointMeta const * point_ = point_meta_[point_index];
//    point_->GetDeltaPoint(local_cluster_index, dz);
//}

//void StochasticBAProblem::SetDeltaPoint(size_t point_index, size_t local_cluster_index, Vec3 const & dz)
//{
//    assert(point_index < PointNum() && "[Setdz] Point index out of range");
//    PointMeta * point_ = point_meta_[point_index];
//    point_->SetDeltaPoint(local_cluster_index, dz);
//}

//void StochasticBAProblem::ClearPointMeta()
//{
//    size_t point_num = point_meta_.size();
//    for (size_t i = 0; i < point_num; i++)
//    {
//        PointMeta * ptr = point_meta_[i];
//        if (ptr != NULL)
//            delete ptr;
//    }
//    point_meta_.clear();
//}

//void StochasticBAProblem::GetPointDiagonal(std::vector<std::vector<Vec3> > & point_diagonal) const
//{
//    size_t point_num = PointNum();
//    point_diagonal.resize(point_num);
//    for (size_t i = 0; i < point_num; i++)
//    {
//        PointMeta const * point_ = point_meta_[i];
//        std::vector<size_t> const & clusters = point_cluster_map_[i];
//        std::vector<Vec3> & cluster_diagonal = point_diagonal[i];
//        cluster_diagonal.resize(clusters.size());
//        for (size_t j = 0; j < clusters.size(); j++)
//        {
//            Vec3 diagonal;
//            point_->GetDiagonal(j, diagonal);
//            cluster_diagonal[j] = diagonal;
//        }
//    }
//}

//void StochasticBAProblem::GetPointAugDiagonal(std::vector<std::vector<Vec3> > & aug_point_diagonal) const
//{
//    size_t point_num = PointNum();
//    for (size_t i = 0; i < point_num; i++)
//    {
//        std::vector<Vec3> & diagonals = aug_point_diagonal[i];
//        for (size_t j = 0; j < diagonals.size(); j++)
//        {
//            Vec3 & diagonal = diagonals[j];
//            diagonal = diagonal / mu_;
//        }
//    }
//}

//void StochasticBAProblem::AddPointDiagonal(std::vector<std::vector<Vec3> > const & aug_point_diagonal)
//{
//    size_t point_num = PointNum();
//    for (size_t i = 0; i < point_num; i++)
//    {
//        PointMeta * point_ = point_meta_[i];
//        std::vector<Vec3> const & diagonals = aug_point_diagonal[i];
//        for (size_t j = 0; j < diagonals.size(); j++)
//        {
//            Vec3 diagonal = diagonals[j];
//            point_->AddDiagonal(j, diagonal);
//        }
//    }
//}

//void StochasticBAProblem::SetPointDiagonal(std::vector<std::vector<Vec3> > const & point_diagonal)
//{
//    size_t point_num = PointNum();
//    for (size_t i = 0; i < point_num; i++)
//    {
//        PointMeta * point_ = point_meta_[i];
//        std::vector<Vec3> const & diagonals = point_diagonal[i];
//        for (size_t j = 0; j < diagonals.size(); j++)
//        {
//            Vec3 diagonal = diagonals[j];
//            point_->SetDiagonal(j, diagonal);
//        }
//    }
//}

//void StochasticBAProblem::EvaluateJpJp(size_t point_index, size_t cluster_index, Mat3 & JpJp) const
//{
//    JpJp = Mat3::Zero();
//    std::unordered_map<size_t, std::unordered_map<size_t, size_t> >::const_iterator it1 = point_projection_map_.find(point_index);
//    assert(it1 != point_projection_map_.end() && "[GetJpJp] Point index not found");
//    std::unordered_map<size_t, size_t> const & map = it1->second;
//    std::unordered_map<size_t, size_t>::const_iterator it2 = map.begin();
//    for (; it2 != map.end(); it2++)
//    {
//        size_t pose_index = it2->first;
//        size_t local_cluster_index = GetPoseCluster(pose_index);
//        if (cluster_index != local_cluster_index)   continue;
//        size_t proj_index = it2->second;
//        Mat23 jacobian;
//        GetPointJacobian(proj_index, jacobian);
//        JpJp += jacobian.transpose() * jacobian;
//    }
//}

//void StochasticBAProblem::EvaluateJpJp()
//{
//    size_t point_num = point_block_.PointNum();

//#ifdef OPENMP
//#pragma omp parallel for
//#endif
//    for (size_t i = 0; i < point_num; i++)
//    {
//        std::vector<size_t> const & cluster_indexes = point_cluster_map_[i];
//        for (size_t j = 0; j < cluster_indexes.size(); j++)
//        {
//            size_t cluster_index = cluster_indexes[j];
//            Mat3 jpjp;
//            EvaluateJpJp(i, cluster_index, jpjp);
//            SetJpJp(i, j, jpjp);
//        }
//    }
//}

//void StochasticBAProblem::EvaluateJpe(size_t point_index, size_t cluster_index, Vec3 & Jpe) const
//{
//    Jpe = Vec3::Zero();
//    std::unordered_map<size_t, std::unordered_map<size_t, size_t> >::const_iterator it1 = point_projection_map_.find(point_index);
//    assert(it1 != point_projection_map_.end() && "[GetJpe] Point index not found");
//    std::unordered_map<size_t, size_t> const & map = it1->second;
//    std::unordered_map<size_t, size_t>::const_iterator it2 = map.begin();
//    for (; it2 != map.end(); it2++)
//    {
//        size_t pose_index = it2->first;
//        size_t local_cluster_index = GetPoseCluster(pose_index);
//        if (cluster_index != local_cluster_index)   continue;

//        size_t proj_index = it2->second;
//        Mat23 point_jacobian;
//        GetPointJacobian(proj_index, point_jacobian);

//        Vec2 residual;
//        GetResidual(proj_index, residual);
//        Jpe += point_jacobian.transpose() * residual;
//    }
//}

//void StochasticBAProblem::EvaluateJpe()
//{
//    size_t point_num = point_block_.PointNum();

//#ifdef OPENMP
//#pragma omp parallel for
//#endif
//    for (size_t i = 0; i < point_num; i++)
//    {
//        std::vector<size_t> const & cluster_indexes = point_cluster_map_[i];
//        for (size_t j = 0; j < cluster_indexes.size(); j++)
//        {
//            size_t cluster_index = cluster_indexes[j];
//            Vec3 jpe;
//            EvaluateJpe(i, cluster_index, jpe);
//            SetJpe(i, j, jpe);
//        }
//    }
//}

//bool StochasticBAProblem::EvaluateEcEc(size_t pose_index1, size_t pose_index2, Mat6 & EcEc) const
//{
//    EcEc.setZero();
//    std::vector<size_t> points;
//    GetCommonPoints(pose_index1, pose_index2, points);
//    if (points.empty()) return false;

//    size_t cluster_index1 = GetPoseCluster(pose_index1);
//    size_t cluster_index2 = GetPoseCluster(pose_index2);
//    assert(cluster_index1 == cluster_index2 && "[EvaluateEcEc] Clusters of two poses disagree");

//    for (size_t i = 0; i < points.size(); i++)
//    {
//        size_t point_index = points[i];
//        size_t local_point_cluster = GetPointLocalCluster(point_index, cluster_index1);

//        Mat63 Jc1Jp, Jc2Jp;
//        Mat3 JpJp;
//        GetJcJp(pose_index1, point_index, Jc1Jp);
//        GetJcJp(pose_index2, point_index, Jc2Jp);
//        GetJpJp(point_index, local_point_cluster, JpJp);
//        Mat3 JpJp_inv = JpJp.inverse();
//        if (IsNumericalValid(JpJp_inv))
//            EcEc += Jc1Jp * JpJp_inv * Jc2Jp.transpose();
//    }
//    return true;
//}

//void StochasticBAProblem::EvaluateEcEc(std::vector<size_t> const & pose_indexes, MatX & EcEc) const
//{
//    size_t pose_num = pose_indexes.size();
//    EcEc = MatX::Zero(pose_num * 6, pose_num * 6);

//    if (pose_indexes.empty())   return;

//    size_t cluster_index = GetPoseCluster(pose_indexes[0]);
//    std::unordered_map<size_t, size_t> local_pose_map;
//    for (size_t i = 0; i < pose_indexes.size(); i++)
//    {
//        size_t pose_index = pose_indexes[i];
//        local_pose_map[pose_index] = i;
//    }

//    std::vector<size_t> const & points = cluster_points_[cluster_index];
//    for (size_t i = 0; i < points.size(); i++)
//    {
//        size_t point_index = points[i];
//        std::unordered_map<size_t, std::unordered_map<size_t, size_t> >::const_iterator it1 = point_projection_map_.find(point_index);
//        assert(it1 != point_projection_map_.end() && "[EvaluateEcEc] Point index not found");
//        std::unordered_map<size_t, size_t> const & map = it1->second;
//        std::unordered_map<size_t, size_t>::const_iterator it2 = map.begin();
//        std::vector<size_t> cluster_pose_indexes, cluster_proj_indexes;
//        for (; it2 != map.end(); it2++)
//        {
//            size_t pose_index = it2->first;
//            size_t proj_index = it2->second;
//            size_t local_cluster_index = GetPoseCluster(pose_index);
//            if (local_cluster_index == cluster_index)
//            {
//                cluster_pose_indexes.push_back(pose_index);
//                cluster_proj_indexes.push_back(proj_index);
//            }
//        }
//        Mat3 JpJp, JpJp_inv;
//        GetJpJp(point_index, GetPointLocalCluster(point_index, cluster_index), JpJp);
//        JpJp_inv = JpJp.inverse();
//        if (IsNumericalValid(JpJp_inv))
//        {
//            for (size_t j = 0; j < cluster_pose_indexes.size(); j++)
//            {
//                size_t pose_index1 = cluster_pose_indexes[j];
//                size_t proj_index1 = cluster_proj_indexes[j];
//                assert(local_pose_map.find(pose_index1) != local_pose_map.end());
//                size_t local_pose_index1 = local_pose_map[pose_index1];
//                Mat63 Jc1Jp;
//                GetJcJp(proj_index1, Jc1Jp);
//                for (size_t k = j; k < cluster_pose_indexes.size(); k++)
//                {
//                    size_t pose_index2 = cluster_pose_indexes[k];
//                    size_t proj_index2 = cluster_proj_indexes[k];
//                    assert(local_pose_map.find(pose_index2) != local_pose_map.end());
//                    size_t local_pose_index2 = local_pose_map[pose_index2];
//                    Mat63 Jc2Jp;
//                    GetJcJp(proj_index2, Jc2Jp);
//                    Mat6 ece = Jc1Jp * JpJp_inv * Jc2Jp.transpose();
//                    EcEc.block(local_pose_index1 * 6, local_pose_index2 * 6, 6, 6) += ece;
//                    if (pose_index1 != pose_index2)
//                    {
//                        EcEc.block(local_pose_index2 * 6, local_pose_index1 * 6, 6, 6) += ece.transpose();
//                    }
//                }
//            }
//        }
//    }
//}

//void StochasticBAProblem::EvaluateEDeltaPose(size_t point_index, size_t cluster_index, Vec3 & Edy) const
//{
//    Edy = Vec3::Zero();
//    std::unordered_map<size_t, std::unordered_map<size_t, size_t> >::const_iterator it1 = point_projection_map_.find(point_index);
//    assert(it1 != point_projection_map_.end() && "[EvaluateEDeltaPose] Point index not found");
//    std::unordered_map<size_t, size_t> const & map = it1->second;
//    std::unordered_map<size_t, size_t>::const_iterator it2 = map.begin();
//    for (; it2 != map.end(); it2++)
//    {
//        size_t pose_index = it2->first;
//        if (GetPoseCluster(pose_index) != cluster_index)    continue;
//        size_t proj_index = it2->second;
//        Mat63 JcJp;
//        Vec6 dy;
//        GetJcJp(proj_index, JcJp);
//        pose_block_.GetTempDeltaPose(pose_index, dy);
//        Edy += JcJp.transpose() * dy;
//    }
//}

///*!
// * @brief S dy = b, omitting intrinsic blocks here
// */
//bool StochasticBAProblem::EvaluateDeltaPose(std::vector<size_t> const & pose_indexes, VecX const & b, VecX & dy) const
//{
//    bool ret;
//    if (pose_indexes.size() < 5000)
//    {
//        MatX S;
//        BAProblem::EvaluateSchurComplement(pose_indexes, S);
//        size_t pose_index = pose_indexes[4];
//        Mat6 JcJc;
//        GetJcJc(pose_index, JcJc);
//        std::cout << "JcJc:\n" << JcJc << "\n";
//        std::cout << S.rows() << "\t" << S.cols() << "\n";
//        std::cout << "S:\n" << S.block(0, 0, 24, 24) << "\n";
//        ret = SolveLinearSystem(S, b, dy);

//        Vec6 dy0 = JcJc.inverse() * b.segment(24, 6);
//        std::cout << "dy0: " << dy0 << "\n";
//        std::cout << "dy0: " << dy.segment(24, 6) << "\n";
//        exit(0);
//    }
//    else
//    {
//        SMat S;
//        BAProblem::EvaluateSchurComplement(pose_indexes, S);
//        ret = SolveLinearSystem(S, b, dy);
//    }

//    return ret;
//}

//void StochasticBAProblem::EvaluateSchurComplement(std::vector<std::unordered_map<size_t, Mat6> > & S) const
//{
//    size_t pose_num = PoseNum();
//    S.clear();
//    S.resize(pose_num);

//    std::vector<std::pair<size_t, size_t> > pose_pairs;
//    std::unordered_map<size_t, std::unordered_map<size_t, std::vector<size_t> > >::const_iterator it1 = common_point_map_.begin();
//    for (; it1 != common_point_map_.end(); it1++)
//    {
//        size_t pose_index1 = it1->first;
//        std::unordered_map<size_t, std::vector<size_t> > const & map = it1->second;
//        std::unordered_map<size_t, std::vector<size_t> >::const_iterator it2 = map.begin();
//        for (; it2 != map.end(); it2++)
//        {
//            size_t pose_index2 = it2->first;
//            pose_pairs.push_back(std::make_pair(pose_index1, pose_index2));
//            if (pose_index1 == pose_index2)
//            {
//                S[pose_index1][pose_index1] = Mat6::Zero();
//            }
//            else
//            {
//                S[pose_index1][pose_index2] = Mat6::Zero();
//                S[pose_index2][pose_index1] = Mat6::Zero();
//            }
//        }
//    }

//#ifdef OPENMP
//#pragma omp parallel for
//#endif
//    for (size_t i = 0; i < pose_pairs.size(); i++)
//    {
//        size_t pose_index1 = pose_pairs[i].first;
//        size_t pose_index2 = pose_pairs[i].second;
//        Mat6 local_EcEc;
//        bool ret = BAProblem::EvaluateEcEc(pose_index1, pose_index2, local_EcEc);
//        if (ret)
//        {
//            if (pose_index1 == pose_index2)
//            {
//                Mat6 JcJc;
//                GetJcJc(pose_index1, JcJc);
//                S[pose_index1][pose_index1] = JcJc - local_EcEc;
//            }
//            else
//            {
//                S[pose_index1][pose_index2] = -local_EcEc;
//                S[pose_index2][pose_index1] = -local_EcEc.transpose();
//            }
//        }
//    }

//}

//void StochasticBAProblem::EvaluateSchurComplement(std::vector<MatX> & S_mats) const
//{
//    size_t cluster_num = cluster_poses_.size();
//    S_mats.resize(cluster_num);

//    std::vector<size_t> local_pose_map(PoseNum());
//    for (size_t i = 0; i < cluster_num; i++)
//    {
//        std::vector<size_t> const & pose_indexes = cluster_poses_[i];
//        size_t cluster_pose_num = pose_indexes.size();
//        S_mats[i] = MatX::Zero(cluster_pose_num * 6, cluster_pose_num * 6);
//        for (size_t j = 0; j < cluster_pose_num; j++)
//        {
//            size_t pose_index = pose_indexes[j];
//            local_pose_map[pose_index] = j;
//            Mat6 JcJc;
//            GetJcJc(pose_index, JcJc);
//            S_mats[i].block(j * 6, j * 6, 6, 6) = JcJc;
//        }
//    }

//    size_t point_num = PointNum();
//#ifdef OPENMP
//#pragma omp parallel for
//#endif
//    for (size_t i = 0; i < point_num; i++)
//    {
//        std::unordered_map<size_t, std::unordered_map<size_t, size_t> >::const_iterator it1 = point_projection_map_.find(i);
//        assert(it1 != point_projection_map_.end() && "[EvaluateSchurComplement] Point index not found");
//        std::unordered_map<size_t, size_t> const & map = it1->second;
//        std::unordered_map<size_t, size_t>::const_iterator it2 = map.begin();
//        std::vector<size_t> pose_indexes, proj_indexes;
//        for (; it2 != map.end(); it2++)
//        {
//            size_t pose_index = it2->first;
//            size_t proj_index = it2->second;
//            pose_indexes.push_back(pose_index);
//            proj_indexes.push_back(proj_index);
//        }
//        Mat3 JpJp, JpJp_inv;
//        BAProblem::GetJpJp(i, JpJp);
//        JpJp_inv = JpJp.inverse();
//        if (IsNumericalValid(JpJp_inv))
//        {
//            for (size_t j = 0; j < pose_indexes.size(); j++)
//            {
//                size_t pose_index1 = pose_indexes[j];
//                size_t proj_index1 = proj_indexes[j];
//                size_t cluster_index1 = pose_cluster_map_[pose_index1];
//                size_t local_pose_index1 = local_pose_map[pose_index1];
//                Mat63 Jc1Jp;
//                GetJcJp(proj_index1, Jc1Jp);
//                for (size_t k = j; k < pose_indexes.size(); k++)
//                {
//                    size_t pose_index2 = pose_indexes[k];
//                    size_t proj_index2 = proj_indexes[k];
//                    size_t cluster_index2 = pose_cluster_map_[pose_index2];
//                    size_t local_pose_index2 = local_pose_map[pose_index2];
//                    if (cluster_index1 != cluster_index2)
//                        continue;
//                    Mat63 Jc2Jp;
//                    GetJcJp(proj_index2, Jc2Jp);
//                    Mat6 ece = Jc1Jp * JpJp_inv * Jc2Jp.transpose();

//                    MatX & cluster_S = S_mats[cluster_index1];
////#ifdef OPENMP
////#pragma omp critical
////#endif
//                    {
//                        cluster_S.block(local_pose_index1 * 6, local_pose_index2 * 6, 6, 6) -= ece;
//                    }
//                    if (pose_index1 != pose_index2)
//                    {
////#ifdef OPENMP
////#pragma omp critical
////#endif
//                        {
//                            cluster_S.block(local_pose_index2 * 6, local_pose_index1 * 6, 6, 6) -= ece.transpose();
//                        }
//                    }
//                }
//            }
//        }
//    }

//}

//void StochasticBAProblem::EvaluateSdy(std::vector<std::unordered_map<size_t, Mat6> > const & S,
//                                      VecX const & dy, VecX & Sdy) const
//{
//    size_t pose_num = PoseNum();
//    assert(S.size() == pose_num);
//    assert(dy.rows() == pose_num * 6);
//    Sdy = VecX::Zero(pose_num * 6);

//#ifdef OPENMP
//#pragma omp parallel for
//#endif
//    for (size_t i = 0; i < pose_num; i++)
//    {
//        Vec6 local_Sdy = Vec6::Zero();
//        std::unordered_map<size_t, Mat6> const & local_S = S[i];
//        std::unordered_map<size_t, Mat6>::const_iterator it = local_S.begin();
//        for (; it != local_S.end(); it++)
//        {
//            size_t pose_index = it->first;
//            Mat6 const & block = it->second;
//            Vec6 local_dy = dy.segment(pose_index * 6, 6);
//            local_Sdy += block * local_dy;
//        }
//        Sdy.segment(6 * i, 6) = local_Sdy;
//    }
//}

//void StochasticBAProblem::InnerIteration(VecX & dy) const
//{
//    for (size_t i = 0; i < inner_step_; i++)
//    {
//        VecX Sdy;
//        EvaluateSdy(full_S_, dy, Sdy);
//        VecX residual_b = full_b_ - Sdy;

//        for (size_t j = 0; j < PoseNum(); j++)
//        {
//            Mat6 JcJc, JcJc_inv;
//            GetJcJc(j, JcJc);
//            JcJc_inv = JcJc.inverse();
//            if (IsNumericalValid(JcJc_inv))
//            {
//                Vec6 local_dy = JcJc_inv * residual_b.segment(6 * j, 6);
//                dy.segment(j * 6, 6) += local_dy;
//            }
//        }
//    }
//}

//void StochasticBAProblem::EvaluateFullb(VecX & b) const
//{
//    VecX Jce, ecw;
//    GetJce(Jce);
//    ecw.resize(PoseNum() * 6);
//#ifdef OPENMP
//#pragma omp parallel for
//#endif
//    for (size_t i = 0; i < PoseNum(); i++)
//    {
//        Vec6 local_ecw;
//        BAProblem::EvaluateEcw(i, local_ecw);
//        ecw.segment(6 * i, 6) = local_ecw;
//    }
//    b = -Jce - ecw;
//}

//bool StochasticBAProblem::EvaluateDeltaPose()
//{
//    VecX dy = VecX::Zero(PoseNum() * 6);
//    if (inner_step_ > 0)
//    {
//        EvaluateSchurComplement(full_S_);
//        EvaluateFullb(full_b_);
//    }

//    for (size_t b = 0; b < batch_size_; b++)
//    {
//        VecX local_dy = VecX::Zero(PoseNum() * 6);

//        // Compute initial steps
//        RunCluster();
//        EvaluateJpJp();
//        AugmentPointDiagonal();
//        EvaluateJpe();
//        BAProblem::EvaluateEcw();

//        size_t cluster_num = cluster_poses_.size();
//        size_t sum_broken = 0;

//#ifdef OPENMP
//#pragma omp parallel for reduction(+:sum_broken)
//#endif
//        for (size_t i = 0; i < cluster_num; i++)
//        {
//            std::vector<size_t> const & pose_cluster = cluster_poses_[i];
//            VecX cluster_dy;
//            if (!BAProblem::EvaluateDeltaPose(pose_cluster, cluster_dy))
//            {
//                std::cout << "[EvaluateDeltaPose] Fail in solver linear system.\n";
//                sum_broken += 1;
//            }
//            for (size_t j = 0; j < pose_cluster.size(); j++)
//            {
//                size_t pose_index = pose_cluster[j];
//                local_dy.segment(pose_index * 6, 6) += cluster_dy.segment(j * 6, 6);
//            }
//        }
//        if (sum_broken != 0)
//        {
//            std::cout << "[StochasticBAProblem::EvaluateDeltaPose] Fail in computing initial step.\n";
//            return false;
//        }

//        InnerIteration(local_dy);
//        dy += local_dy;
//    }

//    dy = dy / double(batch_size_);
//    for (size_t i = 0; i < PoseNum(); i++)
//    {
//        Vec3 angle_axis = dy.segment(i * 6, 3);
//        Vec3 translation = dy.segment(i * 6 + 3, 3);
//        pose_block_.SetDeltaPose(i, angle_axis, translation);
//    }

//    return true;
//}

//void StochasticBAProblem::EvaluateDeltaPoint(size_t point_index, Vec3 & dz)
//{
//    dz = Vec3::Zero();
//    Mat3 sum_JpJp = Mat3::Zero();

//    std::vector<size_t> const & cluster_indexes = point_cluster_map_[point_index];
//    size_t cluster_num = cluster_indexes.size();
//    assert(cluster_num > 0 && "[EvaluateDeltaPoint] Zero cluster number");

//    size_t valid_cluster_num = 0;
//    for (size_t i = 0; i < cluster_num; i++)
//    {
//        size_t cluster_index = cluster_indexes[i];
//        Vec3 Jpe, Edy;
//        Mat3 JpJp, JpJp_inv;
//        GetJpe(point_index, i, Jpe);
//        EvaluateEDeltaPose(point_index, cluster_index, Edy);
//        GetJpJp(point_index, i, JpJp);
//        Vec3 delta = -Jpe - Edy;
//        dz += delta;
//        sum_JpJp += JpJp;
//        valid_cluster_num++;

//        JpJp_inv = JpJp.inverse();
//        if (IsNumericalValid(JpJp_inv))
//        {
//            Vec3 cluster_delta = JpJp_inv * delta;
//            SetDeltaPoint(point_index, i, cluster_delta);
//        }
//    }

//    if (valid_cluster_num > 0)
//    {
//        Mat3 sum_JpJp_inv = sum_JpJp.inverse();
//        if (IsNumericalValid(sum_JpJp_inv))
//            dz = sum_JpJp_inv * dz;
//    }
//}

//void StochasticBAProblem::EvaluateDeltaPoint()
//{
//    size_t point_num = PointNum();

//#ifdef OPENMP
//#pragma omp parallel for
//#endif
//    for (size_t i = 0; i < point_num; i++)
//    {
//        Vec3 dz;
//        BAProblem::EvaluateDeltaPoint(i, dz);
//        point_block_.SetDeltaPoint(i, dz);
//    }
//}

//void StochasticBAProblem::EvaluateEcw(size_t pose_index, Vec6 & Ecw) const
//{
//    Ecw = Vec6::Zero();
//    size_t pose_cluster_index = GetPoseCluster(pose_index);
//    std::unordered_map<size_t, std::unordered_map<size_t, size_t> >::const_iterator it1 = pose_projection_map_.find(pose_index);
//    assert(it1 != pose_projection_map_.end() && "[EvaluateECw] Pose index not found");
//    std::unordered_map<size_t, size_t> const & map = it1->second;
//    std::unordered_map<size_t, size_t>::const_iterator it2 = map.begin();
//    for (; it2 != map.end(); it2++)
//    {
//        size_t point_index = it2->first;
//        size_t proj_index = it2->second;
//        size_t local_point_cluster = GetPointLocalCluster(point_index, pose_cluster_index);

//        Mat63 JcJp;
//        Mat3 JpJp;
//        Vec3 Jpe;
//        GetJcJp(proj_index, JcJp);
//        GetJpJp(point_index, local_point_cluster, JpJp);
//        GetJpe(point_index, local_point_cluster, Jpe);
//        Mat3 JpJp_inv = JpJp.inverse();
//        if (IsNumericalValid(JpJp_inv))
//        {
//            Ecw += JcJp * JpJp_inv * (-Jpe);
//        }
//    }
//}

//double StochasticBAProblem::EvaluateRSquare(VecX const & aug_pose_diagonal,
//                                            std::vector<std::vector<Vec3> > const & aug_point_diagonal) const
//{
//    double R = 0;
//    size_t proj_num = ProjectionNum();
//    for (size_t i = 0; i < proj_num; i++)
//    {
//        size_t pose_index = projection_block_.PoseIndex(i);
//        size_t cluster_index = GetPoseCluster(pose_index);
//        size_t point_index = projection_block_.PointIndex(i);
//        size_t local_point_cluster = GetPointLocalCluster(point_index, cluster_index);

//        Mat26 pose_jacobian;
//        Mat23 point_jacobian;
//        Vec6 delta_pose;
//        Vec3 delta_point;
//        Vec2 residual;
//        GetPoseJacobian(i, pose_jacobian);
//        GetPointJacobian(i, point_jacobian);
//        pose_block_.GetDeltaPose(pose_index, delta_pose);
//        GetDeltaPoint(point_index, local_point_cluster, delta_point);
//        GetResidual(i, residual);
//        Vec2 r = pose_jacobian * delta_pose + point_jacobian * delta_point + residual;
//        R += r.squaredNorm();
//    }

//    size_t pose_num = PoseNum();
//    assert(aug_pose_diagonal.rows() == pose_num * 6);
//    for (size_t i = 0; i < pose_num; i++)
//    {
//        Vec6 delta_pose, diagonal;
//        pose_block_.GetDeltaPose(i, delta_pose);
//        diagonal = aug_pose_diagonal.segment(6 * i, 6);
//        R += delta_pose.transpose() * diagonal.cwiseProduct(delta_pose);
//    }

//    size_t point_num = PointNum();
//    for (size_t i = 0; i < point_num; i++)
//    {
//        std::vector<Vec3> const & diagonals = aug_point_diagonal[i];
//        for (size_t j = 0; j < diagonals.size(); j++)
//        {
//            Vec3 aug_diagonal = diagonals[j];
//            Vec3 delta_point;
//            GetDeltaPoint(i, j, delta_point);
//            R += delta_point.transpose() * aug_diagonal.cwiseProduct(delta_point);
//        }
//    }

//    return R;
//}

//double StochasticBAProblem::EvaluateRSquare2(VecX const & aug_diagonal)
//{
//    double R = 0;
//    size_t proj_num = ProjectionNum();
//    double max_error = 0;
//    for (size_t i = 0; i < proj_num; i++)
//    {
//        size_t pose_index = projection_block_.PoseIndex(i);
//        size_t point_index = projection_block_.PointIndex(i);
//        Mat26 pose_jacobian;
//        Mat23 point_jacobian;
//        Vec6 delta_pose;
//        Vec3 delta_point;
//        Vec2 residual;
//        GetPoseJacobian(i, pose_jacobian);
//        GetPointJacobian(i, point_jacobian);
//        pose_block_.GetDeltaPose(pose_index, delta_pose);
//        point_block_.GetDeltaPoint(point_index, delta_point);
//        GetResidual(i, residual);
//        Vec2 r = pose_jacobian * delta_pose + point_jacobian * delta_point + residual;
//        std::vector<size_t> clusters;
//        GetPointClusters(point_index, clusters);
//        double error = r.squaredNorm();
//        max_error = std::max(max_error, error);
//        if (error > 1e3)
//        {
//            continue;
//            std::cout << "[EvaluateRSquare2] " << pose_index << ", " << point_index << ", "
//                      << r.squaredNorm() << ", " << clusters.size() << "\n";
//        }
//        R += r.squaredNorm();
//    }
//    //    std::cout << "[EvaluateRSquare2] max_error = " << max_error << "\n";

//    size_t pose_num = PoseNum();
//    size_t point_num = PointNum();
//    assert(aug_diagonal.rows() == pose_num * 6 + point_num * 3);
//    for (size_t i = 0; i < pose_num; i++)
//    {
//        Vec6 delta_pose, diagonal;
//        pose_block_.GetDeltaPose(i, delta_pose);
//        diagonal = aug_diagonal.segment(6 * i, 6);
//        R += delta_pose.transpose() * diagonal.cwiseProduct(delta_pose);
//    }
//    for (size_t i = 0; i < point_num; i++)
//    {
//        Vec3 delta_point, diagonal;
//        point_block_.GetDeltaPoint(i, delta_point);
//        diagonal = aug_diagonal.segment(6 * pose_num + 3 * i, 3);
//        R += delta_point.transpose() * diagonal.cwiseProduct(delta_point);
//    }

//    return R;
//}

//bool StochasticBAProblem::StepAccept()
//{
//    return last_square_error_ > square_error_;
//}

//bool StochasticBAProblem::Initialize(BundleBlock const & bundle_block)
//{
//    if (!BAProblem::Initialize(bundle_block))
//        return false;
//    InitializeCluster();
//    return true;
//}

//void StochasticBAProblem::InitializeCluster()
//{
//    std::vector<size_t> nodes(PoseNum());
//    std::unordered_map<size_t, std::unordered_map<size_t, double> > edges;
//    std::iota(nodes.begin(), nodes.end(), 0);

//    std::unordered_map<size_t, std::unordered_map<size_t, std::vector<size_t> > >::const_iterator it1 = common_point_map_.begin();
//    for (; it1 != common_point_map_.end(); it1++)
//    {
//        size_t pose_index1 = it1->first;
//        size_t point_num1 = pose_projection_map_.find(pose_index1)->second.size();
//        std::unordered_map<size_t, double> edge_map;
//        std::unordered_map<size_t, std::vector<size_t> > const & map = it1->second;
//        std::unordered_map<size_t, std::vector<size_t> >::const_iterator it2 = map.begin();
//        for (; it2 != map.end(); it2++)
//        {
//            size_t pose_index2 = it2->first;
//            if (pose_index1 == pose_index2) continue;
//            std::vector<size_t> const & points = it2->second;
//            size_t point_num2 = pose_projection_map_.find(pose_index2)->second.size();
//            edge_map[pose_index2] = double(points.size()) / (point_num1 + point_num2 - points.size());
//        }
//        edges[pose_index1] = edge_map;
//    }
//    cluster_->Initialize(nodes, edges);
//}

//void StochasticBAProblem::RunCluster()
//{
//    std::vector<std::pair<size_t, size_t> > initial_pairs;
//    cluster_->GetEdgesAcrossClusters(initial_pairs);
//    cluster_->Reinitialize();
//    if (complementary_clustering_)
//        cluster_->StochasticCluster(initial_pairs);
//    else
//        cluster_->StochasticCluster();
//    double broken_edge_weight = 0.0;
//    for (size_t i = 0; i < initial_pairs.size(); i++)
//    {
//        size_t index1 = initial_pairs[i].first;
//        size_t index2 = initial_pairs[i].second;
//        broken_edge_weight += cluster_->EdgeWeight(index1, index2);
//    }
////    connectivity_sample_ratio_ = 1.0 - broken_edge_weight / cluster_->SumEdgeWeight();
//    connectivity_sample_ratio_ = 1.0 - initial_pairs.size() / double(cluster_->EdgeNum());

//    cluster_->GetClusters(cluster_poses_);
//    size_t cluster_num = cluster_poses_.size();
//    pose_cluster_map_.resize(PoseNum(), 0);
//    for (size_t i = 0; i < cluster_num; i++)
//    {
//        std::vector<size_t> const & pose_cluster = cluster_poses_[i];
//        for (size_t j = 0; j < pose_cluster.size(); j++)
//        {
//            size_t pose_index = pose_cluster[j];
//            assert(pose_index < PoseNum() && "[RunCluster] Pose index out of range");
//            pose_cluster_map_[pose_index] = i;
//        }
//    }

//    ClearPointMeta();
//    size_t point_num = PointNum();
//    cluster_points_.clear();
//    point_cluster_map_.clear();
//    cluster_points_.resize(cluster_num, std::vector<size_t>());
//    point_cluster_map_.resize(point_num, std::vector<size_t>());
//    point_meta_.resize(point_num);
//    for (size_t i = 0; i < point_num; i++)
//    {
//        std::vector<size_t> clusters;
//        GetPointClusters(i, clusters);
//        point_cluster_map_[i] = clusters;
//        point_meta_[i] = new PointMeta(clusters.size());
//        for (size_t j = 0; j < clusters.size(); j++)
//        {
//            size_t cluster_index = clusters[j];
//            cluster_points_[cluster_index].push_back(i);
//        }
//    }
//}

//void StochasticBAProblem::Print()
//{
//    double delta_loss = last_square_error_ - square_error_;
//    double max_gradient = MaxGradient();
//    double step = Step();
//    double modualarity = cluster_->Modularity();
//    std::vector<std::vector<size_t> > clusters;
//    cluster_->GetClusters(clusters);
//    double mean_error, median_error, max_error;
//    ReprojectionError(mean_error, median_error, max_error, true);
//    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
//    std::chrono::duration<double> elapse = now - time_;
//    double duration = elapse.count();

//    size_t width = 9;
//    std::string status = step_accept_ ? std::string("[Update] ") : std::string("[Reject] ");
//    std::stringstream local_stream;
//    local_stream << std::setprecision(3) << std::scientific
//                 << status << std::left << std::setw(3) << iter_ << ", "
//                 << "d: " << std::setw(width+1) << delta_loss << ", "
//                 << "F0: " << std::setw(width) << last_square_error_ << ", "
//                 << "F1: " << std::setw(width) << square_error_ << ", "
////                 << "f0^2: " << std::setw(width) << last_square_residual_ << ", "
////                 << "f1^2: " << std::setw(width) << square_residual_ << ", "
////                 << "R^2: " << std::setw(width) << R_square_ << ", "
////                 << "phi: " << std::setw(width+1) << phi_ << ", "
//                 << "g: " << std::setw(width) << max_gradient << ", "
//                 << "mu: " << std::setw(width) << mu_ << ", "
//                 << "h: " << std::setw(width) << step << ", "
//                 << std::setprecision(3) << std::fixed
//                 << "me: " << std::setw(6) << median_error << ", "
//                 << "ae: " << std::setw(6) << mean_error << ", "
//                 << "B: " << std::setw(2) << batch_size_ << ", "
//                 << "In: " << std::setw(2) << inner_step_ << ", "
//                 << "#C: " << std::setw(4) << clusters.size() << ", "
//                 << "Q: " << std::setw(5) << modualarity << ", "
//                 << "s: " << std::setw(5) << connectivity_sample_ratio_ << ", "
//                 << std::setprecision(1) << std::fixed
//                 << "t: " << std::setw(5) << duration << "\n";
//    std::cout << local_stream.str();
//    stream_ << local_stream.str();
//}

//void StochasticBAProblem::SaveCameraCluster(std::string const & save_path)
//{
//    size_t camera_num = PoseNum();
//    std::vector<std::vector<size_t> > clusters;
//    cluster_->GetClusters(clusters);
//    size_t cluster_num = clusters.size();

//    std::ofstream fout(save_path);
//    fout << camera_num << "\t" << cluster_num << "\n";

//    for (size_t i = 0; i < cluster_num; i++)
//    {
//        size_t cluster_index = i;
//        std::vector<size_t> const & camera_indexes = clusters[i];
//        for (size_t j = 0; j < camera_indexes.size(); j++)
//        {
//            size_t camera_index = camera_indexes[j];
//            size_t group_index = GetPoseGroup(camera_index);
//            Vec6 intrinsic;
//            GetIntrinsic(group_index, intrinsic);
//            double focal = intrinsic(0);
//            double u = intrinsic(1);
//            double v = intrinsic(2);
//            fout << camera_index << "\t" << cluster_index << "\t" << focal << "\t" << u << "\t" << v << "\n";
//            Vec3 angle_axis, translation;
//            pose_block_.GetPose(camera_index, angle_axis, translation);
//            Mat3 rotation = AngleAxis2Matrix(angle_axis);
//            Vec3 center = -rotation.transpose() * translation;
//            fout << rotation << "\n" << center(0) << " " << center(1) << " " << center(2) << "\n";
//        }
//    }
//    fout.close();
//}

//void StochasticBAProblem::SamplingControl()
//{
////    if (iter_ % 30 == 0 && iter_ != 0)
////        batch_size_ *= 2;
//}
