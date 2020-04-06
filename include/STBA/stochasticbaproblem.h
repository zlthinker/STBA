#ifndef STOCHASTICBAPROBLEM_H
#define STOCHASTICBAPROBLEM_H

#include "lmbaproblem.h"
#include "clustering/louvain.h"

class StochasticBAProblem : public LMBAProblem
{
public:
    StochasticBAProblem();
    StochasticBAProblem(size_t max_iter,
                        double radius,
                        LossType loss_type,
                        size_t max_community,
                        size_t inner_step);
    StochasticBAProblem(size_t max_iter,
                        double radius,
                        LossType loss_type,
                        size_t max_community,
                        double temperature,
                        size_t batch_size,
                        size_t inner_step,
                        bool complementary_clustering);
    StochasticBAProblem(size_t pose_num, size_t group_num, size_t point_num, size_t proj_num);
    virtual ~StochasticBAProblem();

    inline void SetMaxCommunity(size_t val) { cluster_->SetMaxCommunity(val); }
    inline size_t GetMaxCommunity() const { return cluster_->GetMaxCommunity(); }
    inline void SetBatchSize(size_t val)    { batch_size_ = val; }
    inline void SetInnerStep(size_t val)    { inner_step_ = val; }
    inline void SetComplementaryClustering(bool val)    { complementary_clustering_ = val; }

    virtual void Solve();
    double EvaluateRSquare(VecX const & aug_pose_diagonal,
                           std::vector<std::vector<Vec3> > const & aug_point_diagonal) const;
    double EvaluateRSquare2(VecX const & aug_diagonal);
    bool StepAccept();
    virtual bool Initialize(BundleBlock const & bundle_block);
    void SaveCameraCluster(std::string const & save_path);

protected:
    size_t GetPoseCluster(size_t) const;
    size_t GetPointLocalCluster(size_t, size_t) const;
    size_t GetPointClusters(size_t, std::vector<size_t> &) const;
    void GetPointDiagonal(std::vector<std::vector<Vec3> > &) const;
    void GetPointAugDiagonal(std::vector<std::vector<Vec3> > &) const;
    void AddPointDiagonal(std::vector<std::vector<Vec3> > const &);
    void SetPointDiagonal(std::vector<std::vector<Vec3> > const &);
    void EvaluateJpJp(size_t point_index, size_t cluster_index, Mat3 & JpJp) const;
    virtual void EvaluateJpJp();
    void EvaluateJpe(size_t point_index, size_t cluster_index, Vec3 & Jpe) const;
    virtual void EvaluateJpe();
    bool EvaluateDeltaPose(std::vector<size_t> const & pose_indexes, VecX const & b, VecX & dy) const;
    virtual bool EvaluateDeltaPose();
    virtual void EvaluateDeltaPoint(size_t point_index, Vec3 & dz);
    virtual void EvaluateDeltaPoint();
    virtual void EvaluateEcw(size_t pose_index, Vec6 & Ecw) const;
    void EvaluateEDeltaPose(size_t point_index, size_t cluster_index, Vec3 & Edy) const;
    virtual bool EvaluateEcEc(size_t pose_index1, size_t pose_index2, Mat6 & EcEc) const;
    virtual void EvaluateEcEc(std::vector<size_t> const & pose_indexes, MatX & EcEc) const;
    void EvaluateSchurComplement(std::vector<std::unordered_map<size_t, Mat6> > & S) const;
    void EvaluateFullb(VecX & b) const;
    void EvaluateSdy(std::vector<std::unordered_map<size_t, Mat6> > const & S, VecX const & dy, VecX & Sdy) const;
    virtual void AugmentPointDiagonal();
    virtual void ResetPointDiagonal();
    void GetJpJp(size_t point_index, size_t cluster_index, Mat3 & JpJp) const;
    void SetJpJp(size_t point_index, size_t cluster_index, Mat3 const & JpJp);
    void GetJpe(size_t point_index, size_t cluster_index, Vec3 & Jpe) const;
    void SetJpe(size_t point_index, size_t cluster_index, Vec3 const & Jpe);
    void GetDeltaPoint(size_t point_index, size_t cluster_index, Vec3 & dz) const;
    void SetDeltaPoint(size_t point_index, size_t cluster_index, Vec3 const & dz);
    void ClearPointMeta();
    void InnerIteration(VecX & dy) const;
    void SteepestDescentCorrection(size_t const point_index);
    void SteepestDescentCorrection();

private:
    void InitializeCluster();
    void RunCluster();
    virtual void Print();

private:
    Louvain * cluster_;
    std::vector<std::vector<size_t> > cluster_poses_;
    std::vector<std::vector<size_t> > cluster_points_;
    std::vector<size_t> pose_cluster_map_;
    std::vector<std::vector<size_t> > point_cluster_map_;
    std::vector<PointMeta* > point_meta_;
    double last_square_residual_;
    double square_residual_;
    double R_square_;
    double phi_;
    size_t batch_size_;
    double connectivity_sample_ratio_;   // sample ratio of camera connectivity
    std::vector<std::vector<Vec3> > cluster_point_diagonal_;
    // for inner iterations
    size_t inner_step_;
    std::vector<std::unordered_map<size_t, Mat6> > full_S_;
    VecX full_b_;
    bool complementary_clustering_;
};

#endif // STOCHASTICBAPROBLEM_H
