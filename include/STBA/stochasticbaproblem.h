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
                        bool complementary_clustering);
    StochasticBAProblem(size_t pose_num, size_t group_num, size_t point_num, size_t proj_num);
    virtual ~StochasticBAProblem();

    inline void SetMaxCommunity(size_t val) { cluster_->SetMaxCommunity(val); }
    inline size_t GetMaxCommunity() const { return cluster_->GetMaxCommunity(); }
    inline void SetComplementaryClustering(bool val) { complementary_clustering_ = val; }

    virtual bool EvaluateCamera(DT const lambda);
    virtual void Solve();
    bool StepAccept();
    virtual bool Initialize(BundleBlock const &bundle_block);
    void SaveCameraCluster(std::string const &save_path);

protected:
    size_t GetPoseCluster(size_t) const;

private:
    void InitializeCluster();
    void RunCluster();
    virtual void Print();
    void SamplingControl();
    void SteepestDescentCorrection(std::unordered_map<size_t, Mat3> const &Hpp_map,
                                   std::unordered_map<size_t, Vec3> &bp_map);

private:
    Louvain *cluster_;
    std::vector<size_t> pose_cluster_map_;
    double last_square_residual_;
    double square_residual_;
    double R_square_;
    double phi_;
    double connectivity_sample_ratio_; // sample ratio of camera connectivity
    bool complementary_clustering_;
    size_t inner_step_;
};

#endif // STOCHASTICBAPROBLEM_H
