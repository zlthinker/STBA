#ifndef BAPROBLEM_H
#define BAPROBLEM_H

#include "utility.h"
#include "datablock.h"
#include "lossfunction.h"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <Eigen/SparseCholesky>

#define OPENMP

enum LinearSolverType
{
    SPARSE = 0,
    DENSE = 1,
    ITERATIVE=2,
    ADAPTIVE=3
};

class BAProblem
{
public:
    BAProblem();
    BAProblem(LossType loss_type);
    BAProblem(size_t pose_num, size_t group_num, size_t point_num, size_t proj_num);
    virtual ~BAProblem();

public:
    bool Create(size_t pose_num, size_t group_num, size_t point_num, size_t proj_num);
    virtual bool Initialize(BundleBlock const & bundle_block);

    inline size_t PoseNum() const { return pose_block_.PoseNum(); }
    inline void GetPose(size_t idx, Vec3 & angle_axis, Vec3 & translation) const
    { pose_block_.GetPose(idx, angle_axis, translation);}
    inline void SetPose(size_t idx, Vec3 const & angle_axis, Vec3 const & translation)
    { pose_block_.SetPose(idx, angle_axis, translation); }

    inline size_t PointNum() const { return point_block_.PointNum(); }
    inline void GetPoint(size_t idx, Vec3 & point) const
    { point_block_.GetPoint(idx, point); }
    inline void SetPoint(size_t idx, Vec3 const & point)
    { point_block_.SetPoint(idx, point); }
    inline void GetColor(size_t idx, Vec3 & color) const
    { point_block_.GetColor(idx, color); }
    inline void SetColor(size_t idx, Vec3 const & color)
    { point_block_.SetColor(idx, color); }
    void SetCommonPoints(size_t, size_t, std::vector<size_t> const &);

    inline size_t GroupNum() const { return intrinsic_block_.GroupNum(); }
    inline void GetIntrinsic(size_t idx, Vec6 & intrinsic) const
    { intrinsic_block_.GetIntrinsic(idx, intrinsic); }
    void GetPoseIntrinsic(size_t pose_index, Vec6 & intrinsic) const
    {
        size_t group_index = GetPoseGroup(pose_index);
        GetIntrinsic(group_index, intrinsic);
    }
    inline void SetIntrinsic(size_t idx, Vec6 const & intrinsic)
    { intrinsic_block_.SetIntrinsic(idx, intrinsic); }
    void SetIntrinsic(size_t idx, size_t camera_index, Vec6 const & intrinsic);

    inline size_t ProjectionNum() const { return projection_block_.ProjectionNum(); }
    void SetProjection(size_t idx, size_t camera_index, size_t point_index, Vec2 const & proj);

    inline void SetDebugFolder(std::string const & val) { debug_folder_ = val; }
    std::string GetDebugFolder() const { return debug_folder_; }

    void ReprojectionError(double & mean, double & median, double & max, bool const update) const;

    inline void SetIntrinsicFixed(bool val) { fix_intrinsic_ = val; }
    inline bool IsIntrinsicFixed() const { return fix_intrinsic_; }

    inline void SetThreadNum(size_t val)
    {
        thread_num_ = val;
#ifdef OPENMP
        omp_set_dynamic(0);     // Explicitly disable dynamic teams
        omp_set_num_threads(thread_num_);
#endif
        Eigen::initParallel();
        std::cout << "[SetThreadNum] Eigen thread number: " << Eigen::nbThreads() << "\n";
    }

    virtual void Solve() = 0;

    void Update(BundleBlock & bundle_block) const;

    void SaveReport(std::string const & report_path) const;

protected:
    size_t GetProjectionIndex(size_t pose_index, size_t point_index) const;
    void GetCommonPoints(size_t, size_t, std::vector<size_t> &) const;  // TODO
    void GetResidual(size_t, Vec2 &) const;
    void SetResidual(size_t, Vec2 const &);
    void GetPoseJacobian(size_t, Mat26 &) const;
    void SetPoseJacobian(size_t, Mat23 const &, Mat23 const &);
    void GetIntrinsicJacobian(size_t, Mat26 &) const;
    void SetIntrinsicJacobian(size_t, Mat26 const &);
    void GetPointJacobian(size_t, Mat23 &) const;
    void SetPointJacobian(size_t, Mat23 const &);
    void GetPose(VecX &) const;
    void GetPoint(VecX &) const;
    void GetPoseUpdate(VecX &) const;
    void GetPointUpdate(VecX &) const;
    size_t GetPoseGroup(size_t) const;


protected:
    void EvaluateResidual();
    double EvaluateSquareResidual(bool const) const;
    double EvaluateSquareError(bool const) const;

    void EvaluateJacobian();
    bool EvaluateCamera(DT const lambda);
    void EvaluatePoint();

    void UpdateParam();

    void ClearUpdate();
    void ClearResidual();
    void ClearPoseJacobian();
    void ClearIntrinsicJacobian();
    void ClearPointJacobian();
    bool SolveLinearSystem(MatX const & A, VecX const & b, VecX & x) const;
    bool SolveLinearSystem(SMat const & A, VecX const & b, VecX & x) const;
    inline void SetLinearSolveType(int t) { linear_solver_type_ = static_cast<LinearSolverType>(t); }

private:
    void Delete();
    bool SolveLinearSystemSparse(SMat const & A, VecX const & b, VecX & x) const;
    bool SolveLinearSystemDense(MatX const & A, VecX const & b, VecX & x) const;
    bool SolveLinearSystemIterative(SMat const & A, VecX const & b, VecX & x) const;

protected:
    PoseBlock pose_block_;
    PointBlock point_block_;
    IntrinsicBlock intrinsic_block_;
    ProjectionBlock projection_block_;
    std::unordered_map<size_t, std::unordered_map<size_t, size_t> > pose_projection_map_;               // <pose, <point, projection> >
    std::unordered_map<size_t, std::unordered_map<size_t, size_t> > point_projection_map_;              // <point, <pose, projection> >
    std::unordered_map<size_t, std::unordered_map<size_t, std::vector<size_t> > > common_point_map_;    // <pose, <pose, points> >
    std::unordered_map<size_t, size_t> pose_group_map_;                                                         // <pose, group>
    std::unordered_map<size_t, std::vector<size_t> > group_pose_map_;                                   // <group, poses>
    std::unordered_map<size_t, size_t> group_index_map_;                                                        // <local group id, origin id>
    std::unordered_map<size_t, size_t> pose_index_map_;                                                          // <local pose id, origin id>
    std::unordered_map<size_t, size_t> point_index_map_;                                                         // <local point id, origin id>
    std::vector<std::pair<size_t, size_t> > GetProjectionsInTrack(size_t const track_id) const;

protected:
    LossFunction * loss_function_;
    DT * residual_;                             // e - reprojection error
    // Jacobian
    DT * pose_jacobian_;                        // Jc, 2x6
    DT * point_jacobian_;                       // Jp, 2x3
    DT * intrinsic_jacobian_;                   // Ji, 2x6
    std::string debug_folder_;
    bool fix_intrinsic_;
    size_t iter_;
    bool step_accept_;
    std::stringstream stream_;
    std::chrono::system_clock::time_point time_;
    size_t max_degree_;                         // Max degree of camera graph
    size_t thread_num_;
    LinearSolverType linear_solver_type_;

protected:
    void GetTp(size_t point_index, Vec3 & tp) const;
    void SetTp(size_t point_index, Vec3 const & tp);
    void GetTcp(size_t proj_index, Mat63 & Tcp) const;
    void SetTcp(size_t proj_index, Mat63 const & Tcp);
    void GetTip(size_t proj_index, Mat63 & Tip) const;
    void SetTip(size_t proj_index, Mat63 const & Tip);


protected:
    DT * tp_;       // 3 * point_num
    DT * Tcp_;      // 6 * 3 * projection_num
    DT * Tip_;      // 6 * 3 * projection_num
};

#endif // BAPROBLEM_H
