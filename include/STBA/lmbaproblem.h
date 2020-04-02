#ifndef LMBAPROBLAM_H
#define LMBAPROBLAM_H

#include "baproblem.h"

class LMBAProblem : public BAProblem
{
public:
    LMBAProblem();
    LMBAProblem(LossType loss_type);
    LMBAProblem(size_t max_iter, double radius, LossType loss_type);
    LMBAProblem(size_t pose_num, size_t group_num, size_t point_num, size_t proj_num);
    virtual ~LMBAProblem();

public:
    virtual void Solve();
    inline void SetRadius(double r) { mu_ = r; }
    inline void SetMaxIteration(size_t iter) { max_iteration_ = iter; }

protected:
    void DecreaseRadius();  // When step rejected
    void IncreaseRadius();  // When step accepted
    bool StopCriterionGradient();
    bool StopCriterionUpdate();
    bool StopCriterionRadius();
    bool StopCriterionRelativeCostChange();
    virtual void Print();
    double MaxGradient() const;
    bool StepAccept() const;


protected:
    size_t max_iteration_;
    double mu_;     // 1/lambda
    double decrease_factor_;
    double last_square_error_;
    double square_error_;
    VecX pose_diagonal_;
    VecX intrinsic_diagonal_;
    VecX point_diagonal_;
    bool evaluate_;
};

#endif // LMBAPROBLAM_H
