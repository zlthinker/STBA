#include "STBA/lmbaproblem.h"

#include <iostream>
#include <iomanip>

LMBAProblem::LMBAProblem() : BAProblem(),
    max_iteration_(100), mu_(1e4), rho_(0.0), decrease_factor_(3.0)
{

}

LMBAProblem::LMBAProblem(LossType loss_type) : BAProblem(loss_type),
    max_iteration_(100), mu_(1e4), rho_(0.0), decrease_factor_(3.0)
{

}

LMBAProblem::LMBAProblem(size_t max_iter, double radius, LossType loss_type) : BAProblem(loss_type),
    max_iteration_(max_iter), mu_(radius), rho_(0.0), decrease_factor_(3.0)
{

}

LMBAProblem::LMBAProblem(size_t pose_num, size_t group_num, size_t point_num, size_t proj_num) :
    BAProblem(pose_num, group_num, point_num, proj_num),
    max_iteration_(100), mu_(1e4), rho_(0.0), decrease_factor_(3.0)
{

}

LMBAProblem::~LMBAProblem()
{

}

void LMBAProblem::Solve()
{
    last_square_error_ = EvaluateSquareError(false);
    double mean_error, median_error, max_error;
    ReprojectionError(mean_error, median_error, max_error, false);
    std::cout << "[Solve] Before: mean / median / max reprojection error = "
              << mean_error << " / " << median_error << " / " << max_error << "\n";
    evaluate_ = true;
    time_ = std::chrono::system_clock::now();
    for (iter_ = 0; iter_ < max_iteration_; iter_++)
    {
        if (evaluate_)
        {
//            EvaluateResidual();
//            EvaluateJacobian();
//            // evaluate Hessian
//            EvaluateJcJc();
//            EvaluateJcJi();
//            EvaluateJcJp();
//            EvaluateJiJi();
//            EvaluateJiJp();
//            EvaluateJpJp();
//            // evaluate gradient
//            EvaluateJce();
//            EvaluateJie();
//            EvaluateJpe();
            ClearUpdate();
        }

        // Augment diagonal
//        AugmentPoseDiagonal();
//        AugmentIntrinsicDiagonal();
//        AugmentPointDiagonal();

//        EvaluateEcw();
//        EvaluateEiw();

        // Compute update step
        if (!EvaluateCameraNew(1 / mu_))
        {
            std::cout << "Fail in EvaluateDeltaCamera.\n";
            step_accept_ = false;
        }
        else
        {
            EvaluatePointNew();               // compute delta point
            square_error_ = EvaluateSquareError(true);
            if (StopCriterionUpdate() || StopCriterionRadius() || StopCriterionRelativeCostChange())
                break;
            step_accept_ = StepAccept();
            step_accept_ = true;
        }

        if (step_accept_)                            // accept, descrease lambda
        {
            Print();
            IncreaseRadius();
            last_square_error_ = square_error_;
            UpdateParam();
            evaluate_ = true;                    // Need to re-evaluate once the pose or point parameters are updated
        }
        else                                    // reject, increase lambda
        {
            Print();
            ResetPoseDiagonal();
            ResetIntrinsicDiagonal();
            ResetPointDiagonal();
            evaluate_ = false;
            DecreaseRadius();
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
    stream_ << "[Setting] thread number = " << thread_num_<< "\n";
    stream_ << "[Setting] Levenberg Marquardt\n";
}


/*!
    EvaluateResidual();
    EvaluateJacobian();
    EvaluateJcJc();
    EvaluateJpJp();
    EvaluateJcJp();
    EvaluateJce();
    EvaluateJpe();
    EvaluateJiJi();
    EvaluateJcJi();
    EvaluateJiJp();
    EvaluateJie();
 */
void LMBAProblem::Evaluate()
{
    ClearResidual();
    ClearPoseJacobian();
    ClearPointJacobian();
    ClearIntrinsicJacobian();
    ClearJcJp();
    ClearJcJc();
    ClearJpJp();
    ClearJce();
    ClearJpe();
    if (!fix_intrinsic_)
    {
        ClearJcJi();
        ClearJiJp();
        ClearJie();
        ClearJiJi();
    }
    size_t proj_num = projection_block_.ProjectionNum();

#ifdef OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < proj_num; i++)
    {
        size_t pose_index = projection_block_.PoseIndex(i);
        size_t point_index = projection_block_.PointIndex(i);
        size_t group_index = GetPoseGroup(pose_index);
        Vec3 angle_axis, translation, point;
        Vec6 intrinsic;
        pose_block_.GetPose(pose_index, angle_axis, translation);
        point_block_.GetPoint(point_index, point);
        GetPoseIntrinsic(pose_index, intrinsic);

        Vec2 projection, reprojection, reprojection_error;
        projection_block_.GetProjection(i, projection);
        if (!Project(intrinsic(0), intrinsic(1), intrinsic(2), angle_axis, translation, point, intrinsic.tail<3>(), reprojection))
            continue;
        reprojection_error = reprojection - projection;

        // EvaluateResidual();
        Vec2 residual = reprojection_error;
        loss_function_->CorrectResiduals(residual);
        SetResidual(i, residual);

        // EvaluateJacobian();
        Mat23 jacobian_rotation;
        Mat23 jacobian_translation;
        Mat23 jacobian_point;
        Mat26 jacobian_intrinsic;
        ProjectAndGradient(angle_axis, translation, point, intrinsic(0), intrinsic(1), intrinsic(2), intrinsic.tail<3>(),
                           projection, jacobian_rotation, jacobian_translation, jacobian_point, jacobian_intrinsic);

        loss_function_->CorrectJacobian<2, 3>(reprojection_error, jacobian_rotation);
        loss_function_->CorrectJacobian<2, 3>(reprojection_error, jacobian_translation);
        loss_function_->CorrectJacobian<2, 3>(reprojection_error, jacobian_point);
        loss_function_->CorrectJacobian<2, 6>(reprojection_error, jacobian_intrinsic);
        SetPoseJacobian(i, jacobian_rotation, jacobian_translation);
        SetPointJacobian(i, jacobian_point);
        if (!fix_intrinsic_)
            SetIntrinsicJacobian(i, jacobian_intrinsic);

        // EvaluateJcJp();
        Mat26 Jc;
        Jc << jacobian_rotation, jacobian_translation;
        Mat63 JcJp = Jc.transpose() * jacobian_point;
        SetJcJp(i, JcJp);

        // EvaluateJcJc();
        Mat6 JcJc = Jc.transpose() * Jc;

        // EvaluateJpJp();
        Mat3 JpJp = jacobian_point.transpose() * jacobian_point;

        // EvaluateJce();
        Vec6 Jce = Jc.transpose() * residual;

        // EvaluateJpe();
        Vec3 Jpe = jacobian_point.transpose() * residual;

        Mat6 JiJi, JcJi;
        Mat63 JiJp;
        Vec6 Jie;
        if (!fix_intrinsic_)
        {
            // EvaluateJiJi();
            JiJi = jacobian_intrinsic.transpose() * jacobian_intrinsic;

            // EvaluateJcJi();
            JcJi = Jc.transpose() * jacobian_intrinsic;

            // EvaluateJiJp();
            JiJp = jacobian_intrinsic.transpose() * jacobian_point;

            // EvaluateJie();
            Jie = jacobian_intrinsic.transpose() * residual;
        }

        #pragma omp critical
        {
            IncreJcJc(pose_index, JcJc);
            IncreJpJp(point_index, JpJp);
            IncreJce(pose_index, Jce);
            IncreJpe(point_index, Jpe);
            if (!fix_intrinsic_)
            {
                IncreJiJi(group_index, JiJi);
                IncreJcJi(pose_index, JcJi);
                IncreJiJp(group_index, point_index, JiJp);
                IncreJie(group_index, Jie);
            }
        }


    }

}

/*!
 * @brief Decrease step radius (more conservative) when step rejected
 */
void LMBAProblem::DecreaseRadius()
{
    mu_ /= decrease_factor_;
//    decrease_factor_ *= 2;
}

/*!
 * @brief Increase step radius (more aggressive) when step accepted
 */
void LMBAProblem::IncreaseRadius()
{
    double factor = 1/3.0; //std::max(1/3.0, 1 - std::pow((2 * rho_ - 1), 3));
    mu_ = std::min(1e32, mu_ / factor);
    decrease_factor_ = 3.0;
}

/*!
 * @brief Evaluate the step quality by computing the ratio of exact cost decrease w.r.t the expected decrease of the linear model.
 * @param aug_diagonal - The incremental quantity added to the diagonal of J^TJ
 */
void LMBAProblem::EvaluateRho(VecX const & aug_diagonal)
{
    double last_error = EvaluateSquareError(false);
    double error = EvaluateSquareError(true);
    double change = last_error - error;

    VecX delta_pose, delta_point;
    GetPoseUpdate(delta_pose);
    GetPointUpdate(delta_point);
    VecX gradient_pose, gradient_point;
    GetJce(gradient_pose);
    GetJpe(gradient_point);
    VecX delta(delta_pose.size() + delta_point.size());
    VecX gradient(gradient_pose.size() + gradient_point.size());
    delta << delta_pose, delta_point;
    gradient << gradient_pose, gradient_point;
    double delta_Je = delta.dot(gradient);                  // d^T J^Te
    double delta_square = delta.dot(aug_diagonal.cwiseProduct(delta));  // d^T D d
    double model_change = (delta_square - delta_Je) * 0.5;

    rho_ = change / std::max(model_change, double(EPSILON));
}

/*!
 * @brief Stop when the magnitude of gradient (i.e. J^Te) drops below 1e-8
 */
bool LMBAProblem::StopCriterionGradient()
{
    double max_val = MaxGradient();

    if (max_val < 1e-8)
    {
        stream_ << "[StopCriterionGradient] Max gradient drops below 1e-8: " << max_val << "\n";
        std::cout << "[StopCriterionGradient] Max gradient drops below 1e-8: " << max_val << "\n";
    }

    return max_val < 1e-8;
}

/*!
 * @brief Stop when the relative change of parameters (i.e. delta x / x) drops below 1e-8
 */
bool LMBAProblem::StopCriterionUpdate()
{
    double max_val = Step();

    if (max_val < 1e-8)
    {
        stream_ << "[StopCriterionUpdate] Relative change of parameters drops below 1e-8: " << max_val << "\n";
        std::cout << "[StopCriterionUpdate] Relative change of parameters drops below 1e-8: " << max_val << "\n";
    }

    return max_val < 1e-8;
}

bool LMBAProblem::StopCriterionRadius()
{
    if (mu_ < 1e-32)
    {
        stream_ << "[StopCriterionRadius] Trust region radius drops below 1e-32: " << mu_ << "\n";
        std::cout << "[StopCriterionRadius] Trust region radius drops below 1e-32: " << mu_ << "\n";
    }

    return mu_ < 1e-32;
}

bool LMBAProblem::StopCriterionRelativeCostChange()
{
    double delta_cost = std::abs(last_square_error_ - square_error_);
    double relative_cost_change = delta_cost / last_square_error_;
    if (relative_cost_change < 1e-6)
    {
        stream_ << "[StopCriterionRelativeCostChange] Relative cost change drops below 1e-6: " << relative_cost_change << "\n";
        std::cout << "[StopCriterionRelativeCostChange] Relative cost change drops below 1e-6: " << relative_cost_change << "\n";
    }

    return relative_cost_change < 1e-6;
}

void LMBAProblem::Print()
{
    double delta_loss = last_square_error_ - square_error_;
    double max_gradient = 0.0;//MaxGradient();
    double step = Step();
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
                 << "d: " << std::setw(width+1) << delta_loss << ", "
                 << "F0: " << std::setw(width) << last_square_error_ << ", "
                 << "F1: " << std::setw(width) << square_error_ << ", "
                 << "g: " << std::setw(width) << max_gradient << ", "
                 << "mu: " << std::setw(width) << mu_ << ", "
                 << "h: " << std::setw(width) << step << ", "
                 << std::setprecision(3) << std::fixed
                 << "me: " << std::setw(6) << median_error << ", "
                 << "ae: " << std::setw(6) << mean_error << ", "
                 << "rho: " << std::setw(5) << rho_ << ", "
                 << std::setprecision(1) << std::fixed
                 << "t: " << std::setw(5) << duration << "\n";
    std::cout << local_stream.str();
    stream_ << local_stream.str();
}

double LMBAProblem::Step() const
{
    VecX poses, points, delta_pose, delta_point;
    GetPose(poses);
    GetPoint(points);
    GetPoseUpdate(delta_pose);
    GetPointUpdate(delta_point);
    double relative_step = std::sqrt( (delta_pose.squaredNorm() + delta_point.squaredNorm()) / (poses.squaredNorm() + points.squaredNorm()) );
    return relative_step;
}

double LMBAProblem::MaxGradient() const
{
    VecX Jce, Jpe;
    GetJce(Jce);
    GetJpe(Jpe);
    DT max_val = 0.0;
    for (size_t i = 0; i < Jce.size(); i++)
        max_val = std::max(max_val, std::abs(Jce(i)));
    for (size_t i = 0; i < Jpe.size(); i++)
        max_val = std::max(max_val, std::abs(Jpe(i)));
    return max_val;
}

bool LMBAProblem::StepAccept() const
{
    return square_error_ < last_square_error_;
}

void LMBAProblem::AugmentPoseDiagonal()
{
    GetPoseDiagonal(pose_diagonal_);
    VecX aug_pose_diagonal = pose_diagonal_ / mu_;
    SetPoseDiagonal(pose_diagonal_ + aug_pose_diagonal);
}

void LMBAProblem::ResetPoseDiagonal()
{
     SetPoseDiagonal(pose_diagonal_);
}

void LMBAProblem::AugmentIntrinsicDiagonal()
{
    GetIntrinsicDiagonal(intrinsic_diagonal_);
    VecX aug_intrinsic_diagonal = intrinsic_diagonal_ / mu_;
    SetIntrinsicDiagonal(intrinsic_diagonal_ + aug_intrinsic_diagonal);
}

void LMBAProblem::ResetIntrinsicDiagonal()
{
     SetIntrinsicDiagonal(intrinsic_diagonal_);
}

void LMBAProblem::AugmentPointDiagonal()
{
    GetPointDiagonal(point_diagonal_);
    VecX aug_point_diagonal = point_diagonal_ / mu_;
    SetPointDiagonal(point_diagonal_ + aug_point_diagonal);
}

void LMBAProblem::ResetPointDiagonal()
{
    SetPointDiagonal(point_diagonal_);
}


