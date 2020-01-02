#include "dlbaproblem.h"

DLBAProblem::DLBAProblem() : min_mu_(1.0), max_mu_(1e8), mu_(1e8),
    evaluate_(true), radius_(1e4), max_iteration_(100)
{
    SetLinearSolveType(ADAPTIVE);
}

DLBAProblem::DLBAProblem(LossType loss_type) : BAProblem(loss_type),
    min_mu_(1.0), max_mu_(1e8), mu_(1e8), evaluate_(true), radius_(1e4), max_iteration_(100)
{
    SetLinearSolveType(ADAPTIVE);
}

DLBAProblem::DLBAProblem(size_t max_iter, double radius, LossType loss_type) : BAProblem(loss_type),
    min_mu_(1.0), max_mu_(1e8), mu_(1e8), evaluate_(true), radius_(radius), max_iteration_(max_iter)
{
    SetLinearSolveType(ADAPTIVE);
}

DLBAProblem::~DLBAProblem()
{

}

void DLBAProblem::EvaluateGradient()
{
    GetJce(pose_gradient_);
    GetJpe(point_gradient_);
    for (size_t i = 0; i < pose_gradient_.rows(); i++)
    {
        if (pose_diagonal_sqrt_(i) < std::numeric_limits<DT>::epsilon())
            pose_gradient_(i) = 0.0;
        else
            pose_gradient_(i) = pose_gradient_(i) / pose_diagonal_sqrt_(i);
    }
    for (size_t i = 0; i < point_gradient_.rows(); i++)
    {
        if (point_diagonal_sqrt_(i) < std::numeric_limits<DT>::epsilon())
            point_gradient_(i) = 0.0;
        else
            point_gradient_(i) = point_gradient_(i) / point_diagonal_sqrt_(i);
    }
}

/*!
 * @brief A cauchy step = -alpha * gradient = -ahpha * J^Te
 */
bool DLBAProblem::EvaluateCauchyStep()
{
    VecX scale_pose_gradient = pose_gradient_;
    VecX scale_point_gradient = point_gradient_;
    for (size_t i = 0; i < scale_pose_gradient.rows(); i++)
    {
        if (pose_diagonal_sqrt_(i) < std::numeric_limits<DT>::epsilon())
            scale_pose_gradient(i) = 0.0;
        else
            scale_pose_gradient(i) = scale_pose_gradient(i) / pose_diagonal_sqrt_(i);
    }
    for (size_t i = 0; i < scale_point_gradient.rows(); i++)
    {
        if (point_diagonal_sqrt_(i) < std::numeric_limits<DT>::epsilon())
            scale_point_gradient(i) = 0.0;
        else
            scale_point_gradient(i) = scale_point_gradient(i) / point_diagonal_sqrt_(i);
    }

    double jg_norm_square = 0.0;
    for (size_t i = 0; i < ProjectionNum(); i++)
    {
        Mat26 pose_jacobian;
        Mat23 point_jacobian;
        Vec6 local_pose_gradient;
        Vec3 local_point_gradient;
        Vec2 local_jacobian_gradient;
        size_t pose_index = projection_block_.PoseIndex(i);
        size_t point_index = projection_block_.PointIndex(i);
        GetPoseJacobian(i, pose_jacobian);
        GetPointJacobian(i, point_jacobian);
        local_pose_gradient = scale_pose_gradient.segment(6 * pose_index, 6);
        local_point_gradient = scale_point_gradient.segment(3 * point_index, 3);
        local_jacobian_gradient = pose_jacobian * local_pose_gradient + point_jacobian * local_point_gradient;
        jg_norm_square += local_jacobian_gradient.squaredNorm();
    }

    double gradient_norm_square = pose_gradient_.squaredNorm() + point_gradient_.squaredNorm();
    alpha_ = gradient_norm_square / jg_norm_square;

    cauchy_step_.resize(PoseNum() * 6 + PointNum() * 3);
    cauchy_step_ << (-alpha_ * pose_gradient_), (-alpha_ * point_gradient_);
    return IsNumericalValid(cauchy_step_);
}

bool DLBAProblem::EvaluateGaussNewtonStep()
{
    gauss_newton_step_ = VecX::Zero(PoseNum() * 6 + PointNum() * 3);

    bool success = false;
    while (mu_ > min_mu_)
    {
        AugmentPoseDiagonal();
        AugmentPointDiagonal();
        EvaluateEcw();

        std::vector<size_t> pose_indexes(PoseNum());
        std::iota(pose_indexes.begin(), pose_indexes.end(), 0);
        if (!EvaluateDeltaPose(pose_indexes))
        {
            std::cout << "[EvaluateGaussNewtonStep] Fail in EvaluateDeltaPose\n"
                      << "Descrease mu_ from " << mu_  << " to " << 0.1 * mu_ << "\n";
            mu_ *= 0.1;
        }
        else
        {
            success = true;
            EvaluateDeltaPoint();
            break;
        }
    }

    if (success)
    {
        for (size_t i = 0; i < PoseNum(); i++)
        {
            Vec6 pose_update, pose_diagonal;
            pose_block_.GetDeltaPose(i, pose_update);
            pose_diagonal = pose_diagonal_sqrt_.segment(6 * i, 6);
            gauss_newton_step_.segment(6 * i, 6) = pose_update.cwiseProduct(pose_diagonal);
        }
        for (size_t i = 0; i < PointNum(); i++)
        {
            Vec3 point_update, point_diagonal;
            point_block_.GetDeltaPoint(i, point_update);
            point_diagonal = point_diagonal_sqrt_.segment(3 * i, 3);
            gauss_newton_step_.segment(6 * PoseNum() + 3 * i, 3) = point_update.cwiseProduct(point_diagonal);
        }
    }
    return success && IsNumericalValid(gauss_newton_step_);
}

bool DLBAProblem::EvaluateDogLegStep()
{
    const double gauss_newton_norm = gauss_newton_step_.norm();
    const double cauchy_norm = cauchy_step_.norm();

    if (gauss_newton_norm <= radius_)
    {
        // We credit he comments below to Ceres
        // Case 1. The Gauss-Newton step lies inside the trust region, and
        // is therefore the optimal solution to the trust-region problem.
        dl_step_ = gauss_newton_step_;
        dl_step_norm_ = gauss_newton_norm;
        std::cout << "Gauss-Newton step size: " << gauss_newton_norm << ", radius: " << radius_ << "\n";
    }
    else if (cauchy_norm >= radius_)
    {
        // Case 2. The Cauchy point and the Gauss-Newton steps lie outside
        // the trust region. Rescale the Cauchy point to the trust region
        // and return.
        dl_step_ = (radius_ / cauchy_norm) * cauchy_step_;
        dl_step_norm_ = radius_;
        std::cout << "Cauchy step size: " << dl_step_.norm() << ", radius: " << radius_ << "\n";
    }
    else
    {
        // Case 3. The Cauchy point is inside the trust region and the
        // Gauss-Newton step is outside. Compute the line joining the two
        // points and the point on it which intersects the trust region
        // boundary.
        // Below a means the Cauchy step, b means the Gauss-Newton step
        double a_square = cauchy_step_.squaredNorm();
        double c = cauchy_step_.dot(gauss_newton_step_ - cauchy_step_);
        double a_minus_b_square = (gauss_newton_step_ - cauchy_step_).squaredNorm();
        double d = std::sqrt(c * c -  a_minus_b_square * (a_square - radius_ * radius_));
        double beta = (d - c) / a_minus_b_square;
        std::cout << "[EvaluateDogLegStep] beta = " << beta << "\n";
        dl_step_ = (1.0 - beta) * cauchy_step_ + beta * gauss_newton_step_;
        dl_step_norm_ = dl_step_.norm();
        std::cout << "Dogleg step size: " << dl_step_.norm() << ", cauchy step size: " << cauchy_norm
                  << ", Gauss-Newton step size: " << gauss_newton_norm << ", radius: " << radius_ << "\n";
    }
    for (size_t i = 0; i < 6 * PoseNum(); i++)
    {
        if (pose_diagonal_sqrt_(i) < std::numeric_limits<DT>::epsilon())
            dl_step_(i) = 0.0;
        else
            dl_step_(i) = dl_step_(i) / pose_diagonal_sqrt_(i);
    }
    for (size_t i = 0; i < 3 * PointNum(); i++)
    {
        if (point_diagonal_sqrt_(i) < std::numeric_limits<DT>::epsilon())
            dl_step_(6 * PoseNum() + i) = 0.0;
        else
            dl_step_(6 * PoseNum() + i) =  dl_step_(6 * PoseNum() + i) / point_diagonal_sqrt_(i);
    }
    return IsNumericalValid(dl_step_);
}

bool DLBAProblem::EvaluateStep()
{
    if (evaluate_)
    {
        GetPoseDiagonal();
        GetPointDiagonal();
        EvaluateGradient();
        if (!EvaluateCauchyStep())
        {
            std::cout << "Fail in EvaluateCauchyStep.\n";
            return false;
        }
        if (!EvaluateGaussNewtonStep())
        {
            std::cout << "Fail in EvaluateGaussNewtonStep.\n";
            return false;
        }
    }
    EvaluateDogLegStep();

    for (size_t i = 0; i < PoseNum(); i++)
    {
        Vec6 dy = dl_step_.segment(6 * i, 6);
        pose_block_.SetDeltaPose(i, dy);
    }
    for (size_t i = 0; i < PointNum(); i++)
    {
        Vec3 dz = dl_step_.segment(6 * PoseNum() + 3 * i, 3);
        point_block_.SetDeltaPoint(i, dz);
    }
    return true;
}

void DLBAProblem::EvaluateRho()
{
    VecX diagonal = VecX::Zero(PoseNum() * 6 + PointNum() * 3);
    diagonal << pose_diagonal_, point_diagonal_;
    VecX aug_diagonal = diagonal / mu_;
    double change = last_square_error_ - square_error_;

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

bool DLBAProblem::StepAccept()
{
    return square_error_ < last_square_error_;
}

void DLBAProblem::Solve()
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
            EvaluateResidual();
            EvaluateJacobian();
            EvaluateJcJp();
            EvaluateJcJc();
            EvaluateJpJp();
            EvaluateJce();
            EvaluateJpe();
            ClearUpdate();
        }

        if (!EvaluateStep())
        {
            std::cout << "Fail in EvaluateStep.\n";
            step_accept_ = false;
        }
        else
        {
            square_error_ = EvaluateSquareError(true);
            if (StopCriterionRelativeCostChange() || StopCriterionUpdate())
                break;
            step_accept_ = StepAccept();
        }

        if (step_accept_)
        {
            EvaluateRho();
            Print();
            evaluate_ = true;
            if (rho_ < 0.25)
                radius_ *= 0.5;
            else if (rho_ > 0.75)
                radius_ = std::max(radius_, 3.0 * dl_step_norm_);
            mu_ = std::min(max_mu_, 5.0 * mu_);
            last_square_error_ = square_error_;
            UpdateParam();
        }
        else
        {
            Print();
            evaluate_ = false;
            radius_ /= 3.0;
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
}

void DLBAProblem::GetPoseDiagonal()
{
    size_t pose_num = PoseNum();
    pose_diagonal_.resize(6 * pose_num);
    pose_diagonal_sqrt_.resize(6 * pose_num);
    for (size_t i = 0; i < pose_num; i++)
    {
        pose_diagonal_(6 * i) = pose_jacobian_square_[6 * 6 * i];
        pose_diagonal_(6 * i + 1) = pose_jacobian_square_[6 * 6 * i + 7];
        pose_diagonal_(6 * i + 2) = pose_jacobian_square_[6 * 6 * i + 14];
        pose_diagonal_(6 * i + 3) = pose_jacobian_square_[6 * 6 * i + 21];
        pose_diagonal_(6 * i + 4) = pose_jacobian_square_[6 * 6 * i + 28];
        pose_diagonal_(6 * i + 5) = pose_jacobian_square_[6 * 6 * i + 35];
    }
    for (size_t i = 0; i < pose_diagonal_.rows(); i++)
        pose_diagonal_sqrt_(i) = std::sqrt(pose_diagonal_(i));
}

void DLBAProblem::GetPointDiagonal()
{
    size_t point_num = PointNum();
    point_diagonal_.resize(3 * point_num);
    point_diagonal_sqrt_.resize(3 * point_num);
    for (size_t i = 0; i < point_num; i++)
    {
        point_diagonal_(3 * i) = point_jacobian_square_[3 * 3 * i];
        point_diagonal_(3 * i + 1) = point_jacobian_square_[3 * 3 * i + 4];
        point_diagonal_(3 * i + 2) = point_jacobian_square_[3 * 3 * i + 8];
    }
    for (size_t i = 0; i < point_diagonal_.rows(); i++)
        point_diagonal_sqrt_(i) = std::sqrt(point_diagonal_(i));
}

void DLBAProblem::AugmentPoseDiagonal()
{
    assert(pose_diagonal_.rows() == PoseNum() * 6);
    VecX aug_pose_diagonal = pose_diagonal_ / mu_;
    SetPoseDiagonal(pose_diagonal_ + aug_pose_diagonal);
}

void DLBAProblem::ResetPoseDiagonal()
{
    SetPoseDiagonal(pose_diagonal_);
}

void DLBAProblem::AugmentPointDiagonal()
{
    assert(point_diagonal_.rows() == PointNum() * 3);
    VecX aug_point_diagonal = point_diagonal_ / mu_;
    SetPointDiagonal(point_diagonal_ + aug_point_diagonal);
}

void DLBAProblem::ResetPointDiagonal()
{
    SetPointDiagonal(point_diagonal_);
}

double DLBAProblem::Step() const
{
    VecX poses, points, delta_pose, delta_point;
    GetPose(poses);
    GetPoint(points);
    GetPoseUpdate(delta_pose);
    GetPointUpdate(delta_point);
    double relative_step = std::sqrt( (delta_pose.squaredNorm() + delta_point.squaredNorm()) / (poses.squaredNorm() + points.squaredNorm()) );
    return relative_step;
}

bool DLBAProblem::StopCriterionUpdate()
{
    double max_val = Step();

    if (max_val < 1e-8)
    {
        stream_ << "[StopCriterionUpdate] Relative change of parameters drops below 1e-8: " << max_val << "\n";
        std::cout << "[StopCriterionUpdate] Relative change of parameters drops below 1e-8: " << max_val << "\n";
    }

    return max_val < 1e-8;
}

bool DLBAProblem::StopCriterionRelativeCostChange()
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

void DLBAProblem::Print()
{
    double delta_loss = last_square_error_ - square_error_;
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
                 << "radius: " << std::setw(width) << radius_ << ", "
                 << "mu: " << std::setw(width) << mu_ << ", "
                 << "rho: " << std::setw(width) << rho_ << ", "
                 << std::setprecision(3) << std::fixed
                 << "me: " << std::setw(6) << median_error << ", "
                 << std::setprecision(1) << std::fixed
                 << "t: " << std::setw(5) << duration << "\n";
    std::cout << local_stream.str();
    stream_ << local_stream.str();
}
