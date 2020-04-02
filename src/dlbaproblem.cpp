#include "STBA/dlbaproblem.h"

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

/**
 * @brief Gradient = J^T f
 * Diagonal = \sqrt{ Diag(J^T J) }
 */
void DLBAProblem::EvaluateGradientAndDiagonal()
{
    size_t const pose_num = PoseNum();
    size_t const group_num = GroupNum();
    size_t const point_num = PointNum();
    size_t const proj_num = ProjectionNum();
    pose_gradient_ = VecX::Zero(pose_num * 6);
    point_gradient_ = VecX::Zero(point_num * 3);
    scaled_pose_gradient_ = VecX::Zero(pose_num * 6);
    scaled_point_gradient_ = VecX::Zero(point_num * 3);
    pose_diagonal_ = VecX::Zero(pose_num * 6);
    point_diagonal_ = VecX::Zero(point_num * 3);
    if (!fix_intrinsic_)
    {
        intrinsic_gradient_ = VecX::Zero(group_num * 6);
        scaled_intrinsic_gradient_ = VecX::Zero(group_num * 6);
        intrinsic_diagonal_ = VecX::Zero(group_num * 6);
    }
  
#pragma omp parallel for
    for (size_t pidx = 0; pidx < proj_num; pidx++)
    {
        Vec2 residual;
        GetResidual(pidx, residual);
        
        size_t pose_index = projection_block_.PoseIndex(pidx);
        Mat26 pose_jacobian;
        GetPoseJacobian(pidx, pose_jacobian);
        pose_gradient_.segment(pose_index * 6, 6) += pose_jacobian.transpose() * residual;
        for (size_t i = 0; i < 6; i++)
        {
            pose_diagonal_(6 * pose_index + i) += pose_jacobian(0, i) * pose_jacobian(0, i) + pose_jacobian(1, i) * pose_jacobian(1, i);
        }
        
        size_t point_index = projection_block_.PointIndex(pidx);
        Mat23 point_jacobian;
        GetPointJacobian(pidx, point_jacobian);
        point_gradient_.segment(point_index * 3, 3) += point_jacobian.transpose() * residual;
        for (size_t i = 0; i < 3; i++)
        {
            point_diagonal_(3 * point_index + i) += point_jacobian(0, i) * point_jacobian(0, i) + point_jacobian(1, i) * point_jacobian(1, i);
        }
        
        if (!fix_intrinsic_)
        {
            size_t group_index = GetPoseGroup(pose_index);
            Mat26 intrinsic_jacobian;
            GetIntrinsicJacobian(pidx, intrinsic_jacobian);
            intrinsic_gradient_.segment(group_index * 6, 6) += intrinsic_jacobian.transpose() * residual;
            for (size_t i = 0; i < 6; i++)
            {
                intrinsic_diagonal_(6 * group_index + i) += intrinsic_jacobian(0, i) * intrinsic_jacobian(0, i) + intrinsic_jacobian(1, i) * intrinsic_jacobian(1, i);
            }
        }
    }
    
    for (size_t i = 0; i < pose_num * 6; i++)
    {
        pose_diagonal_(i) = std::sqrt(pose_diagonal_(i));
        scaled_pose_gradient_(i) = pose_gradient_(i) / pose_diagonal_(i);
    }
    for (size_t i = 0; i < point_num * 3; i++)
    {
        point_diagonal_(i) = std::sqrt(point_diagonal_(i));
        scaled_point_gradient_(i) = point_gradient_(i) / point_diagonal_(i);
    }
    if (!fix_intrinsic_)
    {
        for (size_t i = 0; i < group_num * 6; i++)
        {
            intrinsic_diagonal_(i) = std::sqrt(intrinsic_diagonal_(i));
            scaled_intrinsic_gradient_(i) = intrinsic_gradient_(i) / intrinsic_diagonal_(i);
        }
    }
}

/*!
 * @brief A cauchy step = -alpha * gradient. The gradient is scaled by the root of Hessian diagonal to make it scale-invariant
 */
bool DLBAProblem::EvaluateCauchyStep()
{
    double jg_norm_square = 0.0;
    for (size_t i = 0; i < ProjectionNum(); i++)
    {
        // Compute the product of jacobian and gradient.
        // Both jacobian and gradient are scaled by the diagnal, i.e.,
        // (J * D^-1) * (D^-1 * g) = J * (D^-1 * (D^-1 * g))
        Mat26 pose_jacobian;
        Mat23 point_jacobian;
        Vec6 local_pose_gradient;
        Vec3 local_point_gradient;
        Vec2 local_jacobian_gradient;
        size_t pose_index = projection_block_.PoseIndex(i);
        size_t point_index = projection_block_.PointIndex(i);
        GetPoseJacobian(i, pose_jacobian);
        GetPointJacobian(i, point_jacobian);
        local_pose_gradient = scaled_pose_gradient_.segment(6 * pose_index, 6);
        local_point_gradient = scaled_point_gradient_.segment(3 * point_index, 3);
        local_pose_gradient = local_pose_gradient.cwiseProduct(pose_diagonal_.segment(6 * pose_index, 6).cwiseInverse());
        local_point_gradient = local_point_gradient.cwiseProduct(point_diagonal_.segment(3 * point_index, 3).cwiseInverse());
        local_jacobian_gradient = pose_jacobian * local_pose_gradient + point_jacobian * local_point_gradient;
        if (!fix_intrinsic_)
        {
            size_t group_index = GetPoseGroup(pose_index);
            Vec6 local_intrinsic_gradient = scaled_intrinsic_gradient_.segment(6 * group_index, 6);
            local_intrinsic_gradient = local_intrinsic_gradient.cwiseProduct(intrinsic_diagonal_.segment(6 * group_index, 6).cwiseInverse());
            Mat26 intrinsic_jacobian;
            GetIntrinsicJacobian(i, intrinsic_jacobian);
            local_jacobian_gradient += intrinsic_jacobian * local_intrinsic_gradient;
        }
        jg_norm_square += local_jacobian_gradient.squaredNorm();
    }

    double gradient_norm_square = scaled_pose_gradient_.squaredNorm() + scaled_point_gradient_.squaredNorm() + (fix_intrinsic_ ? 0.0 : scaled_intrinsic_gradient_.squaredNorm());
    alpha_ = gradient_norm_square / jg_norm_square;

    if (fix_intrinsic_)
    {
        cauchy_step_.resize(PoseNum() * 6 + PointNum() * 3);
        cauchy_step_ << (-alpha_ * pose_gradient_), (-alpha_ * point_gradient_);
    }
    else
    {
        cauchy_step_.resize(PoseNum() * 6 + GroupNum() * 6 + PointNum() * 3);
        cauchy_step_ << (-alpha_ * pose_gradient_), (-alpha_ * intrinsic_gradient_), (-alpha_ * point_gradient_);
    }
    return IsNumericalValid(cauchy_step_);
}

/**
 * @brief The Gauss-Newton step is computed as -(J^T J)^-1 g.
 * However, the jacobian matrix J is often ill-conditioned and scale-variant, therefore, we add scaling to it by multiplying the root of Hessan diagonal, i.e.,
 * -(D^-1 J^T J D^-1)^-1 (D^-1 g) = -D (J^T J)^-1 D D^-1 g = -D (J^T J)^-1 g.
 * After computing the step -(J^T J)^-1 g as usual, we multiply it with D at last.
 */
bool DLBAProblem::EvaluateGaussNewtonStep()
{
    const size_t pose_num = PoseNum();
    const size_t group_num = fix_intrinsic_ ? 0 : GroupNum();
    const size_t point_num = PointNum();
    gauss_newton_step_ = VecX::Zero(pose_num * 6 + group_num * 6 + point_num * 3);

    // If solving the Gauss-Newton step fails, we reduce the trust region radius for robustness.
    bool success = false;
    while (mu_ > min_mu_)
    {
        if (!EvaluateCamera(1.0 / mu_))
        {
            std::cout << "[EvaluateGaussNewtonStep] Fail in EvaluateCamera\n"
                      << "Descrease mu_ from " << mu_  << " to " << 0.1 * mu_ << "\n";
            mu_ *= 0.1;
        }
        else
        {
            success = true;
            EvaluatePoint();
            break;
        }
    }

    if (success)
    {
        for (size_t i = 0; i < pose_num; i++)
        {
            Vec6 pose_update, pose_diagonal;
            pose_block_.GetDeltaPose(i, pose_update);
            pose_diagonal = pose_diagonal_.segment(6 * i, 6);
            gauss_newton_step_.segment(6 * i, 6) = pose_update.cwiseProduct(pose_diagonal);
        }
        for (size_t i = 0; i < group_num; i++)
        {
            Vec6 intrinisc_update, intrinsic_diagonal;
            intrinsic_block_.GetDeltaIntrinsic(i, intrinisc_update);
            intrinsic_diagonal = intrinsic_diagonal_.segment(6 * i, 6);
            gauss_newton_step_.segment(6 * pose_num + 6 * i, 6) = intrinisc_update.cwiseProduct(intrinsic_diagonal);
        }
        for (size_t i = 0; i < point_num; i++)
        {
            Vec3 point_update, point_diagonal;
            point_block_.GetDeltaPoint(i, point_update);
            point_diagonal = point_diagonal_.segment(3 * i, 3);
            gauss_newton_step_.segment(6 * (pose_num + group_num) + 3 * i, 3) = point_update.cwiseProduct(point_diagonal);
        }
    }
    return success && IsNumericalValid(gauss_newton_step_);
}

/**
 * @brief The DogLeg step is computed as the interpolation of the Cauchy step and the Gauss-Newton step
 */
bool DLBAProblem::EvaluateDogLegStep()
{
    const size_t pose_num = PoseNum();
    const size_t group_num = fix_intrinsic_ ? 0 : GroupNum();
    const size_t point_num = PointNum();
    const double gauss_newton_norm = gauss_newton_step_.norm();
    const double cauchy_norm = cauchy_step_.norm();
//    std::cout << "GN norm = " << gauss_newton_norm << ", cauchy norm = " << cauchy_norm << ", radius = " << radius_ << "\n";

    if (gauss_newton_norm <= radius_)
    {
        // We credit he comments below to Ceres
        // Case 1. The Gauss-Newton step lies inside the trust region, and
        // is therefore the optimal solution to the trust-region problem.
        dl_step_ = gauss_newton_step_;
        dl_step_norm_ = gauss_newton_norm;
//        std::cout << "Gauss-Newton step size: " << gauss_newton_norm << ", radius: " << radius_ << "\n";
    }
    else if (cauchy_norm >= radius_)
    {
        // Case 2. The Cauchy point and the Gauss-Newton steps lie outside
        // the trust region. Rescale the Cauchy point to the trust region
        // and return.
        dl_step_ = (radius_ / cauchy_norm) * cauchy_step_;
        dl_step_norm_ = radius_;
//        std::cout << "Cauchy step size: " << cauchy_norm << ", radius: " << radius_ << "\n";
    }
    else
    {
        // Case 3. The Cauchy point is inside the trust region and the
        // Gauss-Newton step is outside. Compute the line joining the two
        // points and the point on it which intersects the trust region
        // boundary.
        // Below a means the Cauchy step, b means the Gauss-Newton step
        double a_square = cauchy_norm * cauchy_norm;
        double c = cauchy_step_.dot(gauss_newton_step_ - cauchy_step_);
        double a_minus_b_square = (gauss_newton_step_ - cauchy_step_).squaredNorm();
        double d = std::sqrt(c * c -  a_minus_b_square * (a_square - radius_ * radius_));
        double beta = (d - c) / a_minus_b_square;
        dl_step_ = (1.0 - beta) * cauchy_step_ + beta * gauss_newton_step_;
        dl_step_norm_ = dl_step_.norm();
//        std::cout << "Dogleg step size: " << dl_step_.norm() << ", cauchy step size: " << cauchy_norm << ", Gauss-Newton step size: " << gauss_newton_norm << ", radius: " << radius_ << "\n";
    }
    
    // Since the Cauchy step and the Gauss-Newton steps computed before have been rescaled by the root of hessian diagonal, we remove the rescaling here.
    dl_step_.segment(0, 6 * pose_num) = dl_step_.segment(0, 6 * pose_num).cwiseProduct(pose_diagonal_.cwiseInverse());
    if (!fix_intrinsic_)
    {
        dl_step_.segment(6 * pose_num, 6 * group_num) = dl_step_.segment(6 * pose_num, 6 * group_num).cwiseProduct(intrinsic_diagonal_.cwiseInverse());
    }
    dl_step_.segment(6 * (pose_num + group_num), 3 * point_num) = dl_step_.segment(6 * (pose_num + group_num), 3 * point_num).cwiseProduct(point_diagonal_.cwiseInverse());
    
    return IsNumericalValid(dl_step_);
}

bool DLBAProblem::EvaluateStep()
{
    if (evaluate_)
    {
        EvaluateGradientAndDiagonal();
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
    if (!fix_intrinsic_)
    {
        for (size_t i = 0; i < GroupNum(); i++)
        {
            Vec6 di = dl_step_.segment(6 * PoseNum() + 6 * i, 6);
            intrinsic_block_.SetDeltaIntrinsic(i, di);
        }
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
    const size_t pose_num = PoseNum();
    const size_t group_num = fix_intrinsic_ ? 0 : GroupNum();
    const size_t point_num = PointNum();
    
    VecX diagonal = VecX::Zero(pose_num * 6 + group_num * 6 + point_num * 3);
    VecX delta(pose_num * 6 + group_num * 6 + point_num * 3);
    VecX gradient(pose_num * 6 + group_num * 6 + point_num * 3);
    
    VecX delta_pose, delta_intrinsic, delta_point;
    GetPoseUpdate(delta_pose);
    GetIntrinsicUpdate(delta_intrinsic);
    GetPointUpdate(delta_point);
    
    if (fix_intrinsic_)
    {
        diagonal << pose_diagonal_, point_diagonal_;
        delta << delta_pose, delta_point;
        gradient << pose_gradient_, point_gradient_;
    }
    else
    {
        diagonal << pose_diagonal_, intrinsic_diagonal_, point_diagonal_;
        delta << delta_pose, delta_intrinsic, delta_point;
        gradient << pose_gradient_, intrinsic_gradient_, point_gradient_;
    }
    VecX aug_diagonal = diagonal.array().square() / mu_;
    double change = last_square_error_ - square_error_;
    
    double delta_Je = delta.dot(gradient);  // d^T J^Te
    double delta_square = delta.dot(aug_diagonal.cwiseProduct(delta));  // d^T D d
    double model_change = (delta_square - delta_Je) * 0.5;
    rho_ = change / std::max(model_change, double(EPSILON));
    std::cout << "rho = " << rho_ << "\n";
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
            Print();
            evaluate_ = true;
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
    stream_ << "[Setting] Dogleg\n";
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
