#include "STBA/lossfunction.h"

#include <iostream>

/*!
 * @brief Please read corrector.h/cc for reference
 */
void LossFunction::CorrectResiduals(Vec2 & residual) const
{
    DT sq_norm = residual.squaredNorm();
    DT rho1 = FirstOrderDerivative(sq_norm);
    DT rho2 = SecondOrderDerivative(sq_norm);
    DT sqrt_rho1 = std::sqrt(rho1);

    DT residual_scaling, alpha_sq_norm;
    if (sq_norm > MAX_VALID_ERROR)
    {
        residual_scaling = 0.0;
    }
    else if ((sq_norm == 0.0) || (rho2 <= 0.0))
    {
        residual_scaling = sqrt_rho1;
        alpha_sq_norm = 0.0;
    }
    else
    {
        assert(rho1 > 0.0 && "[CorrectResiduals] First order derivative mush be positive");
        DT D = 1.0 + 2.0 * sq_norm * rho2 / rho1;
        DT alpha = 1.0 - std::sqrt(D);
        residual_scaling = sqrt_rho1 / (1.0 - alpha);
        alpha_sq_norm = alpha / sq_norm;
    }
    residual *= residual_scaling;
}


/*!
 * @param error - the inrobust error term. For least square problems, it is the square of residual.
 */
DT HuberLoss::Loss(DT error) const
{
    assert(error >= 0 && "[Loss] Error term must be nonnegative");
    if (error > MAX_VALID_ERROR)
        return 0.0;
    else if (error > b_)
    {
        DT r = std::sqrt(error);
        return 2 * a_ * r - b_;
    }
    return error;
}

DT HuberLoss::FirstOrderDerivative(DT error) const
{
    assert(error >= 0 && "[FirstOrderDerivative] Error term must be nonnegative");
    if (error > b_)
    {
        DT r = std::sqrt(error);
        return a_ / r;
    }
    return 1.0;
}

DT HuberLoss::SecondOrderDerivative(DT error) const
{
    assert(error >= 0 && "[FirstOrderDerivative] Error term must be nonnegative");
    if (error > b_)
    {
        DT r = std::sqrt(error);
        return -a_ / (2 * error * r);
    }
    return 0.0;
}

DT CauchyLoss::Loss(DT error) const
{
    assert(error >= 0 && "[Loss] Error term must be nonnegative");
    if (error > MAX_VALID_ERROR)
        return 0.0;
    return b_ * log(1 + error / b_);
}

DT CauchyLoss::FirstOrderDerivative(DT error) const
{
    assert(error >= 0 && "[FirstOrderDerivative] Error term must be nonnegative");
    return b_ / (b_ + error);
}

DT CauchyLoss::SecondOrderDerivative(DT error) const
{
    assert(error >= 0 && "[FirstOrderDerivative] Error term must be nonnegative");
    return -b_ / ((b_ + error) * (b_ + error));
}


