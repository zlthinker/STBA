#ifndef LOSSFUNCTION_TPP
#define LOSSFUNCTION_TPP

#include "lossfunction.h"

template<size_t R, size_t C>
void LossFunction::CorrectJacobian(Vec2 const & residual,
                                   Matrix<DT, R, C, RowMajor> & jacobian) const
{
    size_t num_rows = jacobian.rows();  // #residuals
    size_t num_cols = jacobian.cols();  // #parameters
    assert(residual.size() == num_rows && "[CorrectJacobian] Dimension disagrees.");

    double sq_norm = residual.squaredNorm();
    if (sq_norm > MAX_VALID_ERROR)
    {
        jacobian = Matrix<DT, R, C, RowMajor>::Zero();
        return;
    }

    double rho1 = FirstOrderDerivative(sq_norm);
    double rho2 = SecondOrderDerivative(sq_norm);
    double sqrt_rho1 = std::sqrt(rho1);

    double residual_scaling, alpha_sq_norm;
    if ((sq_norm == 0.0) || (rho2 <= 0.0))
    {
        residual_scaling = sqrt_rho1;
        alpha_sq_norm = 0.0;
    }
    else
    {
        assert(rho1 > 0.0 && "[CorrectResiduals] First order derivative mush be positive");
        double D = 1.0 + 2.0 * sq_norm * rho2 / rho1;
        double alpha = 1.0 - std::sqrt(D);
        residual_scaling = sqrt_rho1 / (1.0 - alpha);
        alpha_sq_norm = alpha / sq_norm;
    }

    if (alpha_sq_norm == 0.0)
    {
        jacobian *= sqrt_rho1;
    }
    else
    {
        MatX residual_transpose_J = residual.transpose() * jacobian;
        MatX residual_square_J = residual * residual_transpose_J;
        residual_square_J *= alpha_sq_norm;
        jacobian = sqrt_rho1 * (jacobian - residual_square_J);
    }
}

#endif // LOSSFUNCTION_TPP

