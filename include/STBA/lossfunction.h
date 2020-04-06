#ifndef COSTFUNCTION_H
#define COSTFUNCTION_H

#include "utility.h"

#define MAX_VALID_ERROR 1e14
#define SCALE_FACTOR 2

using namespace Eigen;

class LossFunction
{
public:
    LossFunction() : a_(1.0), b_(1.0) {}
    LossFunction(DT a) : a_(a), b_(a * a) {}
    virtual ~LossFunction() {}
public:
    virtual DT Loss(DT) const = 0;
    virtual DT FirstOrderDerivative(DT error) const = 0;
    virtual DT SecondOrderDerivative(DT error) const = 0;

public:
    void CorrectResiduals(Vec2 & residual) const;
    template<size_t R, size_t C>
    void CorrectJacobian(Vec2 const & residual,
                         Matrix<DT, R, C, RowMajor> & jacobian) const;

protected:
    const DT a_;
    const DT b_;
};

class HuberLoss : public LossFunction
{
public:
    HuberLoss() : LossFunction(SCALE_FACTOR) {}
    HuberLoss(DT a) : LossFunction(a) {}

public:
    DT Loss(DT error) const;
    DT FirstOrderDerivative(DT error) const;
    DT SecondOrderDerivative(DT error) const;
};

class CauchyLoss : public LossFunction
{
public:
    CauchyLoss() : LossFunction(SCALE_FACTOR) {}
    CauchyLoss(DT a) : LossFunction(a) {}

public:
    DT Loss(DT error) const;
    DT FirstOrderDerivative(DT error) const;
    DT SecondOrderDerivative(DT error) const;
};

class NULLLoss : public LossFunction
{
public:
    NULLLoss() {}

public:
    DT Loss(DT error) const { return error > MAX_VALID_ERROR ? 0.0 : error; }
    DT FirstOrderDerivative(DT error) const { return 1.0; }
    DT SecondOrderDerivative(DT error) const { return 0.0; }
};

enum LossType
{
    NULLLossType = 0,
    HuberLossType = 1,
    CauchyLossType = 2
};

#include "lossfunction.tpp"

#endif // COSTFUNCTION_H
