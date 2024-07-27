/**
 * Created by Cu Cui on 2022/12/25.
 */

#ifndef LAPLACE_OPERATOR_CUH
#define LAPLACE_OPERATOR_CUH

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include "dealii/cuda_fee.cuh"
#include "dealii/cuda_mf.cuh"

using namespace dealii;

namespace PSMF
{
  template <int dim, int fe_degree, typename Number>
  class LaplaceOperatorQuad
  {
  public:
    const Number tau;

    __device__
    LaplaceOperatorQuad(const Number tau)
      : tau(tau)
    {}
    __device__ void
    operator()(
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> *fe_eval) const
    {
      fe_eval->submit_value(fe_eval->get_value());

      auto grad = fe_eval->get_gradient();
      for (unsigned int d = 0; d < dim; ++d)
        grad[d] *= tau;
      fe_eval->submit_gradient(grad);
    }
  };

  template <int dim, int fe_degree, typename Number>
  class LocalLaplaceOperator
  {
  public:
    static const unsigned int n_dofs_1d    = fe_degree + 1;
    static const unsigned int n_local_dofs = Utilities::pow(fe_degree + 1, dim);
    static const unsigned int n_q_points   = Utilities::pow(fe_degree + 1, dim);

    const unsigned int dofs_per_dim;
    const Number       tau;

    LocalLaplaceOperator(const unsigned int dofs_per_dim, const Number tau)
      : dofs_per_dim(dofs_per_dim)
      , tau(tau)
    {}

    __device__ void
    operator()(const unsigned int                            cell,
               const typename MatrixFree<dim, Number>::Data *gpu_data,
               SharedDataCuda<dim, Number>                  *shared_data,
               const Number                                 *src,
               Number                                       *dst) const
    {
      const unsigned int pos =
        local_q_point_id<dim, Number>(cell, gpu_data, n_dofs_1d, n_q_points);
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> fe_eval(
        cell, gpu_data, shared_data, dofs_per_dim);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true, true);
      fe_eval.apply_for_each_quad_point(
        LaplaceOperatorQuad<dim, fe_degree, Number>(tau));
      fe_eval.integrate(true, true);
      fe_eval.distribute_local_to_global(dst);
    }
  };

  template <int dim, int fe_degree, typename Number>
  class LocalMassOperator
  {
  public:
    static const unsigned int n_dofs_1d    = fe_degree + 1;
    static const unsigned int n_local_dofs = Utilities::pow(fe_degree + 1, dim);
    static const unsigned int n_q_points   = Utilities::pow(fe_degree + 1, dim);

    const unsigned int dofs_per_dim;

    LocalMassOperator(const unsigned int dofs_per_dim)
      : dofs_per_dim(dofs_per_dim)
    {}

    __device__ void
    operator()(const unsigned int                            cell,
               const typename MatrixFree<dim, Number>::Data *gpu_data,
               SharedDataCuda<dim, Number>                  *shared_data,
               const Number                                 *src,
               Number                                       *dst) const
    {
      const unsigned int pos =
        local_q_point_id<dim, Number>(cell, gpu_data, n_dofs_1d, n_q_points);
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> fe_eval(
        cell, gpu_data, shared_data, dofs_per_dim);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true, false);
      fe_eval.submit_value(fe_eval.get_value());
      __syncthreads();
      fe_eval.integrate(true, false);
      fe_eval.distribute_local_to_global(dst);
    }
  };

  template <int dim, int fe_degree, typename Number>
  class LocalRHSOperator
  {
  public:
    static const unsigned int n_dofs_1d    = fe_degree + 1;
    static const unsigned int n_local_dofs = Utilities::pow(fe_degree + 1, dim);
    static const unsigned int n_q_points   = Utilities::pow(fe_degree + 1, dim);

    const unsigned int dofs_per_dim;

    const Number wave_number;
    const Number tau;
    const Number a_t;
    const Number T;

    LocalRHSOperator(const unsigned int dofs_per_dim,
                     const Number       wave_number,
                     const Number       tau,
                     const Number       a_t,
                     const Number       T)
      : dofs_per_dim(dofs_per_dim)
      , wave_number(wave_number)
      , tau(tau)
      , a_t(a_t)
      , T(T)
    {}

    __device__ void
    operator()(const unsigned int                            cell,
               const typename MatrixFree<dim, Number>::Data *gpu_data,
               SharedDataCuda<dim, Number>                  *shared_data,
               const Number                                 *src,
               Number                                       *dst) const
    {
      (void)src;

      const auto point =
        get_quadrature_point<dim, Number>(cell, gpu_data, n_dofs_1d);
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> fe_eval(
        cell, gpu_data, shared_data, dofs_per_dim);

      Number val = exp(-a_t * T);
      for (unsigned int d = 0; d < dim; ++d)
        val *= sin(numbers::PI * point[d] * wave_number);
      val *=
        (numbers::PI * cos(numbers::PI * T) - a_t - a_t * sin(numbers::PI * T) +
         dim * numbers::PI * wave_number * numbers::PI * wave_number +
         dim * numbers::PI * wave_number * numbers::PI * wave_number *
           sin(numbers::PI * T));

      fe_eval.submit_value(val);
      __syncthreads();
      fe_eval.integrate(true, false);
      fe_eval.distribute_local_to_global(dst);
    }
  };



  template <int dim, int fe_degree, typename Number, DoFLayout dof_layout>
  class LaplaceOperator;

  template <int dim, int fe_degree, typename Number>
  class LaplaceOperator<dim, fe_degree, Number, DoFLayout::Q>
    : public Subscriptor
  {
  public:
    using value_type = Number;

    LaplaceOperator()
    {}

    void
    initialize(std::shared_ptr<const PSMF::MatrixFree<dim, Number>> data_,
               const Number                                         tau_,
               const Number wave_number_,
               const Number a_t_)
    {
      mf_data     = data_;
      tau         = tau_;
      wave_number = wave_number_;
      a_t         = a_t_;
    }

    void
    vmult(LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
          const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
            &src) const
    {
      dst = 0.;

      unsigned int level          = mf_data->get_mg_level();
      unsigned int n_dofs_per_dim = (1 << level) * fe_degree + 1;

      LocalLaplaceOperator<dim, fe_degree, Number> Laplace_operator(
        n_dofs_per_dim, tau);

      mf_data->cell_loop(Laplace_operator, src, dst);
      mf_data->copy_constrained_values(src, dst);
    }

    void
    vmult_mass(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>       &dst,
      const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &src)
      const
    {
      dst = 0.;

      unsigned int level          = mf_data->get_mg_level();
      unsigned int n_dofs_per_dim = (1 << level) * fe_degree + 1;

      LocalMassOperator<dim, fe_degree, Number> Laplace_operator(
        n_dofs_per_dim);

      mf_data->cell_loop(Laplace_operator, src, dst);
      mf_data->copy_constrained_values(src, dst);
    }

    void
    rhs_ops(LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
            const double T) const
    {
      dst      = 0.;
      auto src = dst;

      unsigned int level          = mf_data->get_mg_level();
      unsigned int n_dofs_per_dim = (1 << level) * fe_degree + 1;

      LocalRHSOperator<dim, fe_degree, Number> RHS_operator(
        n_dofs_per_dim, wave_number, tau, a_t, T);

      mf_data->cell_loop(RHS_operator, src, dst);
    }


    void
    Tvmult(LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
           const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
             &src) const
    {
      dst = 0.;
      LocalLaplaceOperator<dim, fe_degree, Number> Laplace_operator;
      mf_data->cell_loop(Laplace_operator, src, dst);
      mf_data->copy_constrained_values(src, dst);
    }

    void
    initialize_dof_vector(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &vec) const
    {
      mf_data->initialize_dof_vector(vec);
    }

    void
    compute_rhs(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &src,
      const double                                                   T) const
    {
      dst      = 0.;
      auto tmp = dst;

      rhs_ops(tmp, T);

      vmult_mass(dst, src);

      dst.add(tau, tmp);
    }

    void
    clear()
    {
      mf_data.reset();
    }

    std::shared_ptr<const PSMF::MatrixFree<dim, Number>>
    get_matrix_free() const
    {
      return mf_data;
    }

    std::size_t
    memory_consumption() const
    {
      std::size_t result = sizeof(*this);
      result += mf_data->memory_consumption();
      return result;
    }

  private:
    std::shared_ptr<const PSMF::MatrixFree<dim, Number>> mf_data;

    double tau;
    double wave_number;
    double a_t;
  };
} // namespace PSMF


#endif // LAPLACE_OPERATOR_CUH
