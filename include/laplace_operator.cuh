/**
 * Created by Cu Cui on 2022/12/25.
 */

#ifndef LAPLACE_OPERATOR_CUH
#define LAPLACE_OPERATOR_CUH

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include "dealii/cuda_fee.cuh"
#include "dealii/cuda_mf.cuh"
#include "patch_base.cuh"

using namespace dealii;

namespace PSMF
{

  namespace internal
  {
#define BKSIZE_ELEMWISE_OP 512
#define CHUNKSIZE_ELEMWISE_OP 8

    template <typename Number, typename Number2>
    __global__ void
    vec_add_kernel(Number            *dst,
                   const Number2     *src,
                   const unsigned int N,
                   const Number       a,
                   const unsigned int shift,
                   const unsigned int shift2)
    {
      const unsigned int idx_base =
        threadIdx.x + blockIdx.x * (blockDim.x * CHUNKSIZE_ELEMWISE_OP);

      for (int c = 0; c < CHUNKSIZE_ELEMWISE_OP; ++c)
        {
          const int idx = idx_base + c * BKSIZE_ELEMWISE_OP;
          if (idx < N)
            dst[idx + shift] += a * src[idx + shift2];
        }
    }

    template <typename VectorType, typename VectorType2>
    void
    vec_add(VectorType                           &dst,
            const VectorType2                    &src,
            const unsigned int                    N,
            const typename VectorType::value_type a,
            const unsigned int                    shift,
            const unsigned int                    shift2 = 0)
    {
      const int nblocks =
        1 + (N - 1) / (CHUNKSIZE_ELEMWISE_OP * BKSIZE_ELEMWISE_OP);
      vec_add_kernel<typename VectorType::value_type,
                     typename VectorType2::value_type>
        <<<nblocks, BKSIZE_ELEMWISE_OP>>>(
          dst.get_values(), src.get_values(), N, a, shift, shift2);

      AssertCudaKernel();
    }
  } // namespace internal


  template <int dim, int fe_degree, typename Number>
  class LaplaceOperatorQuad
  {
  public:
    const Number lambda;
    const Number tau;

    __device__
    LaplaceOperatorQuad(const Number lambda, const Number tau)
      : lambda(lambda)
      , tau(tau)
    {}

    __device__ void
    operator()(
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> *fe_eval) const
    {
      fe_eval->submit_value(lambda * fe_eval->get_value());

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
    const Number       lambda;
    const Number       tau;

    LocalLaplaceOperator(const unsigned int dofs_per_dim,
                         const Number       lambda,
                         const Number       tau)
      : dofs_per_dim(dofs_per_dim)
      , lambda(lambda)
      , tau(tau)
    {}

    __device__ void
    operator()(const unsigned int                            cell,
               const typename MatrixFree<dim, Number>::Data *gpu_data,
               SharedDataCuda<dim, Number>                  *shared_data,
               const Number                                 *src,
               Number                                       *dst) const
    {
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> fe_eval(
        cell, gpu_data, shared_data, dofs_per_dim);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true, true);
      fe_eval.apply_for_each_quad_point(
        LaplaceOperatorQuad<dim, fe_degree, Number>(lambda, tau));
      fe_eval.integrate(true, true);
      fe_eval.distribute_local_to_global(dst);
    }
  };

  template <int dim, int fe_degree, typename Number>
  class LocalStiffnessOperator
  {
  public:
    static const unsigned int n_dofs_1d    = fe_degree + 1;
    static const unsigned int n_local_dofs = Utilities::pow(fe_degree + 1, dim);
    static const unsigned int n_q_points   = Utilities::pow(fe_degree + 1, dim);

    const unsigned int dofs_per_dim;

    LocalStiffnessOperator(const unsigned int dofs_per_dim)
      : dofs_per_dim(dofs_per_dim)
    {}

    __device__ void
    operator()(const unsigned int                            cell,
               const typename MatrixFree<dim, Number>::Data *gpu_data,
               SharedDataCuda<dim, Number>                  *shared_data,
               const Number                                 *src,
               Number                                       *dst) const
    {
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> fe_eval(
        cell, gpu_data, shared_data, dofs_per_dim);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(false, true);
      fe_eval.submit_gradient(fe_eval.get_gradient());
      __syncthreads();
      fe_eval.integrate(false, true);
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


  template <int dim, int fe_degree, typename Number>
  class SystemMatrixOp : public Subscriptor
  {
  public:
    using value_type = Number;
    using VectorType =
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>;

    SystemMatrixOp()
    {}

    void
    initialize(std::shared_ptr<const PSMF::MatrixFree<dim, Number>> data_,
               const std::vector<double>                           &A_inv_,
               const std::vector<double>                           &S_,
               const std::vector<double>                           &S_inv_,
               const std::vector<double>                           &b_vec_,
               const std::vector<double>                           &c_vec_,
               const Number                                         tau_,
               const Number wave_number_,
               const Number a_t_,
               const Number n_stages_)
    {
      mf_data = data_;

      A_inv = A_inv_;
      S     = S_;
      S_inv = S_inv_;
      b_vec = b_vec_;
      c_vec = c_vec_;

      tau         = tau_;
      wave_number = wave_number_;
      a_t         = a_t_;
      n_stages    = n_stages_;

      n_dofs = mf_data->get_dof_handler().n_dofs();
    }

    void
    vmult(VectorType &dst, const VectorType &src) const
    {
      AssertDimension(dst.size(), src.size());

      dst = 0.;

      VectorType tmp_src(n_dofs);
      VectorType tmp_dst(n_dofs);

      unsigned int level          = mf_data->get_mg_level();
      unsigned int n_dofs_per_dim = (1 << level) * fe_degree + 1;

      LocalStiffnessOperator<dim, fe_degree, Number> local_stiffness(
        n_dofs_per_dim);
      LocalMassOperator<dim, fe_degree, Number> local_mass(n_dofs_per_dim);

      for (unsigned int i = 0; i < n_stages; ++i)
        {
          tmp_src = 0;
          tmp_dst = 0;
          internal::vec_add(tmp_src, src, n_dofs, 1., 0, i * n_dofs);

          mf_data->cell_loop(local_stiffness, tmp_src, tmp_dst);
          mf_data->copy_constrained_values(tmp_src, tmp_dst);

          internal::vec_add(dst, tmp_dst, n_dofs, tau, i * n_dofs, 0);

          tmp_dst = 0;
          mf_data->cell_loop(local_mass, tmp_src, tmp_dst);
          mf_data->copy_constrained_values(tmp_src, tmp_dst);

          for (unsigned int j = 0; j < n_stages; ++j)
            internal::vec_add(
              dst, tmp_dst, n_dofs, A_inv[j * n_stages + i], j * n_dofs, 0);
        }
    }

    void
    Tvmult(VectorType &dst, const VectorType &src) const
    {
      vmult(dst, src);
    }

    void
    compute_system_rhs(VectorType &dst, const VectorType &src, const double t)
    {
      AssertDimension(dst.size(), n_stages * src.size());

      dst = 0.;

      VectorType tmp_dst(src.size());

      unsigned int level          = mf_data->get_mg_level();
      unsigned int n_dofs_per_dim = (1 << level) * fe_degree + 1;

      LocalStiffnessOperator<dim, fe_degree, Number> local_stiffness(
        n_dofs_per_dim);

      mf_data->cell_loop(local_stiffness, src, tmp_dst);
      mf_data->copy_constrained_values(src, tmp_dst);

      for (unsigned int i = 0; i < n_stages; ++i)
        {
          double scale = 0.;
          for (unsigned int j = 0; j < n_stages; ++j)
            scale += A_inv[i * n_stages + j];

          internal::vec_add(dst, tmp_dst, n_dofs, -scale, i * n_dofs, 0);
        }

      for (unsigned int i = 0; i < n_stages; ++i)
        {
          LocalRHSOperator<dim, fe_degree, Number> local_rhs(
            n_dofs_per_dim, wave_number, tau, a_t, t + c_vec[i] * tau);

          tmp_dst = 0;
          mf_data->cell_loop(local_rhs, src, tmp_dst);
          mf_data->copy_constrained_values(src, tmp_dst);

          for (unsigned int j = 0; j < n_stages; ++j)
            internal::vec_add(
              dst, tmp_dst, n_dofs, A_inv[j * n_stages + i], j * n_dofs, 0);
        }
    }

    void
    transform_basis(VectorType &dst, const VectorType &src) const
    {
      AssertDimension(dst.size(), src.size());

      dst = 0.;

      for (unsigned int i = 0; i < n_stages; ++i)
        for (unsigned int j = 0; j < n_stages; ++j)
          if (abs(S_inv[i * n_stages + j]) > 1e-12)
            internal::vec_add(dst,
                              src,
                              n_dofs,
                              S_inv[i * n_stages + j],
                              i * n_dofs,
                              j * n_dofs);
    }

    void
    transform_basis_back(VectorType &dst, const VectorType &src) const
    {
      AssertDimension(dst.size(), src.size());

      dst = 0.;

      for (unsigned int i = 0; i < n_stages; ++i)
        for (unsigned int j = 0; j < n_stages; ++j)
          if (abs(S[i * n_stages + j]) > 1e-12)
            internal::vec_add(
              dst, src, n_dofs, S[i * n_stages + j], i * n_dofs, j * n_dofs);
    }

    void
    compute_solution(VectorType &dst, const VectorType &src)
    {
      AssertDimension(dst.size() * n_stages, src.size());

      for (unsigned int i = 0; i < n_stages; ++i)
        internal::vec_add(dst, src, n_dofs, b_vec[i] * tau, 0, i * n_dofs);
    }

  private:
    std::shared_ptr<const PSMF::MatrixFree<dim, Number>> mf_data;

    std::vector<double> A_inv;
    std::vector<double> b_vec;
    std::vector<double> c_vec;
    std::vector<double> S;
    std::vector<double> S_inv;

    double tau;
    double wave_number;
    double a_t;

    double n_dofs;
    double n_stages;
  };



  template <int dim, int fe_degree, typename Number, DoFLayout dof_layout>
  class LaplaceOperator;

  template <int dim, int fe_degree, typename Number>
  class LaplaceOperator<dim, fe_degree, Number, DoFLayout::Q>
    : public Subscriptor
  {
  public:
    using value_type = Number;
    using VectorType =
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>;

    LaplaceOperator()
    {}

    void
    initialize(std::shared_ptr<const PSMF::MatrixFree<dim, Number>> data_,
               const std::vector<double>                           &D_vec_,
               const Number                                         tau_)
    {
      mf_data = data_;

      D_vec = D_vec_;
      tau   = tau_;
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
        n_dofs_per_dim, D_vec[current_stage], tau);

      mf_data->cell_loop(Laplace_operator, src, dst);
      mf_data->copy_constrained_values(src, dst);
    }

    void
    Tvmult(LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
           const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
             &src) const
    {
      vmult(dst, src);
    }

    void
    initialize_dof_vector(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &vec) const
    {
      mf_data->initialize_dof_vector(vec);
    }

    std::shared_ptr<const PSMF::MatrixFree<dim, Number>>
    get_matrix_free() const
    {
      return mf_data;
    }

  private:
    std::shared_ptr<const PSMF::MatrixFree<dim, Number>> mf_data;

    std::vector<double> D_vec;
    double              tau;
  };
} // namespace PSMF


#endif // LAPLACE_OPERATOR_CUH
