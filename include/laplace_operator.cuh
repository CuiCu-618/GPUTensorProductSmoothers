/**
 * @file laplace_operator.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief Implementation of the Laplace operations.
 * @version 1.0
 * @date 2023-02-02
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef LAPLACE_OPERATOR_CUH
#define LAPLACE_OPERATOR_CUH

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include "patch_base.cuh"

using namespace dealii;

namespace PSMF
{

  template <int dim, int fe_degree, typename Number, LaplaceVariant kernel>
  struct LocalLaplace
  {
    static constexpr unsigned int n_dofs_1d = 2 * fe_degree + 2;

    mutable std::size_t shared_mem;

    LocalLaplace()
      : shared_mem(0){};

    void
    setup_kernel(const unsigned int) const
    {
      constexpr unsigned int n =
        kernel == LaplaceVariant::ConflictFree ? 2 : (dim - 1);

      shared_mem = 0;

#if N_PATCH == 1
      unsigned int patch_per_block1 = 1;
#else
      unsigned int patch_per_block1 = 2;
#endif

      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      // local_src, local_dst
      shared_mem += 2 * patch_per_block1 * local_dim * sizeof(Number);
      // local_mass, local_derivative
      shared_mem +=
        2 * patch_per_block1 * n_dofs_1d * n_dofs_1d * 3 * sizeof(Number);
      // temp
      shared_mem += n * patch_per_block1 * local_dim * sizeof(Number);

      AssertCuda(cudaFuncSetAttribute(
        laplace_kernel_basic<dim, fe_degree, Number, kernel>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem));
    }

    template <typename VectorType, typename DataType>
    void
    loop_kernel(const VectorType &src,
                VectorType       &dst,
                const DataType   &gpu_data,
                const dim3       &grid_dim,
                const dim3       &block_dim) const
    {
      auto grid_dim1 = dim3((grid_dim.x + N_PATCH - 1) / N_PATCH);

      laplace_kernel_basic<dim, fe_degree, Number, kernel>
        <<<grid_dim1, block_dim, shared_mem>>>(src.get_values(),
                                               dst.get_values(),
                                               gpu_data);
    }
  };

  template <int dim, int fe_degree, typename Number>
  struct LocalLaplace<dim, fe_degree, Number, LaplaceVariant::BasicCell>
  {
    static constexpr unsigned int n_dofs_1d = 2 * fe_degree + 2;

    mutable std::size_t shared_mem;

    LocalLaplace()
      : shared_mem(0){};

    void
    setup_kernel(const unsigned int patch_per_block) const
    {
      constexpr unsigned int n = dim - 1;

      shared_mem = 0;

      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      // local_src, local_dst
      shared_mem += 2 * patch_per_block * local_dim * sizeof(Number);
      // local_mass, local_derivative
      shared_mem +=
        2 * patch_per_block * n_dofs_1d * n_dofs_1d * 3 * sizeof(Number);
      // temp
      shared_mem += n * patch_per_block * local_dim * sizeof(Number);

      AssertCuda(
        cudaFuncSetAttribute(laplace_kernel_basic_cell<dim,
                                                       fe_degree,
                                                       Number,
                                                       LaplaceVariant::Basic>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shared_mem));
    }

    template <typename VectorType, typename DataType>
    void
    loop_kernel(const VectorType &src,
                VectorType       &dst,
                const DataType   &gpu_data,
                const dim3       &grid_dim,
                const dim3       &block_dim) const
    {
      laplace_kernel_basic_cell<dim, fe_degree, Number, LaplaceVariant::Basic>
        <<<grid_dim, block_dim, shared_mem>>>(src.get_values(),
                                              dst.get_values(),
                                              gpu_data);
    }
  };

  template <int dim, int fe_degree, typename Number>
  struct LocalLaplace<dim, fe_degree, Number, LaplaceVariant::ConflictFreeMem>
  {
    static constexpr unsigned int n_dofs_1d = 2 * fe_degree + 2;

    mutable std::size_t shared_mem;

    LocalLaplace()
      : shared_mem(0)
    {
      Assert(fe_degree == 3 || fe_degree == 7, ExcNotImplemented());
    };

    void
    setup_kernel(const unsigned int patch_per_block) const
    {
      shared_mem = 0;

      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      // local_src, local_dst
      shared_mem += 2 * patch_per_block * local_dim * sizeof(Number);
      // local_mass, local_derivative
      shared_mem +=
        2 * patch_per_block * n_dofs_1d * n_dofs_1d * 3 * sizeof(Number);
      // temp
      shared_mem += 2 * patch_per_block * local_dim * sizeof(Number);

      AssertCuda(cudaFuncSetAttribute(
        laplace_kernel_cfmem<dim,
                             fe_degree,
                             Number,
                             LaplaceVariant::ConflictFreeMem>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem));
    }

    template <typename VectorType, typename DataType>
    void
    loop_kernel(const VectorType &src,
                VectorType       &dst,
                const DataType   &gpu_data,
                const dim3       &grid_dim,
                const dim3       &block_dim) const
    {
      laplace_kernel_cfmem<dim,
                           fe_degree,
                           Number,
                           LaplaceVariant::ConflictFreeMem>
        <<<grid_dim, block_dim, shared_mem>>>(src.get_values(),
                                              dst.get_values(),
                                              gpu_data);
    }
  };

  template <int dim, int fe_degree, typename Number>
  struct LocalLaplace<dim, fe_degree, Number, LaplaceVariant::TensorCore>
  {
    static constexpr unsigned int n_dofs_1d         = 2 * fe_degree + 2;
    static constexpr unsigned int n_dofs_1d_padding = n_dofs_1d + Util::padding;

    mutable std::size_t shared_mem;

    LocalLaplace()
      : shared_mem(0)
    {
      Assert(fe_degree == 3 || fe_degree == 7, ExcNotImplemented());
    };

    void
    setup_kernel(const unsigned int patch_per_block) const
    {
      shared_mem = 0;

      const unsigned int local_dim =
        Util::pow(n_dofs_1d, dim - 1) * n_dofs_1d_padding;
      // local_src, local_dst
      shared_mem += 2 * patch_per_block * local_dim * sizeof(Number);
      // local_mass, local_derivative
      shared_mem += 2 * patch_per_block * n_dofs_1d * n_dofs_1d_padding * 3 *
                    sizeof(Number);
      // temp
      shared_mem += (dim - 1) * patch_per_block * local_dim * sizeof(Number);

      AssertCuda(cudaFuncSetAttribute(
        laplace_kernel_tensorcore<dim,
                                  fe_degree,
                                  Number,
                                  LaplaceVariant::TensorCore>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem));
    }

    template <typename VectorType, typename DataType>
    void
    loop_kernel(const VectorType &src,
                VectorType       &dst,
                const DataType   &gpu_data,
                const dim3       &grid_dim,
                const dim3       &block_dim) const
    {
      laplace_kernel_tensorcore<dim,
                                fe_degree,
                                Number,
                                LaplaceVariant::TensorCore>
        <<<grid_dim, block_dim, shared_mem>>>(src.get_values(),
                                              dst.get_values(),
                                              gpu_data);
    }
  };


#if MMAKERNEL <= 2 || MMAKERNEL > 5
  template <int dim, int fe_degree, typename Number>
  struct LocalLaplace<dim, fe_degree, Number, LaplaceVariant::TensorCoreMMA>
  {
    static constexpr unsigned int n_dofs_1d = 2 * fe_degree + 2;

    mutable std::size_t shared_mem;
    mutable dim3        block_dim;

    LocalLaplace()
      : shared_mem(0)
    {
      Assert(fe_degree == 2 || fe_degree == 3 || fe_degree == 4 ||
               fe_degree == 6 || fe_degree == 7,
             ExcNotImplemented());
    };

    void
    setup_kernel(const unsigned int patch_per_block) const
    {
      shared_mem = 0;

      auto n_dofs_1d_p = fe_degree <= 3 ? 8 : 16;

      const unsigned int local_dim_p =
        Util::pow(n_dofs_1d_p, dim - 1) * n_dofs_1d;
      // local_src, local_dst
      shared_mem += 2 * patch_per_block * local_dim_p * sizeof(Number);
#if N_PATCH > 1
      // pipeline buffer
      shared_mem += patch_per_block * local_dim_p * sizeof(Number);
      shared_mem +=
        2 * patch_per_block * n_dofs_1d_p * n_dofs_1d_p * 3 * sizeof(Number);
#endif
      // local_mass, local_derivative
      shared_mem +=
        2 * patch_per_block * n_dofs_1d_p * n_dofs_1d_p * 3 * sizeof(Number);
      // temp
      shared_mem += 2 * patch_per_block * local_dim_p * sizeof(Number);

      AssertCuda(cudaFuncSetAttribute(
        laplace_kernel_tensorcoremma<dim,
                                     fe_degree,
                                     Number,
                                     LaplaceVariant::TensorCoreMMA>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem));

      block_dim = dim3(n_dofs_1d_p, n_dofs_1d_p, 1);
    }

    template <typename VectorType, typename DataType>
    void
    loop_kernel(const VectorType &src,
                VectorType       &dst,
                const DataType   &gpu_data,
                const dim3       &grid_dim,
                const dim3 &) const
    {
      auto grid_dim1 = dim3((grid_dim.x + N_PATCH - 1) / N_PATCH);

      laplace_kernel_tensorcoremma<dim,
                                   fe_degree,
                                   Number,
                                   LaplaceVariant::TensorCoreMMA>
        <<<grid_dim1, block_dim, shared_mem>>>(src.get_values(),
                                               dst.get_values(),
                                               gpu_data);
    }
  };
#endif

#if MMAKERNEL == 3 || MMAKERNEL == 4
  template <int dim, int fe_degree, typename Number>
  struct LocalLaplace<dim, fe_degree, Number, LaplaceVariant::TensorCoreMMA>
  {
    static constexpr unsigned int n_dofs_1d = 2 * fe_degree + 2;

    mutable std::size_t shared_mem;
    mutable dim3        block_dim;

    LocalLaplace()
      : shared_mem(0)
    {
      Assert(fe_degree == 3 || fe_degree == 7, ExcNotImplemented());
    };

    void
    setup_kernel(const unsigned int patch_per_block) const
    {
      shared_mem = 0;

      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      // local_src, local_dst
      shared_mem += 2 * patch_per_block * local_dim * sizeof(Number);
      // local_mass, local_derivative
      shared_mem +=
        2 * patch_per_block * n_dofs_1d * n_dofs_1d * 3 * sizeof(Number);
      // temp
      shared_mem += 2 * patch_per_block * local_dim * sizeof(Number);

      constexpr int n = fe_degree == 3 ? 2 : 1;

      block_dim = dim3(n_dofs_1d, n_dofs_1d * n, 1);

      AssertCuda(cudaFuncSetAttribute(
        laplace_kernel_tensorcoremma<dim,
                                     fe_degree,
                                     Number,
                                     LaplaceVariant::TensorCoreMMA>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem));
    }

    template <typename VectorType, typename DataType>
    void
    loop_kernel(const VectorType &src,
                VectorType       &dst,
                const DataType   &gpu_data,
                const dim3       &grid_dim,
                const dim3 &) const
    {
      laplace_kernel_tensorcoremma<dim,
                                   fe_degree,
                                   Number,
                                   LaplaceVariant::TensorCoreMMA>
        <<<grid_dim, block_dim, shared_mem>>>(src.get_values(),
                                              dst.get_values(),
                                              gpu_data);
    }
  };
#endif

#if MMAKERNEL == 3 || MMAKERNEL == 4
  template <int dim>
  struct LocalLaplace<dim, 7, float, LaplaceVariant::TensorCoreMMA>
  {
    static constexpr unsigned int n_dofs_1d = 2 * 7 + 2;

    mutable std::size_t shared_mem;
    mutable dim3        block_dim;

    LocalLaplace()
      : shared_mem(0){};

    void
    setup_kernel(const unsigned int patch_per_block) const
    {
      shared_mem = 0;

      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      // local_src, local_dst
      shared_mem += 2 * patch_per_block * local_dim * sizeof(float);
      // local_mass, local_derivative
      shared_mem +=
        2 * patch_per_block * n_dofs_1d * n_dofs_1d * 3 * sizeof(float);
      // temp
      shared_mem += 2 * patch_per_block * local_dim * sizeof(float);

      block_dim = dim3(n_dofs_1d, n_dofs_1d * 2, 1);

      AssertCuda(cudaFuncSetAttribute(
        laplace_kernel_tensorcoremma<dim,
                                     7,
                                     float,
                                     LaplaceVariant::TensorCoreMMA>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem));
    }

    template <typename VectorType, typename DataType>
    void
    loop_kernel(const VectorType &src,
                VectorType       &dst,
                const DataType   &gpu_data,
                const dim3       &grid_dim,
                const dim3 &) const
    {
      laplace_kernel_tensorcoremma<dim, 7, float, LaplaceVariant::TensorCoreMMA>
        <<<grid_dim, block_dim, shared_mem>>>(src.get_values(),
                                              dst.get_values(),
                                              gpu_data);
    }
  };
#endif

#if MMAKERNEL == 5
  template <int dim>
  struct LocalLaplace<dim, 3, double, LaplaceVariant::TensorCoreMMA>
  {
    static constexpr unsigned int n_dofs_1d = 2 * 3 + 2;

    mutable std::size_t shared_mem;
    mutable dim3        block_dim;

    LocalLaplace()
      : shared_mem(0){};

    void
    setup_kernel(const unsigned int patch_per_block) const
    {
      shared_mem = 0;

      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      // local_src, local_dst
      shared_mem += 2 * patch_per_block * local_dim * sizeof(double);
      // local_mass, local_derivative
      shared_mem +=
        2 * patch_per_block * n_dofs_1d * n_dofs_1d * 3 * sizeof(double);
      // temp
      shared_mem += 2 * patch_per_block * local_dim * sizeof(double);

      block_dim = dim3(n_dofs_1d, n_dofs_1d / 2, 1);

      AssertCuda(cudaFuncSetAttribute(
        laplace_kernel_tensorcoremma_s<dim,
                                       3,
                                       double,
                                       LaplaceVariant::TensorCoreMMA>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem));
    }

    template <typename VectorType, typename DataType>
    void
    loop_kernel(const VectorType &src,
                VectorType       &dst,
                const DataType   &gpu_data,
                const dim3       &grid_dim,
                const dim3 &) const
    {
      laplace_kernel_tensorcoremma_s<dim,
                                     3,
                                     double,
                                     LaplaceVariant::TensorCoreMMA>
        <<<grid_dim, block_dim, shared_mem>>>(src.get_values(),
                                              dst.get_values(),
                                              gpu_data);
    }
  };
#endif

#if MMAKERNEL == 5
  template <int dim>
  struct LocalLaplace<dim, 7, float, LaplaceVariant::TensorCoreMMA>
  {
    static constexpr unsigned int n_dofs_1d = 2 * 7 + 2;

    mutable std::size_t shared_mem;
    mutable dim3        block_dim;

    LocalLaplace()
      : shared_mem(0){};

    void
    setup_kernel(const unsigned int patch_per_block) const
    {
      shared_mem = 0;

      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      // local_src, local_dst
      shared_mem += 2 * patch_per_block * local_dim * sizeof(float);
      // local_mass, local_derivative
      shared_mem +=
        2 * patch_per_block * n_dofs_1d * n_dofs_1d * 3 * sizeof(float);
      // temp
      shared_mem += 2 * patch_per_block * local_dim * sizeof(float);

      block_dim = dim3(n_dofs_1d, n_dofs_1d / 2, 1);

      AssertCuda(cudaFuncSetAttribute(
        laplace_kernel_tensorcoremma_s<dim,
                                       7,
                                       float,
                                       LaplaceVariant::TensorCoreMMA>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem));
    }

    template <typename VectorType, typename DataType>
    void
    loop_kernel(const VectorType &src,
                VectorType       &dst,
                const DataType   &gpu_data,
                const dim3       &grid_dim,
                const dim3 &) const
    {
      laplace_kernel_tensorcoremma_s<dim,
                                     7,
                                     float,
                                     LaplaceVariant::TensorCoreMMA>
        <<<grid_dim, block_dim, shared_mem>>>(src.get_values(),
                                              dst.get_values(),
                                              gpu_data);
    }
  };
#endif

#if MMAKERNEL == 8
  template <int dim>
  struct LocalLaplace<dim, 7, float, LaplaceVariant::TensorCoreMMA>
  {
    static constexpr unsigned int n_dofs_1d = 2 * 7 + 2;

    mutable std::size_t shared_mem;
    mutable dim3        block_dim;

    LocalLaplace()
      : shared_mem(0){};

    void
    setup_kernel(const unsigned int patch_per_block) const
    {
      shared_mem = 0;

      const unsigned int local_dim = Util::pow(n_dofs_1d, dim);
      // local_src, local_dst
      shared_mem += 2 * patch_per_block * local_dim * sizeof(float);
      // local_mass, local_derivative
      shared_mem +=
        2 * patch_per_block * n_dofs_1d * n_dofs_1d * 3 * sizeof(float);
      // temp
      shared_mem += 2 * patch_per_block * local_dim * sizeof(float);

      block_dim = dim3(n_dofs_1d, n_dofs_1d * 2, 1);

      AssertCuda(cudaFuncSetAttribute(
        laplace_kernel_tensorcoremma<dim,
                                     7,
                                     float,
                                     LaplaceVariant::TensorCoreMMA>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem));
    }

    template <typename VectorType, typename DataType>
    void
    loop_kernel(const VectorType &src,
                VectorType       &dst,
                const DataType   &gpu_data,
                const dim3       &grid_dim,
                const dim3 &) const
    {
      laplace_kernel_tensorcoremma<dim, 7, float, LaplaceVariant::TensorCoreMMA>
        <<<grid_dim, block_dim, shared_mem>>>(src.get_values(),
                                              dst.get_values(),
                                              gpu_data);
    }
  };
#endif

  template <int dim, int fe_degree, typename Number, LaplaceVariant kernel>
  class LaplaceOperator : public Subscriptor
  {
  public:
    using value_type = Number;

    LaplaceOperator()
    {}

    void
    initialize(
      std::shared_ptr<const LevelVertexPatch<dim, fe_degree, Number>> data_,
      const DoFHandler<dim>                                          &mg_dof,
      const unsigned int                                              level)
    {
      data        = data_;
      dof_handler = &mg_dof;
      mg_level    = level;
    }

    void
    vmult(LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
          const LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
            &src) const
    {
      dst = 0.;

      LocalLaplace<dim, fe_degree, Number, kernel> local_laplace;

      data->cell_loop(local_laplace, src, dst);

      // mf_data->copy_constrained_values(src, dst);
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
      const unsigned int n_dofs = dof_handler->n_dofs(mg_level);
      vec.reinit(n_dofs);
    }

    void
    compute_residual(
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &dst,
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> &src,
      const Function<dim, Number> &rhs_function,
      const Function<dim, Number> &exact_solution,
      const unsigned int) const
    {
      const MappingQGeneric<dim> mapping(fe_degree);
      AffineConstraints<Number>  dummy;
      dummy.close();

      typename MatrixFree<dim, Number>::AdditionalData additional_data;
      additional_data.tasks_parallel_scheme =
        MatrixFree<dim, Number>::AdditionalData::none;
      additional_data.mapping_update_flags =
        (update_gradients | update_JxW_values | update_quadrature_points);
      additional_data.mapping_update_flags_inner_faces =
        (update_gradients | update_JxW_values | update_normal_vectors);
      additional_data.mapping_update_flags_boundary_faces =
        (update_gradients | update_JxW_values | update_normal_vectors |
         update_quadrature_points);

      MatrixFree<dim, Number> data;
      data.reinit(mapping,
                  *dof_handler,
                  dummy,
                  QGauss<1>(fe_degree + 1),
                  additional_data);

      dst = 0.;

      const unsigned int n_dofs = src.size();

      LinearAlgebra::distributed::Vector<Number, MemorySpace::Host>
        system_rhs_host(n_dofs);
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
        system_rhs_dev(n_dofs);

      LinearAlgebra::ReadWriteVector<Number> rw_vector(n_dofs);


      FEEvaluation<dim, fe_degree> phi(data);

      for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
        {
          phi.reinit(cell);
          for (const unsigned int q : phi.quadrature_point_indices())
            {
              VectorizedArray<Number> rhs_val = VectorizedArray<Number>();
              Point<dim, VectorizedArray<Number>> point_batch =
                phi.quadrature_point(q);
              for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
                {
                  Point<dim> single_point;
                  for (unsigned int d = 0; d < dim; ++d)
                    single_point[d] = point_batch[d][v];
                  rhs_val[v] = rhs_function.value(single_point);
                }
              phi.submit_value(rhs_val, q);
            }
          phi.integrate_scatter(EvaluationFlags::values, system_rhs_host);
        }

      FEFaceEvaluation<dim, fe_degree> phi_face(data, true);
      for (unsigned int face = data.n_inner_face_batches();
           face < data.n_inner_face_batches() + data.n_boundary_face_batches();
           ++face)
        {
          phi_face.reinit(face);

          const VectorizedArray<double> h_inner =
            1. / std::abs((phi_face.get_normal_vector(0) *
                           phi_face.inverse_jacobian(0))[dim - 1]);
          const auto one_over_h = (0.5 / h_inner) + (0.5 / h_inner);
          const auto gamma = fe_degree == 0 ? 1 : fe_degree * (fe_degree + 1);
          const VectorizedArray<double> sigma = 2.0 * gamma * one_over_h;

          for (const unsigned int q : phi_face.quadrature_point_indices())
            {
              VectorizedArray<Number> test_value = VectorizedArray<Number>(),
                                      test_normal_derivative =
                                        VectorizedArray<Number>();
              Point<dim, VectorizedArray<Number>> point_batch =
                phi_face.quadrature_point(q);

              for (unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
                {
                  Point<dim> single_point;
                  for (unsigned int d = 0; d < dim; ++d)
                    single_point[d] = point_batch[d][v];

                  if (data.get_boundary_id(face) == 0)
                    test_value[v] = 1.0 * exact_solution.value(single_point);
                  else
                    {
                      Tensor<1, dim> normal;
                      for (unsigned int d = 0; d < dim; ++d)
                        normal[d] = phi_face.get_normal_vector(q)[d][v];
                      test_normal_derivative[v] =
                        -normal * exact_solution.gradient(single_point);
                    }
                }
              phi_face.submit_value(test_value * sigma - test_normal_derivative,
                                    q);
              phi_face.submit_normal_derivative(-1.0 * test_value, q);
            }
          phi_face.integrate_scatter(EvaluationFlags::values |
                                       EvaluationFlags::gradients,
                                     system_rhs_host);
        }

      system_rhs_host.compress(VectorOperation::add);
      rw_vector.import(system_rhs_host, VectorOperation::insert);
      system_rhs_dev.import(rw_vector, VectorOperation::insert);

      vmult(dst, src);
      dst.sadd(-1., system_rhs_dev);
    }

    unsigned int
    get_mg_level() const
    {
      return mg_level;
    }

    const DoFHandler<dim> *
    get_dof_handler() const
    {
      return dof_handler;
    }

    std::size_t
    memory_consumption() const
    {
      std::size_t result = sizeof(*this);
      return result;
    }

  private:
    std::shared_ptr<const LevelVertexPatch<dim, fe_degree, Number>> data;
    const DoFHandler<dim>                                          *dof_handler;
    unsigned int                                                    mg_level;
  };
} // namespace PSMF


#endif // LAPLACE_OPERATOR_CUH
