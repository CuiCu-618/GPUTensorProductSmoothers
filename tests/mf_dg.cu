
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>

#include <helper_cuda.h>

#include <iostream>

#include "cuda_fe_evaluation.cuh"
#include "cuda_matrix_free.cuh"

// Matrix-free implementation with discontinous elements.

using namespace dealii;

template <int dim, int fe_degree, typename Number>
class LocalLaplaceOperator
{
public:
  static const unsigned int n_dofs_1d    = fe_degree + 1;
  static const unsigned int n_local_dofs = Utilities::pow(fe_degree + 1, dim);
  static const unsigned int n_q_points   = Utilities::pow(fe_degree + 1, dim);

  static const unsigned int cells_per_block =
    PSMF::cells_per_block_shmem(dim, fe_degree);


  LocalLaplaceOperator()
  {}

  __device__ void
  operator()(const unsigned int                                  cell,
             const typename PSMF::MatrixFree<dim, Number>::Data *gpu_data,
             PSMF::SharedData<dim, Number>                      *shared_data,
             const Number                                       *src,
             Number                                             *dst) const
  {
    PSMF::FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> fe_eval(
      cell, gpu_data, shared_data);
    fe_eval.read_dof_values(src);
    fe_eval.evaluate(false, true);
    fe_eval.submit_gradient(fe_eval.get_gradient());
    fe_eval.integrate(false, true);
    fe_eval.distribute_local_to_global(dst);
  }
};


template <int dim, int fe_degree, typename Number>
class LocalLaplaceFaceOperator
{
public:
  static const unsigned int n_dofs_1d    = fe_degree + 1;
  static const unsigned int n_local_dofs = Utilities::pow(fe_degree + 1, dim);
  static const unsigned int n_q_points   = Utilities::pow(fe_degree + 1, dim);

  static const unsigned int cells_per_block = 1;


  LocalLaplaceFaceOperator()
  {}

  __device__ Number
  get_penalty_factor() const
  {
    return 1.0 * fe_degree * (fe_degree + 1);
  }

  __device__ void
  operator()(const unsigned int                                  face,
             const typename PSMF::MatrixFree<dim, Number>::Data *gpu_data,
             PSMF::SharedData<dim, Number>                      *shared_data,
             const Number                                       *src,
             Number                                             *dst) const
  {
    PSMF::FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> fe_eval(
      face, gpu_data, shared_data, true);
    fe_eval.read_dof_values(src);
    fe_eval.evaluate(true, true);


    auto hi    = fabs(fe_eval.inverse_length_normal_to_face());
    auto sigma = hi * get_penalty_factor();

    // if (blockIdx.x == 1)
    //   {
    //     auto idx = PSMF::compute_index<dim, fe_degree + 1>();
    //     printf("%d: %.2f, %.2f\n", idx, hi, sigma);
    //   }

    auto u_inner                 = fe_eval.get_value();
    auto normal_derivative_inner = fe_eval.get_normal_derivative();
    auto test_by_value = 2 * u_inner * sigma - normal_derivative_inner;

    // if (blockIdx.x == 1)
    //   {
    //     __syncthreads();

    //     auto der = fe_eval.get_gradient();
    //     auto idx = PSMF::compute_index<dim, fe_degree + 1>();
    //     printf("%d: %.2f, %.2f, %.2f |  %.2f, %.2f\n",
    //            idx,
    //            u_inner,
    //            normal_derivative_inner,
    //            test_by_value,
    //            der[0],
    //            der[1]);
    //   }

    fe_eval.submit_normal_derivative(-u_inner);
    fe_eval.submit_value(test_by_value);

    // if (blockIdx.x == 1)
    //   {
    //     auto val = fe_eval.get_value();
    //     auto der = fe_eval.get_gradient();
    //     auto nor = fe_eval.get_normal_derivative();
    //     auto idx = PSMF::compute_index<dim, fe_degree + 1>();
    //     printf("%d: %.2f, %.2f |  %.2f, %.2f\n", idx, val, nor, der[0], der[1]);
    //   }

    fe_eval.integrate(true, true);
    fe_eval.distribute_local_to_global(dst);
  }
};


template <int dim, int fe_degree>
class LaplaceOperator
{
public:
  LaplaceOperator(const DoFHandler<dim>           &dof_handler,
                  const AffineConstraints<double> &constraints);

  void
  vmult(LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &dst,
        const LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>
          &src) const;

  void
  initialize_dof_vector(
    LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &vec) const;

private:
  PSMF::MatrixFree<dim, double> mf_data;
};



template <int dim, int fe_degree>
LaplaceOperator<dim, fe_degree>::LaplaceOperator(
  const DoFHandler<dim>           &dof_handler,
  const AffineConstraints<double> &constraints)
{
  MappingQ<dim>                                          mapping(fe_degree);
  typename PSMF::MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.mapping_update_flags =
    update_values | update_gradients | update_JxW_values;
  additional_data.mapping_update_flags_inner_faces =
    update_values | update_gradients | update_JxW_values |
    update_normal_vectors;
  additional_data.mg_level = 1;

  const QGauss<1> quad(fe_degree + 1);
  mf_data.reinit(mapping,
                 dof_handler,
                 constraints,
                 quad,
                 IteratorFilters::LocallyOwnedLevelCell(),
                 additional_data);
}


template <int dim, int fe_degree>
void
LaplaceOperator<dim, fe_degree>::vmult(
  LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>       &dst,
  const LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &src)
  const
{
  dst = 0.;
  // LocalLaplaceOperator<dim, fe_degree, double> laplace_operator;
  // mf_data.cell_loop(laplace_operator, src, dst);

  LocalLaplaceFaceOperator<dim, fe_degree, double> laplace_face_operator;
  mf_data.boundary_face_loop(laplace_face_operator, src, dst);

  // mf_data.copy_constrained_values(src, dst);
}



template <int dim, int fe_degree>
void
LaplaceOperator<dim, fe_degree>::initialize_dof_vector(
  LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &vec) const
{
  mf_data.initialize_dof_vector(vec);
}

template <int dim, int fe_degree>
void
test()
{
  Triangulation<dim> triangulation(
    Triangulation<dim>::limit_level_difference_at_vertices);
  GridGenerator::hyper_cube(triangulation, 0., 1.);
  triangulation.refine_global(3);

  FE_DGQ<dim>     fe(fe_degree);
  DoFHandler<dim> dof_handler(triangulation);
  MappingQ1<dim>  mapping;

  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs();

  AffineConstraints<double> level_constraints;
  level_constraints.close();

  LaplaceOperator<dim, fe_degree> laplace_operator(dof_handler,
                                                   level_constraints);

  LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> solution_dev;
  LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> system_rhs_dev;

  laplace_operator.initialize_dof_vector(solution_dev);
  system_rhs_dev.reinit(solution_dev);

  // system_rhs_dev = 1.;
  // laplace_operator.vmult(solution_dev, system_rhs_dev);

  for (unsigned int i = 0; i < system_rhs_dev.size(); ++i)
    {
      LinearAlgebra::ReadWriteVector<double> rw_vector(system_rhs_dev.size());
      rw_vector[i] = 1.;
      system_rhs_dev.import(rw_vector, VectorOperation::insert);

      laplace_operator.vmult(solution_dev, system_rhs_dev);

      solution_dev.print(std::cout);
    }
  // std::cout << solution_dev.l2_norm() << std::endl;
}

int
main(int argc, char *argv[])
{
  int device_id = findCudaDevice(argc, (const char **)argv);
  AssertCuda(cudaSetDevice(device_id));

  test<2, 2>();

  return 0;
}