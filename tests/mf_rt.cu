
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas_new.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/matrix_free/shape_info.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <helper_cuda.h>

#include <iostream>

#include "TPSS/move_to_deal_ii.h"
#include "cuda_fe_evaluation.cuh"
#include "cuda_matrix_free.cuh"
#include "renumber.h"

// Matrix-free implementation with Raviart-Thomas (RT) elements

using namespace dealii;

template <int dim, int fe_degree, typename Number>
class LocalLaplaceOperator
{
public:
  static const unsigned int n_dofs_1d = fe_degree + 2;
  static const unsigned int n_local_dofs =
    Utilities::pow(fe_degree + 2, dim) * dim;
  static const unsigned int n_q_points =
    Utilities::pow(fe_degree + 2, dim) * dim;

  static const unsigned int cells_per_block =
    PSMF::cells_per_block_shmem(dim, fe_degree);

  using value_type = typename PSMF::
    FEFaceEvaluation<dim, fe_degree, fe_degree + 2, 1, Number>::value_type;

  LocalLaplaceOperator(
    std::shared_ptr<const PSMF::MatrixFree<dim, Number>> mf_data_v,
    std::shared_ptr<const PSMF::MatrixFree<dim, Number>> mf_data_p)
  {
    n_dofs_v = mf_data_v->get_n_dofs();

    data_v = mf_data_v->get_data(0);
    data_p = mf_data_p->get_data(0);
  }


  __device__ void
  operator()(const unsigned int             cell,
             const unsigned int             color,
             PSMF::SharedData<dim, Number> *shared_data,
             const Number                  *src,
             Number                        *dst) const
  {
    PSMF::FEEvaluation<dim, fe_degree, fe_degree + 2, dim, Number> fe_eval(
      cell, data_v, shared_data);
    PSMF::FEEvaluation<dim, fe_degree, fe_degree + 2, 1, Number> fe_eval_p(
      cell, data_p, shared_data);

    fe_eval.read_dof_values(src);
    fe_eval.evaluate(false, true);

    auto grad_phi = fe_eval.get_gradient();

    value_type div_phi;
    div_phi[0] = -fe_eval.get_divergence();

    fe_eval_p.submit_value(div_phi);
    fe_eval_p.integrate(true, false);

    fe_eval_p.distribute_local_to_global(&dst[n_dofs_v]);


    fe_eval_p.read_dof_values(&src[n_dofs_v]);
    fe_eval_p.evaluate(true, false);

    Number phi = -fe_eval_p.get_value()[0];

    fe_eval.submit_divergence(phi);
    fe_eval.integrate(false, true);
    fe_eval.distribute_local_to_global(dst);

    fe_eval.submit_gradient(grad_phi);
    fe_eval.integrate(false, true);
    fe_eval.distribute_local_to_global(dst);


    // fe_eval.submit_value(fe_eval.get_value());
    // fe_eval.submit_divergence();
    // fe_eval.integrate(false, true);

    // fe_eval.distribute_local_to_global(dst);
  }

  unsigned int n_dofs_v;

  typename PSMF::MatrixFree<dim, Number>::Data data_v;
  typename PSMF::MatrixFree<dim, Number>::Data data_p;
};


template <int dim, int fe_degree, typename Number>
class LocalLaplaceBDOperator
{
public:
  static const unsigned int n_dofs_1d = fe_degree + 2;
  static const unsigned int n_local_dofs =
    Utilities::pow(fe_degree + 2, dim) * dim;
  static const unsigned int n_q_points =
    Utilities::pow(fe_degree + 2, dim) * dim;

  static const unsigned int cells_per_block = 1;

  using value_type = typename PSMF::
    FEFaceEvaluation<dim, fe_degree, fe_degree + 2, dim, Number>::value_type;

  LocalLaplaceBDOperator(
    std::shared_ptr<const PSMF::MatrixFree<dim, Number>> mf_data_v)
  {
    data_v = mf_data_v->template get_face_data<true>(0);
  }

  __device__ Number
  get_penalty_factor() const
  {
    return 1.0 * (fe_degree + 1) * (fe_degree + 2);
  }

  __device__ void
  operator()(const unsigned int             face,
             const unsigned int             color,
             PSMF::SharedData<dim, Number> *shared_data,
             const Number                  *src,
             Number                        *dst) const
  {
    PSMF::FEFaceEvaluation<dim, fe_degree, fe_degree + 2, dim, Number> fe_eval(
      face, data_v, shared_data, true);
    fe_eval.read_dof_values(src);
    fe_eval.evaluate(true, true);

    auto hi    = fabs(fe_eval.inverse_length_normal_to_face());
    auto sigma = hi * get_penalty_factor();

    auto u_inner                 = fe_eval.get_value();
    auto normal_derivative_inner = fe_eval.get_normal_derivative();

    // if (blockIdx.x == 0 && threadIdx.x == 0 && threadIdx.y == 0)
    //   printf("%f\n", sigma);

    // auto test_by_value = 2 * u_inner * sigma - normal_derivative_inner;
    value_type test_by_value;
    for (unsigned int d = 0; d < dim; ++d)
      {
        test_by_value[d] = 2. * u_inner[d] * sigma - normal_derivative_inner[d];
        u_inner[d]       = -u_inner[d];
      }

    fe_eval.submit_value(test_by_value);
    fe_eval.submit_normal_derivative(u_inner);

    fe_eval.integrate(true, true);
    fe_eval.distribute_local_to_global(dst);
  }

  typename PSMF::MatrixFree<dim, Number>::Data data_v;
};


template <int dim, int fe_degree, typename Number>
class LocalLaplaceFaceOperator
{
public:
  static const unsigned int n_dofs_1d = fe_degree + 2;
  static const unsigned int n_local_dofs =
    Utilities::pow(fe_degree + 2, dim) * dim * 2;
  static const unsigned int n_q_points =
    Utilities::pow(fe_degree + 2, dim) * dim * 2;

  static const unsigned int cells_per_block = 1;

  using value_type = typename PSMF::
    FEFaceEvaluation<dim, fe_degree, fe_degree + 2, dim, Number>::value_type;


  LocalLaplaceFaceOperator(
    std::shared_ptr<const PSMF::MatrixFree<dim, Number>> mf_data_v)
  {
    data_v = mf_data_v->template get_face_data<false>(0);
  }

  __device__ Number
  get_penalty_factor() const
  {
    return 1.0 * (fe_degree + 1) * (fe_degree + 2);
  }

  __device__ void
  operator()(const unsigned int             face,
             const unsigned int             color,
             PSMF::SharedData<dim, Number> *shared_data,
             const Number                  *src,
             Number                        *dst) const
  {
    PSMF::FEFaceEvaluation<dim, fe_degree, fe_degree + 2, dim, Number>
      phi_inner(face, data_v, shared_data, true);
    PSMF::FEFaceEvaluation<dim, fe_degree, fe_degree + 2, dim, Number>
      phi_outer(face, data_v, shared_data, false);

    phi_inner.read_dof_values(src);
    phi_inner.evaluate(true, true);

    phi_outer.read_dof_values(src);
    phi_outer.evaluate(true, true);

    auto hi    = 0.5 * (fabs(phi_inner.inverse_length_normal_to_face()) +
                     fabs(phi_outer.inverse_length_normal_to_face()));
    auto sigma = hi * get_penalty_factor();

    value_type solution_jump;
    value_type average_normal_derivative;
    value_type test_by_value;

    for (unsigned int d = 0; d < dim; ++d)
      {
        solution_jump[d] = phi_inner.get_value()[d] - phi_outer.get_value()[d];
        average_normal_derivative[d] =
          0.5 * (phi_inner.get_normal_derivative()[d] +
                 phi_outer.get_normal_derivative()[d]);
        test_by_value[d] =
          solution_jump[d] * sigma - average_normal_derivative[d];

        solution_jump[d] = -solution_jump[d] * 0.5;
      }


    phi_inner.submit_value(test_by_value);

    for (unsigned int d = 0; d < dim; ++d)
      test_by_value[d] *= -1.;

    phi_outer.submit_value(test_by_value);

    phi_inner.submit_normal_derivative(solution_jump);
    phi_outer.submit_normal_derivative(solution_jump);

    phi_inner.integrate(true, true);
    phi_inner.distribute_local_to_global(dst);

    phi_outer.integrate(true, true);
    phi_outer.distribute_local_to_global(dst);
  }

  typename PSMF::MatrixFree<dim, Number>::Data data_v;
};

template <int dim, int fe_degree>
class LaplaceOperator
{
public:
  LaplaceOperator(const DoFHandler<dim>           &dof_handler_v,
                  const DoFHandler<dim>           &dof_handler_p,
                  const AffineConstraints<double> &constraints);

  void
  vmult(
    LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &dst,
    LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &src) const;

  void
  initialize_dof_vector(
    LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &vec) const;

private:
  std::shared_ptr<PSMF::MatrixFree<dim, double>> mf_data_v;
  std::shared_ptr<PSMF::MatrixFree<dim, double>> mf_data_p;
};



template <int dim, int fe_degree>
LaplaceOperator<dim, fe_degree>::LaplaceOperator(
  const DoFHandler<dim>           &dof_handler_v,
  const DoFHandler<dim>           &dof_handler_p,
  const AffineConstraints<double> &constraints)
{
  MappingQ<dim>   mapping(fe_degree);
  const QGauss<1> quad(fe_degree + 2);

  typename PSMF::MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.mapping_update_flags =
    update_values | update_gradients | update_JxW_values | update_jacobians;
  // additional_data.mg_level    = 3;
  additional_data.matrix_type = PSMF::MatrixType::active_matrix;
  // additional_data.matrix_type = PSMF::MatrixType::edge_down_matrix;
  additional_data.my_id = 0;

  mf_data_p = std::make_shared<PSMF::MatrixFree<dim, double>>();
  mf_data_p->reinit(mapping,
                    dof_handler_p,
                    constraints,
                    quad,
                    IteratorFilters::LocallyOwnedCell(),
                    additional_data);

  additional_data.mapping_update_flags_inner_faces =
    update_values | update_gradients | update_JxW_values |
    update_normal_vectors | update_jacobians;
  additional_data.my_id = 1;

  mf_data_v = std::make_shared<PSMF::MatrixFree<dim, double>>();
  mf_data_v->reinit(mapping,
                    dof_handler_v,
                    constraints,
                    quad,
                    IteratorFilters::LocallyOwnedCell(),
                    additional_data);
}


template <int dim, int fe_degree>
void
LaplaceOperator<dim, fe_degree>::vmult(
  LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &dst,
  LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &src) const
{
  dst = 0.;

  LocalLaplaceOperator<dim, fe_degree, double> laplace_operator(mf_data_v,
                                                                mf_data_p);
  mf_data_v->cell_loop(laplace_operator, src, dst);

  LocalLaplaceBDOperator<dim, fe_degree, double> laplace_bd_operator(mf_data_v);
  mf_data_v->boundary_face_loop(laplace_bd_operator, src, dst);

  LocalLaplaceFaceOperator<dim, fe_degree, double> laplace_face_operator(
    mf_data_v);
  mf_data_v->inner_face_loop(laplace_face_operator, src, dst);

  mf_data_v->copy_constrained_values(src, dst);
}



template <int dim, int fe_degree>
void
LaplaceOperator<dim, fe_degree>::initialize_dof_vector(
  LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &vec) const
{
  mf_data_v->initialize_dof_vector(vec);
}

template <typename Tri, typename Dof>
void
output_mesh(Tri &tri, Dof &dof)
{
  int               degree = 2;
  const std::string filename_grid =
    "./grid_2D_Q" + std::to_string(degree) + ".gnuplot";
  std::ofstream out(filename_grid);
  out << "set terminal png" << std::endl
      << "set output 'grid_2D_Q" << std::to_string(degree) << ".png'"
      << std::endl
      << "plot '-' using 1:2 with lines, "
      << "'-' with labels point pt 2 offset 1,1" << std::endl;
  GridOut().write_gnuplot(tri, out);
  out << "e" << std::endl;

  std::map<types::global_dof_index, Point<2>> support_points;
  DoFTools::map_dofs_to_support_points(MappingQ1<2>(), dof, support_points);
  DoFTools::write_gnuplot_dof_support_point_info(out, support_points);
  out << "e" << std::endl;

  // std::ofstream out("mesh.vtu");

  // DataOut<2> data_out;
  // data_out.attach_dof_handler(dof);
  // data_out.build_patches();
  // data_out.write_vtu(out);
}


template <int dim, int fe_degree>
void
test()
{
  Triangulation<dim> triangulation(
    Triangulation<dim>::limit_level_difference_at_vertices);
  GridGenerator::hyper_cube(triangulation, 0., 1.);

  triangulation.refine_global(1);

  auto begin_cell = triangulation.begin_active();
  // begin_cell++;
  // begin_cell->set_refine_flag();
  // begin_cell++;
  begin_cell->set_refine_flag();
  // triangulation.execute_coarsening_and_refinement();

  // begin_cell = triangulation.begin_active();
  // begin_cell++;
  // begin_cell->set_refine_flag();
  // begin_cell++;
  // begin_cell->set_refine_flag();
  // triangulation.execute_coarsening_and_refinement();

  FE_DGQLegendre<dim>       fe_p(fe_degree);
  FE_RaviartThomas_new<dim> fe_v(fe_degree);

  FESystem<dim> fe(FE_RaviartThomas_new<dim>(fe_degree),
                   1,
                   FE_DGQLegendre<dim>(fe_degree),
                   1);

  DoFHandler<dim> dof_handler(triangulation);
  DoFHandler<dim> dof_handler_v(triangulation);
  DoFHandler<dim> dof_handler_p(triangulation);
  MappingQ1<dim>  mapping;

  dof_handler.distribute_dofs(fe);
  dof_handler_v.distribute_dofs(fe_v);
  dof_handler_p.distribute_dofs(fe_p);

  dof_handler.distribute_mg_dofs();
  dof_handler_v.distribute_mg_dofs();
  dof_handler_p.distribute_mg_dofs();

  AffineConstraints<double> constraints;
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler_v, constraints);

  VectorToolsFix::project_boundary_values_div_conforming(
    dof_handler_v,
    0,
    Functions::ZeroFunction<dim>(dim),
    0,
    constraints,
    MappingQ1<dim>());
  constraints.close();
  constraints.print(std::cout);

  LaplaceOperator<dim, fe_degree> laplace_operator(dof_handler_v,
                                                   dof_handler_p,
                                                   constraints);

  // return;

  LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> solution_dev;
  LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> system_rhs_dev;

  // laplace_operator.initialize_dof_vector(solution_dev);
  // system_rhs_dev.reinit(solution_dev);

  system_rhs_dev.reinit(dof_handler.n_dofs());
  solution_dev.reinit(dof_handler.n_dofs());

  // system_rhs_dev = 1.;
  // laplace_operator.vmult(solution_dev, system_rhs_dev);

  for (unsigned int i = 0; i < system_rhs_dev.size(); ++i)
    {
      LinearAlgebra::ReadWriteVector<double> rw_vector(system_rhs_dev.size());

      // for (unsigned int i = 0; i < system_rhs_dev.size(); ++i)
      rw_vector[i] = 1. + 0;

      system_rhs_dev.import(rw_vector, VectorOperation::insert);

      laplace_operator.vmult(solution_dev, system_rhs_dev);

      solution_dev.print(std::cout, 5, false);
      // std::cout << solution_dev.l2_norm() << std::endl;
      // if (i == 0)
      // break;
    }
  // std::cout << solution_dev.l2_norm() << std::endl;
}

int
main(int argc, char *argv[])
{
  int device_id = findCudaDevice(argc, (const char **)argv);
  AssertCuda(cudaSetDevice(device_id));

  test<3, 1>();

  return 0;
}