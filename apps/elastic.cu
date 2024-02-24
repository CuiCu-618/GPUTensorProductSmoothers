/**
 * @file poisson_adaptive.cu
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief Continuous Galerkin methods for Elastic problems with local refinement.
 * @version 1.0
 * @date 2023-02-02
 *
 * @copyright Copyright (c) 2023
 *
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/cuda.h>
#include <deal.II/base/function.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <helper_cuda.h>

#include <fstream>

#include "app_utilities.h"
#include "ct_parameter.h"
#include "cuda_fe_evaluation.cuh"
#include "cuda_matrix_free.cuh"
#include "utilities.cuh"

// double percision

namespace Step8
{
  using namespace dealii;


  template <int dim, int fe_degree, typename Number>
  class LocalElasticOperator
  {
  public:
    static const unsigned int n_dofs_1d = fe_degree + 1;
    static const unsigned int n_local_dofs =
      Utilities::pow(fe_degree + 1, dim) * dim;
    static const unsigned int n_q_points =
      Utilities::pow(fe_degree + 1, dim) * dim;

    static const unsigned int cells_per_block =
      PSMF::cells_per_block_shmem(dim, fe_degree);


    LocalElasticOperator()
    {}

    __device__ void
    operator()(const unsigned int                                  cell,
               const typename PSMF::MatrixFree<dim, Number>::Data *gpu_data,
               PSMF::SharedData<dim, Number>                      *shared_data,
               const Number                                       *src,
               Number                                             *dst) const
    {
      PSMF::FEEvaluation<dim, fe_degree, fe_degree + 1, dim, Number> fe_eval(
        cell, gpu_data, shared_data);

      fe_eval.read_dof_values(src);
      fe_eval.evaluate(false, true);

      const Number mu     = 1.;
      const Number lambda = 1.;

      auto grad_copy = fe_eval.get_gradient(); // (\mu\nabla u_i, \nabla v_j)
      auto gradij    = grad_copy; // (\mu\partial_i u_j, \partial_j u_i)

      // (\lambda\partial_i u_i, \partial_j u_j)
      Number val_ii = 0;
      for (unsigned int c0 = 0; c0 < dim; ++c0)
        val_ii += grad_copy[c0][c0] * lambda;

      for (unsigned int c0 = 0; c0 < dim; ++c0)
        for (unsigned int c1 = 0; c1 < dim; ++c1)
          grad_copy[c0][c1] =
            grad_copy[c0][c1] * mu +
            (c0 == c1 ? val_ii + gradij[c1][c0] * mu : gradij[c1][c0] * mu);

      fe_eval.submit_gradient(grad_copy);
      fe_eval.integrate(false, true);
      fe_eval.distribute_local_to_global(dst);
    }
  };


  template <int dim, int fe_degree>
  class ElasticOperator
  {
  public:
    ElasticOperator(const DoFHandler<dim>           &dof_handler,
                    const AffineConstraints<double> &constraints);

    void
    vmult(
      LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &dst,
      LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &src) const;

    void
    initialize_dof_vector(
      LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &vec) const;

  private:
    PSMF::MatrixFree<dim, double> mf_data;
  };



  template <int dim, int fe_degree>
  ElasticOperator<dim, fe_degree>::ElasticOperator(
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
    additional_data.matrix_type = PSMF::MatrixType::active_matrix;

    const QGauss<1> quad(fe_degree + 1);
    mf_data.reinit(mapping,
                   dof_handler,
                   constraints,
                   quad,
                   IteratorFilters::LocallyOwnedCell(),
                   additional_data);
  }


  template <int dim, int fe_degree>
  void
  ElasticOperator<dim, fe_degree>::vmult(
    LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &dst,
    LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &src) const
  {
    dst = 0.;

    LocalElasticOperator<dim, fe_degree, double> laplace_operator;
    mf_data.cell_loop(laplace_operator, src, dst);

    mf_data.copy_constrained_values(src, dst);
  }



  template <int dim, int fe_degree>
  void
  ElasticOperator<dim, fe_degree>::initialize_dof_vector(
    LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA> &vec) const
  {
    mf_data.initialize_dof_vector(vec);
  }


  template <int dim, int fe_degree>
  class ElasticProblem
  {
  public:
    using full_number = double;
    using MatrixFree  = PSMF::MatrixFree<dim, full_number>;

    ElasticProblem();
    ~ElasticProblem();
    void
    run(const unsigned int n_cycles);

  private:
    void
    setup_system();
    void
    assemble_rhs();
    void
    solve_cg();
    void
    refine_grid();
    void
    output_results(const unsigned int cycle);

    MPI_Comm                                  mpi_communicator;
    parallel::distributed::Triangulation<dim> triangulation;
    AffineConstraints<full_number>            constraints;
    FESystem<dim>                             fe;
    DoFHandler<dim>                           dof_handler;
    MappingQ<dim>                             mapping;
    double                                    setup_time;

    std::fstream                        fout;
    std::shared_ptr<ConditionalOStream> pcout;


    std::unique_ptr<ElasticOperator<dim, fe_degree>> system_matrix_dev;

    LinearAlgebra::distributed::Vector<full_number, MemorySpace::CUDA>
      solution_dev;
    LinearAlgebra::distributed::Vector<full_number, MemorySpace::CUDA>
      system_rhs_dev;
    LinearAlgebra::distributed::Vector<full_number, MemorySpace::Host>
      ghost_solution_host;
  };


  template <int dim>
  void
  right_hand_side(const std::vector<Point<dim>> &points,
                  std::vector<Tensor<1, dim>>   &values)
  {
    AssertDimension(values.size(), points.size());
    Assert(dim >= 2, ExcNotImplemented());

    Point<dim> point_1, point_2;
    point_1(0) = 0.5;
    point_2(0) = -0.5;

    for (unsigned int point_n = 0; point_n < points.size(); ++point_n)
      {
        if (((points[point_n] - point_1).norm_square() < 0.2 * 0.2) ||
            ((points[point_n] - point_2).norm_square() < 0.2 * 0.2))
          values[point_n][0] = 1.0;
        else
          values[point_n][0] = 0.0;

        if (points[point_n].norm_square() < 0.2 * 0.2)
          values[point_n][1] = 1.0;
        else
          values[point_n][1] = 0.0;
      }
  }


  template <int dim, int fe_degree>
  ElasticProblem<dim, fe_degree>::ElasticProblem()
    : mpi_communicator(MPI_COMM_WORLD)
    , triangulation(MPI_COMM_WORLD,
                    Triangulation<dim>::limit_level_difference_at_vertices,
                    parallel::distributed::Triangulation<
                      dim>::construct_multigrid_hierarchy)
    , fe(FE_Q<dim>(fe_degree), dim)
    , dof_handler(triangulation)
    , mapping(fe_degree)
    , setup_time(0.)
    , pcout(std::make_shared<ConditionalOStream>(std::cout, false))
  {
    const auto filename = Util::get_filename();
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      {
        fout.open(filename + ".log", std::ios_base::out);
        pcout = std::make_shared<ConditionalOStream>(
          fout, Utilities::MPI::this_mpi_process(mpi_communicator) == 0);
      }
  }

  template <int dim, int fe_degree>
  ElasticProblem<dim, fe_degree>::~ElasticProblem()
  {
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      fout.close();
  }

  template <int dim, int fe_degree>
  void
  ElasticProblem<dim, fe_degree>::setup_system()
  {
    Timer time;
    setup_time = 0;

    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();

    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<dim>(dim),
                                             constraints);
    constraints.close();

    setup_time += time.wall_time();

    *pcout << "DoF setup time:         " << setup_time << "s" << std::endl;


    time.restart();

    system_matrix_dev.reset(
      new ElasticOperator<dim, fe_degree>(dof_handler, constraints));

    system_matrix_dev->initialize_dof_vector(solution_dev);
    system_rhs_dev.reinit(solution_dev);

    ghost_solution_host.reinit(solution_dev.size());

    *pcout << "Matrix-free setup time: " << time.wall_time() << "s"
           << std::endl;
  }


  template <int dim, int fe_degree>
  void
  ElasticProblem<dim, fe_degree>::assemble_rhs()
  {
    LinearAlgebra::distributed::Vector<full_number, MemorySpace::Host>
      system_rhs_host(dof_handler.n_dofs());

    QGauss<dim> quadrature_formula(fe.degree + 1);

    FEValues<dim> fe_values(fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    const unsigned int n_q_points    = quadrature_formula.size();

    Vector<double> cell_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    std::vector<double> lambda_values(n_q_points);
    std::vector<double> mu_values(n_q_points);

    Functions::ConstantFunction<dim> lambda(1.), mu(1.);

    std::vector<Tensor<1, dim>> rhs_values(n_q_points);

    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_rhs = 0;

        fe_values.reinit(cell);

        lambda.value_list(fe_values.get_quadrature_points(), lambda_values);
        mu.value_list(fe_values.get_quadrature_points(), mu_values);
        right_hand_side(fe_values.get_quadrature_points(), rhs_values);

        for (const unsigned int i : fe_values.dof_indices())
          {
            const unsigned int component_i =
              fe.system_to_component_index(i).first;

            for (const unsigned int q_point :
                 fe_values.quadrature_point_indices())
              cell_rhs(i) += fe_values.shape_value(i, q_point) *
                             rhs_values[q_point][component_i] *
                             fe_values.JxW(q_point);
          }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(cell_rhs,
                                               local_dof_indices,
                                               system_rhs_host);
      }

    system_rhs_host.compress(VectorOperation::add);

    LinearAlgebra::ReadWriteVector<full_number> rw_vector(
      system_rhs_host.size());
    rw_vector.import(system_rhs_host, VectorOperation::insert);
    system_rhs_dev.import(rw_vector, VectorOperation::insert);
  }



  template <int dim, int fe_degree>
  void
  ElasticProblem<dim, fe_degree>::solve_cg()
  {
    PreconditionIdentity preconditioner;

    SolverControl solver_control(system_rhs_dev.size(), 1e-12);
    SolverCG<LinearAlgebra::distributed::Vector<full_number, MemorySpace::CUDA>>
      cg(solver_control);

    Timer time;

    cg.solve(*system_matrix_dev, solution_dev, system_rhs_dev, preconditioner);
    
    *pcout << "Solve CG time:          " << time.wall_time() << "s"
           << std::endl;

    *pcout << "  Solved in " << solver_control.last_step() << " iterations.\n\n";

    LinearAlgebra::ReadWriteVector<double> rw_vector(solution_dev.size());
    rw_vector.import(solution_dev, VectorOperation::insert);
    ghost_solution_host.import(rw_vector, VectorOperation::insert);

    constraints.distribute(ghost_solution_host);

    ghost_solution_host.update_ghost_values();
  }

  template <int dim, int fe_degree>
  void
  ElasticProblem<dim, fe_degree>::refine_grid()
  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    KellyErrorEstimator<dim>::estimate(dof_handler,
                                       QGauss<dim - 1>(fe.degree + 1),
                                       {},
                                       ghost_solution_host,
                                       estimated_error_per_cell);

    GridRefinement::refine_and_coarsen_fixed_number(triangulation,
                                                    estimated_error_per_cell,
                                                    0.3,
                                                    0.03);

    triangulation.execute_coarsening_and_refinement();
  }

  template <int dim, int fe_degree>
  void
  ElasticProblem<dim, fe_degree>::output_results(const unsigned int cycle)
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);

    std::vector<std::string> solution_names;
    switch (dim)
      {
        case 1:
          solution_names.emplace_back("displacement");
          break;
        case 2:
          solution_names.emplace_back("x_displacement");
          solution_names.emplace_back("y_displacement");
          break;
        case 3:
          solution_names.emplace_back("x_displacement");
          solution_names.emplace_back("y_displacement");
          solution_names.emplace_back("z_displacement");
          break;
        default:
          Assert(false, ExcNotImplemented());
      }

    data_out.add_data_vector(ghost_solution_host, solution_names);
    data_out.build_patches();

    std::ofstream output("solution-" + std::to_string(cycle) + ".vtk");
    data_out.write_vtk(output);
  }



  template <int dim, int fe_degree>
  void
  ElasticProblem<dim, fe_degree>::run(const unsigned int n_cycles)
  {
    *pcout << Util::generic_info_to_fstring() << std::endl;

    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
      {
        *pcout << "Cycle " << cycle << ':' << std::endl;

        if (cycle == 0)
          {
            GridGenerator::hyper_cube(triangulation, -1, 1);
            triangulation.refine_global(2);
          }
        else
          refine_grid();

        *pcout << "   Number of active cells:       "
               << triangulation.n_active_cells() << std::endl;

        setup_system();

        *pcout << "   Number of degrees of freedom: " << dof_handler.n_dofs()
               << std::endl;

        assemble_rhs();
        solve_cg();
        output_results(cycle);
      }
  }
} // namespace Step8
int
main(int argc, char *argv[])
{
  try
    {
      using namespace Step8;

      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      {
        int         n_devices       = 0;
        cudaError_t cuda_error_code = cudaGetDeviceCount(&n_devices);
        AssertCuda(cuda_error_code);
        const unsigned int my_mpi_id =
          Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
        const int device_id = my_mpi_id % n_devices;
        cuda_error_code     = cudaSetDevice(device_id);
        AssertCuda(cuda_error_code);
      }

      {
        ElasticProblem<CT::DIMENSION_, CT::FE_DEGREE_> Laplace_problem;
        Laplace_problem.run(4);
      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}