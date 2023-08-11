/**
 * @file poisson.cu
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief Discontinuous Galerkin methods for poisson problems.
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
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_raviart_thomas_new.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <helper_cuda.h>

#include <fstream>

#include "TPSS/move_to_deal_ii.h"
#include "app_utilities.h"
#include "ct_parameter.h"
#include "equation_data.h"
#include "solver.cuh"
#include "utilities.cuh"

// -\delta u = f, u = 0 on \parital \Omege, f = 1.
// double percision

namespace Step64
{
  using namespace dealii;

  template <int dim>
  using Solution = Stokes::NoSlipExp::Solution<dim>;

  template <int dim>
  using SolutionVelocity = Stokes::NoSlipExp::SolutionVelocity<dim>;

  template <int dim>
  using SolutionPressure = Stokes::NoSlipExp::SolutionPressure<dim>;

  template <int dim>
  using RightHandSide = Stokes::ManufacturedLoad<dim>;

  template <int dim, int fe_degree>
  class LaplaceProblem
  {
  public:
    using full_number   = double;
    using vcycle_number = CT::VCYCLE_NUMBER_;
    using MatrixFreeDP  = PSMF::LevelVertexPatch<dim, fe_degree, full_number>;
    using MatrixFreeSP  = PSMF::LevelVertexPatch<dim, fe_degree, vcycle_number>;

    LaplaceProblem();
    ~LaplaceProblem();
    void
    run(const unsigned int n_cycles);

  private:
    void
    setup_system();
    void
    assemble_mg();
    void
    solve_mg(unsigned int n_mg_cycles);
    std::tuple<double, double, double>
    compute_error();

    template <PSMF::LocalSolverVariant local_solver,
              PSMF::LaplaceVariant     laplace,
              PSMF::LaplaceVariant     smooth_vmult,
              PSMF::SmootherVariant    smooth_inv>
    void
    do_solve(unsigned int k, unsigned int call_count);

    Triangulation<dim>                  triangulation;
    std::shared_ptr<FiniteElement<dim>> fe;
    DoFHandler<dim>                     dof_handler;
    DoFHandler<dim>                     dof_handler_velocity;
    DoFHandler<dim>                     dof_handler_pressure;
    MappingQ1<dim>                      mapping;

    double setup_time;

    std::vector<ConvergenceTable> info_table;

    std::fstream                        fout;
    std::shared_ptr<ConditionalOStream> pcout;

    MGLevelObject<std::shared_ptr<MatrixFreeDP>> mfdata_dp;
    MGLevelObject<std::shared_ptr<MatrixFreeSP>> mfdata_sp;
    MGConstrainedDoFs                            mg_constrained_dofs;
    AffineConstraints<double>                    constraints;

    LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
      solution_velocity_host;
    LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
      solution_pressure_host;


    PSMF::MGTransferCUDA<dim, vcycle_number> transfer;
  };

  template <int dim, int fe_degree>
  LaplaceProblem<dim, fe_degree>::LaplaceProblem()
    : triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
    , fe([&]() -> std::shared_ptr<FiniteElement<dim>> {
      if (CT::DOF_LAYOUT_ == PSMF::DoFLayout::Q)
        return std::make_shared<FE_Q<dim>>(fe_degree);
      else if (CT::DOF_LAYOUT_ == PSMF::DoFLayout::DGQ)
        return std::make_shared<FE_DGQHermite<dim>>(fe_degree);
      else if (CT::DOF_LAYOUT_ == PSMF::DoFLayout::RT)
        return std::make_shared<FESystem<dim>>(FE_RaviartThomas_new<dim>(
                                                 fe_degree),
                                               1,
                                               FE_DGQLegendre<dim>(fe_degree),
                                               1);
      return std::shared_ptr<FiniteElement<dim>>();
    }())
    , dof_handler(triangulation)
    , dof_handler_velocity(triangulation)
    , dof_handler_pressure(triangulation)
    , setup_time(0.)
    , pcout(std::make_shared<ConditionalOStream>(std::cout, false))
  {
    const auto filename = Util::get_filename();
    fout.open(filename + ".log", std::ios_base::out);
    pcout = std::make_shared<ConditionalOStream>(fout, true);

    info_table.resize(CT::LOCAL_SOLVER_.size());
  }

  template <int dim, int fe_degree>
  LaplaceProblem<dim, fe_degree>::~LaplaceProblem()
  {
    fout.close();
  }

  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::setup_system()
  {
    Timer time;
    setup_time = 0;

    dof_handler_velocity.distribute_dofs(fe->get_sub_fe(0, dim));
    dof_handler_velocity.distribute_mg_dofs();

    dof_handler_pressure.distribute_dofs(fe->get_sub_fe(dim, 1));
    dof_handler_pressure.distribute_mg_dofs();

    dof_handler.distribute_dofs(*fe);
    dof_handler.distribute_mg_dofs();

    *pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << " = ("
           << dof_handler_velocity.n_dofs() << " + "
           << dof_handler_pressure.n_dofs() << ")" << std::endl;

    constraints.clear();
    VectorToolsFix::project_boundary_values_div_conforming(
      dof_handler_velocity,
      0,
      SolutionVelocity<dim>(),
      0,
      constraints,
      mapping);
    constraints.close();

    setup_time += time.wall_time();

    *pcout << "DoF setup time:         " << setup_time << "s" << std::endl;
  }
  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::assemble_mg()
  {
    // Initialization of Dirichlet boundaries
    std::set<types::boundary_id> dirichlet_boundary;
    dirichlet_boundary.insert(0);
    mg_constrained_dofs.initialize(dof_handler_velocity);
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler_velocity,
                                                       dirichlet_boundary);

    // set up a mapping for the geometry representation
    MappingQ1<dim> mapping;

    unsigned int minlevel = 1;
    unsigned int maxlevel = triangulation.n_global_levels() - 1;

    mfdata_dp.resize(1, maxlevel);

    if (std::is_same_v<vcycle_number, float>)
      mfdata_sp.resize(1, maxlevel);

    Timer time;
    for (unsigned int level = minlevel; level <= maxlevel; ++level)
      {
        // IndexSet relevant_dofs;
        // DoFTools::extract_locally_relevant_level_dofs(dof_handler,
        //                                               level,
        //                                               relevant_dofs);
        // double-precision matrix-free data
        {
          // AffineConstraints<full_number> level_constraints;
          // level_constraints.reinit(relevant_dofs);
          // level_constraints.add_lines(
          //   mg_constrained_dofs.get_boundary_indices(level));
          // level_constraints.close();

          typename MatrixFreeDP::AdditionalData additional_data;
          additional_data.relaxation         = 1.;
          additional_data.use_coloring       = false;
          additional_data.patch_per_block    = CT::PATCH_PER_BLOCK_;
          additional_data.granularity_scheme = CT::GRANULARITY_;

          mfdata_dp[level] = std::make_shared<MatrixFreeDP>();
          mfdata_dp[level]->reinit(dof_handler_velocity,
                                   dof_handler_pressure,
                                   mg_constrained_dofs,
                                   level,
                                   additional_data);
        }

        // single-precision matrix-free data
        if (std::is_same_v<vcycle_number, float>)
          {
            // AffineConstraints<vcycle_number> level_constraints;
            // level_constraints.reinit(relevant_dofs);
            // level_constraints.add_lines(
            //   mg_constrained_dofs.get_boundary_indices(level));
            // level_constraints.close();

            typename MatrixFreeSP::AdditionalData additional_data;
            additional_data.relaxation         = 1.;
            additional_data.use_coloring       = false;
            additional_data.patch_per_block    = CT::PATCH_PER_BLOCK_;
            additional_data.granularity_scheme = CT::GRANULARITY_;

            mfdata_sp[level] = std::make_shared<MatrixFreeSP>();
            mfdata_sp[level]->reinit(dof_handler_velocity,
                                     dof_handler_pressure,
                                     mg_constrained_dofs,
                                     level,
                                     additional_data);
          }
      }

    *pcout << "Matrix-free setup time: " << time.wall_time() << "s"
           << std::endl;

    time.restart();

    transfer.initialize_constraints(mg_constrained_dofs);
    transfer.build(dof_handler_velocity, dof_handler_pressure);

    *pcout << "MG transfer setup time: " << time.wall_time() << "s"
           << std::endl;
  }

  template <int dim, int fe_degree>
  template <PSMF::LocalSolverVariant local_solver,
            PSMF::LaplaceVariant     laplace,
            PSMF::LaplaceVariant     smooth_vmult,
            PSMF::SmootherVariant    smooth_inv>
  void
  LaplaceProblem<dim, fe_degree>::do_solve(unsigned int k,
                                           unsigned int call_count)
  {
    PSMF::MultigridSolver<dim,
                          fe_degree,
                          full_number,
                          local_solver,
                          laplace,
                          smooth_vmult,
                          smooth_inv,
                          vcycle_number>
      solver(dof_handler,
             dof_handler_velocity,
             mfdata_dp,
             mfdata_sp,
             transfer,
             SolutionVelocity<dim>(),
             RightHandSide<dim>(std::make_shared<Solution<dim>>()),
             pcout,
             1);

    *pcout << "\nMG with [" << LaplaceToString(CT::LAPLACE_TYPE_[0]) << " "
           << LaplaceToString(CT::SMOOTH_VMULT_[0]) << " "
           << SmootherToString(CT::SMOOTH_INV_[0]) << " "
           << LocalSolverToString(CT::LOCAL_SOLVER_[k]) << "]\n";

    unsigned int index = k;

    info_table[index].add_value("level", triangulation.n_global_levels());
    info_table[index].add_value("cells", triangulation.n_global_active_cells());
    info_table[index].add_value("dofs", dof_handler.n_dofs());
    info_table[index].add_value("dofs_v", dof_handler_velocity.n_dofs());
    info_table[index].add_value("dofs_p", dof_handler_pressure.n_dofs());

    std::vector<PSMF::SolverData> comp_data = solver.static_comp();
    for (auto &data : comp_data)
      {
        *pcout << data.print_comp();

        auto times = data.solver_name + "[s]";
        auto perfs = data.solver_name + "Perf[Dof/s]";

        info_table[index].add_value(times, data.timing);
        info_table[index].add_value(perfs, data.perf);

        if (call_count == 0)
          {
            info_table[index].set_scientific(times, true);
            info_table[index].set_precision(times, 3);
            info_table[index].set_scientific(perfs, true);
            info_table[index].set_precision(perfs, 3);

            info_table[index].add_column_to_supercolumn(times,
                                                        data.solver_name);
            info_table[index].add_column_to_supercolumn(perfs,
                                                        data.solver_name);
          }
      }

    *pcout << std::endl;

    std::vector<PSMF::SolverData> solver_data = solver.solve();
    for (auto &data : solver_data)
      {
        *pcout << data.print_solver();

        auto it    = data.solver_name + "it";
        auto step  = data.solver_name + "step";
        auto times = data.solver_name + "[s]";
        auto mem   = data.solver_name + "Mem Usage[MB]";

        info_table[index].add_value(it, data.n_iteration);
        info_table[index].add_value(step, data.n_step);
        info_table[index].add_value(times, data.timing);
        info_table[index].add_value(mem, data.mem_usage);

        if (call_count == 0)
          {
            info_table[index].set_scientific(times, true);
            info_table[index].set_precision(times, 3);

            info_table[index].add_column_to_supercolumn(it, data.solver_name);
            info_table[index].add_column_to_supercolumn(step, data.solver_name);
            info_table[index].add_column_to_supercolumn(times,
                                                        data.solver_name);
            info_table[index].add_column_to_supercolumn(mem, data.solver_name);
          }
      }

    if (CT::SETS_ == "error_analysis")
      {
        auto solution = solver.get_solution();

        LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
                                               solution_host(solution.size());
        LinearAlgebra::ReadWriteVector<double> rw_vector(solution.size());
        rw_vector.import(solution, VectorOperation::insert);
        solution_host.import(rw_vector, VectorOperation::insert);

        solution_velocity_host.reinit(dof_handler_velocity.n_dofs());
        solution_pressure_host.reinit(dof_handler_pressure.n_dofs());

        for (unsigned int i = 0; i < solution_velocity_host.size(); ++i)
          solution_velocity_host[i] = solution_host[i];

        for (unsigned int i = 0; i < solution_pressure_host.size(); ++i)
          solution_pressure_host[i] =
            solution_host[solution_velocity_host.size() + i];

        constraints.distribute(solution_velocity_host);

        // solution_host.print(std::cout);

        const auto [l2_error_v, l2_error_p, H1_error_v] = compute_error();

        *pcout << "L2 error velocity: " << l2_error_v << std::endl
               << "L2 error pressure: " << l2_error_p << std::endl
               << "H1 error velocity: " << H1_error_v << std::endl
               << std::endl;

        // ghost_solution_host.print(std::cout);

        info_table[index].add_value("l2_error_v", l2_error_v);
        info_table[index].set_scientific("l2_error_v", true);
        info_table[index].set_precision("l2_error_v", 3);

        info_table[index].evaluate_convergence_rates(
          "l2_error_v", "dofs", ConvergenceTable::reduction_rate_log2, dim);

        info_table[index].add_value("l2_error_p", l2_error_p);
        info_table[index].set_scientific("l2_error_p", true);
        info_table[index].set_precision("l2_error_p", 3);

        info_table[index].evaluate_convergence_rates(
          "l2_error_p", "dofs", ConvergenceTable::reduction_rate_log2, dim);

        info_table[index].add_value("H1_error_v", H1_error_v);
        info_table[index].set_scientific("H1_error_v", true);
        info_table[index].set_precision("H1_error_v", 3);

        info_table[index].evaluate_convergence_rates(
          "H1_error_v", "dofs", ConvergenceTable::reduction_rate_log2, dim);
      }
  }

  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::solve_mg(unsigned int n_mg_cycles)
  {
    static unsigned int call_count = 0;

    using LA = PSMF::LaplaceVariant;
    using SM = PSMF::SmootherVariant;

    // do_solve<CT::LOCAL_SOLVER_[0],
    //          CT::LAPLACE_TYPE_[0],
    //          CT::SMOOTH_VMULT_[0],
    //          CT::SMOOTH_INV_[0]>(0, 0, 0, call_count);

    for (unsigned int k = 0; k < CT::LOCAL_SOLVER_.size(); ++k)
      {
        switch (CT::LOCAL_SOLVER_[k])
          {
            case PSMF::LocalSolverVariant::Direct:
              {
                do_solve<PSMF::LocalSolverVariant::Direct,
                         CT::LAPLACE_TYPE_[0],
                         CT::SMOOTH_VMULT_[0],
                         CT::SMOOTH_INV_[0]>(k, call_count);
                break;
              }
            case PSMF::LocalSolverVariant::Bila:
              {
                do_solve<PSMF::LocalSolverVariant::Bila,
                         CT::LAPLACE_TYPE_[0],
                         CT::SMOOTH_VMULT_[0],
                         CT::SMOOTH_INV_[0]>(k, call_count);
                break;
              }
            case PSMF::LocalSolverVariant::KSVD:
              {
                do_solve<PSMF::LocalSolverVariant::KSVD,
                         CT::LAPLACE_TYPE_[0],
                         CT::SMOOTH_VMULT_[0],
                         CT::SMOOTH_INV_[0]>(k, call_count);
                break;
              }
            case PSMF::LocalSolverVariant::NN:
              {
                do_solve<PSMF::LocalSolverVariant::NN,
                         CT::LAPLACE_TYPE_[0],
                         CT::SMOOTH_VMULT_[0],
                         CT::SMOOTH_INV_[0]>(k, call_count);
                break;
              }
            default:
              AssertThrow(false, ExcMessage("Invalid Smoother Variant."));
          }
      }



    call_count++;
  }

  template <int dim, int fe_degree>
  std::tuple<double, double, double>
  LaplaceProblem<dim, fe_degree>::compute_error()
  {
    Vector<double> cellwise_norm(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler_velocity,
                                      solution_velocity_host,
                                      SolutionVelocity<dim>(),
                                      cellwise_norm,
                                      QGauss<dim>(fe->degree + 2),
                                      VectorTools::L2_norm);
    const double global_norm_v =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_norm,
                                        VectorTools::L2_norm);

    Vector<double> cellwise_norm_p(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler_pressure,
                                      solution_pressure_host,
                                      SolutionPressure<dim>(),
                                      cellwise_norm_p,
                                      QGauss<dim>(fe->degree + 2),
                                      VectorTools::L2_norm);
    const double global_norm_p =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_norm_p,
                                        VectorTools::L2_norm);

    Vector<double> cellwise_h1norm(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler_velocity,
                                      solution_velocity_host,
                                      SolutionVelocity<dim>(),
                                      cellwise_h1norm,
                                      QGauss<dim>(fe->degree + 2),
                                      VectorTools::H1_seminorm);
    const double global_h1norm =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_h1norm,
                                        VectorTools::H1_seminorm);

    return std::make_tuple(global_norm_v, global_norm_p, global_h1norm);
  }

  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::run(const unsigned int n_cycles)
  {
    *pcout << Util::generic_info_to_fstring() << std::endl;

    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
      {
        *pcout << "Cycle " << cycle << std::endl;

        unsigned int n_levels = triangulation.n_global_levels();

        long long unsigned int n_dofs =
          2 * std::pow(std::pow(2, n_levels) * (fe_degree + 1), dim);

        if (n_dofs > CT::MAX_SIZES_)
          {
            *pcout << "Max size reached, terminating." << std::endl;
            *pcout << std::endl;

            for (unsigned int k = 0; k < CT::LOCAL_SOLVER_.size(); ++k)
              {
                unsigned int index = k;

                std::ostringstream oss;

                oss << "\n[" << LaplaceToString(CT::LAPLACE_TYPE_[0]) << " "
                    << LaplaceToString(CT::SMOOTH_VMULT_[0]) << " "
                    << SmootherToString(CT::SMOOTH_INV_[0]) << " "
                    << LocalSolverToString(CT::LOCAL_SOLVER_[k]) << "]\n";
                info_table[index].write_text(oss);

                *pcout << oss.str() << std::endl;
              }

            return;
          }

        if (cycle == 0)
          {
            GridGenerator::hyper_cube(triangulation, 0., 1.);
            triangulation.refine_global(2);
          }
        else
          triangulation.refine_global(1);

        setup_system();
        assemble_mg();

        solve_mg(1);
        *pcout << std::endl;
      }
  }
} // namespace Step64
int
main(int argc, char *argv[])
{
  try
    {
      using namespace Step64;

      {
        int device_id = findCudaDevice(argc, (const char **)argv);
        AssertCuda(cudaSetDevice(device_id));
      }

      {
        LaplaceProblem<CT::DIMENSION_, CT::FE_DEGREE_> Laplace_problem;
        Laplace_problem.run(20);
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