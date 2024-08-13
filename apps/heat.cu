/**
 * Created by Cu Cui on 2022/12/25.
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/cuda.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <helper_cuda.h>

#include <fstream>
#include <functional>
#include <optional>

#include "app_utilities.h"
#include "ct_parameter.h"
#include "solver.cuh"
#include "utilities.cuh"

// Solving Heat equation with implicit Euler time discretizations

namespace Step64
{
  using namespace dealii;

  const double tau  = CT::DT_;
  const double endT = CT::ENDT_;

  const unsigned int N = endT / tau;

  const double wave_number = 2;
  const double a_t         = 0.5;

  template <int dim, typename Number = double>
  class Solution : public Function<dim, Number>
  {
  public:
    Solution()
      : Function<dim>()
    {}

    virtual Number
    value(const Point<dim> &p, const unsigned int = 0) const override final
    {
      const double T   = this->get_time();
      double       val = (1 + std::sin(numbers::PI * T)) * std::exp(-a_t * T);
      for (unsigned int d = 0; d < dim; ++d)
        val *= std::sin(numbers::PI * p[d] * wave_number);
      return val;
    }

    virtual Tensor<1, dim, Number>
    gradient(const Point<dim> &p, const unsigned int = 0) const override final
    {
      const double   T = this->get_time();
      Tensor<1, dim> return_value;
      for (unsigned int d = 0; d < dim; ++d)
        {
          return_value[d] =
            (1 + std::sin(numbers::PI * T)) * std::exp(-a_t * T);
          for (unsigned int e = 0; e < dim; ++e)
            if (d == e)
              return_value[d] *= numbers::PI * wave_number *
                                 std::cos(numbers::PI * p[e] * wave_number);
            else
              return_value[d] *= std::sin(numbers::PI * p[e] * wave_number);
        }

      return return_value;
    }
  };

  template <int dim, typename Number = double>
  class RightHandSide : public Function<dim, Number>
  {
  public:
    RightHandSide()
      : Function<dim>()
    {}

    virtual Number
    value(const Point<dim> &p, const unsigned int = 0) const override final
    {
      const double T   = this->get_time();
      double       val = std::exp(-a_t * T);
      for (unsigned int d = 0; d < dim; ++d)
        val *= std::sin(numbers::PI * p[d] * wave_number);
      return val *
             (numbers::PI * std::cos(numbers::PI * T) - a_t -
              a_t * std::sin(numbers::PI * T) +
              dim * numbers::PI * wave_number * numbers::PI * wave_number +
              dim * numbers::PI * wave_number * numbers::PI * wave_number *
                std::sin(numbers::PI * T));
    }
  };

  template <int dim, int fe_degree>
  class LaplaceProblem
  {
  public:
    using full_number   = double;
    using vcycle_number = CT::VCYCLE_NUMBER_;

    LaplaceProblem();
    ~LaplaceProblem();
    void
    run(const unsigned int n_cycles);

  private:
    void
    setup_system();
    void
    solve(unsigned int n_mg_cycles);

    std::pair<double, double>
    compute_error(const double time);

    Triangulation<dim>                  triangulation;
    std::shared_ptr<FiniteElement<dim>> fe;
    DoFHandler<dim>                     dof_handler;
    MappingQ1<dim>                      mapping;
    double                              setup_time;

    ConvergenceTable convergence_table;
    ConvergenceTable convergence_table_N;

    std::fstream                        fout;
    std::shared_ptr<ConditionalOStream> pcout;

    AffineConstraints<double> constraints;

    LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
      ghost_solution_host;
  };

  template <int dim, int fe_degree>
  LaplaceProblem<dim, fe_degree>::LaplaceProblem()
    : triangulation(Triangulation<dim>::limit_level_difference_at_vertices)
    , fe([&]() -> std::shared_ptr<FiniteElement<dim>> {
      if (CT::DOF_LAYOUT_ == PSMF::DoFLayout::Q)
        return std::make_shared<FE_Q<dim>>(fe_degree);
      else if (CT::DOF_LAYOUT_ == PSMF::DoFLayout::DGQ)
        return std::make_shared<FE_DGQ<dim>>(fe_degree);
      return std::shared_ptr<FiniteElement<dim>>();
    }())
    , dof_handler(triangulation)
    , setup_time(0.)
    , pcout(std::make_shared<ConditionalOStream>(std::cout, false))
  {
    const auto filename = Util::get_filename();
    fout.open(filename + ".log", std::ios_base::out);
    pcout = std::make_shared<ConditionalOStream>(fout, true);
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

    dof_handler.distribute_dofs(*fe);
    dof_handler.distribute_mg_dofs();
    const unsigned int nlevels = triangulation.n_global_levels();
    for (unsigned int level = 0; level < nlevels; ++level)
      Util::Lexicographic(dof_handler, level);
    Util::Lexicographic(dof_handler);

    *pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << " = ("
           << ((int)std::pow(dof_handler.n_dofs() * 1.0000001, 1. / dim) - 1) /
                fe->degree
           << " x " << fe->degree << " + 1)^" << dim << std::endl;

    setup_time += time.wall_time();

    *pcout << "DoF setup time:           " << setup_time << "s" << std::endl;

    constraints.clear();
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Solution<dim>(),
                                             constraints);
    constraints.close();
  }

  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::solve(unsigned int n_mg_cycles)
  {
    Solution<dim> analytic_solution;
    analytic_solution.set_time(0);

    PSMF::MultigridSolvers<dim, fe_degree, vcycle_number, full_number> solver(
      dof_handler,
      analytic_solution,
      pcout,
      N,
      tau,
      wave_number,
      a_t,
      n_mg_cycles);

    Timer time;

    Utilities::System::MemoryStats stats;
    Utilities::System::get_memory_stats(stats);
    Utilities::MPI::MinMaxAvg memory =
      Utilities::MPI::min_max_avg(stats.VmRSS / 1024., MPI_COMM_WORLD);

    *pcout << "CPU Memory stats [MB]: " << memory.min << " [p"
           << memory.min_index << "] " << memory.avg << " " << memory.max
           << " [p" << memory.max_index << "]" << std::endl;

    size_t free_mem, total_mem;
    AssertCuda(cudaMemGetInfo(&free_mem, &total_mem));

    int mem_usage = (total_mem - free_mem) / 1024 / 1024;
    *pcout << "GPU Memory stats [MB]: " << mem_usage << "\n\n";

    double time_gmres = 1e10;
    for (unsigned int i = 0; i < 1; ++i)
      {
        time.restart();
        solver.solve_gmres(false);
        cudaDeviceSynchronize();
        time_gmres = std::min(time.wall_time(), time_gmres);
        *pcout << "Time solve GMRES (one time step): " << time.wall_time()
               << "\n";
      }

    std::optional<std::pair<ReductionControl, double>> it_data =
      solver.solve_gmres(true);

    auto solver_control = it_data->first;
    auto n_iter         = solver_control.last_step();
    auto residual_0     = solver_control.initial_value();
    auto residual_n     = solver_control.last_value();
    auto reduction      = solver_control.reduction();
    auto rho =
      std::pow(residual_n / residual_0, static_cast<double>(1. / n_iter));
    const auto n_frac = std::log(reduction) / std::log(rho);

    // solver.print_wall_times();
    auto history_data = solver_control.get_history_data();
    for (auto i = 1U; i < n_iter + 1; ++i)
      *pcout << "step " << i << ": " << history_data[i] / residual_0 << "\n";

    {
      auto solution = solver.get_solution();

      LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
                                             solution_host(solution.size());
      LinearAlgebra::ReadWriteVector<double> rw_vector(solution.size());
      rw_vector.import(solution, VectorOperation::insert);
      solution_host.import(rw_vector, VectorOperation::insert);
      ghost_solution_host = solution_host;
      constraints.distribute(ghost_solution_host);
    }
    const auto [l2_error_gmres, H1_error_gmres] = compute_error(tau);

    *pcout << "Iterations: " << n_iter << std::endl
           << "frac Its. : " << n_frac << std::endl
           << std::endl;


    *pcout << "L2 error: " << l2_error_gmres << std::endl
           << "H1 error: " << H1_error_gmres << std::endl
           << std::endl;

    double time_gmres_N = 1e10;
    {
      time.restart();
      solver.solve_gmres(false, N);
      cudaDeviceSynchronize();
      time_gmres_N = std::min(time.wall_time(), time_gmres_N);
    }

    {
      auto solution = solver.get_solution();

      LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
                                             solution_host(solution.size());
      LinearAlgebra::ReadWriteVector<double> rw_vector(solution.size());
      rw_vector.import(solution, VectorOperation::insert);
      solution_host.import(rw_vector, VectorOperation::insert);
      ghost_solution_host = solution_host;
      constraints.distribute(ghost_solution_host);
    }
    const auto [l2_error_gmres_N, H1_error_gmres_N] = compute_error(tau * N);

    *pcout << "L2 error: " << l2_error_gmres_N << std::endl
           << "H1 error: " << H1_error_gmres_N << std::endl
           << std::endl;

    *pcout << "GMRES L2 error with ndof = " << dof_handler.n_dofs() << "  "
           << l2_error_gmres << std::endl;

    auto timing_result = solver.get_timing();

    convergence_table.add_value("cells", triangulation.n_global_active_cells());
    convergence_table.add_value("dofs", dof_handler.n_dofs());

    convergence_table.add_value("system_mat-vec", timing_result[0]);
    convergence_table.add_value("system_prec", timing_result[1]);
    convergence_table.add_value("local_prec", timing_result[2]);
    convergence_table.add_value("local_mv", timing_result[3]);
    convergence_table.add_value("local_mv_sp", timing_result[4]);
    convergence_table.add_value("smoother", timing_result[5]);

    convergence_table.add_value("gmres_L2error", l2_error_gmres);
    convergence_table.add_value("gmres_H1error", H1_error_gmres);
    convergence_table.add_value("gmres_time", time_gmres);
    convergence_table.add_value("gmres_its", n_iter);
    convergence_table.add_value("frac_its", n_frac);
    convergence_table.add_value("inner_its_avg", it_data->second);

    convergence_table_N.add_value("cells",
                                  triangulation.n_global_active_cells());
    convergence_table_N.add_value("dofs", dof_handler.n_dofs());
    convergence_table_N.add_value("gmres_L2error", l2_error_gmres_N);
    convergence_table_N.add_value("gmres_H1error", H1_error_gmres_N);
    convergence_table_N.add_value("gmres_time", time_gmres_N);
  }


  template <int dim, int fe_degree>
  std::pair<double, double>
  LaplaceProblem<dim, fe_degree>::compute_error(const double time)
  {
    Solution<dim> u_analytical;
    u_analytical.set_time(time);

    Vector<double> cellwise_norm(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler,
                                      ghost_solution_host,
                                      u_analytical,
                                      cellwise_norm,
                                      QGauss<dim>(fe->degree + 2),
                                      VectorTools::L2_norm);
    const double global_norm =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_norm,
                                        VectorTools::L2_norm);

    Vector<double> cellwise_h1norm(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler,
                                      ghost_solution_host,
                                      u_analytical,
                                      cellwise_h1norm,
                                      QGauss<dim>(fe->degree + 2),
                                      VectorTools::H1_seminorm);
    const double global_h1norm =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_h1norm,
                                        VectorTools::H1_seminorm);

    return std::make_pair(global_norm, global_h1norm);
  }

  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::run(const unsigned int n_cycles)
  {
    *pcout << Util::generic_info_to_fstring() << std::endl;

    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
      {
        smoother_mem = 0;

        *pcout << "Cycle " << cycle << std::endl;

        long long unsigned int n_dofs =
          std::pow(std::pow(2, triangulation.n_global_levels()) * fe_degree + 1,
                   dim);

        if (n_dofs > CT::MAX_SIZES_)
          {
            *pcout << "Max size reached, terminating." << std::endl;
            *pcout << std::endl;

            break;
          }

        if (cycle == 0)
          {
            GridGenerator::hyper_cube(triangulation, 0., 1.);
            triangulation.refine_global(3);
          }
        else
          triangulation.refine_global(1);

        setup_system();
        solve(1);

        *pcout << std::endl;
      }

    if (true)
      {
        auto set_format = [&](auto name) {
          convergence_table.set_scientific(name, true);
          convergence_table.set_precision(name, 3);
        };

        set_format("system_mat-vec");
        set_format("system_prec");
        set_format("local_prec");
        set_format("local_mv");
        set_format("local_mv_sp");
        set_format("smoother");

        convergence_table.set_scientific("gmres_L2error", true);
        convergence_table.set_precision("gmres_L2error", 3);
        convergence_table.evaluate_convergence_rates(
          "gmres_L2error", "cells", ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("gmres_H1error", true);
        convergence_table.set_precision("gmres_H1error", 3);
        convergence_table.evaluate_convergence_rates(
          "gmres_H1error", "cells", ConvergenceTable::reduction_rate_log2, dim);
        convergence_table.set_scientific("gmres_time", true);
        convergence_table.set_precision("gmres_time", 3);

        std::ostringstream oss;

        oss << "\n[" << SmootherToString(CT::KERNEL_TYPE_[0]) << "]\n";
        oss << "\n Time = " << tau * 1 << "\n";
        convergence_table.write_text(oss);

        convergence_table_N.set_scientific("gmres_L2error", true);
        convergence_table_N.set_precision("gmres_L2error", 3);
        convergence_table_N.evaluate_convergence_rates(
          "gmres_L2error", "cells", ConvergenceTable::reduction_rate_log2, dim);
        convergence_table_N.set_scientific("gmres_H1error", true);
        convergence_table_N.set_precision("gmres_H1error", 3);
        convergence_table_N.evaluate_convergence_rates(
          "gmres_H1error", "cells", ConvergenceTable::reduction_rate_log2, dim);
        convergence_table_N.set_scientific("gmres_time", true);
        convergence_table_N.set_precision("gmres_time", 3);

        oss << "\n Time = " << tau * N << "\n";
        convergence_table_N.write_text(oss);

        *pcout << oss.str() << std::endl;

        *pcout << std::endl << std::endl;
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
