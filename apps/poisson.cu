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

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <helper_cuda.h>

#include <fstream>

#include "app_utilities.h"
#include "ct_parameter.h"
#include "solver.cuh"
#include "utilities.cuh"


// -\delta u = f, u = 0 on \parital \Omege, f = 1.
// double percision

namespace Step64
{
  using namespace dealii;

  // template <int dim>
  // class SolutionBase
  // {
  // protected:
  //   static constexpr std::size_t n_source_centers = 3;
  //   static const Point<dim>      source_centers[n_source_centers];
  //   static const double          width;
  // };

  // template <>
  // const Point<1>
  //   SolutionBase<1>::source_centers[SolutionBase<1>::n_source_centers] =
  //     {Point<1>(0.0), Point<1>(0.25), Point<1>(0.6)};

  // template <>
  // const Point<2>
  //   SolutionBase<2>::source_centers[SolutionBase<2>::n_source_centers] =
  //     {Point<2>(0.0, +0.0), Point<2>(0.25, 0.85), Point<2>(+0.6, 0.4)};

  // template <>
  // const Point<3>
  //   SolutionBase<3>::source_centers[SolutionBase<3>::n_source_centers] = {
  //     Point<3>(0.0, 0.0, 0.0),
  //     Point<3>(0.25, 0.85, 0.85),
  //     Point<3>(0.6, 0.4, 0.4)};

  // template <int dim>
  // const double SolutionBase<dim>::width = 1. / 3.;

  // using dealii::numbers::PI;

  // template <int dim>
  // class Solution : public Function<dim>, protected SolutionBase<dim>
  // {
  // public:
  //   using SolutionBase<dim>::width;
  //   using SolutionBase<dim>::n_source_centers;

  //   virtual double
  //   value(const Point<dim> &p, const unsigned int = 0) const override final
  //   {
  //     double val = 0;
  //     for (unsigned int i = 0; i < n_source_centers; ++i)
  //       val += value_impl(p, i);
  //     return -val;
  //   }

  //   virtual Tensor<1, dim>
  //   gradient(const Point<dim> &p, const unsigned int = 0) const override
  //   final
  //   {
  //     dealii::Tensor<1, dim> grad;
  //     for (unsigned int i = 0; i < n_source_centers; ++i)
  //       grad += gradient_impl(p, i);
  //     return grad;
  //   }

  //   virtual double
  //   laplacian(const dealii::Point<dim> &p,
  //             const unsigned int = 0) const override final
  //   {
  //     double lapl = 0;
  //     for (unsigned int i = 0; i < n_source_centers; ++i)
  //       lapl += laplacian_impl(p, i);
  //     return lapl;
  //   }

  // private:
  //   constexpr double
  //   u0() const
  //   {
  //     return 1. / std::pow(std::sqrt(2 * PI) * width, dim);
  //   }

  //   constexpr double
  //   v0() const
  //   {
  //     return -1. / (width * width);
  //   }

  //   double
  //   bell(const Point<dim> &p, const unsigned int i) const
  //   {
  //     AssertIndexRange(i, n_source_centers);
  //     const dealii::Tensor<1, dim> x_minus_xi =
  //       p - SolutionBase<dim>::source_centers[i];
  //     return std::exp(v0() * x_minus_xi.norm_square());
  //   }

  //   double
  //   value_impl(const Point<dim> &p, const unsigned int i) const
  //   {
  //     return u0() * bell(p, i);
  //   }

  //   Tensor<1, dim>
  //   gradient_impl(const Point<dim> &p, const unsigned int i) const
  //   {
  //     const dealii::Tensor<1, dim> x_minus_xi =
  //       p - SolutionBase<dim>::source_centers[i];
  //     return u0() * 2. * v0() * bell(p, i) * x_minus_xi;
  //   }

  //   double
  //   laplacian_impl(const dealii::Point<dim> &p, const unsigned int i) const
  //   {
  //     double     lapl       = 0;
  //     const auto x_minus_xi = p - SolutionBase<dim>::source_centers[i];
  //     lapl += bell(p, i) *
  //             (static_cast<double>(dim) + 2. * v0() *
  //             x_minus_xi.norm_square());
  //     lapl *= u0() * 2. * v0();
  //     return lapl;
  //   }

  //   SymmetricTensor<2, dim>
  //   hessian_impl(const Point<dim> &p, const unsigned int i) const
  //   {
  //     const auto x_minus_xi = p - SolutionBase<dim>::source_centers[i];
  //     SymmetricTensor<2, dim> hess;
  //     for (auto d = 0U; d < dim; ++d)
  //       hess[d][d] = 2. * u0() * v0() * bell(p, i) *
  //                    (1. + 2. * v0() * x_minus_xi[d] * x_minus_xi[d]);
  //     for (auto d1 = 0U; d1 < dim; ++d1)
  //       for (auto d2 = d1 + 1; d2 < dim; ++d2)
  //         hess[d1][d2] = 2. * 2. * u0() * v0() * v0() * bell(p, i) *
  //                        x_minus_xi[d1] * x_minus_xi[d2];
  //     return hess;
  //   }

  //   double
  //   bilaplacian_impl(const dealii::Point<dim> &p, const unsigned int i) const
  //   {
  //     double           bilapl     = 0;
  //     constexpr double d          = dim;
  //     const auto       x_minus_xi = p - SolutionBase<dim>::source_centers[i];
  //     bilapl += d + 2. + v0() * x_minus_xi.norm_square();
  //     bilapl *= 2. * 2. * v0() * x_minus_xi.norm_square();
  //     bilapl += d * d + 2. * d;
  //     bilapl *= 2. * 2. * v0() * v0() * u0() * bell(p, i);
  //     return bilapl;
  //   }
  // };

  // template <int dim>
  // class RightHandSide : public Function<dim>
  // {
  // public:
  //   // ManufacturedLoad(
  //   //   const std::shared_ptr<const Function<dim>> solution_function_in)
  //   //   : Function<dim>()
  //   //   , solution_function(solution_function_in)
  //   // {}

  //   virtual double
  //   value(const Point<dim> &p, const unsigned int = 0) const override final
  //   {
  //     return solution_function.laplacian(p);
  //   }

  // private:
  //   Solution<dim> solution_function;
  // };

  template <int dim, typename Number = double>
  class Solution : public Function<dim, Number>
  {
  public:
    virtual Number
    value(const Point<dim> &p, const unsigned int = 0) const override final
    {
      Number val = 1.;
      for (unsigned int d = 0; d < dim; ++d)
        val *= std::sin(numbers::PI * p[d]);
      return -val;
    }

    virtual Tensor<1, dim, Number>
    gradient(const Point<dim> &p, const unsigned int = 0) const override final
    {
      Tensor<1, dim, Number> grad;
      for (unsigned int d = 0; d < dim; ++d)
        {
          grad[d] = 1.;
          for (unsigned int e = 0; e < dim; ++e)
            if (d == e)
              grad[d] *= -numbers::PI * std::cos(numbers::PI * p[e]);
            else
              grad[d] *= std::sin(numbers::PI * p[e]);
        }
      return grad;
    }
  };

  template <int dim, typename Number = double>
  class RightHandSide : public Function<dim, Number>
  {
  public:
    virtual Number
    value(const Point<dim> &p, const unsigned int = 0) const override final
    {
      const Number arg = numbers::PI;
      Number       val = 1.;
      for (unsigned int d = 0; d < dim; ++d)
        val *= std::sin(arg * p[d]);
      return -dim * arg * arg * val;
    }
  };

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
    std::pair<double, double>
    compute_error();

    template <PSMF::LaplaceVariant  laplace,
              PSMF::LaplaceVariant  smooth_vmult,
              PSMF::SmootherVariant smooth_inv>
    void
    do_solve(unsigned int k,
             unsigned int j,
             unsigned int i,
             unsigned int call_count);

    Triangulation<dim>                  triangulation;
    std::shared_ptr<FiniteElement<dim>> fe;
    DoFHandler<dim>                     dof_handler;
    MappingQ1<dim>                      mapping;
    double                              setup_time;

    std::shared_ptr<Function<dim>> analytical_solution;
    std::shared_ptr<Function<dim>> load_function;

    std::vector<ConvergenceTable> info_table;

    std::fstream                        fout;
    std::shared_ptr<ConditionalOStream> pcout;

    MGLevelObject<std::shared_ptr<MatrixFreeDP>> mfdata_dp;
    MGLevelObject<std::shared_ptr<MatrixFreeSP>> mfdata_sp;
    MGConstrainedDoFs                            mg_constrained_dofs;
    AffineConstraints<double>                    constraints;

    PSMF::MGTransferCUDA<dim, vcycle_number> transfer;

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
        return std::make_shared<FE_DGQHermite<dim>>(fe_degree);
      return std::shared_ptr<FiniteElement<dim>>();
    }())
    , dof_handler(triangulation)
    , setup_time(0.)
    , pcout(std::make_shared<ConditionalOStream>(std::cout, false))
  {
    const auto filename = Util::get_filename();
    fout.open(filename + ".log", std::ios_base::out);
    pcout = std::make_shared<ConditionalOStream>(fout, true);

    info_table.resize(CT::LAPLACE_TYPE_.size() * CT::SMOOTH_VMULT_.size() *
                      CT::SMOOTH_INV_.size());
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

    *pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << " = ("
           << (1 << (nlevels - 1)) << " x (" << fe->degree << " + 1))^" << dim
           << std::endl;

    constraints.clear();
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
    mg_constrained_dofs.initialize(dof_handler);
    // mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,
    //                                                    dirichlet_boundary);

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
          mfdata_dp[level]->reinit(dof_handler, level, additional_data);
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
            mfdata_sp[level]->reinit(dof_handler, level, additional_data);
          }
      }

    *pcout << "Matrix-free setup time: " << time.wall_time() << "s"
           << std::endl;

    time.restart();

    transfer.initialize_constraints(mg_constrained_dofs);
    transfer.build(dof_handler);

    *pcout << "MG transfer setup time: " << time.wall_time() << "s"
           << std::endl;
  }

  template <int dim, int fe_degree>
  template <PSMF::LaplaceVariant  laplace,
            PSMF::LaplaceVariant  smooth_vmult,
            PSMF::SmootherVariant smooth_inv>
  void
  LaplaceProblem<dim, fe_degree>::do_solve(unsigned int k,
                                           unsigned int j,
                                           unsigned int i,
                                           unsigned int call_count)
  {
    PSMF::MultigridSolver<dim,
                          fe_degree,
                          CT::DOF_LAYOUT_,
                          full_number,
                          laplace,
                          smooth_vmult,
                          smooth_inv,
                          vcycle_number>
      solver(dof_handler,
             mfdata_dp,
             mfdata_sp,
             transfer,
             Solution<dim>(),
             RightHandSide<dim>(),
             pcout,
             1);

    *pcout << "\nMG with [" << LaplaceToString(CT::LAPLACE_TYPE_[k]) << " "
           << LaplaceToString(CT::SMOOTH_VMULT_[j]) << " "
           << SmootherToString(CT::SMOOTH_INV_[i]) << "]\n";

    unsigned int index =
      (k * CT::SMOOTH_VMULT_.size() + j) * CT::SMOOTH_INV_.size() + i;

    info_table[index].add_value("level", triangulation.n_global_levels());
    info_table[index].add_value("cells", triangulation.n_global_active_cells());
    info_table[index].add_value("dofs", dof_handler.n_dofs());

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
        ghost_solution_host = solution_host;
        constraints.distribute(ghost_solution_host);

        const auto [l2_error, H1_error] = compute_error();

        *pcout << "L2 error: " << l2_error << std::endl
               << "H1 error: " << H1_error << std::endl
               << std::endl;

        // ghost_solution_host.print(std::cout);

        info_table[index].add_value("L2_error", l2_error);
        info_table[index].set_scientific("L2_error", true);
        info_table[index].set_precision("L2_error", 3);

        info_table[index].evaluate_convergence_rates(
          "L2_error", "dofs", ConvergenceTable::reduction_rate_log2, dim);

        info_table[index].add_value("H1_error", H1_error);
        info_table[index].set_scientific("H1_error", true);
        info_table[index].set_precision("H1_error", 3);

        info_table[index].evaluate_convergence_rates(
          "H1_error", "dofs", ConvergenceTable::reduction_rate_log2, dim);
      }
  }

  template <int dim, int fe_degree>
  std::pair<double, double>
  LaplaceProblem<dim, fe_degree>::compute_error()
  {
    Vector<double> cellwise_norm(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler,
                                      ghost_solution_host,
                                      Solution<dim>(),
                                      cellwise_norm,
                                      QGauss<dim>(fe->degree + 1),
                                      VectorTools::L2_norm);
    const double global_norm =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_norm,
                                        VectorTools::L2_norm);

    Vector<double> cellwise_h1norm(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler,
                                      ghost_solution_host,
                                      Solution<dim>(),
                                      cellwise_h1norm,
                                      QGauss<dim>(fe->degree + 1),
                                      VectorTools::H1_seminorm);
    const double global_h1norm =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_h1norm,
                                        VectorTools::H1_seminorm);

    return std::make_pair(global_norm, global_h1norm);
  }

  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::solve_mg(unsigned int n_mg_cycles)
  {
    static unsigned int call_count = 0;

    using LA = PSMF::LaplaceVariant;
    using SM = PSMF::SmootherVariant;



    if (CT::LAPLACE_TYPE_.size() > 1)
      for (unsigned int k = 0; k < CT::LAPLACE_TYPE_.size(); ++k)
        {
          switch (CT::LAPLACE_TYPE_[k])
            {
              case LA::Basic:
                {
                  do_solve<LA::Basic, CT::SMOOTH_VMULT_[0], CT::SMOOTH_INV_[0]>(
                    k, 0, 0, call_count);
                  break;
                }
              case LA::BasicCell:
                {
                  do_solve<LA::BasicCell,
                           CT::SMOOTH_VMULT_[0],
                           CT::SMOOTH_INV_[0]>(k, 0, 0, call_count);
                  break;
                }
              case LA::ConflictFree:
                {
                  do_solve<LA::ConflictFree,
                           CT::SMOOTH_VMULT_[0],
                           CT::SMOOTH_INV_[0]>(k, 0, 0, call_count);
                  break;
                }
              case LA::ConflictFreeMem:
                {
                  do_solve<LA::ConflictFreeMem,
                           CT::SMOOTH_VMULT_[0],
                           CT::SMOOTH_INV_[0]>(k, 0, 0, call_count);
                  break;
                }
              case LA::TensorCore:
                {
                  do_solve<LA::TensorCore,
                           CT::SMOOTH_VMULT_[0],
                           CT::SMOOTH_INV_[0]>(k, 0, 0, call_count);
                  break;
                }
              case LA::TensorCoreMMA:
                {
                  do_solve<LA::TensorCoreMMA,
                           CT::SMOOTH_VMULT_[0],
                           CT::SMOOTH_INV_[0]>(k, 0, 0, call_count);
                  break;
                }
              default:
                AssertThrow(false, ExcMessage("Invalid Smoother Variant."));
            }
        }
    else if (CT::SMOOTH_INV_.size() > 1)
      for (unsigned int k = 0; k < CT::SMOOTH_INV_.size(); ++k)
        {
          switch (CT::SMOOTH_INV_[k])
            {
              case SM::AllPatch:
                {
                  do_solve<CT::LAPLACE_TYPE_[0],
                           CT::SMOOTH_VMULT_[0],
                           SM::AllPatch>(0, 0, k, call_count);
                  break;
                }
              case SM::GLOBAL:
                {
                  do_solve<CT::LAPLACE_TYPE_[0],
                           CT::SMOOTH_VMULT_[0],
                           SM::GLOBAL>(0, 0, k, call_count);
                  break;
                }
              case SM::ConflictFree:
                {
                  do_solve<CT::LAPLACE_TYPE_[0],
                           CT::SMOOTH_VMULT_[0],
                           SM::ConflictFree>(0, 0, k, call_count);
                  break;
                }
              case SM::ExactRes:
                {
                  do_solve<CT::LAPLACE_TYPE_[0],
                           CT::SMOOTH_VMULT_[0],
                           SM::ExactRes>(0, 0, k, call_count);
                  break;
                }
              default:
                AssertThrow(false, ExcMessage("Invalid Smoother Variant."));
            }
        }
    else
      do_solve<CT::LAPLACE_TYPE_[0], CT::SMOOTH_VMULT_[0], CT::SMOOTH_INV_[0]>(
        0, 0, 0, call_count);

    call_count++;
  }

  template <int dim, int fe_degree>
  void
  LaplaceProblem<dim, fe_degree>::run(const unsigned int n_cycles)
  {
    *pcout << Util::generic_info_to_fstring() << std::endl;

    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
      {
        *pcout << "Cycle " << cycle << std::endl;

#ifdef DUPLICATE
        long long unsigned int n_dofs =
          std::pow(std::pow(2, triangulation.n_global_levels()) *
                     (fe_degree + 1),
                   dim) *
          4;
#else
        long long unsigned int n_dofs = std::pow(
          std::pow(2, triangulation.n_global_levels()) * (fe_degree + 1), dim);
#endif

        if (n_dofs > CT::MAX_SIZES_)
          {
            *pcout << "Max size reached, terminating." << std::endl;
            *pcout << std::endl;

            for (unsigned int k = 0; k < CT::LAPLACE_TYPE_.size(); ++k)
              for (unsigned int j = 0; j < CT::SMOOTH_VMULT_.size(); ++j)
                for (unsigned int i = 0; i < CT::SMOOTH_INV_.size(); ++i)
                  {
                    unsigned int index = (k * CT::SMOOTH_VMULT_.size() + j) *
                                           CT::SMOOTH_INV_.size() +
                                         i;

                    std::ostringstream oss;

                    oss << "\n[" << LaplaceToString(CT::LAPLACE_TYPE_[k]) << " "
                        << LaplaceToString(CT::SMOOTH_VMULT_[j]) << " "
                        << SmootherToString(CT::SMOOTH_INV_[i]) << "]\n";
                    info_table[index].write_text(oss);

                    *pcout << oss.str() << std::endl;
                  }

            return;
          }

        if (cycle == 0)
          {
#ifdef DUPLICATE
            Triangulation<dim> tria(
              Triangulation<dim>::limit_level_difference_at_vertices);

            GridGenerator::hyper_cube(tria, 0, 1);
            if (dim == 2)
              GridGenerator::replicate_triangulation(tria,
                                                     {2, 2},
                                                     triangulation);
            else if (dim == 3)
              GridGenerator::replicate_triangulation(tria,
                                                     {2, 2, 1},
                                                     triangulation);
#else
            GridGenerator::hyper_cube(triangulation, 0, 1);
            triangulation.refine_global(1);
#endif
            triangulation.refine_global(1);
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
