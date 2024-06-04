/**
 * @file benchmark_mg.cu
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief Benchmarks on different MG componts.
 * @version 1.0
 * @date 2023-01-02
 *
 * @copyright Copyright (c) 2023
 *
 */


#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/cuda.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/matrix_free/cuda_fe_evaluation.h>
#include <deal.II/matrix_free/cuda_matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/multigrid.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "app_utilities.h"
#include "ct_parameter.h"
#include "cuda_mg_transfer.cuh"
#include "laplace_operator.cuh"
#include "patch_base.cuh"
#include "patch_smoother.cuh"

using namespace dealii;

#ifndef TIMING
#  define TIMING 1
#endif

#define MODE 0
// 0 - ncu
// 1 - perf

template <int dim, int fe_degree>
class LaplaceProblem
{
public:
  using full_number   = double;
  using vcycle_number = float;
  using MatrixFreeDP  = PSMF::LevelVertexPatch<dim, fe_degree, full_number>;
  using MatrixFreeSP  = PSMF::LevelVertexPatch<dim, fe_degree, vcycle_number>;

  using VectorTypeDP =
    LinearAlgebra::distributed::Vector<full_number, MemorySpace::CUDA>;
  using VectorTypeSP =
    LinearAlgebra::distributed::Vector<vcycle_number, MemorySpace::CUDA>;

  LaplaceProblem();
  ~LaplaceProblem();
  void
  run();

private:
  void
  setup_system();
  void
  bench_Ax();
  void
  bench_transfer();
  void
  bench_smooth();

  template <PSMF::LaplaceVariant kernel>
  void
  do_Ax();
  template <PSMF::LaplaceVariant smooth_vmult, PSMF::SmootherVariant smooth_inv>
  void
  do_smooth();

  Triangulation<dim>                  triangulation;
  std::shared_ptr<FiniteElement<dim>> fe;
  DoFHandler<dim>                     dof_handler;
  MappingQ1<dim>                      mapping;

  MGConstrainedDoFs mg_constrained_dofs;

  std::shared_ptr<MatrixFreeDP> mfdata_dp;
  std::shared_ptr<MatrixFreeSP> mfdata_sp;

  VectorTypeDP solution_dp;
  VectorTypeDP system_rhs_dp;

  VectorTypeSP solution_sp;
  VectorTypeSP system_rhs_sp;

  double base_time_dp;
  double base_time_sp;

  unsigned int N;
  unsigned int n_mv;
  unsigned int n_dofs;
  unsigned int maxlevel;

  std::fstream                        fout;
  std::shared_ptr<ConditionalOStream> pcout;

  std::array<ConvergenceTable, 5> info_table;
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
  , base_time_dp(0.)
  , base_time_sp(0.)
  , pcout(std::make_shared<ConditionalOStream>(std::cout, false))
{
  const auto filename = Util::get_filename();
  fout.open("Benchmark_" + filename + ".log", std::ios_base::out);
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
  dof_handler.distribute_dofs(*fe);
  dof_handler.distribute_mg_dofs();

  n_dofs = dof_handler.n_dofs();
#if MODE == 0
  N    = 1;
  n_mv = 1;
#elif MODE == 1
  N    = 5;
  n_mv = n_dofs < 10000000 ? 100 : 20;
#endif

  const unsigned int nlevels = triangulation.n_global_levels();

  *pcout << "Setting up dofs...\n";
  *pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << " = ("
         << (1 << (nlevels - 1)) << " x (" << fe->degree << " + 1))^" << dim
         << std::endl;

  *pcout << "Setting up Matrix-Free...\n";
  // Initialization of Dirichlet boundaries
  std::set<types::boundary_id> dirichlet_boundary;
  dirichlet_boundary.insert(0);
  mg_constrained_dofs.initialize(dof_handler);
  mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,
                                                     dirichlet_boundary);
  MappingQ1<dim> mapping;
  maxlevel = triangulation.n_global_levels() - 1;

  IndexSet relevant_dofs;
  DoFTools::extract_locally_relevant_level_dofs(dof_handler,
                                                maxlevel,
                                                relevant_dofs);
  // DP
  {
    typename MatrixFreeDP::AdditionalData additional_data;
    additional_data.relaxation         = 1.;
    additional_data.use_coloring       = false;
    additional_data.patch_per_block    = CT::PATCH_PER_BLOCK_;
    additional_data.granularity_scheme = CT::GRANULARITY_;

    mfdata_dp = std::make_shared<MatrixFreeDP>();
    mfdata_dp->reinit(dof_handler, maxlevel, additional_data);
  }
  // SP
  {
    typename MatrixFreeSP::AdditionalData additional_data;
    additional_data.relaxation         = 1.;
    additional_data.use_coloring       = false;
    additional_data.patch_per_block    = CT::PATCH_PER_BLOCK_;
    additional_data.granularity_scheme = CT::GRANULARITY_;

    mfdata_sp = std::make_shared<MatrixFreeSP>();
    mfdata_sp->reinit(dof_handler, maxlevel, additional_data);
  }
}
template <int dim, int fe_degree>
template <PSMF::LaplaceVariant kernel>
void
LaplaceProblem<dim, fe_degree>::do_Ax()
{
  PSMF::LaplaceOperator<dim, fe_degree, full_number, kernel> matrix_dp;
  matrix_dp.initialize(mfdata_dp, dof_handler, maxlevel);
  matrix_dp.initialize_dof_vector(system_rhs_dp);
  solution_dp.reinit(system_rhs_dp);

  unsigned int max_level = triangulation.n_levels() - 1;
  unsigned int n_patches = std::pow((1 << max_level) - 1, dim);

  system_rhs_dp = 1.;
  solution_dp   = 0.;

  std::srand(0);

  LinearAlgebra::ReadWriteVector<double> rw_vector_dp(dof_handler.n_dofs());
  {
    for (unsigned int i = 0; i < rw_vector_dp.size(); ++i)
      rw_vector_dp[i] = ((double)std::rand()) / RAND_MAX;
    system_rhs_dp.import(rw_vector_dp, VectorOperation::insert);
  }

  Timer  time;
  double best_time = 1e10;

#if TIMING != 0
  float max_cycles = 0;
  float min_cycles = 0;
#endif

  LinearAlgebra::distributed::Vector<double, MemorySpace::Host> solution_host(
    solution_dp.size());

  auto copy_back = [&]() {
    LinearAlgebra::ReadWriteVector<double> rw_vector(solution_dp.size());
    rw_vector.import(solution_dp, VectorOperation::insert);
    solution_host.import(rw_vector, VectorOperation::insert);
  };

  for (unsigned int i = 0; i < N; ++i)
    {
      time.restart();
      for (unsigned int i = 0; i < n_mv; ++i)
        {
#if TIMING != 0
          printf("Testing.............\n");
#endif
          matrix_dp.vmult(solution_dp, system_rhs_dp);
          cudaDeviceSynchronize();
#if TIMING != 0
          copy_back();
          auto max_ =
            *std::max_element(solution_host.begin(), solution_host.end());
          auto min_  = *std::min_element(solution_host.begin(),
                                        solution_host.begin() + n_patches);
          max_cycles = std::max(max_, max_cycles);
          min_cycles = std::max(min_, min_cycles);
#endif
        }
      best_time = std::min(time.wall_time() / n_mv, best_time);
    }

    // std::cout << min_cycles << " " << max_cycles << std::endl;
    // std::cout << std::fixed << std::setprecision(10) << solution_dp.l2_norm()
    //           << std::endl;

#if TIMING != 0
  for (unsigned int i = 0; i < n_patches; ++i)
    std::cout << solution_host[i] << " ";
  std::cout << "\n";
#endif

  // solution_dp.print(std::cout);

  info_table[0].add_value("dofs", dof_handler.n_dofs());
  info_table[0].add_value("Name", std::string(LaplaceToString(kernel)) + " DP");
  info_table[0].add_value("Time[s]", best_time);
  info_table[0].add_value("Perf[Dof/s]", n_dofs / best_time);


  PSMF::LaplaceOperator<dim, fe_degree, vcycle_number, kernel> matrix_sp;
  matrix_sp.initialize(mfdata_sp, dof_handler, maxlevel);
  matrix_sp.initialize_dof_vector(system_rhs_sp);
  solution_sp.reinit(system_rhs_sp);

  system_rhs_sp = 1.;
  solution_sp   = 0.;

  {
    LinearAlgebra::ReadWriteVector<float> rw_vector(dof_handler.n_dofs());
    for (unsigned int i = 0; i < rw_vector.size(); ++i)
      rw_vector[i] = rw_vector_dp[i];
    system_rhs_sp.import(rw_vector, VectorOperation::insert);
  }

  LinearAlgebra::distributed::Vector<float, MemorySpace::Host> solution_hosts(
    solution_sp.size());

  auto copy_back_sp = [&]() {
    LinearAlgebra::ReadWriteVector<float> rw_vector(solution_sp.size());
    rw_vector.import(solution_sp, VectorOperation::insert);
    solution_hosts.import(rw_vector, VectorOperation::insert);
  };

#if TIMING != 0
  float max_cycless = 0;
  float min_cycless = 0;
#endif

  for (unsigned int i = 0; i < N; ++i)
    {
      time.restart();
      for (unsigned int i = 0; i < n_mv; ++i)
        {
#if TIMING != 0
          printf("Testing.............\n");
#endif
          matrix_sp.vmult(solution_sp, system_rhs_sp);
          cudaDeviceSynchronize();
#if TIMING != 0
          copy_back_sp();
          auto max_ =
            *std::max_element(solution_hosts.begin(), solution_hosts.end());
          auto min_   = *std::min_element(solution_hosts.begin(),
                                        solution_hosts.begin() + n_patches);
          max_cycless = std::max(max_, max_cycless);
          min_cycless = std::max(min_, min_cycless);
#endif
        }
      best_time = std::min(time.wall_time() / n_mv, best_time);
    }

    // std::cout << min_cycless << " " << max_cycless << std::endl;
    // std::cout << std::fixed << std::setprecision(10) << solution_sp.l2_norm()
    //           << std::endl;

#if TIMING != 0
  for (unsigned int i = 0; i < n_patches; ++i)
    std::cout << solution_hosts[i] << " ";
  std::cout << "\n";
#endif

  info_table[1].add_value("dofs", dof_handler.n_dofs());
  info_table[1].add_value("Name", std::string(LaplaceToString(kernel)) + " SP");
  info_table[1].add_value("Time[s]", best_time);
  info_table[1].add_value("Perf[Dof/s]", n_dofs / best_time);


  if (fe_degree == 7)
    {
      copy_back();
      copy_back_sp();

      std::cout << solution_hosts.l2_norm() << " " << solution_host.l2_norm()
                << std::endl;

      for (unsigned int i = 0; i < solution_dp.size(); ++i)
        {
          auto abs_error    = std::abs(solution_hosts[i] - solution_host[i]);
          solution_hosts[i] = abs_error;
          solution_host[i]  = abs_error / solution_host[i];
        }

      info_table[1].add_value("abs error", solution_hosts.l2_norm());
      info_table[1].add_value("rel error", solution_host.l2_norm());
    }
}
template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::bench_Ax()
{
  *pcout << "Benchmarking Mat-vec...\n";

  for (unsigned int k = 0; k < CT::LAPLACE_TYPE_.size(); ++k)
    switch (CT::LAPLACE_TYPE_[k])
      {
        case PSMF::LaplaceVariant::Basic:
          do_Ax<PSMF::LaplaceVariant::Basic>();
          break;
        case PSMF::LaplaceVariant::BasicCell:
          do_Ax<PSMF::LaplaceVariant::BasicCell>();
          break;
        case PSMF::LaplaceVariant::TensorCore:
          do_Ax<PSMF::LaplaceVariant::TensorCore>();
          break;
        case PSMF::LaplaceVariant::TensorCoreMMA:
          do_Ax<PSMF::LaplaceVariant::TensorCoreMMA>();
          break;
        case PSMF::LaplaceVariant::ConflictFree:
          do_Ax<PSMF::LaplaceVariant::ConflictFree>();
          break;
        case PSMF::LaplaceVariant::ConflictFreeMem:
          do_Ax<PSMF::LaplaceVariant::ConflictFreeMem>();
          break;
        default:
          AssertThrow(false, ExcMessage("Invalid Laplace Variant."));
      }
}
template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::bench_transfer()
{
  *pcout << "Benchmarking Transfer in double precision...\n";

  unsigned int max_level = triangulation.n_levels() - 1;
  VectorTypeDP u_coarse(dof_handler.n_dofs(max_level - 1));
  VectorTypeSP u_coarse_(dof_handler.n_dofs(max_level - 1));
  u_coarse  = 1.;
  u_coarse_ = 1.;

  PSMF::MGTransferCUDA<dim, full_number> mg_transfer(mg_constrained_dofs);
  mg_transfer.build(dof_handler);

  Timer  time;
  double best_time  = 1e10;
  double best_time2 = 1e10;

  for (unsigned int i = 0; i < N; ++i)
    {
      time.restart();
      for (unsigned int i = 0; i < n_mv; ++i)
        {
          mg_transfer.prolongate(max_level, system_rhs_dp, u_coarse);
          mg_transfer.restrict_and_add(max_level, u_coarse, system_rhs_dp);
          cudaDeviceSynchronize();
        }
      best_time = std::min(time.wall_time() / n_mv, best_time);
    }

  info_table[2].add_value("dofs", dof_handler.n_dofs());
  info_table[2].add_value("Name", "Transfer DP");
  info_table[2].add_value("Time[s]", best_time);
  info_table[2].add_value("Perf[Dof/s]", n_dofs / best_time);

  *pcout << "Benchmarking Transfer in single precision...\n";

  PSMF::MGTransferCUDA<dim, vcycle_number> mg_transfer_(mg_constrained_dofs);
  mg_transfer_.build(dof_handler);

  for (unsigned int i = 0; i < N; ++i)
    {
      time.restart();
      for (unsigned int i = 0; i < n_mv; ++i)
        {
          mg_transfer_.prolongate(max_level, system_rhs_sp, u_coarse_);
          mg_transfer_.restrict_and_add(max_level, u_coarse_, system_rhs_sp);
          cudaDeviceSynchronize();
        }
      best_time2 = std::min(time.wall_time() / n_mv, best_time2);
    }

  info_table[2].add_value("dofs", dof_handler.n_dofs());
  info_table[2].add_value("Name", "Transfer SP");
  info_table[2].add_value("Time[s]", best_time2);
  info_table[2].add_value("Perf[Dof/s]", n_dofs / best_time2);
}

template <int dim, int fe_degree>
template <PSMF::LaplaceVariant smooth_vmult, PSMF::SmootherVariant smooth_inv>
void
LaplaceProblem<dim, fe_degree>::do_smooth()
{
  // DP
  using MatrixTypeDP =
    PSMF::LaplaceOperator<dim, fe_degree, full_number, smooth_vmult>;
  MatrixTypeDP matrix_dp;
  matrix_dp.initialize(mfdata_dp, dof_handler, maxlevel);

  using SmootherTypeDP =
    PSMF::PatchSmoother<MatrixTypeDP, dim, fe_degree, smooth_vmult, smooth_inv>;
  SmootherTypeDP                          smooth_dp;
  typename SmootherTypeDP::AdditionalData smoother_data_dp;
  smoother_data_dp.data = mfdata_dp;

  smooth_dp.initialize(matrix_dp, smoother_data_dp);

  Timer  time;
  double best_time = 1e10;

  system_rhs_dp = 1.;

  for (unsigned int i = 0; i < N; ++i)
    {
      time.restart();
      for (unsigned int i = 0; i < n_mv; ++i)
        {
          // smooth_dp.step(solution_dp, system_rhs_dp);
          cudaDeviceSynchronize();
        }
      best_time = std::min(time.wall_time() / n_mv, best_time);
    }

  info_table[3].add_value("dofs", dof_handler.n_dofs());
  info_table[3].add_value("Name",
                          std::string(LaplaceToString(smooth_vmult)) + " " +
                            std::string(SmootherToString(smooth_inv)) + " DP");
  info_table[3].add_value("Time[s]", best_time);
  info_table[3].add_value("Perf[Dof/s]", n_dofs / best_time);

  // SP
  using MatrixTypeSP =
    PSMF::LaplaceOperator<dim, fe_degree, vcycle_number, smooth_vmult>;
  MatrixTypeSP matrix_sp;
  matrix_sp.initialize(mfdata_sp, dof_handler, maxlevel);

  using SmootherTypeSP =
    PSMF::PatchSmoother<MatrixTypeSP, dim, fe_degree, smooth_vmult, smooth_inv>;
  SmootherTypeSP                          smooth_sp;
  typename SmootherTypeSP::AdditionalData smoother_data_sp;
  smoother_data_sp.data = mfdata_sp;

  smooth_sp.initialize(matrix_sp, smoother_data_sp);

  system_rhs_sp = 1.;

  for (unsigned int i = 0; i < N; ++i)
    {
      time.restart();
      for (unsigned int i = 0; i < n_mv; ++i)
        {
          // smooth_sp.step(solution_sp, system_rhs_sp);
          cudaDeviceSynchronize();
        }
      best_time = std::min(time.wall_time() / n_mv, best_time);
    }

  info_table[4].add_value("dofs", dof_handler.n_dofs());
  info_table[4].add_value("Name",
                          std::string(LaplaceToString(smooth_vmult)) + " " +
                            std::string(SmootherToString(smooth_inv)) + " SP");
  info_table[4].add_value("Time[s]", best_time);
  info_table[4].add_value("Perf[Dof/s]", n_dofs / best_time);
}
template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::bench_smooth()
{
  *pcout << "Benchmarking Smoothing...\n";

  for (unsigned int k = 0; k < CT::SMOOTH_VMULT_.size(); ++k)
    switch (CT::SMOOTH_VMULT_[k])
      {
        case PSMF::LaplaceVariant::Basic:
          for (unsigned int j = 0; j < CT::SMOOTH_INV_.size(); ++j)
            switch (CT::SMOOTH_INV_[j])
              {
                case PSMF::SmootherVariant::GLOBAL:
                  do_smooth<PSMF::LaplaceVariant::Basic,
                            PSMF::SmootherVariant::GLOBAL>();
                  break;
                case PSMF::SmootherVariant::FUSED_L:
                  do_smooth<PSMF::LaplaceVariant::Basic,
                            PSMF::SmootherVariant::FUSED_L>();
                  break;
                case PSMF::SmootherVariant::ConflictFree:
                  do_smooth<PSMF::LaplaceVariant::Basic,
                            PSMF::SmootherVariant::ConflictFree>();
                  break;
              }
          break;
        case PSMF::LaplaceVariant::ConflictFree:
          for (unsigned int j = 0; j < CT::SMOOTH_INV_.size(); ++j)
            switch (CT::SMOOTH_INV_[j])
              {
                case PSMF::SmootherVariant::GLOBAL:
                  do_smooth<PSMF::LaplaceVariant::ConflictFree,
                            PSMF::SmootherVariant::GLOBAL>();
                  break;
                case PSMF::SmootherVariant::FUSED_L:
                  do_smooth<PSMF::LaplaceVariant::ConflictFree,
                            PSMF::SmootherVariant::FUSED_L>();
                  break;
                case PSMF::SmootherVariant::ConflictFree:
                  do_smooth<PSMF::LaplaceVariant::ConflictFree,
                            PSMF::SmootherVariant::ConflictFree>();
                  break;
              }
          break;
        case PSMF::LaplaceVariant::TensorCore:
          for (unsigned int j = 0; j < CT::SMOOTH_INV_.size(); ++j)
            switch (CT::SMOOTH_INV_[j])
              {
                case PSMF::SmootherVariant::GLOBAL:
                  do_smooth<PSMF::LaplaceVariant::TensorCore,
                            PSMF::SmootherVariant::GLOBAL>();
                  break;
                case PSMF::SmootherVariant::FUSED_L:
                  do_smooth<PSMF::LaplaceVariant::TensorCore,
                            PSMF::SmootherVariant::FUSED_L>();
                  break;
                case PSMF::SmootherVariant::ConflictFree:
                  do_smooth<PSMF::LaplaceVariant::TensorCore,
                            PSMF::SmootherVariant::ConflictFree>();
                  break;
              }
          break;
        case PSMF::LaplaceVariant::TensorCoreMMA:
          for (unsigned int j = 0; j < CT::SMOOTH_INV_.size(); ++j)
            switch (CT::SMOOTH_INV_[j])
              {
                case PSMF::SmootherVariant::GLOBAL:
                  do_smooth<PSMF::LaplaceVariant::TensorCoreMMA,
                            PSMF::SmootherVariant::GLOBAL>();
                  break;
                case PSMF::SmootherVariant::FUSED_L:
                  do_smooth<PSMF::LaplaceVariant::TensorCoreMMA,
                            PSMF::SmootherVariant::FUSED_L>();
                  break;
                case PSMF::SmootherVariant::ConflictFree:
                  do_smooth<PSMF::LaplaceVariant::TensorCoreMMA,
                            PSMF::SmootherVariant::ConflictFree>();
                  break;
              }
          break;
        default:
          AssertThrow(false, ExcMessage("Invalid Smoother Variant."));
      }
}

template <int dim, int fe_degree>
void
LaplaceProblem<dim, fe_degree>::run()
{
  *pcout << Util::generic_info_to_fstring() << std::endl;

#ifdef DUPLICATE
  Triangulation<dim> tria(
    Triangulation<dim>::limit_level_difference_at_vertices);

  GridGenerator::hyper_cube(tria, 0, 1);
  if (dim == 2)
    GridGenerator::replicate_triangulation(tria, {2, 2}, triangulation);
  else if (dim == 3)
    GridGenerator::replicate_triangulation(tria, {2, 2, 1}, triangulation);
#else
  GridGenerator::hyper_cube(triangulation, 0, 1);
#endif

  double n_dofs_1d = 0;
  if (dim == 2)
    n_dofs_1d = std::sqrt(CT::MAX_SIZES_);
  else if (dim == 3)
    n_dofs_1d = std::cbrt(CT::MAX_SIZES_);

  auto n_refinement =
    static_cast<unsigned int>(std::log2(n_dofs_1d / (fe_degree + 1)));

#if MODE == 0
  triangulation.refine_global(n_refinement);
  setup_system();
  bench_Ax();
  bench_transfer();
  bench_smooth();
#elif MODE == 1
  for (unsigned int cycle = 0; cycle < n_refinement; ++cycle)
    {
      triangulation.refine_global(1);
      setup_system();
      bench_Ax();
      bench_transfer();
      bench_smooth();
    }
#endif

  *pcout << std::endl;

  for (unsigned int k = 0; k < 5; ++k)
    {
      std::ostringstream oss;

      info_table[k].set_scientific("Time[s]", true);
      info_table[k].set_precision("Time[s]", 3);
      info_table[k].set_scientific("Perf[Dof/s]", true);
      info_table[k].set_precision("Perf[Dof/s]", 3);
      info_table[k].set_scientific("abs error", true);
      info_table[k].set_precision("abs error", 3);
      info_table[k].set_scientific("rel error", true);
      info_table[k].set_precision("rel error", 3);

      info_table[k].write_text(oss);
      *pcout << oss.str() << std::endl;
      *pcout << std::endl;
    }
}

int
main(int argc, char *argv[])
{
  try
    {
      {
        int device_id = findCudaDevice(argc, (const char **)argv);
        AssertCuda(cudaSetDevice(device_id));
      }

      {
        LaplaceProblem<CT::DIMENSION_, CT::FE_DEGREE_> Laplace_problem;
        Laplace_problem.run();
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
