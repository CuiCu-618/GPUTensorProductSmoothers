/**
 * @file utilities.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief collection of solvers
 * @version 1.0
 * @date 2022-12-26
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef SOLVER_CUH
#define SOLVER_CUH

#include <deal.II/base/function.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

#include <functional>
#include <optional>

#include "cuda_mg_transfer.cuh"
#include "dealii/cuda_mf.cuh"
#include "laplace_operator.cuh"
#include "patch_base.cuh"
#include "patch_smoother.cuh"

using namespace dealii;

namespace PSMF
{
  template <bool is_zero, typename Number>
  __global__ void
  set_inhomogeneous_dofs(const unsigned int *indicex,
                         const Number       *values,
                         const unsigned int  n_inhomogeneous_dofs,
                         Number             *dst)
  {
    const unsigned int dof =
      threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);

    if (dof < n_inhomogeneous_dofs)
      {
        if (is_zero)
          dst[indicex[dof]] = 0;
        else
          dst[indicex[dof]] = values[dof];
      }
  }

  template <bool is_d2f, typename number, typename number2>
  __global__ void
  copy_vector(number *dst, const number2 *src, const unsigned n_dofs)
  {
    const unsigned int dof =
      threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);

    if (dof < n_dofs)
      {
        if (is_d2f)
          dst[dof] = __double2float_rn(src[dof]);
        else
          dst[dof] = src[dof];
      }
  }

  // A coarse solver defined via the smoother
  template <typename VectorType, typename SmootherType>
  class MGCoarseFromSmoother : public MGCoarseGridBase<VectorType>
  {
  public:
    MGCoarseFromSmoother(const SmootherType &mg_smoother, const bool is_empty)
      : smoother(mg_smoother)
      , is_empty(is_empty)
    {}

    virtual void
    operator()(const unsigned int level,
               VectorType        &dst,
               const VectorType  &src) const
    {
      if (is_empty)
        return;
      smoother[level].vmult(dst, src);
    }

    const SmootherType &smoother;
    const bool          is_empty;
  };

  struct SolverData
  {
    std::string solver_name = "";

    int    n_iteration      = 0;
    double residual         = 0.;
    double reduction_rate   = 0.;
    double convergence_rate = 0.;
    double l2_error         = 0.;
    int    mem_usage        = 0.;
    double timing           = 0.;
    double perf             = 0.;

    std::string
    print_comp()
    {
      std::ostringstream oss;

      oss.width(12);
      oss.precision(4);
      oss.setf(std::ios::left);
      oss.setf(std::ios::scientific);

      oss << std::left << std::setw(12) << solver_name << std::setprecision(4)
          << std::setw(12) << timing << std::setprecision(4) << std::setw(12)
          << perf << std::endl;

      return oss.str();
    }

    std::string
    print_solver()
    {
      std::ostringstream oss;

      oss.width(12);
      oss.precision(4);
      oss.setf(std::ios::left);
      oss.setf(std::ios::scientific);

      oss << std::left << std::setw(12) << solver_name << std::setw(4)
          << n_iteration << std::setprecision(4) << std::setw(12) << timing;

      // if (CT::SETS_ == "error_analysis")
      oss << std::left << std::setprecision(4) << std::setw(12) << residual
          << std::setprecision(4) << std::setw(12) << reduction_rate
          << std::setprecision(4) << std::setw(12) << convergence_rate;

      oss << std::left << std::setw(8) << mem_usage << std::endl;

      return oss.str();
    }
  };


  // Mixed-precision multigrid solver setup
  template <int dim, int fe_degree, typename Number, typename Number2>
  class MultigridSolvers
  {
  public:
    MultigridSolvers(const DoFHandler<dim>              &dof_handler,
                     const Function<dim>                &boundary_values,
                     std::shared_ptr<ConditionalOStream> pcout,
                     const unsigned int                  N,
                     const double                        tau,
                     const double                        wave_number,
                     const double                        a_t,
                     const unsigned int                  n_cycles = 1)
      : dof_handler(&dof_handler)
      , minlevel(1)
      , maxlevel(dof_handler.get_triangulation().n_global_levels() - 1)
      , solution(minlevel, maxlevel)
      , solution_old(minlevel, maxlevel)
      , rhs(minlevel, maxlevel)
      , residual(minlevel, maxlevel)
      , defect(minlevel, maxlevel)
      , t(minlevel, maxlevel)
      , solution_update(minlevel, maxlevel)
      , matrix(minlevel, maxlevel)
      , matrix_dp(minlevel, maxlevel)
      , smooth(minlevel, maxlevel)
      , coarse(smooth, false)
      , N(N)
      , tau(tau)
      , wave_number(wave_number)
      , a_t(a_t)
      , n_cycles(n_cycles)
      , timings(maxlevel + 1)
      , analytic_solution(boundary_values)
      , pcout(pcout)
    {
      AssertDimension(fe_degree, dof_handler.get_fe().degree);

      // Initialization of Dirichlet boundaries
      std::set<types::boundary_id> dirichlet_boundary;
      dirichlet_boundary.insert(0);
      mg_constrained_dofs.initialize(dof_handler);
      mg_constrained_dofs.make_zero_boundary_constraints(dof_handler,
                                                         dirichlet_boundary);

      // set up a mapping for the geometry representation
      MappingQGeneric<dim> mapping(std::min(fe_degree, 10));

      Timer time;

      for (unsigned int level = minlevel; level <= maxlevel; ++level)
        {
          IndexSet relevant_dofs;
          DoFTools::extract_locally_relevant_level_dofs(dof_handler,
                                                        level,
                                                        relevant_dofs);
          AffineConstraints<Number> level_constraints;
          level_constraints.reinit(relevant_dofs);
          level_constraints.add_lines(
            mg_constrained_dofs.get_boundary_indices(level));
          level_constraints.close();

          AffineConstraints<Number2> level_constraints2;
          level_constraints2.reinit(relevant_dofs);
          level_constraints2.add_lines(
            mg_constrained_dofs.get_boundary_indices(level));
          level_constraints2.close();

          // single-precision matrix-free data
          {
            typename PSMF::MatrixFree<dim, Number>::AdditionalData
              additional_data;
            additional_data.mapping_update_flags =
              (update_gradients | update_JxW_values | update_values |
               update_quadrature_points);
            additional_data.mg_level = level;
            std::shared_ptr<PSMF::MatrixFree<dim, Number>> mg_mf_storage_level(
              new PSMF::MatrixFree<dim, Number>());
            mg_mf_storage_level->reinit(mapping,
                                        dof_handler,
                                        level_constraints,
                                        QGauss<1>(fe_degree + 1),
                                        additional_data);

            matrix[level].initialize(mg_mf_storage_level,
                                     tau,
                                     wave_number,
                                     a_t);

            matrix[level].initialize_dof_vector(defect[level]);
            t[level]               = defect[level];
            solution_update[level] = defect[level];
          }

          // double-precision matrix-free data
          {
            typename PSMF::MatrixFree<dim, Number2>::AdditionalData
              additional_data;
            additional_data.mapping_update_flags =
              (update_gradients | update_JxW_values | update_values |
               update_quadrature_points);
            additional_data.mg_level = level;
            std::shared_ptr<PSMF::MatrixFree<dim, Number2>> mg_mf_storage_level(
              new PSMF::MatrixFree<dim, Number2>());
            mg_mf_storage_level->reinit(mapping,
                                        dof_handler,
                                        level_constraints2,
                                        QGauss<1>(fe_degree + 1),
                                        additional_data);

            matrix_dp[level].initialize(mg_mf_storage_level,
                                        tau,
                                        wave_number,
                                        a_t);

            matrix_dp[level].initialize_dof_vector(solution[level]);
            solution_old[level] = solution[level];
            rhs[level]          = solution[level];
            residual[level]     = solution[level];
          }
        }

      *pcout << "MatrixFree setup time:    " << time.wall_time() << std::endl;
      time.restart();

      // build two level transfers; one is without boundary conditions for the
      // transfer of the solution (with inhomogeneous boundary conditions),
      // and one is for the homogeneous part in the v-cycle
      {
        mg_transfer_no_boundary.build(dof_handler);
      }
      {
        transfer.initialize_constraints(mg_constrained_dofs);
        transfer.build(dof_handler);
      }
      *pcout << "MG transfer setup time:   " << time.wall_time() << std::endl;

      // interpolate the inhomogeneous boundary conditions
      inhomogeneous_bc.clear();
      inhomogeneous_bc.resize(maxlevel + 1);
      for (unsigned int level = maxlevel; level <= maxlevel; ++level)
        {
          Quadrature<dim - 1> face_quad(
            dof_handler.get_fe().get_unit_face_support_points());
          FEFaceValues<dim>                    fe_values(mapping,
                                      dof_handler.get_fe(),
                                      face_quad,
                                      update_quadrature_points);
          std::vector<types::global_dof_index> face_dof_indices(
            dof_handler.get_fe().dofs_per_face);
          typename DoFHandler<dim>::cell_iterator cell =
                                                    dof_handler.begin(level),
                                                  endc = dof_handler.end(level);
          for (; cell != endc; ++cell)
            if (cell->level_subdomain_id() != numbers::artificial_subdomain_id)
              for (unsigned int face_no = 0;
                   face_no < GeometryInfo<dim>::faces_per_cell;
                   ++face_no)
                if (cell->at_boundary(face_no))
                  {
                    const typename DoFHandler<dim>::face_iterator face =
                      cell->face(face_no);
                    face->get_mg_dof_indices(level, face_dof_indices);
                    fe_values.reinit(cell, face_no);
                    for (unsigned int i = 0; i < face_dof_indices.size(); ++i)
                      if (dof_handler.locally_owned_mg_dofs(level).is_element(
                            face_dof_indices[i]))
                        {
                          const double value = analytic_solution.value(
                            fe_values.quadrature_point(i));
                          if (value != 0.0)
                            inhomogeneous_bc[level][face_dof_indices[i]] =
                              value;
                        }
                  }

          // evaluate the right hand side in the equation, including the
          // residual from the inhomogeneous boundary conditions
          set_inhomogeneous_bc<false>(level);

          // U0
          {
            const unsigned int n_dofs = solution[level].size();

            LinearAlgebra::distributed::Vector<Number2, MemorySpace::Host>
              system_u_host(n_dofs);
            LinearAlgebra::distributed::Vector<Number2, MemorySpace::CUDA>
                                                    system_u_dev(n_dofs);
            LinearAlgebra::ReadWriteVector<Number2> rw_vector(n_dofs);

            // U
            std::map<types::global_dof_index, Point<dim>> current_point_map;
            DoFTools::map_dofs_to_support_points<dim>(MappingQGeneric<dim>(
                                                        fe_degree + 1),
                                                      dof_handler,
                                                      current_point_map);
            for (unsigned int i = 0; i < n_dofs; ++i)
              system_u_host[i] = analytic_solution.value(current_point_map[i]);

            rw_vector.import(system_u_host, VectorOperation::insert);
            solution_old[level].import(rw_vector, VectorOperation::insert);
          }

          auto T = tau;

          time.restart();
          matrix_dp[level].compute_rhs(rhs[level], solution_old[level], T);
        }


      const double rhs_norm = rhs[maxlevel].l2_norm();
      *pcout << "Time compute rhs:         " << time.wall_time()
             << " rhs_norm = " << rhs_norm << std::endl;

      time.restart();
      for (unsigned int level = minlevel; level <= maxlevel; ++level)
        {
          typename SmootherType::AdditionalData smoother_data;

          smoother_data.tau        = tau;
          smoother_data.relaxation = 1.;
          smoother_data.patch_per_block =
            fe_degree == 1 ? 16 : (fe_degree == 2 ? 2 : 1);
          smoother_data.granularity_scheme = CT::GRANULARITY_;

          smooth[level].initialize(matrix[level], smoother_data);
        }
      *pcout << "Time initial smoother:    " << time.wall_time() << std::endl;
    }

    // Print a summary of computation times on the various levels
    void
    print_wall_times()
    {
      {
        *pcout << "Coarse solver " << (int)timings[minlevel][1]
               << " times: " << timings[minlevel][0] << " tot prec "
               << timings[minlevel][2] << std::endl;
        *pcout
          << "level  smoother    mg_mv     mg_vec    restrict  prolongate  inhomBC"
          << std::endl;
        for (unsigned int level = minlevel + 1; level <= maxlevel; ++level)
          {
            *pcout << "L" << std::setw(2) << std::left << level << "    ";
            *pcout << std::setprecision(4) << std::setw(12) << timings[level][5]
                   << std::setw(10) << timings[level][0] << std::setw(10)
                   << timings[level][4] << std::setw(10) << timings[level][1]
                   << std::setw(12) << timings[level][2] << std::setw(10)
                   << timings[level][3] << std::endl;
          }
        *pcout << std::setprecision(5);
      }
      for (unsigned int l = 0; l < timings.size(); ++l)
        for (unsigned int j = 0; j < timings[l].size(); ++j)
          timings[l][j] = 0.;
    }



    // Return the solution vector for further processing
    const LinearAlgebra::distributed::Vector<Number2, MemorySpace::CUDA> &
    get_solution()
    {
      set_inhomogeneous_bc<false>(maxlevel);
      return solution[maxlevel];
    }

    // Solve with the FMG cycle and return the reduction rate of a V-cycle
    double
    solve(const bool do_analyze)
    {
      double reduction_rate = 1.;

      Timer time;

      // copy double to float, invoke coarse solver twice (improves accuracy
      // for high-order methods where 1e-3 might not be enough, and this is
      // done only once anyway), and copy back to double

      // defect[minlevel] = rhs[minlevel];
      convert_precision<true>(defect[minlevel], rhs[minlevel]);

      coarse(minlevel, t[minlevel], defect[minlevel]);
      smooth[minlevel].step(t[minlevel], defect[minlevel]);

      // solution[minlevel] = t[minlevel];
      convert_precision<false>(solution[minlevel], t[minlevel]);

      cudaDeviceSynchronize();
      timings[minlevel][0] += time.wall_time();
      timings[minlevel][1] += 2;

      for (unsigned int level = minlevel + 1; level <= maxlevel; ++level)
        {
          // interpolate inhomogeneous boundary values
          Timer time;
          set_inhomogeneous_bc<false>(level - 1);
          cudaDeviceSynchronize();
          timings[level][3] += time.wall_time();

          // prolongate (without boundary conditions) to next finer level in
          // double precision
          time.restart();
          mg_transfer_no_boundary.prolongate(level,
                                             solution[level],
                                             solution[level - 1]);
          cudaDeviceSynchronize();
          timings[level][2] += time.wall_time();

          set_inhomogeneous_bc<true>(level);

          // compute residual in double precision
          time.restart();
          matrix_dp[level].vmult(residual[level], solution[level]);
          residual[level].sadd(-1., 1., rhs[level]);

          cudaDeviceSynchronize();
          timings[level][0] += time.wall_time();

          time.restart();

          // copy to single precision
          // defect[level] = residual[level];
          convert_precision<false>(defect[level], residual[level]);

          cudaDeviceSynchronize();
          timings[level][4] += time.wall_time();


          // run v-cycle to obtain correction
          v_cycle(level, n_cycles);

          time.restart();

          // add correction
          // internal::add_vector(solution[level], solution_update[level]);
          convert_precision<false>(residual[level], solution_update[level]);
          solution[level] += residual[level];


          cudaDeviceSynchronize();
          timings[level][4] += time.wall_time();
        }
      return reduction_rate;
    }

    // Solve with the conjugate gradient method preconditioned by the V-cycle
    // (invoking this->vmult() or vmult_with_residual_update()) and return the
    // number of iterations and the reduction rate per GMRES iteration
    std::optional<ReductionControl>
    solve_gmres(const bool do_analyze, const unsigned int N = 1)
    {
      ReductionControl solver_control(1000, 1e-16, CT::REDUCE_);
      solver_control.enable_history_data();
      solver_control.log_history(true);

      SolverGMRES<VectorType2> solver_gmres(solver_control);
      solution[maxlevel] = 0;

      if (N == 1)
        {
          solver_gmres.solve(matrix_dp[maxlevel],
                             solution[maxlevel],
                             rhs[maxlevel],
                             *this);

          if (do_analyze)
            return solver_control;
          else
            return std::nullopt;
        }

      for (unsigned int it = 1; it <= N; ++it)
        {
          Timer time;
          auto  T = it * tau;
          *pcout << "Time step " << it << " at t=" << T << std::endl;

          matrix_dp[maxlevel].compute_rhs(rhs[maxlevel],
                                          solution_old[maxlevel],
                                          T);

          *pcout << "Time compute rhs:         " << time.wall_time() << "\n";
          time.restart();

          solver_gmres.solve(matrix_dp[maxlevel],
                             solution[maxlevel],
                             rhs[maxlevel],
                             *this);

          *pcout << "Time solve GMRES:         " << time.wall_time() << "\n";

          *pcout << "     " << solver_control.last_step()
                 << " GMRES iterations." << "\n\n";

          solution_old[maxlevel] = solution[maxlevel];
        }

      return std::nullopt;
    }



    // Implement the vmult() function needed by the preconditioner interface
    void
    vmult(LinearAlgebra::distributed::Vector<Number2, MemorySpace::CUDA> &dst,
          const LinearAlgebra::distributed::Vector<Number2, MemorySpace::CUDA>
            &src) const
    {
      Timer time1, time;
      // defect[maxlevel].copy_locally_owned_data_from(src);
      convert_precision<true>(defect[maxlevel], src);
      timings[maxlevel][4] += time.wall_time();
      v_cycle(maxlevel, 1);
      time.restart();
      // dst.copy_locally_owned_data_from(solution_update[maxlevel]);
      convert_precision<false>(dst, solution_update[maxlevel]);
      timings[maxlevel][4] += time.wall_time();
      timings[minlevel][2] += time1.wall_time();
    }

    // run matrix-vector product in double precision
    void
    do_matvec()
    {
      matrix_dp[maxlevel].vmult(residual[maxlevel], solution[maxlevel]);
      cudaDeviceSynchronize();
    }



    // run matrix-vector product in single precision
    void
    do_matvec_smoother()
    {
      smooth[maxlevel].vmult(solution_update[maxlevel], defect[maxlevel]);
      cudaDeviceSynchronize();
    }

  private:
    // Implement the V-cycle
    void
    v_cycle(const unsigned int level, const unsigned int my_n_cycles) const
    {
      if (level == minlevel)
        {
          Timer time;
          (coarse)(level, solution_update[level], defect[level]);
          cudaDeviceSynchronize();
          timings[level][0] += time.wall_time();
          timings[level][1] += 1;
          return;
        }

      for (unsigned int c = 0; c < my_n_cycles; ++c)
        {
          Timer time;
          if (c == 0)
            (smooth)[level].vmult(solution_update[level], defect[level]);
          else
            (smooth)[level].step(solution_update[level], defect[level]);
          cudaDeviceSynchronize();
          timings[level][5] += time.wall_time();

          time.restart();
          (matrix)[level].vmult(t[level], solution_update[level]);
          t[level].sadd(-1.0, 1.0, defect[level]);
          cudaDeviceSynchronize();
          timings[level][0] += time.wall_time();

          time.restart();
          defect[level - 1] = 0;
          transfer.restrict_and_add(level, defect[level - 1], t[level]);
          cudaDeviceSynchronize();
          timings[level][1] += time.wall_time();

          v_cycle(level - 1, 1);

          time.restart();
          transfer.prolongate_and_add(level,
                                      solution_update[level],
                                      solution_update[level - 1]);
          cudaDeviceSynchronize();
          timings[level][2] += time.wall_time();

          time.restart();
          (smooth)[level].step(solution_update[level], defect[level]);
          cudaDeviceSynchronize();
          timings[level][5] += time.wall_time();
        }
    }

    template <bool is_zero = false>
    void
    set_inhomogeneous_bc(const unsigned int level)
    {
      unsigned int n_inhomogeneous_bc = inhomogeneous_bc[level].size();
      if (n_inhomogeneous_bc != 0)
        {
          const unsigned int block_size = 256;
          const unsigned int inhomogeneous_n_blocks =
            std::ceil(static_cast<double>(n_inhomogeneous_bc) /
                      static_cast<double>(block_size));
          const unsigned int inhomogeneous_x_n_blocks =
            std::round(std::sqrt(inhomogeneous_n_blocks));
          const unsigned int inhomogeneous_y_n_blocks =
            std::ceil(static_cast<double>(inhomogeneous_n_blocks) /
                      static_cast<double>(inhomogeneous_x_n_blocks));

          dim3 inhomogeneous_grid_dim(inhomogeneous_x_n_blocks,
                                      inhomogeneous_y_n_blocks);
          dim3 inhomogeneous_block_dim(block_size);

          std::vector<unsigned int> inhomogeneous_index_host(
            n_inhomogeneous_bc);
          std::vector<Number2> inhomogeneous_value_host(n_inhomogeneous_bc);
          unsigned int         count = 0;
          for (auto &i : inhomogeneous_bc[level])
            {
              inhomogeneous_index_host[count] = i.first;
              inhomogeneous_value_host[count] = i.second;
              count++;
            }

          unsigned int *inhomogeneous_index;
          Number2      *inhomogeneous_value;

          cudaError_t cuda_error =
            cudaMalloc(&inhomogeneous_index,
                       n_inhomogeneous_bc * sizeof(unsigned int));
          AssertCuda(cuda_error);

          cuda_error = cudaMemcpy(inhomogeneous_index,
                                  inhomogeneous_index_host.data(),
                                  n_inhomogeneous_bc * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice);
          AssertCuda(cuda_error);

          cuda_error = cudaMalloc(&inhomogeneous_value,
                                  n_inhomogeneous_bc * sizeof(Number2));
          AssertCuda(cuda_error);

          cuda_error = cudaMemcpy(inhomogeneous_value,
                                  inhomogeneous_value_host.data(),
                                  n_inhomogeneous_bc * sizeof(Number2),
                                  cudaMemcpyHostToDevice);
          AssertCuda(cuda_error);


          set_inhomogeneous_dofs<is_zero, Number2>
            <<<inhomogeneous_grid_dim, inhomogeneous_block_dim>>>(
              inhomogeneous_index,
              inhomogeneous_value,
              n_inhomogeneous_bc,
              solution[level].get_values());
          AssertCudaKernel();
        }
    }

    template <bool is_d2f = true, typename number, typename number2>
    void
    convert_precision(
      LinearAlgebra::distributed::Vector<number, MemorySpace::CUDA>        &dst,
      const LinearAlgebra::distributed::Vector<number2, MemorySpace::CUDA> &src)
      const
    {
      unsigned int n_dofs = dst.size();
      if (n_dofs != 0)
        {
          const unsigned int block_size = 256;
          const unsigned int n_blocks   = std::ceil(
            static_cast<double>(n_dofs) / static_cast<double>(block_size));
          const unsigned int x_n_blocks = std::round(std::sqrt(n_blocks));
          const unsigned int y_n_blocks = std::ceil(
            static_cast<double>(n_blocks) / static_cast<double>(x_n_blocks));

          dim3 grid_dim(x_n_blocks, y_n_blocks);
          dim3 block_dim(block_size);
          copy_vector<is_d2f, number, number2>
            <<<grid_dim, block_dim>>>(dst.get_values(),
                                      src.get_values(),
                                      n_dofs);
          AssertCudaKernel();
        }
    }


    const SmartPointer<const DoFHandler<dim>> dof_handler;

    std::vector<std::map<types::global_dof_index, Number2>> inhomogeneous_bc;

    MGConstrainedDoFs mg_constrained_dofs;

    MGTransferCUDA<dim, Number2> mg_transfer_no_boundary;
    MGTransferCUDA<dim, Number>  transfer;

    typedef LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>
      VectorType;
    typedef LinearAlgebra::distributed::Vector<Number2, MemorySpace::CUDA>
      VectorType2;

    /**
     * Lowest level of cells.
     */
    unsigned int minlevel;

    /**
     * Highest level of cells.
     */
    unsigned int maxlevel;

    /**
     * The solution vector
     */
    mutable MGLevelObject<VectorType2> solution;
    mutable MGLevelObject<VectorType2> solution_old;

    /**
     * Original right hand side vector
     */
    mutable MGLevelObject<VectorType2> rhs;

    /**
     * Residual vector before it is passed down into float through the v-cycle
     */
    mutable MGLevelObject<VectorType2> residual;

    /**
     * Input vector for the cycle. Contains the defect of the outer method
     * projected to the multilevel vectors.
     */
    mutable MGLevelObject<VectorType> defect;

    /**
     * Auxiliary vector.
     */
    mutable MGLevelObject<VectorType> t;

    /**
     * Auxiliary vector for the solution update
     */
    mutable MGLevelObject<VectorType> solution_update;

    /**
     * The matrix for each level
     */
    MGLevelObject<LaplaceOperator<dim, fe_degree, Number, CT::DOF_LAYOUT_>>
      matrix;

    /**
     * The double-precision matrix for the outer correction
     */
    MGLevelObject<LaplaceOperator<dim, fe_degree, Number2, CT::DOF_LAYOUT_>>
      matrix_dp;

    /**
     * The smoother object
     */
    typedef PatchSmoother<
      LaplaceOperator<dim, fe_degree, Number, CT::DOF_LAYOUT_>,
      dim,
      fe_degree,
      CT::KERNEL_TYPE_[0],
      CT::DOF_LAYOUT_>
                                SmootherType;
    MGLevelObject<SmootherType> smooth;

    /**
     * The coarse solver
     */
    MGCoarseFromSmoother<VectorType, MGLevelObject<SmootherType>> coarse;

    const unsigned int N;
    const double       tau;
    const double       wave_number;
    const double       a_t;

    /**
     * Number of cycles to be done in the FMG cycle
     */
    const unsigned int n_cycles;

    /**
     * Collection of compute times on various levels
     */
    mutable std::vector<std::array<double, 6>> timings;

    /**
     * Function for boundary values that we keep as analytic solution
     */
    const Function<dim> &analytic_solution;

    std::shared_ptr<ConditionalOStream> pcout;
  };


} // namespace PSMF

#endif // SOLVER_CUH
