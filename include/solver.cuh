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

#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_matrix.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/multigrid.h>

#include <functional>
#include <optional>

#include "ct_parameter.h"
#include "cuda_mg_transfer.cuh"
#include "dealii/cuda_mf.cuh"
#include "laplace_operator.cuh"
#include "patch_base.cuh"
#include "patch_smoother.cuh"

int current_stage = 0;

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

  template <bool is_d2f = true, typename number, typename number2>
  void
  convert_precision(
    LinearAlgebra::distributed::Vector<number, MemorySpace::CUDA>        &dst,
    const LinearAlgebra::distributed::Vector<number2, MemorySpace::CUDA> &src)
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
          <<<grid_dim, block_dim>>>(dst.get_values(), src.get_values(), n_dofs);
        AssertCudaKernel();
      }
  }


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

  // #define MIXED

  template <typename VectorType>
  class MYSolverFGMRES : public SolverBase<VectorType>
  {
  public:
    using VectorTypeF =
      LinearAlgebra::distributed::Vector<float, MemorySpace::CUDA>;

    MYSolverFGMRES(SolverControl     &solver_control,
                   const unsigned int GCRmaxit = 20)
      : SolverBase<VectorType>(solver_control)
      , GCRmaxit(GCRmaxit)
#ifdef MIXED
      , mem(static_vector_memory)
#endif
    {
      solver_control.set_max_steps(GCRmaxit);
    }

    template <typename MatrixType, typename PreconditionerType>
    void
    solve(const MatrixType         &A,
          VectorType               &x,
          const VectorType         &b,
          const PreconditionerType &preconditioner)
    {
      SolverControl::State iteration_state = SolverControl::iterate;

      const unsigned int basis_size = GCRmaxit;
#ifdef MIXED
      dealii::internal::SolverGMRESImplementation::TmpVectors<VectorTypeF> v(
        basis_size, this->mem);
      dealii::internal::SolverGMRESImplementation::TmpVectors<VectorTypeF> z(
        basis_size, this->mem);
#else
      dealii::internal::SolverGMRESImplementation::TmpVectors<VectorType> v(
        basis_size, this->memory);
      dealii::internal::SolverGMRESImplementation::TmpVectors<VectorType> z(
        basis_size, this->memory);
#endif

      unsigned int accumulated_iterations = 0;

      H.reinit(basis_size + 1, basis_size);

      Vector<double> projected_rhs;
      Vector<double> y;

      double res = std::numeric_limits<double>::lowest();

#ifdef MIXED
      typename VectorMemory<VectorType>::Pointer aux_t(this->memory);
      aux_t->reinit(x);

      typename VectorMemory<VectorType>::Pointer tmp(this->memory);
      tmp->reinit(x);

      typename VectorMemory<VectorTypeF>::Pointer aux(this->mem);
      aux->reinit(x.size());
#else
      typename VectorMemory<VectorType>::Pointer aux(this->memory);
      aux->reinit(x);
#endif

      do
        {
#ifdef MIXED
          A.vmult(*aux_t, x);
          aux_t->sadd(-1., 1., b);
          convert_precision<true>(*aux, *aux_t);
#else
          A.vmult(*aux, x);
          aux->sadd(-1., 1., b);
#endif

          double beta = aux->l2_norm();
          res         = beta;
          iteration_state =
            this->iteration_status(accumulated_iterations, res, x);
          if (iteration_state == SolverControl::success)
            break;

          H.reinit(basis_size + 1, basis_size);
          double a = beta;

          for (unsigned int j = 0; j < basis_size; ++j)
            {
              if (a != 0) // treat lucky breakdown
                v(j, *aux).equ(1. / a, *aux);
              else
                v(j, *aux) = 0.;


              preconditioner.vmult(z(j, *aux), v[j]);
#ifdef MIXED
              convert_precision<false>(*tmp, z[j]);
              A.vmult(*aux_t, *tmp);
              convert_precision<true>(*aux, *aux_t);
#else
              A.vmult(*aux, z[j]);
#endif

              // Gram-Schmidt
              H(0, j) = *aux * v[0];
              for (unsigned int i = 1; i <= j; ++i)
                H(i, j) = aux->add_and_dot(-H(i - 1, j), v[i - 1], v[i]);
              H(j + 1, j) = a =
                std::sqrt(aux->add_and_dot(-H(j, j), v[j], *aux));

              // Compute projected solution

              if (j > 0)
                {
                  H1.reinit(j + 1, j);
                  projected_rhs.reinit(j + 1);
                  y.reinit(j);
                  projected_rhs(0) = beta;
                  H1.fill(H);

                  Householder<double> house(H1);
                  res = house.least_squares(y, projected_rhs);
                  iteration_state =
                    this->iteration_status(++accumulated_iterations, res, x);
                  if (iteration_state != SolverControl::iterate)
                    break;
                }
            }

          // Update solution vector
          for (unsigned int j = 0; j < y.size(); ++j)
            internal::vec_add(x, z[j], x.size(), y(j), 0, 0);
          // x.add(y(j), z[j]);
        }
      while (iteration_state == SolverControl::iterate);

      // in case of failure: throw exception
      if (iteration_state != SolverControl::success)
        AssertThrow(false,
                    SolverControl::NoConvergence(accumulated_iterations, res));
    }



  private:
    const unsigned int GCRmaxit;

    FullMatrix<double> H;
    FullMatrix<double> H1;

#ifdef MIXED
    mutable GrowingVectorMemory<VectorTypeF> static_vector_memory;

    VectorMemory<VectorTypeF> &mem;
#endif
  };

  template <typename VectorType>
  class SolverGCR : public SolverBase<VectorType>
  {
  public:
    using VectorTypeF =
      LinearAlgebra::distributed::Vector<float, MemorySpace::CUDA>;

    SolverGCR(SolverControl &solver_control, const unsigned int GCRmaxit = 20)
      : SolverBase<VectorType>(solver_control)
      , GCRmaxit(GCRmaxit)
    {
      solver_control.set_max_steps(GCRmaxit);
    }

    template <typename MatrixType, typename PreconditionerType>
    void
    solve(const MatrixType         &A,
          VectorType               &x,
          const VectorType         &b,
          const PreconditionerType &preconditioner)
    {
      using number = typename VectorType::value_type;

      SolverControl::State conv = SolverControl::iterate;

      typename VectorMemory<VectorType>::Pointer search_pointer(this->memory);
      typename VectorMemory<VectorType>::Pointer Asearch_pointer(this->memory);
      typename VectorMemory<VectorType>::Pointer p_pointer(this->memory);

      VectorType &search  = *search_pointer;
      VectorType &Asearch = *Asearch_pointer;
      VectorType &p       = *p_pointer;

      search.reinit(x);
      Asearch.reinit(x);
      p.reinit(x);

      A.vmult(p, x);
      p.add(-1., b); // p = A*x- b # Initalize residual
      p *= -1;	// p = b-A*x # fix sign, this should perhaps be reorganized.
       
      double res = p.l2_norm();
      unsigned int it = 0;

      // Allocate "vectors of vectors"
      dealii::internal::SolverGMRESImplementation::TmpVectors<VectorType> z_vec(GCRmaxit, this->memory);
      dealii::internal::SolverGMRESImplementation::TmpVectors<VectorType> c_vec_h(GCRmaxit, this->memory);
      dealii::internal::SolverGMRESImplementation::TmpVectors<VectorType> c_vec(GCRmaxit, this->memory);

      typename VectorMemory<VectorType>::Pointer aux(this->memory);
      aux->reinit(x);

      FullMatrix<double> gamma(GCRmaxit,GCRmaxit);
      
      std::vector<typename VectorType::value_type> alpha_vec;
      alpha_vec.reserve(GCRmaxit);       
      
      std::vector<typename VectorType::value_type> u_vec;
      u_vec.reserve(GCRmaxit);             
      alpha_vec.resize(GCRmaxit);      

      conv = this->iteration_status(it, res, x);
      if (conv != SolverControl::iterate)
        return;

      while (conv == SolverControl::iterate)
        {
		  preconditioner.vmult(search, p);      
		  z_vec(it,*aux)=search;  
		  A.vmult(Asearch, search);
                  c_vec_h(0,*aux)=Asearch;
		 for( unsigned int i=0 ; i< it ; i++ ){
                        gamma(i,it) = c_vec[i]*c_vec_h[i];			
		        c_vec_h(i+1,*aux)=c_vec_h[i]; 
			c_vec_h(i+1,*aux).add(-gamma(i,it),c_vec[i]);
		 }                    
		gamma(it,it) = std::sqrt(aux->add_and_dot(1.0 ,  c_vec_h[it], c_vec_h[it]) );  
                c_vec(it,*aux).equ( (1./gamma(it,it)), c_vec_h[it] );
		alpha_vec[it] = c_vec[it] * p; 
		p.add( -alpha_vec[it] , c_vec[it] ); 
		res = p.l2_norm();
		it++;
	        conv = this->iteration_status(it, res, x); // I dont think we should send x here as we have not yet updated the solution
	}
	if (conv != SolverControl::success)
        	AssertThrow(false, SolverControl::NoConvergence(it, res));

        u_vec.resize(it);
	for( int j = it-1; j >= 0; j--){
	    for( int i = it-1; i >= j; i--){
		if( i == j){
		    u_vec[j] = alpha_vec[j] / gamma(j, i); 
		}else{
		    alpha_vec[j] = alpha_vec[j] - gamma(j, i) * u_vec[i];
		}
	    }
	}

	for( int i = 0; i<it; i++){
		x.add(u_vec[i],z_vec[i]);
	}
        
        conv = this->iteration_status(it, res, x);
        if (conv != SolverControl::success)
            AssertThrow(false, SolverControl::NoConvergence(it, res));
         
        }

  private:
    const unsigned int GCRmaxit;
  };




  // 0: GMRES
  // 1: FGMRES
  // 2: GCR
  // 3: CG
  template <int N, typename Vectype>
  struct SelectSolver;

  template <typename Vectype>
  struct SelectSolver<0, Vectype>
  {
    using type = SolverGMRES<Vectype>;
  };

  template <typename Vectype>
  struct SelectSolver<1, Vectype>
  {
    using type = MYSolverFGMRES<Vectype>;
    // using type = SolverFGMRES<Vectype>;
  };

  template <typename Vectype>
  struct SelectSolver<2, Vectype>
  {
    using type = SolverGCR<Vectype>;
  };

  template <typename Vectype>
  struct SelectSolver<3, Vectype>
  {
    using type = SolverCG<Vectype>;
  };

  template <int dim, int fe_degree, typename Number, typename Number2>
  class Preconditioner
  {
  public:
    using VectorType =
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>;
    using VectorType2 =
      LinearAlgebra::distributed::Vector<Number2, MemorySpace::CUDA>;

    using LocalMatrixType =
      LaplaceOperator<dim, fe_degree, Number, CT::DOF_LAYOUT_>;
    using SmootherType = PatchSmoother<LocalMatrixType,
                                       dim,
                                       fe_degree,
                                       CT::KERNEL_TYPE_[0],
                                       CT::DOF_LAYOUT_>;

    Preconditioner(
      const PreconditionMG<dim, VectorType, MGTransferCUDA<dim, Number>>
                        &precon,
      const unsigned int n_dofs)
      : precon(precon)
    {
      if (std::is_same<Number, Number2>::value == false)
        {
          defect.reinit(n_dofs);
          solution_update.reinit(n_dofs);
        }
    }

    void
    vmult(VectorType2 &dst, const VectorType2 &src) const
    {
      if (std::is_same<Number, Number2>::value == true)
        {
          precon.vmult(dst, src);
        }
      else
        {
          convert_precision<true>(defect, src);

          precon.vmult(solution_update, defect);

          convert_precision<false>(dst, solution_update);
        }
    }

  private:
    const PreconditionMG<dim, VectorType, MGTransferCUDA<dim, Number>> &precon;

    mutable VectorType defect;
    mutable VectorType solution_update;
  };


  /**
   * @brief Multigrid Method with vertex-patch smoother.
   *
   * @tparam dim
   * @tparam fe_degree
   * @tparam Number vcycle number
   * @tparam Number2 inner gmres number
   */
  template <int dim, int fe_degree, typename Number, typename Number2>
  class MultigridSolvers
  {
  public:
    using VectorType =
      LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA>;
    using VectorType2 =
      LinearAlgebra::distributed::Vector<Number2, MemorySpace::CUDA>;
    using VectorTypeD =
      LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>;

    using LocalMatrixType =
      LaplaceOperator<dim, fe_degree, Number, CT::DOF_LAYOUT_>;
    using LocalMatrixType2 =
      LaplaceOperator<dim, fe_degree, Number2, CT::DOF_LAYOUT_>;

    using SystemMatrixType = SystemMatrixOp<dim, fe_degree, double>;

    using SmootherType = PatchSmoother<LocalMatrixType,
                                       dim,
                                       fe_degree,
                                       CT::KERNEL_TYPE_[0],
                                       CT::DOF_LAYOUT_>;

    using SmootherTypeCheb = PreconditionChebyshev<LocalMatrixType, VectorType>;

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
      , N(N)
      , tau(tau)
      , wave_number(wave_number)
      , a_t(a_t)
      , n_cycles(n_cycles)
      , n_stages(CT::N_STAGES_)
      , analytic_solution(boundary_values)
      , pcout(pcout)
      , A(Util::load_matrix_from_file(CT::N_STAGES_, "A"))
      , A_inv(Util::load_matrix_from_file(CT::N_STAGES_, "A_inv"))
      , b_vec(Util::load_vector_from_file(CT::N_STAGES_, "b_vec_"))
      , c_vec(Util::load_vector_from_file(CT::N_STAGES_, "c_vec_"))
      , D_vec(Util::load_vector_from_file(CT::N_STAGES_, "D_vec_"))
      , L(Util::load_matrix_from_file(CT::N_STAGES_, "L"))
      , S(Util::load_matrix_from_file(CT::N_STAGES_, "S"))
      , S_inv(Util::load_matrix_from_file(CT::N_STAGES_, "S_inv"))
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

      matrix_dp.resize(minlevel, maxlevel);
      matrix.resize(minlevel, maxlevel);

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
            additional_data.tau      = tau;
            additional_data.n_stages = n_stages;
            additional_data.mg_level = level;
            std::shared_ptr<PSMF::MatrixFree<dim, Number>> mg_mf_storage_level(
              new PSMF::MatrixFree<dim, Number>());
            mg_mf_storage_level->reinit(mapping,
                                        dof_handler,
                                        level_constraints,
                                        QGauss<1>(fe_degree + 1),
                                        additional_data);

            matrix[level].initialize(mg_mf_storage_level, D_vec, tau);
          }

          // double-precision matrix-free data
          {
            typename PSMF::MatrixFree<dim, Number2>::AdditionalData
              additional_data;
            additional_data.mapping_update_flags =
              (update_gradients | update_JxW_values | update_values |
               update_quadrature_points);
            additional_data.tau      = tau;
            additional_data.n_stages = n_stages;
            additional_data.mg_level = level;
            std::shared_ptr<PSMF::MatrixFree<dim, Number2>> mg_mf_storage_level(
              new PSMF::MatrixFree<dim, Number2>());
            mg_mf_storage_level->reinit(mapping,
                                        dof_handler,
                                        level_constraints2,
                                        QGauss<1>(fe_degree + 1),
                                        additional_data);

            matrix_dp[level].initialize(mg_mf_storage_level, D_vec, tau);

            if (level == maxlevel)
              {
                AffineConstraints<double> level_constraintsD;
                level_constraintsD.reinit(relevant_dofs);
                level_constraintsD.add_lines(
                  mg_constrained_dofs.get_boundary_indices(level));
                level_constraintsD.close();

                typename PSMF::MatrixFree<dim, double>::AdditionalData
                  additional_dataD;
                additional_dataD.mapping_update_flags =
                  (update_gradients | update_JxW_values | update_values |
                   update_quadrature_points);
                additional_dataD.tau      = tau;
                additional_dataD.n_stages = n_stages;
                additional_dataD.mg_level = level;
                std::shared_ptr<PSMF::MatrixFree<dim, double>>
                  mg_mf_storage_levelD(new PSMF::MatrixFree<dim, double>());
                mg_mf_storage_levelD->reinit(mapping,
                                             dof_handler,
                                             level_constraintsD,
                                             QGauss<1>(fe_degree + 1),
                                             additional_dataD);

                system_matrix.initialize(mg_mf_storage_levelD,
                                         A_inv,
                                         S,
                                         S_inv,
                                         b_vec,
                                         c_vec,
                                         tau,
                                         wave_number,
                                         a_t,
                                         n_stages);

                matrix_dp[maxlevel].initialize_dof_vector(rhs);
                matrix_dp[maxlevel].initialize_dof_vector(tmp);

                matrix[maxlevel].initialize_dof_vector(defect);
                matrix[maxlevel].initialize_dof_vector(solution_update);

                solution.reinit(rhs.size());
                solution_0.reinit(rhs.size());
                solution_old.reinit(rhs.size());

                system_solution.reinit(n_stages * solution_old.size());
                system_rhs.reinit(n_stages * solution_old.size());

                system_defect.reinit(n_stages * solution_old.size());
                system_update.reinit(n_stages * solution_old.size());
              }
          }
        }

      *pcout << "MatrixFree setup time:    " << time.wall_time() << std::endl;
      time.restart();

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
            const unsigned int n_dofs = solution.size();

            LinearAlgebra::distributed::Vector<double, MemorySpace::Host>
              system_u_host(n_dofs);
            LinearAlgebra::distributed::Vector<double, MemorySpace::CUDA>
                                                   system_u_dev(n_dofs);
            LinearAlgebra::ReadWriteVector<double> rw_vector(n_dofs);

            // U
            std::map<types::global_dof_index, Point<dim>> current_point_map;
            DoFTools::map_dofs_to_support_points<dim>(MappingQGeneric<dim>(
                                                        fe_degree + 1),
                                                      dof_handler,
                                                      current_point_map);
            for (unsigned int i = 0; i < n_dofs; ++i)
              system_u_host[i] = analytic_solution.value(current_point_map[i]);

            rw_vector.import(system_u_host, VectorOperation::insert);
            solution_0.import(rw_vector, VectorOperation::insert);
            solution_old.import(rw_vector, VectorOperation::insert);
          }
        }

      time.restart();

      if constexpr (CT::KERNEL_TYPE_[0] != PSMF::SmootherVariant::Chebyshev)
        {
          MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
          smoother_data.resize(minlevel, maxlevel);
          for (unsigned int level = minlevel; level <= maxlevel; ++level)
            {
              smoother_data[level].tau        = tau;
              smoother_data[level].n_stages   = n_stages;
              smoother_data[level].relaxation = 1.;
              smoother_data[level].patch_per_block =
                fe_degree == 1 ? 16 : (fe_degree == 2 ? 2 : 1);
              smoother_data[level].granularity_scheme = CT::GRANULARITY_;
            }
          mg_smoother.initialize(matrix, smoother_data);
          mg_coarse.initialize(mg_smoother);

          mg_matrix.initialize(matrix);
          mg = std::make_unique<Multigrid<VectorType>>(mg_matrix,
                                                       mg_coarse,
                                                       transfer,
                                                       mg_smoother,
                                                       mg_smoother,
                                                       minlevel,
                                                       maxlevel);


          preconditioner_mg = std::make_unique<
            PreconditionMG<dim, VectorType, MGTransferCUDA<dim, Number>>>(
            dof_handler, *mg, transfer);

#ifdef MIXED
          precon =
            std::make_unique<Preconditioner<dim, fe_degree, Number, Number>>(
              *preconditioner_mg, rhs.size());
#else
          precon =
            std::make_unique<Preconditioner<dim, fe_degree, Number, Number2>>(
              *preconditioner_mg, rhs.size());
#endif
        }

      *pcout << "Time initial smoother:    " << time.wall_time() << std::endl;
    }

    void
    print_timings() const
    {
      {
        *pcout << " - #N of calls of multigrid: " << all_mg_counter << std::endl
               << std::endl;
        *pcout << " - Times of multigrid (levels):" << std::endl;

        const auto print_line = [&](const auto &vector) {
          for (const auto &i : vector)
            *pcout << std::scientific << std::setprecision(2) << std::setw(10)
                   << i.first;

          double sum = 0;

          for (const auto &i : vector)
            sum += i.first;

          *pcout << "   | " << std::scientific << std::setprecision(2)
                 << std::setw(10) << sum;

          *pcout << "\n";
        };

        for (unsigned int l = 0; l < all_mg_timers.size(); ++l)
          {
            *pcout << std::setw(4) << l << ": ";

            print_line(all_mg_timers[l]);
          }

        std::vector<
          std::pair<double, std::chrono::time_point<std::chrono::system_clock>>>
          sums(all_mg_timers[0].size());

        for (unsigned int i = 0; i < sums.size(); ++i)
          for (unsigned int j = 0; j < all_mg_timers.size(); ++j)
            sums[i].first += all_mg_timers[j][i].first;

        *pcout
          << "   ----------------------------------------------------------------------------+-----------\n";
        *pcout << "      ";
        print_line(sums);

        *pcout << std::endl;

        *pcout << " - Times of multigrid (solver <-> mg): ";

        for (const auto &i : all_mg_precon_timers)
          *pcout << i.first << " ";
        *pcout << std::endl;
        *pcout << std::endl;
      }
    }

    void
    clear_timings() const
    {
      for (auto &is : all_mg_timers)
        for (auto &i : is)
          i.first = 0.0;

      for (auto &i : all_mg_precon_timers)
        i.first = 0.0;

      all_mg_counter = 0;
    }

    // Return the solution vector for further processing
    const VectorTypeD &
    get_solution()
    {
      set_inhomogeneous_bc<false>(maxlevel);
      return solution;
    }

    // Solve with the conjugate gradient method preconditioned by the V-cycle
    // (invoking this->vmult() or vmult_with_residual_update()) and return the
    // number of iterations and the reduction rate per GMRES iteration
    std::optional<std::pair<ReductionControl, double>>
    solve_gmres(const bool do_analyze, const unsigned int N = 1)
    {
      ReductionControl solver_control(CT::MAX_STEPS_, 1e-15, CT::REDUCE_);
      solver_control.enable_history_data();
      solver_control.log_history(true);

      SolverFGMRES<VectorTypeD> solver_gmres(solver_control);

      solution        = solution_0;
      solution_old    = solution_0;
      system_solution = 0;

      current_stage = 0;

      for (unsigned int it = 0; it < N; ++it)
        {
          n_inner_its.assign(n_stages, 0);

          auto t = it * tau;

          Timer time;

          if (do_analyze)
            *pcout << "Time step " << it << " at t=" << t + tau << std::endl;

          if (it == 0)
            solution_old = solution_0;

          system_matrix.compute_system_rhs(system_rhs, solution_old, t);

          if (do_analyze)
            *pcout << "Time compute system rhs:  " << time.wall_time() << "\n";
          time.restart();

          solver_gmres.solve(system_matrix, system_solution, system_rhs, *this);

          if (do_analyze)
            *pcout << "Time solve FGMRES: " << time.wall_time() << "\n";

          if (do_analyze)
            *pcout << "     " << solver_control.last_step()
                   << " outer FGMRES iterations and " << n_inner_its[0];

          if (do_analyze)
            {
              for (unsigned int i = 1; i < n_stages; ++i)
                *pcout << "+" << n_inner_its[i];
              *pcout << " inner iterations. \n\n";
            }

          system_matrix.compute_solution(solution, system_solution);

          solution_old = solution;
        }

      if (do_analyze)
        {
          double n_inner_it_avg =
            std::accumulate(n_inner_its.begin(), n_inner_its.end(), 0.0) /
            n_inner_its.size();

          return std::make_pair(solver_control, n_inner_it_avg);
        }
      else
        return std::nullopt;
    }


    std::optional<std::pair<double, double>>
    solve_chebyshev(const bool do_analyze, const unsigned int N = 1)
    {
      ReductionControl solver_control(CT::MAX_STEPS_, 1e-15, CT::REDUCE_);
      solver_control.enable_history_data();
      solver_control.log_history(true);

      SolverGMRES<VectorTypeD> solver_gmres(solver_control);

      solution        = solution_0;
      solution_old    = solution_0;
      system_solution = 0;

      const double lambda_min = 4. / 25 * (3 * std::sqrt(6) - 2);
      const double lambda_max = 1.;

      const double d = (lambda_max + lambda_min) / 2;
      const double c = (lambda_max - lambda_min) / 2;

      current_stage = 0;

      unsigned int        IT = 0;
      std::vector<double> history;

      for (unsigned int it = 0; it < N; ++it)
        {
          n_inner_its.assign(n_stages, 0);

          auto t = it * tau;

          Timer time;

          if (do_analyze)
            *pcout << "Time step " << it << " at t=" << t + tau << std::endl;

          if (it == 0)
            solution_old = solution_0;

          system_matrix.compute_system_rhs(system_rhs, solution_old, t);

          if (do_analyze)
            *pcout << "Time compute system rhs:  " << time.wall_time() << "\n";
          time.restart();

          // solver_gmres.solve(system_matrix, system_solution, system_rhs,
          // *this);
          auto   r     = system_rhs;
          auto   z     = system_rhs;
          auto   p     = z;
          double alpha = 0;
          double beta  = 0;

          const double n0 = r.l2_norm();

          for (unsigned int iitt = 1; iitt < CT::MAX_STEPS_; ++iitt)
            {
              // Mz = r
              this->vmult(z, r);

              if (iitt == 1)
                {
                  p     = z;
                  alpha = 2. / d;
                }
              else
                {
                  beta  = std::pow(c * alpha / 2, 2);
                  alpha = 1. / (d - beta);
                  p.sadd(beta, 1., z);
                }

              system_solution.add(alpha, p);
              system_matrix.vmult(r, system_solution);

              // r = system_rhs - Ax;
              r.sadd(-1., system_rhs);

              auto norm = r.l2_norm() / n0;
              history.push_back(norm);

              if (norm < CT::REDUCE_)
                {
                  IT = iitt - 1;
                  break;
                }
            }

          if (do_analyze)
            *pcout << "Time solve Chebyshev: " << time.wall_time() << "\n";

          if (do_analyze)
            *pcout << "     " << IT << " outer Chebyshev iterations and "
                   << n_inner_its[0];

          if (do_analyze)
            {
              for (unsigned int i = 1; i < n_stages; ++i)
                *pcout << "+" << n_inner_its[i];
              *pcout << " inner FGMRES iterations. \n\n";
            }

          system_matrix.compute_solution(solution, system_solution);

          solution_old = solution;
        }

      if (do_analyze)
        {
          for (auto i = 1U; i < IT + 1; ++i)
            *pcout << "step " << i << ": " << history[i] << "\n";

          double n_inner_it_avg =
            std::accumulate(n_inner_its.begin(), n_inner_its.end(), 0.0) /
            n_inner_its.size();

          return std::make_pair(IT, n_inner_it_avg);
        }
      else
        return std::nullopt;
    }


    // Implement the vmult() function needed by the preconditioner interface
    void
    vmult(VectorTypeD &dst, const VectorTypeD &src) const
    {
      dst = 0.;

      system_defect = 0;
      system_update = 0;

      system_matrix.transform_basis(system_defect, src);

      ReductionControl solver_control(CT::MAX_STEPS_INNER_,
                                      1e-10,
                                      CT::REDUCE_INNER_);

      typename SelectSolver<CT::SOLVER_, VectorType2>::type solver(
        solver_control);

      for (unsigned int i = 0; i < n_stages; ++i)
        {
          current_stage = i;

          if constexpr (CT::KERNEL_TYPE_[0] == PSMF::SmootherVariant::Chebyshev)
            {
              MGLevelObject<typename SmootherTypeCheb::AdditionalData>
                smoother_data;
              smoother_data.resize(minlevel, maxlevel);
              for (unsigned int level = minlevel; level <= maxlevel; ++level)
                {
                  matrix[level].compute_diagonal();

                  smoother_data[level].smoothing_range     = 20.;
                  smoother_data[level].degree              = 5;
                  smoother_data[level].eig_cg_n_iterations = 20;
                  smoother_data[level].preconditioner =
                    matrix[level].get_diagonal_inverse();
                }

              mg_smoother_cheb.initialize(matrix, smoother_data);
              mg_coarse_cheb.initialize(mg_smoother_cheb);

              mg_matrix.initialize(matrix);
              mg = std::make_unique<Multigrid<VectorType>>(mg_matrix,
                                                           mg_coarse_cheb,
                                                           transfer,
                                                           mg_smoother_cheb,
                                                           mg_smoother_cheb,
                                                           minlevel,
                                                           maxlevel);

              preconditioner_mg = std::make_unique<
                PreconditionMG<dim, VectorType, MGTransferCUDA<dim, Number>>>(
                *dof_handler, *mg, transfer);

              precon = std::make_unique<
                Preconditioner<dim, fe_degree, Number, Number2>>(
                *preconditioner_mg, rhs.size());
            }

          tmp = 0.;
          rhs = 0.;
          internal::vec_add(
            rhs, system_defect, tmp.size(), 1., 0, i * tmp.size());

          solver.solve(matrix_dp[maxlevel], tmp, rhs, *precon);

          n_inner_its[i] += solver_control.last_step();

          internal::vec_add(
            system_update, tmp, tmp.size(), 1., i * tmp.size(), 0);
        }

      system_matrix.transform_basis_back(dst, system_update);
    }

    std::vector<double>
    get_timing()
    {
      auto tester = [&](auto kernel) {
        Timer              time;
        const unsigned int N    = 5;
        const unsigned int n_mv = dof_handler->n_dofs() > 10000000 ? 20 : 5;
        double             best_time = 1e10;
        for (unsigned int i = 0; i < N; ++i)
          {
            time.restart();
            for (unsigned int i = 0; i < n_mv; ++i)
              kernel(this);
            cudaDeviceSynchronize();
            best_time = std::min(time.wall_time() / n_mv, best_time);
          }
        return dof_handler->n_dofs() / best_time;
      };

      std::vector<double> result(6);

      result[0] = tester(std::mem_fn(&MultigridSolvers::do_system_matvec));
      result[1] = tester(std::mem_fn(&MultigridSolvers::do_system_precon));
      result[2] = tester(std::mem_fn(&MultigridSolvers::do_precon));
      result[3] = tester(std::mem_fn(&MultigridSolvers::do_matvec));
      result[4] = tester(std::mem_fn(&MultigridSolvers::do_matvec_sp));
      result[5] = tester(std::mem_fn(&MultigridSolvers::do_smoother));

      return result;
    }

    // run system matrix-vector product in double precision
    void
    do_system_matvec()
    {
      system_matrix.vmult(system_solution, system_rhs);
    }

    // run system preconditioner in double precision
    void
    do_system_precon()
    {
      this->vmult(system_solution, system_rhs);
    }

    // run system preconditioner in double precision
    void
    do_precon()
    {
#ifndef MIXED
      precon->vmult(tmp, rhs);
#endif
    }

    // run matrix-vector product in double precision
    void
    do_matvec()
    {
      matrix_dp[maxlevel].vmult(tmp, rhs);
    }

    // run matrix-vector product in single precision
    void
    do_matvec_sp()
    {
      matrix[maxlevel].vmult(solution_update, defect);
    }

    // run smoother in single precision
    void
    do_smoother()
    {
      if constexpr (CT::KERNEL_TYPE_[0] != PSMF::SmootherVariant::Chebyshev)
        (mg_smoother).smooth(maxlevel, solution_update, defect);
      else
        (mg_smoother_cheb).smooth(maxlevel, solution_update, defect);
    }

  private:
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
          std::vector<double> inhomogeneous_value_host(n_inhomogeneous_bc);
          unsigned int        count = 0;
          for (auto &i : inhomogeneous_bc[level])
            {
              inhomogeneous_index_host[count] = i.first;
              inhomogeneous_value_host[count] = i.second;
              count++;
            }

          unsigned int *inhomogeneous_index;
          double       *inhomogeneous_value;

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
                                  n_inhomogeneous_bc * sizeof(double));
          AssertCuda(cuda_error);

          cuda_error = cudaMemcpy(inhomogeneous_value,
                                  inhomogeneous_value_host.data(),
                                  n_inhomogeneous_bc * sizeof(double),
                                  cudaMemcpyHostToDevice);
          AssertCuda(cuda_error);


          set_inhomogeneous_dofs<is_zero, double>
            <<<inhomogeneous_grid_dim, inhomogeneous_block_dim>>>(
              inhomogeneous_index,
              inhomogeneous_value,
              n_inhomogeneous_bc,
              solution.get_values());
          AssertCudaKernel();
        }
    }


    const SmartPointer<const DoFHandler<dim>> dof_handler;

    std::vector<std::map<types::global_dof_index, double>> inhomogeneous_bc;

    MGConstrainedDoFs mg_constrained_dofs;

    MGTransferCUDA<dim, Number> transfer;

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
    mutable VectorTypeD solution;
    mutable VectorTypeD solution_0;
    mutable VectorTypeD solution_old;
    mutable VectorTypeD system_solution;

    /**
     * buffer
     */
    mutable VectorType2 tmp;
    mutable VectorType2 rhs;

    mutable VectorType2 system_defect;
    mutable VectorType2 system_update;

    /**
     * Original right hand side vector
     */
    mutable VectorTypeD system_rhs;

    /**
     * Input vector for the cycle. Contains the defect of the outer method
     * projected to the multilevel vectors.
     */
    mutable VectorType defect;

    /**
     * Auxiliary vector for the solution update
     */
    mutable VectorType solution_update;

    /**
     * The local matrix for each level
     */
    mutable MGLevelObject<LocalMatrixType> matrix;

    /**
     * The double-precision local matrix
     */
    MGLevelObject<LocalMatrixType2> matrix_dp;

    /**
     * The system matrix for outer iteration
     */
    SystemMatrixType system_matrix;

    /**
     * The smoother object
     */
    mutable MGSmootherPrecondition<LocalMatrixType, SmootherType, VectorType>
      mg_smoother;

    mutable MGSmootherPrecondition<LocalMatrixType,
                                   SmootherTypeCheb,
                                   VectorType>
      mg_smoother_cheb;

    /**
     * The coarse solver
     */
    mutable MGCoarseGridApplySmoother<VectorType> mg_coarse;

    mutable MGCoarseGridApplySmoother<VectorType> mg_coarse_cheb;

    const unsigned int N;
    const double       tau;
    const double       wave_number;
    const double       a_t;

    /**
     * Number of cycles to be done in the FMG cycle
     */
    const unsigned int n_cycles;

    /**
     * Number of stages
     */
    const unsigned int n_stages;

    /**
     * Function for boundary values that we keep as analytic solution
     */
    const Function<dim> &analytic_solution;

    std::shared_ptr<ConditionalOStream> pcout;

    mutable mg::Matrix<VectorType> mg_matrix;

    mutable std::unique_ptr<Multigrid<VectorType>> mg;

    mutable std::unique_ptr<
      PreconditionMG<dim, VectorType, MGTransferCUDA<dim, Number>>>
      preconditioner_mg;

#ifdef MIXED
    mutable std::unique_ptr<Preconditioner<dim, fe_degree, Number, Number>>
      precon;
#else
    mutable std::unique_ptr<Preconditioner<dim, fe_degree, Number, Number2>>
      precon;
#endif

    mutable unsigned int all_mg_counter = 0;

    mutable std::vector<std::vector<
      std::pair<double, std::chrono::time_point<std::chrono::system_clock>>>>
      all_mg_timers;

    mutable std::vector<
      std::pair<double, std::chrono::time_point<std::chrono::system_clock>>>
      all_mg_precon_timers;

    mutable std::vector<unsigned int> n_inner_its;

    /**
     * IRK data
     */
    mutable std::vector<double> A;
    mutable std::vector<double> A_inv;

    mutable std::vector<double> b_vec;
    mutable std::vector<double> c_vec;

    mutable std::vector<double> D_vec;
    mutable std::vector<double> L;

    mutable std::vector<double> S;
    mutable std::vector<double> S_inv;
  };


} // namespace PSMF

#endif // SOLVER_CUH
