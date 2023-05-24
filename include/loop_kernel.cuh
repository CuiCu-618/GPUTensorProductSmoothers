/**
 * @file loop_kernel.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief collection of global functions
 * @version 1.0
 * @date 2022-12-26
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef LOOP_KERNEL_CUH
#define LOOP_KERNEL_CUH

#include "patch_base.cuh"

namespace PSMF
{

  /**
   * Shared data for laplace operator kernel.
   * We have to use the following thing to avoid
   * a compile time error since @tparam Number
   * could be double and float at same time.
   */
  extern __shared__ double data_d[];
  extern __shared__ float  data_f[];

  template <typename Number>
  __device__ inline Number *
  get_shared_data_ptr();

  template <>
  __device__ inline double *
  get_shared_data_ptr()
  {
    return data_d;
  }

  template <>
  __device__ inline float *
  get_shared_data_ptr()
  {
    return data_f;
  }


  template <int dim,
            int fe_degree,
            typename Number,
            SmootherVariant kernel,
            typename Functor,
            DoFLayout dof_layout>
  __global__ void
  loop_kernel_seperate(
    Functor                                           func,
    const Number                                     *src,
    Number                                           *dst,
    Number                                           *tmp,
    const typename LevelVertexPatch<dim,
                                    fe_degree,
                                    Number,
                                    kernel,
                                    dof_layout>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d = Functor::n_dofs_1d;
    constexpr unsigned int local_dim = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int n_dofs_z  = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedMemData<dim, Number, kernel> shared_data(
      get_shared_data_ptr<Number>(), patch_per_block, n_dofs_1d, local_dim);

    if (patch < gpu_data.n_patches)
      {
        shared_data.local_mass[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.global_mass_1d[threadIdx.y * n_dofs_1d + local_tid_x];
        shared_data.local_derivative[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.global_derivative_1d[threadIdx.y * n_dofs_1d + local_tid_x];

        // #pragma unroll
        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            unsigned int index = local_patch * local_dim +
                                 z * n_dofs_1d * n_dofs_1d +
                                 threadIdx.y * n_dofs_1d + local_tid_x;

            unsigned int global_dof_indices =
              z * func.get_ndofs() * func.get_ndofs() +
              threadIdx.y * func.get_ndofs() + local_tid_x +
              gpu_data.first_dof[patch];

            shared_data.local_src[index] = src[global_dof_indices];

            shared_data.local_dst[index] = dst[global_dof_indices];
          }

        func(local_patch, &gpu_data, &shared_data);

        // #pragma unroll
        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            unsigned int index = local_patch * local_dim +
                                 z * n_dofs_1d * n_dofs_1d +
                                 threadIdx.y * n_dofs_1d + local_tid_x;

            unsigned int global_dof_indices =
              z * func.get_ndofs() * func.get_ndofs() +
              threadIdx.y * func.get_ndofs() + local_tid_x +
              gpu_data.first_dof[patch];

            tmp[global_dof_indices] = shared_data.local_src[index];
          }
      }
  }

  template <int dim,
            int fe_degree,
            typename Number,
            SmootherVariant kernel,
            typename Functor,
            DoFLayout dof_layout>
  __global__ typename std::enable_if<kernel == SmootherVariant::Exact>::type
  loop_kernel_seperate_inv(
    Functor                                           func,
    const Number                                     *src,
    Number                                           *dst,
    const typename LevelVertexPatch<dim,
                                    fe_degree,
                                    Number,
                                    kernel,
                                    dof_layout>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d = Functor::n_dofs_1d;
    constexpr unsigned int local_dim = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int n_dofs_z  = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;
    const unsigned int linear_tid_x = threadIdx.y * n_dofs_1d + local_tid_x;

    SharedMemData<dim, Number, kernel> shared_data(
      get_shared_data_ptr<Number>(), patch_per_block, n_dofs_1d, local_dim);

    if (patch < gpu_data.n_patches)
      {
        // #pragma unroll
        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            unsigned int index = local_patch * local_dim +
                                 z * n_dofs_1d * n_dofs_1d +
                                 threadIdx.y * n_dofs_1d + local_tid_x;

            unsigned int global_dof_indices =
              (z + dim - 2) * func.get_ndofs() * func.get_ndofs() +
              (threadIdx.y + 1) * func.get_ndofs() + local_tid_x + 1 +
              gpu_data.first_dof[patch];

            shared_data.local_src[index] = src[global_dof_indices];

            shared_data.local_dst[index] = dst[global_dof_indices];
          }

        __syncthreads();

        for (unsigned int row = 0; row < local_dim; ++row)
          {
            for (unsigned int z = 0; z < n_dofs_z; ++z)
              {
                auto val =
                  gpu_data
                    .eigenvalues[n_dofs_1d + row * local_dim +
                                 z * n_dofs_1d * n_dofs_1d + linear_tid_x] *
                  shared_data
                    .local_src[local_patch * local_dim +
                               z * n_dofs_1d * n_dofs_1d + linear_tid_x];

                atomicAdd(&shared_data.local_dst[local_patch * local_dim + row],
                          val);
              }
          }

        __syncthreads();

        // #pragma unroll
        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            unsigned int index = local_patch * local_dim +
                                 z * n_dofs_1d * n_dofs_1d +
                                 threadIdx.y * n_dofs_1d + local_tid_x;

            unsigned int global_dof_indices =
              (z + dim - 2) * func.get_ndofs() * func.get_ndofs() +
              (threadIdx.y + 1) * func.get_ndofs() + local_tid_x + 1 +
              gpu_data.first_dof[patch];

            dst[global_dof_indices] =
              shared_data.local_dst[index] * gpu_data.relaxation;
          }
      }
  }

  template <int dim,
            int fe_degree,
            typename Number,
            SmootherVariant kernel,
            typename Functor,
            DoFLayout dof_layout>
  __global__ typename std::enable_if<kernel != SmootherVariant::Exact>::type
  loop_kernel_seperate_inv(
    Functor                                           func,
    const Number                                     *src,
    Number                                           *dst,
    const typename LevelVertexPatch<dim,
                                    fe_degree,
                                    Number,
                                    kernel,
                                    dof_layout>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d = Functor::n_dofs_1d;
    constexpr unsigned int local_dim = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int n_dofs_z  = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedMemData<dim, Number, kernel> shared_data(
      get_shared_data_ptr<Number>(), patch_per_block, n_dofs_1d, local_dim);

    if (patch < gpu_data.n_patches)
      {
        shared_data.local_mass[local_tid_x] = gpu_data.eigenvalues[local_tid_x];
        shared_data.local_derivative[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.eigenvectors[threadIdx.y * n_dofs_1d + local_tid_x];

        // #pragma unroll
        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            unsigned int index = local_patch * local_dim +
                                 z * n_dofs_1d * n_dofs_1d +
                                 threadIdx.y * n_dofs_1d + local_tid_x;

            unsigned int global_dof_indices =
              (z + dim - 2) * func.get_ndofs() * func.get_ndofs() +
              (threadIdx.y + 1) * func.get_ndofs() + local_tid_x + 1 +
              gpu_data.first_dof[patch];

            shared_data.local_src[index] = src[global_dof_indices];

            shared_data.local_dst[index] = dst[global_dof_indices];
          }

        func(local_patch, &gpu_data, &shared_data);

        // #pragma unroll
        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            unsigned int index = local_patch * local_dim +
                                 z * n_dofs_1d * n_dofs_1d +
                                 threadIdx.y * n_dofs_1d + local_tid_x;

            unsigned int global_dof_indices =
              (z + dim - 2) * func.get_ndofs() * func.get_ndofs() +
              (threadIdx.y + 1) * func.get_ndofs() + local_tid_x + 1 +
              gpu_data.first_dof[patch];

            dst[global_dof_indices] =
              shared_data.local_dst[index] * gpu_data.relaxation;
          }
      }
  }

  template <int dim,
            int fe_degree,
            typename Number,
            SmootherVariant kernel,
            typename Functor,
            DoFLayout dof_layout>
  __global__ void
  loop_kernel_fused_base(
    Functor                                           func,
    const Number                                     *src,
    Number                                           *dst,
    const typename LevelVertexPatch<dim,
                                    fe_degree,
                                    Number,
                                    kernel,
                                    dof_layout>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d = Functor::n_dofs_1d;
    constexpr unsigned int local_dim = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int n_dofs_z  = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedMemData<dim, Number, kernel> shared_data(
      get_shared_data_ptr<Number>(), patch_per_block, n_dofs_1d, local_dim);

    if (patch < gpu_data.n_patches)
      {
        shared_data.local_mass[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.global_mass_1d[threadIdx.y * n_dofs_1d + local_tid_x];
        shared_data.local_derivative[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.global_derivative_1d[threadIdx.y * n_dofs_1d + local_tid_x];

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            unsigned int index = local_patch * local_dim +
                                 z * n_dofs_1d * n_dofs_1d +
                                 threadIdx.y * n_dofs_1d + local_tid_x;

            unsigned int global_dof_indices =
              z * func.get_ndofs() * func.get_ndofs() +
              threadIdx.y * func.get_ndofs() + local_tid_x +
              gpu_data.first_dof[patch];

            shared_data.local_src[index] = src[global_dof_indices];

            shared_data.local_dst[index] = dst[global_dof_indices];
          }

        func(local_patch, &gpu_data, &shared_data);

        if (dim == 2)
          {
            if (0 < local_tid_x && local_tid_x < n_dofs_1d - 1 &&
                0 < threadIdx.y && threadIdx.y < n_dofs_1d - 1)
              {
                unsigned int index = local_patch * local_dim +
                                     threadIdx.y * n_dofs_1d + local_tid_x;

                unsigned int global_dof_indices =
                  threadIdx.y * func.get_ndofs() + local_tid_x +
                  gpu_data.first_dof[patch];

                dst[global_dof_indices] =
                  shared_data.local_dst[index] * gpu_data.relaxation;
              }
          }
        else if (dim == 3)
          {
            if (0 < local_tid_x && local_tid_x < n_dofs_1d - 1 &&
                0 < threadIdx.y && threadIdx.y < n_dofs_1d - 1)
              for (unsigned int z = 1; z < n_dofs_1d - 1; ++z)
                {
                  unsigned int index = local_patch * local_dim +
                                       z * n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x;

                  unsigned int global_dof_indices =
                    z * func.get_ndofs() * func.get_ndofs() +
                    threadIdx.y * func.get_ndofs() + local_tid_x +
                    gpu_data.first_dof[patch];

                  dst[global_dof_indices] =
                    shared_data.local_dst[index] * gpu_data.relaxation;
                }
          }
      }
  }



  template <int dim,
            int fe_degree,
            typename Number,
            SmootherVariant kernel,
            typename Functor,
            DoFLayout dof_layout>
  __global__ void
  loop_kernel_fused_boundary(
    Functor                                           func,
    const Number                                     *src,
    Number                                           *dst,
    const typename LevelVertexPatch<dim,
                                    fe_degree,
                                    Number,
                                    kernel,
                                    dof_layout>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d    = 2 * fe_degree + 1;
    constexpr unsigned int n_dofs_1d_in = 2 * fe_degree - 1;
    constexpr unsigned int n_bound_dofs =
      Util::pow(n_dofs_1d, dim) - Util::pow(n_dofs_1d_in, dim);
    constexpr unsigned int n_inner_dofs = Util::pow(n_dofs_1d_in, dim);
    constexpr unsigned int n_dofs_z     = dim == 2 ? 1 : n_dofs_1d_in;

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch = local_patch + patch_per_block * blockIdx.x;
    const unsigned int linear_tid =
      threadIdx.y * n_dofs_1d + threadIdx.x % n_dofs_1d;

    SharedMemData<dim, Number, kernel> shared_data(
      get_shared_data_ptr<Number>(),
      patch_per_block,
      n_dofs_1d,
      n_dofs_1d_in,
      n_bound_dofs,
      n_inner_dofs);

    if (patch < gpu_data.n_patches)
      {
        shared_data.mass_I[linear_tid] = gpu_data.mass_I[linear_tid];
        shared_data.der_I[linear_tid]  = gpu_data.der_I[linear_tid];

        if (linear_tid < n_dofs_1d_in * n_dofs_1d_in)
          {
            shared_data.mass_ii[linear_tid] = gpu_data.mass_ii[linear_tid];
            shared_data.der_ii[linear_tid]  = gpu_data.der_ii[linear_tid];
          }

        if (linear_tid < n_dofs_1d_in * 2)
          {
            shared_data.mass_ib[linear_tid] = gpu_data.mass_ib[linear_tid];
            shared_data.der_ib[linear_tid]  = gpu_data.der_ib[linear_tid];
          }

        // interior RHS verctor: b
        if (linear_tid < n_dofs_1d_in * n_dofs_1d_in)
          for (unsigned int z = 0; z < n_dofs_z; ++z)
            {
              const unsigned int y = linear_tid / n_dofs_1d_in;
              const unsigned int x = linear_tid % n_dofs_1d_in;

              unsigned int index = local_patch * n_inner_dofs +
                                   z * n_dofs_1d_in * n_dofs_1d_in +
                                   y * n_dofs_1d_in + x;

              unsigned int global_dof_indices =
                (z + dim - 2) * func.get_ndofs() * func.get_ndofs() +
                (y + 1) * func.get_ndofs() + x + 1 + gpu_data.first_dof[patch];

              shared_data.local_src[index] = src[global_dof_indices];
            }

        // boundary coefficents vector: x
        for (unsigned int i = 0;
             i < n_bound_dofs / (n_dofs_1d * n_dofs_1d_in) + 1;
             ++i)
          {
            unsigned int index = i * n_dofs_1d * n_dofs_1d_in + linear_tid;
            if (index < n_bound_dofs)
              {
                auto local_index = boundary_dofs_index[index];

                auto z = local_index / (n_dofs_1d * n_dofs_1d);
                auto y = (local_index / n_dofs_1d) % n_dofs_1d;
                auto x = local_index % n_dofs_1d;

                unsigned int global_dof_indices =
                  z * func.get_ndofs() * func.get_ndofs() +
                  y * func.get_ndofs() + x + gpu_data.first_dof[patch];

                shared_data.local_dst[local_patch * n_bound_dofs + index] =
                  dst[global_dof_indices];
              }
          }

        func(local_patch, &gpu_data, &shared_data);

        if (linear_tid < n_dofs_1d_in * n_dofs_1d_in)
          for (unsigned int z = 0; z < n_dofs_z; ++z)
            {
              const unsigned int y = linear_tid / n_dofs_1d_in;
              const unsigned int x = linear_tid % n_dofs_1d_in;

              unsigned int index = local_patch * n_inner_dofs +
                                   z * n_dofs_1d_in * n_dofs_1d_in +
                                   y * n_dofs_1d_in + x;

              unsigned int global_dof_indices =
                (z + dim - 2) * func.get_ndofs() * func.get_ndofs() +
                (y + 1) * func.get_ndofs() + x + 1 + gpu_data.first_dof[patch];

              dst[global_dof_indices] =
                shared_data.local_src[index] * gpu_data.relaxation;
            }
      }
  }


  template <int dim,
            int fe_degree,
            typename Number,
            SmootherVariant kernel,
            typename Functor,
            DoFLayout dof_layout>
  __global__ void
  loop_kernel_fused_l(
    Functor                                           func,
    const Number                                     *src,
    Number                                           *dst,
    const typename LevelVertexPatch<dim,
                                    fe_degree,
                                    Number,
                                    kernel,
                                    dof_layout>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d = Functor::n_dofs_1d;
    constexpr unsigned int local_dim = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int n_dofs_z  = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedMemData<dim, Number, kernel> shared_data(
      get_shared_data_ptr<Number>(), patch_per_block, n_dofs_1d, local_dim);

    if (patch < gpu_data.n_patches)
      {
        shared_data.local_mass[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.global_mass_1d[threadIdx.y * n_dofs_1d + local_tid_x];
        shared_data.local_derivative[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.global_derivative_1d[threadIdx.y * n_dofs_1d + local_tid_x];

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            unsigned int index = local_patch * local_dim +
                                 z * n_dofs_1d * n_dofs_1d +
                                 threadIdx.y * n_dofs_1d + local_tid_x;

            unsigned int global_dof_indices =
              z * func.get_ndofs() * func.get_ndofs() +
              threadIdx.y * func.get_ndofs() + local_tid_x +
              gpu_data.first_dof[patch];

            shared_data.local_src[index] = src[global_dof_indices];

            shared_data.local_dst[index] = dst[global_dof_indices];
          }

        func(local_patch, &gpu_data, &shared_data);

        if (dim == 2)
          {
            unsigned int linear_tid = local_tid_x + threadIdx.y * n_dofs_1d;

            if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
              {
                int row = linear_tid / (n_dofs_1d - 2) + 1;
                int col = linear_tid % (n_dofs_1d - 2) + 1;

                unsigned int index =
                  local_patch * local_dim + row * n_dofs_1d + col;

                unsigned int global_dof_indices =
                  row * func.get_ndofs() + col + gpu_data.first_dof[patch];

                dst[global_dof_indices] =
                  shared_data.local_dst[index] * gpu_data.relaxation;
              }
          }
        else if (dim == 3)
          {
            for (unsigned int z = 1; z < n_dofs_1d - 1; ++z)
              {
                unsigned int linear_tid = local_tid_x + threadIdx.y * n_dofs_1d;

                if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
                  {
                    int row = linear_tid / (n_dofs_1d - 2) + 1;
                    int col = linear_tid % (n_dofs_1d - 2) + 1;

                    unsigned int index = local_patch * local_dim +
                                         z * n_dofs_1d * n_dofs_1d +
                                         row * n_dofs_1d + col;

                    unsigned int global_dof_indices =
                      z * func.get_ndofs() * func.get_ndofs() +
                      row * func.get_ndofs() + col + gpu_data.first_dof[patch];

                    dst[global_dof_indices] =
                      shared_data.local_dst[index] * gpu_data.relaxation;
                  }
              }
          }
      }
  }

  template <int dim,
            int fe_degree,
            typename Number,
            SmootherVariant kernel,
            typename Functor,
            DoFLayout dof_layout>
  __global__ void
  loop_kernel_fused_3d(
    Functor                                           func,
    const Number                                     *src,
    Number                                           *dst,
    const typename LevelVertexPatch<dim,
                                    fe_degree,
                                    Number,
                                    kernel,
                                    dof_layout>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d = Functor::n_dofs_1d;
    constexpr unsigned int local_dim = Util::pow(n_dofs_1d, dim);

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedMemData<dim, Number, kernel> shared_data(
      get_shared_data_ptr<Number>(), patch_per_block, n_dofs_1d, local_dim);

    if (patch < gpu_data.n_patches)
      {
        shared_data.local_mass[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.global_mass_1d[threadIdx.y * n_dofs_1d + local_tid_x];
        shared_data.local_derivative[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.global_derivative_1d[threadIdx.y * n_dofs_1d + local_tid_x];

        unsigned int index = local_patch * local_dim +
                             threadIdx.z * n_dofs_1d * n_dofs_1d +
                             threadIdx.y * n_dofs_1d + local_tid_x;

        unsigned int global_dof_indices =
          threadIdx.z * func.get_ndofs() * func.get_ndofs() +
          threadIdx.y * func.get_ndofs() + local_tid_x +
          gpu_data.first_dof[patch];

        shared_data.local_src[index] = src[global_dof_indices];

        shared_data.local_dst[index] = dst[global_dof_indices];

        func(local_patch, &gpu_data, &shared_data);

        const unsigned int linear_tid =
          local_tid_x + (threadIdx.y + threadIdx.z * n_dofs_1d) * n_dofs_1d;

        if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2) * (n_dofs_1d - 2))
          {
            const unsigned int row =
              (linear_tid / (n_dofs_1d - 2)) % ((n_dofs_1d - 2)) + 1;
            const unsigned int col = linear_tid % (n_dofs_1d - 2) + 1;
            const unsigned int z =
              linear_tid / ((n_dofs_1d - 2) * (n_dofs_1d - 2)) + 1;

            unsigned int index = local_patch * local_dim +
                                 z * n_dofs_1d * n_dofs_1d + row * n_dofs_1d +
                                 col;

            unsigned int global_dof_indices =
              z * func.get_ndofs() * func.get_ndofs() + row * func.get_ndofs() +
              col + gpu_data.first_dof[patch];

            dst[global_dof_indices] =
              shared_data.local_dst[index] * gpu_data.relaxation;
          }
      }
  }


  template <int dim,
            int fe_degree,
            typename Number,
            SmootherVariant kernel,
            typename Functor,
            DoFLayout dof_layout>
  __global__ void
  loop_kernel_fused_cf(
    Functor                                           func,
    const Number                                     *src,
    Number                                           *dst,
    const typename LevelVertexPatch<dim,
                                    fe_degree,
                                    Number,
                                    kernel,
                                    dof_layout>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d = Functor::n_dofs_1d;
    constexpr unsigned int local_dim = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int n_dofs_z  = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedMemData<dim, Number, kernel> shared_data(
      get_shared_data_ptr<Number>(), patch_per_block, n_dofs_1d, local_dim);

    if (patch < gpu_data.n_patches)
      {
        shared_data.local_mass[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.global_mass_1d[threadIdx.y * n_dofs_1d + local_tid_x];
        shared_data.local_derivative[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.global_derivative_1d[threadIdx.y * n_dofs_1d + local_tid_x];

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            unsigned int index = local_patch * local_dim +
                                 z * n_dofs_1d * n_dofs_1d +
                                 threadIdx.y * n_dofs_1d + local_tid_x;

            unsigned int global_dof_indices =
              z * func.get_ndofs() * func.get_ndofs() +
              threadIdx.y * func.get_ndofs() + local_tid_x +
              gpu_data.first_dof[patch];

            shared_data.local_src[index] = src[global_dof_indices];

            shared_data.local_dst[index] = dst[global_dof_indices];
          }

        func(local_patch, &gpu_data, &shared_data);

        if (dim == 2)
          {
            unsigned int linear_tid = local_tid_x + threadIdx.y * n_dofs_1d;

            if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
              {
                int row = linear_tid / (n_dofs_1d - 2) + 1;
                int col = linear_tid % (n_dofs_1d - 2) + 1;

                unsigned int index =
                  local_patch * local_dim + row * n_dofs_1d + col;

                unsigned int global_dof_indices =
                  row * func.get_ndofs() + col + gpu_data.first_dof[patch];

                dst[global_dof_indices] =
                  shared_data.local_dst[index] * gpu_data.relaxation;
              }
          }
        else if (dim == 3)
          {
            for (unsigned int z = 0; z < n_dofs_1d - 2; ++z)
              {
                unsigned int linear_tid = local_tid_x + threadIdx.y * n_dofs_1d;

                if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
                  {
                    unsigned int row = linear_tid / (n_dofs_1d - 2) + 1;
                    unsigned int col = linear_tid % (n_dofs_1d - 2) + 1;

                    unsigned int index = (dim - 1) * local_patch * local_dim +
                                         z * n_dofs_1d * n_dofs_1d +
                                         (row - 1) * (n_dofs_1d - 2) + col - 1;

                    unsigned int global_dof_indices =
                      (z + 1) * func.get_ndofs() * func.get_ndofs() +
                      row * func.get_ndofs() + col + gpu_data.first_dof[patch];

                    dst[global_dof_indices] =
                      shared_data.temp[index] * gpu_data.relaxation;
                  }
              }
          }
      }
  }

} // namespace PSMF

#endif // LOOP_KERNEL_CUH