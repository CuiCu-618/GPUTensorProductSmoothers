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

#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

#include "evaluate_kernel.cuh"
#include "patch_base.cuh"
#include <cuda/pipeline>

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

  /// Cast shared memory pointer to uint32_t
  __device__ inline uint32_t
  cast_smem_ptr_to_uint(void const *const ptr)
  {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  }

  /// Copy via cp.async with caching at all levels
  template <class TS, class TD = TS>
  inline __device__ void
  copy(TS const &gmem_src, TD &smem_dst)
  {
    TS const *gmem_ptr     = &gmem_src;
    uint32_t  smem_int_ptr = cast_smem_ptr_to_uint(&smem_dst);
    asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(
                   smem_int_ptr),
                 "l"(gmem_ptr),
                 "n"(sizeof(TS)));
  }

  /// Establishes an ordering w.r.t previously issued cp.async instructions.
  /// Does not block.
  inline __device__ void
  cp_async_fence()
  {
    asm volatile("cp.async.commit_group;\n" ::);
  }

  /// Blocks until all but N previous cp.async.commit_group operations have
  /// committed.
  template <int N>
  inline __device__ void
  cp_async_wait()
  {
    if constexpr (N == 0)
      {
        asm volatile("cp.async.wait_all;\n" ::);
      }
    else
      {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
      }
  }

  template <typename Number>
  struct scaler_to_vector;

  template <>
  struct scaler_to_vector<double>
  {
    using type = double2;
  };

  template <>
  struct scaler_to_vector<float>
  {
    using type = float2;
  };

#if N_PATCH == 1
  template <int dim, int fe_degree, typename Number, LaplaceVariant laplace>
  __global__
  __launch_bounds__(Util::pow(2 * fe_degree + 2, 2)) void laplace_kernel_basic(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr int n_dofs_1d = 2 * fe_degree + 2;
    constexpr int local_dim = Util::pow(n_dofs_1d, dim);
    constexpr int n_dofs_z  = dim == 2 ? 1 : n_dofs_1d;

    const int patch       = blockIdx.x;
    const int local_tid_x = threadIdx.x;
    const int tid         = threadIdx.y * n_dofs_1d + local_tid_x;

    SharedMemData<dim, Number, true> shared_data(get_shared_data_ptr<Number>(),
                                                 1,
                                                 n_dofs_1d,
                                                 local_dim);

    if (patch < gpu_data.n_patches)
      {
        for (unsigned int d = 0; d < dim; ++d)
          {
            const int src_idx =
              gpu_data.patch_type[patch * dim + d] * n_dofs_1d * n_dofs_1d +
              tid;
#  if PIPELINE == 2
            const int dst_idx =
              d * n_dofs_1d * n_dofs_1d +
              (tid ^ Util::get_base<n_dofs_1d, Number>(threadIdx.y));
#  else
            const int          dst_idx = d * n_dofs_1d * n_dofs_1d + tid;
#  endif

            shared_data.local_mass[dst_idx] = gpu_data.laplace_mass_1d[src_idx];
            shared_data.local_derivative[dst_idx] =
              gpu_data.laplace_stiff_1d[src_idx];
          }

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
#  if PIPELINE == 2
            const unsigned int index =
              (z * n_dofs_1d * n_dofs_1d + tid) ^
              Util::get_base<n_dofs_1d, Number>(threadIdx.y, z);
#  else
            const unsigned int index   = z * n_dofs_1d * n_dofs_1d + tid;
#  endif

            const unsigned int global_dof_indices =
              Util::compute_indices<dim, fe_degree>(
                &gpu_data.first_dof[patch * (1 << dim)],
                0,
                local_tid_x,
                threadIdx.y,
                z);

            shared_data.local_src[index] = src[global_dof_indices];

            // shared_data.local_dst[index] = 0.;
          }

        evaluate_laplace<dim, fe_degree, Number, laplace>(0, &shared_data);

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
#  if PIPELINE == 2
            const unsigned int index =
              (z * n_dofs_1d * n_dofs_1d + tid) ^
              Util::get_base<n_dofs_1d, Number>(threadIdx.y, z);
#  else
            const unsigned int index   = z * n_dofs_1d * n_dofs_1d + tid;
#  endif

            const unsigned int global_dof_indices =
              Util::compute_indices<dim, fe_degree>(
                &gpu_data.first_dof[patch * (1 << dim)],
                0,
                local_tid_x,
                threadIdx.y,
                z);

            atomicAdd(&dst[global_dof_indices], shared_data.local_dst[index]);
            // dst[global_dof_indices] += shared_data.local_dst[index]; //
            // colored
          }
      }
  }
#else // N_PATCH > 1

  template <int dim, int fe_degree, typename Number, LaplaceVariant laplace>
  __global__
  __launch_bounds__(Util::pow(2 * fe_degree + 2, 2)) void laplace_kernel_basic(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr int n_dofs_1d = 2 * fe_degree + 2;
    constexpr int local_dim = Util::pow(n_dofs_1d, dim);
    constexpr int n_dofs_z  = dim == 2 ? 1 : n_dofs_1d;

    const int patch_0     = blockIdx.x * N_PATCH;
    const int local_tid_x = threadIdx.x;
    const int tid         = threadIdx.y * n_dofs_1d + local_tid_x;

    SharedMemData<dim, Number, true> shared_data[2] = {
      SharedMemData<dim, Number, true>(
        get_shared_data_ptr<Number>(), 1, n_dofs_1d, local_dim),
      SharedMemData<dim, Number, true>(get_shared_data_ptr<Number>() +
                                         4 * local_dim +
                                         6 * n_dofs_1d * n_dofs_1d,
                                       1,
                                       n_dofs_1d,
                                       local_dim),
    };

    int patch = patch_0;
    int pipe  = 0;

    for (unsigned int d = 0; d < dim; ++d)
      {
        copy(gpu_data.laplace_mass_1d[gpu_data.patch_type[patch * dim + d] *
                                        n_dofs_1d * n_dofs_1d +
                                      tid],
             shared_data[pipe].local_mass[d * n_dofs_1d * n_dofs_1d + tid]);

        copy(
          gpu_data.laplace_stiff_1d[gpu_data.patch_type[patch * dim + d] *
                                      n_dofs_1d * n_dofs_1d +
                                    tid],
          shared_data[pipe].local_derivative[d * n_dofs_1d * n_dofs_1d + tid]);
      }
    for (unsigned int z = 0; z < n_dofs_z; ++z)
      {
        const unsigned int index = z * n_dofs_1d * n_dofs_1d + tid;
        const unsigned int global_dof_indices =
          Util::compute_indices<dim, fe_degree>(
            &gpu_data.first_dof[patch * (1 << dim)],
            0,
            local_tid_x,
            threadIdx.y,
            z);

        copy(src[global_dof_indices], shared_data[pipe].local_src[index]);
        shared_data[pipe].local_dst[index] = 0.;
      }
    cp_async_fence();

    for (; patch < patch_0 + N_PATCH - 1; ++patch)
      {
        if (patch + 1 < gpu_data.n_patches)
          {
            for (unsigned int d = 0; d < dim; ++d)
              {
                copy(gpu_data.laplace_mass_1d
                       [gpu_data.patch_type[(patch + 1) * dim + d] * n_dofs_1d *
                          n_dofs_1d +
                        tid],
                     shared_data[pipe ^ 1]
                       .local_mass[d * n_dofs_1d * n_dofs_1d + tid]);

                copy(gpu_data.laplace_stiff_1d
                       [gpu_data.patch_type[(patch + 1) * dim + d] * n_dofs_1d *
                          n_dofs_1d +
                        tid],
                     shared_data[pipe ^ 1]
                       .local_derivative[d * n_dofs_1d * n_dofs_1d + tid]);
              }

            for (unsigned int z = 0; z < n_dofs_z; ++z)
              {
                const unsigned int index = z * n_dofs_1d * n_dofs_1d + tid;

                const unsigned int global_dof_indices =
                  Util::compute_indices<dim, fe_degree>(
                    &gpu_data.first_dof[(patch + 1) * (1 << dim)],
                    0,
                    local_tid_x,
                    threadIdx.y,
                    z);

                copy(src[global_dof_indices],
                     shared_data[pipe ^ 1].local_src[index]);

                shared_data[pipe ^ 1].local_dst[index] = 0.;
              }
            cp_async_fence();
            cp_async_wait<1>();
            __syncthreads();
          }

        evaluate_laplace<dim, fe_degree, Number, laplace>(0,
                                                          &(shared_data[pipe]));

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index = z * n_dofs_1d * n_dofs_1d + tid;

            const unsigned int global_dof_indices =
              Util::compute_indices<dim, fe_degree>(
                &gpu_data.first_dof[patch * (1 << dim)],
                0,
                local_tid_x,
                threadIdx.y,
                z);

            atomicAdd(&dst[global_dof_indices],
                      shared_data[pipe].local_dst[index]);
          }
        pipe ^= 1;
      }

    if (patch < gpu_data.n_patches)
      {
        cp_async_wait<0>();
        __syncthreads();

        evaluate_laplace<dim, fe_degree, Number, laplace>(0,
                                                          &(shared_data[pipe]));

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index = z * n_dofs_1d * n_dofs_1d + tid;

            const unsigned int global_dof_indices =
              Util::compute_indices<dim, fe_degree>(
                &gpu_data.first_dof[patch * (1 << dim)],
                0,
                local_tid_x,
                threadIdx.y,
                z);

            atomicAdd(&dst[global_dof_indices],
                      shared_data[pipe].local_dst[index]);
          }
      }
  }
#endif



  // Load patch data cell by cell
  template <int dim, int fe_degree, typename Number, LaplaceVariant laplace>
  __global__ void
  laplace_kernel_basic_cell(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d = 2 * fe_degree + 2;
    constexpr unsigned int local_dim = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int n_dofs_z  = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedMemData<dim, Number, true> shared_data(get_shared_data_ptr<Number>(),
                                                 patch_per_block,
                                                 n_dofs_1d,
                                                 local_dim);

    if (patch < gpu_data.n_patches)
      {
        for (unsigned int d = 0; d < dim; ++d)
          {
            shared_data
              .local_mass[(local_patch * dim + d) * n_dofs_1d * n_dofs_1d +
                          threadIdx.y * n_dofs_1d + local_tid_x] =
              gpu_data.laplace_mass_1d[gpu_data.patch_type[patch * dim + d] *
                                         n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x];

            shared_data
              .local_derivative[(local_patch * dim + d) * n_dofs_1d *
                                  n_dofs_1d +
                                threadIdx.y * n_dofs_1d + local_tid_x] =
              gpu_data.laplace_stiff_1d[gpu_data.patch_type[patch * dim + d] *
                                          n_dofs_1d * n_dofs_1d +
                                        threadIdx.y * n_dofs_1d + local_tid_x];
          }

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int linear_tid =
              z * n_dofs_1d * n_dofs_1d + threadIdx.y * n_dofs_1d + local_tid_x;
            const unsigned int index =
              local_patch * local_dim + gpu_data.l_to_h[linear_tid];

            const unsigned int global_dof_indices =
              Util::compute_indices_cell<dim, fe_degree>(
                &gpu_data.first_dof[patch * (1 << dim)], linear_tid);

            shared_data.local_src[index] = src[global_dof_indices];

            shared_data.local_dst[index] = 0.;
          }

        evaluate_laplace<dim, fe_degree, Number, laplace>(local_patch,
                                                          &shared_data);

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int linear_tid =
              z * n_dofs_1d * n_dofs_1d + threadIdx.y * n_dofs_1d + local_tid_x;
            const unsigned int index =
              local_patch * local_dim + gpu_data.l_to_h[linear_tid];

            const unsigned int global_dof_indices =
              Util::compute_indices_cell<dim, fe_degree>(
                &gpu_data.first_dof[patch * (1 << dim)], linear_tid);

            atomicAdd(&dst[global_dof_indices], shared_data.local_dst[index]);
          }
      }
  }

  template <int dim, int fe_degree, typename Number, LaplaceVariant laplace>
  __global__ void
  laplace_kernel_cfmem(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d = 2 * fe_degree + 2;
    constexpr unsigned int local_dim = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int n_dofs_z  = dim == 2 ? 1 : n_dofs_1d;

    constexpr int multiple = std::is_same<Number, double>::value ?
                               Util::calculate_multiple<n_dofs_1d, 16>() :
                               Util::calculate_multiple<n_dofs_1d, 32>();

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedMemData<dim, Number, true> shared_data(get_shared_data_ptr<Number>(),
                                                 patch_per_block,
                                                 n_dofs_1d,
                                                 local_dim);

    if (patch < gpu_data.n_patches)
      {
        for (unsigned int d = 0; d < dim; ++d)
          {
            shared_data
              .local_mass[(local_patch * dim + d) * n_dofs_1d * n_dofs_1d +
                          threadIdx.y * n_dofs_1d + local_tid_x] =
              gpu_data.laplace_mass_1d[gpu_data.patch_type[patch * dim + d] *
                                         n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x];

            shared_data
              .local_derivative[(local_patch * dim + d) * n_dofs_1d *
                                  n_dofs_1d +
                                threadIdx.y * n_dofs_1d + local_tid_x] =
              gpu_data.laplace_stiff_1d[gpu_data.patch_type[patch * dim + d] *
                                          n_dofs_1d * n_dofs_1d +
                                        threadIdx.y * n_dofs_1d + local_tid_x];
          }

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index =
              local_patch * local_dim + z * n_dofs_1d * n_dofs_1d +
              ((threadIdx.y + z) & (n_dofs_1d - 1)) * n_dofs_1d +
              ((local_tid_x +
                ((threadIdx.y + z) & (n_dofs_1d - 1)) / multiple) &
               (n_dofs_1d - 1));

            const unsigned int global_dof_indices =
              Util::compute_indices<dim, fe_degree>(
                &gpu_data.first_dof[patch * (1 << dim)],
                local_patch,
                local_tid_x,
                threadIdx.y,
                z);

            shared_data.local_src[index] = src[global_dof_indices];

            shared_data.local_dst[index] = 0.;
          }

        evaluate_laplace<dim, fe_degree, Number, laplace>(local_patch,
                                                          &shared_data);

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index =
              local_patch * local_dim + z * n_dofs_1d * n_dofs_1d +
              ((threadIdx.y + z) & (n_dofs_1d - 1)) * n_dofs_1d +
              ((local_tid_x +
                ((threadIdx.y + z) & (n_dofs_1d - 1)) / multiple) &
               (n_dofs_1d - 1));

            const unsigned int global_dof_indices =
              Util::compute_indices<dim, fe_degree>(
                &gpu_data.first_dof[patch * (1 << dim)],
                local_patch,
                local_tid_x,
                threadIdx.y,
                z);

            atomicAdd(&dst[global_dof_indices], shared_data.local_dst[index]);
          }
      }
  }

  template <int dim, int fe_degree, typename Number, LaplaceVariant laplace>
  __global__ void
  laplace_kernel_tensorcore(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d         = 2 * fe_degree + 2;
    constexpr unsigned int n_dofs_1d_padding = n_dofs_1d + Util::padding;
    constexpr unsigned int local_dim         = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int local_dim_padding =
      Util::pow(n_dofs_1d, dim - 1) * n_dofs_1d_padding;
    constexpr unsigned int n_dofs_z = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedMemData<dim, Number, true> shared_data(get_shared_data_ptr<Number>(),
                                                 patch_per_block,
                                                 n_dofs_1d,
                                                 local_dim_padding);

    if (patch < gpu_data.n_patches)
      {
#if GACCESS == 1
#  if TIMING == 2
        auto start = clock64();
#  endif
        for (unsigned int d = 0; d < dim; ++d)
          {
            shared_data
              .local_mass[(local_patch * dim + d) * n_dofs_1d *
                            n_dofs_1d_padding +
                          threadIdx.y * n_dofs_1d_padding + local_tid_x] =
              gpu_data.laplace_mass_1d[gpu_data.patch_type[patch * dim + d] *
                                         n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x];

            shared_data
              .local_derivative[(local_patch * dim + d) * n_dofs_1d *
                                  n_dofs_1d_padding +
                                threadIdx.y * n_dofs_1d_padding + local_tid_x] =
              gpu_data.laplace_stiff_1d[gpu_data.patch_type[patch * dim + d] *
                                          n_dofs_1d * n_dofs_1d +
                                        threadIdx.y * n_dofs_1d + local_tid_x];
          }

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index_padding =
              local_patch * local_dim_padding +
              z * n_dofs_1d * n_dofs_1d_padding +
              threadIdx.y * n_dofs_1d_padding + local_tid_x;

            const unsigned int global_dof_indices =
              Util::compute_indices<dim, fe_degree>(
                &gpu_data.first_dof[patch * (1 << dim)],
                local_patch,
                local_tid_x,
                threadIdx.y,
                z);

            shared_data.local_src[index_padding] = src[global_dof_indices];

            shared_data.local_dst[index_padding] = 0.;
          }
#  if TIMING == 2
        __syncthreads();
        auto elapsed = clock64() - start;
        if (threadIdx.x == 0 && threadIdx.y == 0)
          printf("loading from global timing info: %ld cycles\n", elapsed);
#  endif
#endif

#if GACCESS == 0
        for (unsigned int d = 0; d < dim; ++d)
          {
            shared_data
              .local_mass[(local_patch * dim + d) * n_dofs_1d * n_dofs_1d +
                          (threadIdx.y * n_dofs_1d + local_tid_x)] =
              local_tid_x;

            shared_data
              .local_derivative[(local_patch * dim + d) * n_dofs_1d *
                                  n_dofs_1d +
                                (threadIdx.y * n_dofs_1d + local_tid_x)] =
              local_tid_x;
          }

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index =
              local_patch * local_dim + (z * n_dofs_1d * n_dofs_1d +
                                         threadIdx.y * n_dofs_1d + local_tid_x);

            shared_data.local_src[index] = index;

            shared_data.local_dst[index] = 0.;
          }
#endif

#if TIMING == 2
        __syncthreads();
        auto start_c = clock64();
#endif
        evaluate_laplace<dim, fe_degree, Number, laplace>(local_patch,
                                                          &shared_data);
#if TIMING == 2
        __syncthreads();
        auto elapsed_c = clock64() - start_c;
        if (threadIdx.x == 0 && threadIdx.y == 0)
          dst[patch] = elapsed_c;
#endif

#if GACCESS == 1
#  if TIMING == 2
        start = clock64();
#  endif
        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index_padding =
              local_patch * local_dim_padding +
              z * n_dofs_1d * n_dofs_1d_padding +
              threadIdx.y * n_dofs_1d_padding + local_tid_x;

            const unsigned int global_dof_indices =
              Util::compute_indices<dim, fe_degree>(
                &gpu_data.first_dof[patch * (1 << dim)],
                local_patch,
                local_tid_x,
                threadIdx.y,
                z);

            atomicAdd(&dst[global_dof_indices],
                      shared_data.local_dst[index_padding]);
          }
#  if TIMING == 2
        __syncthreads();
        elapsed = clock64() - start;
        if (threadIdx.x == 0 && threadIdx.y == 0)
          printf("Storing to global timing info: %ld cycles\n", elapsed);
#  endif
#endif
      }
  }

#if MMAKERNEL == 0
  template <int dim, int fe_degree, typename Number, LaplaceVariant laplace>
  __global__ void
  laplace_kernel_tensorcoremma(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr int n_dofs_1d = 2 * fe_degree + 2;
    constexpr int local_dim = Util::pow(n_dofs_1d, dim);
    constexpr int n_dofs_z  = dim == 2 ? 1 : n_dofs_1d;

    const int patch       = blockIdx.x;
    const int local_tid_x = threadIdx.x;
    const int tid         = threadIdx.y * n_dofs_1d + local_tid_x;
    const int warpid      = threadIdx.y / n_dofs_1d;

    SharedMemData<dim, Number, true> shared_data(get_shared_data_ptr<Number>(),
                                                 1,
                                                 n_dofs_1d,
                                                 local_dim);

    if (patch < gpu_data.n_patches)
      {
        if (warpid == 0)
          for (int d = 0; d < dim; ++d)
            {
              shared_data.local_mass[d * n_dofs_1d * n_dofs_1d + tid] =
                gpu_data.laplace_mass_1d[gpu_data.patch_type[patch * dim + d] *
                                           n_dofs_1d * n_dofs_1d +
                                         tid];

              shared_data.local_derivative[d * n_dofs_1d * n_dofs_1d + tid] =
                gpu_data.laplace_stiff_1d[gpu_data.patch_type[patch * dim + d] *
                                            n_dofs_1d * n_dofs_1d +
                                          tid];
            }

        if (warpid == 0)
          for (int z = 0; z < n_dofs_z; ++z)
            {
              const int index = z * n_dofs_1d * n_dofs_1d + tid;

              const int global_dof_indices =
                Util::compute_indices<dim, fe_degree>(
                  &gpu_data.first_dof[patch * (1 << dim)],
                  0,
                  local_tid_x,
                  threadIdx.y,
                  z);

              shared_data.local_src[index] = src[global_dof_indices];
              shared_data.local_dst[index] = 0.;
            }

        evaluate_laplace<dim, fe_degree, Number, laplace>(0, &shared_data);

        if (warpid == 0)
          for (unsigned int z = 0; z < n_dofs_z; ++z)
            {
              const unsigned int index = z * n_dofs_1d * n_dofs_1d + tid;

              const int global_dof_indices =
                Util::compute_indices<dim, fe_degree>(
                  &gpu_data.first_dof[patch * (1 << dim)],
                  0,
                  local_tid_x,
                  threadIdx.y,
                  z);

              atomicAdd(&dst[global_dof_indices], shared_data.local_dst[index]);
            }
      }
  }
#endif

#if MMAKERNEL != 0 && N_PATCH == 1
  template <int dim, int fe_degree, typename Number, LaplaceVariant laplace>
  __global__ typename std::enable_if<fe_degree == 3 || fe_degree == 7>::type
  laplace_kernel_tensorcoremma(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr int n_dofs_1d = 2 * fe_degree + 2;
    constexpr int local_dim = Util::pow(n_dofs_1d, dim);
    constexpr int n_dofs_z  = dim == 2 ? 1 : n_dofs_1d;

    const int patch       = blockIdx.x;
    const int local_tid_x = threadIdx.x;
    const int tid         = threadIdx.y * n_dofs_1d + local_tid_x;
    const int warpid      = threadIdx.y / n_dofs_1d;

    SharedMemData<dim, Number, true> shared_data(get_shared_data_ptr<Number>(),
                                                 1,
                                                 n_dofs_1d,
                                                 local_dim);

    if (patch < gpu_data.n_patches)
      {
#  if TIMING == 2
        auto start = clock64();
#  endif
        if (warpid == 0)
          for (int d = 0; d < dim; ++d)
            {
#  if GACCESS == 0
              shared_data.local_mass[d * n_dofs_1d * n_dofs_1d + tid] =
                local_tid_x;
              shared_data.local_derivative[d * n_dofs_1d * n_dofs_1d + tid] =
                local_tid_x;
#  elif GACCESS == 1

              auto inds =
                gpu_data.patch_type[patch * dim + d] * n_dofs_1d * n_dofs_1d +
                tid;
              if constexpr (std::is_same_v<Number, double> ||
                            (MMAKERNEL != 7 && MMAKERNEL != 8))
                {
                  auto ind =
                    d * n_dofs_1d * n_dofs_1d +
                    (tid ^ Util::get_base<n_dofs_1d, Number>(threadIdx.y));

                  shared_data.local_mass[ind] = gpu_data.laplace_mass_1d[inds];
                  shared_data.local_derivative[ind] =
                    gpu_data.laplace_stiff_1d[inds];
                }
              else
                {
                  auto ind =
                    d * n_dofs_1d * n_dofs_1d +
                    (tid ^ Util::get_base<n_dofs_1d, half>(threadIdx.y));

                  Number tmp_mass            = gpu_data.laplace_mass_1d[inds];
                  Number tmp_der             = gpu_data.laplace_stiff_1d[inds];
                  shared_data.mass_half[ind] = __float2half(tmp_mass);
                  shared_data.der_half[ind]  = __float2half(tmp_der);

#    if ERRCOR == 1
                  constexpr int scale        = 1 << 11;

                  shared_data.mass_half[dim * n_dofs_1d * n_dofs_1d + ind] =
                    __float2half(
                      (tmp_mass - __half2float(__float2half(tmp_mass))) *
                      scale);

                  shared_data.der_half[dim * n_dofs_1d * n_dofs_1d + ind] =
                    __float2half(
                      (tmp_der - __half2float(__float2half(tmp_der))) * scale);
#    endif
                }
#  elif GACCESS == 2
              shared_data.local_mass[d * n_dofs_1d * n_dofs_1d + tid] =
                gpu_data.laplace_mass_1d[gpu_data.patch_type[patch * dim + d] *
                                           n_dofs_1d * n_dofs_1d +
                                         tid];
              shared_data.local_derivative[d * n_dofs_1d * n_dofs_1d + tid] =
                gpu_data.laplace_stiff_1d[gpu_data.patch_type[patch * dim + d] *
                                            n_dofs_1d * n_dofs_1d +
                                          tid];
#  endif
            }

        if (warpid == 0)
          for (int z = 0; z < n_dofs_z; ++z)
            {
#  if GACCESS == 0
              const int index              = z * n_dofs_1d * n_dofs_1d + tid;
              shared_data.local_src[index] = local_tid_x;
              shared_data.local_dst[index] = 0.;
#  elif GACCESS == 1
              const int index =
                (z * n_dofs_1d * n_dofs_1d + tid) ^
                Util::get_base<n_dofs_1d, Number>(threadIdx.y, z);

              const int global_dof_indices =
                Util::compute_indices<dim, fe_degree>(
                  &gpu_data.first_dof[patch * (1 << dim)],
                  0,
                  local_tid_x,
                  threadIdx.y,
                  z);

              shared_data.local_src[index] = src[global_dof_indices];
              shared_data.local_dst[index] = 0.;
#  elif GACCESS == 2
              const int index = z * n_dofs_1d * n_dofs_1d + tid;
              const int global_dof_indices =
                index + gpu_data.first_dof[patch * (1 << dim)];
              shared_data.local_src[index] = src[global_dof_indices];
              shared_data.local_dst[index] = 0.;
#  endif
            }
#  if TIMING == 2
        __syncthreads();
        auto elapsed = clock64() - start;
        if (threadIdx.x == 0 && threadIdx.y == 0)
          printf("loading from global timing info: %ld cycles\n", elapsed);

        __syncthreads();
        auto start_c = clock64();
#  endif

        evaluate_laplace<dim, fe_degree, Number, laplace>(0, &shared_data);
#  if TIMING == 2
        __syncthreads();
        auto elapsed_c = clock64() - start_c;
        if (threadIdx.x == 0 && threadIdx.y == 0)
          dst[patch] = elapsed_c;
#  endif


#  if TIMING == 2
        start = clock64();
#  endif

        if (warpid == 0)
          for (int z = 0; z < n_dofs_z; ++z)
            {
#  if GACCESS == 1
              const int index =
                ((z * n_dofs_1d * n_dofs_1d + tid) ^
                 Util::get_base<n_dofs_1d, Number>(threadIdx.y, z));

              const int global_dof_indices =
                Util::compute_indices<dim, fe_degree>(
                  &gpu_data.first_dof[patch * (1 << dim)],
                  0,
                  local_tid_x,
                  threadIdx.y,
                  z);
              atomicAdd(&dst[global_dof_indices], shared_data.local_dst[index]);
#  elif GACCESS == 2
              const int index              = z * n_dofs_1d * n_dofs_1d + tid;
              const int global_dof_indices =
                index + gpu_data.first_dof[patch * (1 << dim)];
              atomicAdd(&dst[global_dof_indices], shared_data.local_dst[index]);
#  endif
            }
#  if TIMING == 2
        __syncthreads();
        elapsed = clock64() - start;
        if (threadIdx.x == 0 && threadIdx.y == 0)
          printf("Storing to global timing info: %ld cycles\n", elapsed);
#  endif
      }
  }
#else

  template <int dim, int fe_degree, typename Number, LaplaceVariant laplace>
  __global__ __launch_bounds__(1 * Util::pow(2 * fe_degree + 2, 2))
    typename std::enable_if<fe_degree == 3 || fe_degree == 7>::type
    laplace_kernel_tensorcoremma(
      const Number                                                 *src,
      Number                                                       *dst,
      const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr int n_stages = 2;

    constexpr int n_dofs_1d = 2 * fe_degree + 2;
    constexpr int local_dim = Util::pow(n_dofs_1d, dim);
    constexpr int n_dofs_z  = dim == 2 ? 1 : n_dofs_1d;

    const int patch_0     = blockIdx.x * N_PATCH;
    const int local_tid_x = threadIdx.x;
    const int tid         = threadIdx.y * n_dofs_1d + local_tid_x;

    int dof_offset[n_stages], shape_offset[n_stages];
    for (int i = 0; i < n_stages; ++i)
      {
        dof_offset[i]   = i * local_dim;
        shape_offset[i] = i * n_dofs_1d * n_dofs_1d * 3;
      }

    SharedMemData<dim, Number, true> shared_data(get_shared_data_ptr<Number>(),
                                                 1,
                                                 n_dofs_1d,
                                                 local_dim);

    auto load_shape_data = [&](int patch_id, int stage) {
      for (int d = 0; d < dim; ++d)
        {
          auto inds =
            gpu_data.patch_type[patch_id * dim + d] * n_dofs_1d * n_dofs_1d +
            tid;

          auto ind = shape_offset[stage] + d * n_dofs_1d * n_dofs_1d +
                     (tid ^ Util::get_base<n_dofs_1d, Number>(threadIdx.y));

          // shared_data.local_mass[ind]       = gpu_data.laplace_mass_1d[inds];
          // shared_data.local_derivative[ind] =
          // gpu_data.laplace_stiff_1d[inds];

          copy(gpu_data.laplace_mass_1d[inds], shared_data.local_mass[ind]);
          copy(gpu_data.laplace_stiff_1d[inds],
               shared_data.local_derivative[ind]);
        }
    };

    auto load_dof_val = [&](int patch_id, int stage) {
      auto xy_offset =
        threadIdx.y / (fe_degree + 1) * 2 + local_tid_x / (fe_degree + 1);
      auto xy_shift =
        (threadIdx.y & fe_degree) * (fe_degree + 1) + (local_tid_x & fe_degree);

      for (int z = 0; z < n_dofs_z; ++z)
        {
          const int index =
            dof_offset[stage] + (z * n_dofs_1d * n_dofs_1d + tid) ^
            Util::get_base<n_dofs_1d, Number>(threadIdx.y, z);

          const int global_dof_indices = Util::compute_indices<dim, fe_degree>(
            &gpu_data.first_dof[patch_id * (1 << dim)], xy_offset, xy_shift, z);

          // shared_data.local_src[index] = src[global_dof_indices];

          copy(src[global_dof_indices], shared_data.local_src[index]);
        }
    };

    auto store_dof_val = [&](int patch_id) {
      auto xy_offset =
        threadIdx.y / (fe_degree + 1) * 2 + local_tid_x / (fe_degree + 1);
      auto xy_shift =
        (threadIdx.y & fe_degree) * (fe_degree + 1) + (local_tid_x & fe_degree);

      for (int z = 0; z < n_dofs_z; ++z)
        {
          const int index = ((z * n_dofs_1d * n_dofs_1d + tid) ^
                             Util::get_base<n_dofs_1d, Number>(threadIdx.y, z));

          const int global_dof_indices = Util::compute_indices<dim, fe_degree>(
            &gpu_data.first_dof[patch_id * (1 << dim)], xy_offset, xy_shift, z);
          atomicAdd(&dst[global_dof_indices], shared_data.local_dst[index]);
        }
    };

    auto compute = [&](int stage) {
      evaluate_laplace_pipe<dim, fe_degree, Number, laplace>(
        shared_data.local_dst,
        shared_data.local_src + dof_offset[stage],
        shared_data.local_mass + shape_offset[stage],
        shared_data.local_derivative + shape_offset[stage],
        shared_data.tmp);
    };

    for (unsigned int compute_patch = 0, fetch_patch = 0;
         compute_patch < N_PATCH;
         ++compute_patch)
      {
        for (;
             fetch_patch < N_PATCH && fetch_patch < (compute_patch + n_stages);
             ++fetch_patch)
          {
            auto patch = patch_0 + fetch_patch;
            if (patch < gpu_data.n_patches)
              {
                load_shape_data(patch, fetch_patch % n_stages);
                load_dof_val(patch, fetch_patch % n_stages);
                cp_async_fence();
              }
          }

        auto patch = patch_0 + compute_patch;
        if (patch < gpu_data.n_patches)
          {
            cp_async_wait<n_stages - 1>();
            __syncthreads();

            compute(compute_patch % n_stages);
            __syncthreads();
            store_dof_val(patch);
          }
      }
  }
#endif // MMAKERNEL != 0 && N_PATCH == 1



#if MMAKERNEL == 5
  template <int dim, int fe_degree, typename Number, LaplaceVariant laplace>
  __global__ typename std::enable_if<fe_degree == 3 || fe_degree == 7>::type
  laplace_kernel_tensorcoremma_s(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr int n_dofs_1d = 2 * fe_degree + 2;
    constexpr int n_dofs_xd = (2 * fe_degree + 2) * (fe_degree + 1);
    constexpr int local_dim = Util::pow(n_dofs_1d, dim);
    constexpr int n_dofs_z  = dim == 2 ? 1 : n_dofs_1d;

    const int patch       = blockIdx.x;
    const int local_tid_x = threadIdx.x;
    const int tid         = threadIdx.y * n_dofs_1d + local_tid_x;

    SharedMemData<dim, Number, true> shared_data(get_shared_data_ptr<Number>(),
                                                 1,
                                                 n_dofs_1d,
                                                 local_dim);

    if (patch < gpu_data.n_patches)
      {
        for (int d = 0; d < dim; ++d)
          {
#  if GACCESS == 0
            shared_data.local_mass[d * n_dofs_1d * n_dofs_1d + tid] =
              local_tid_x;
            shared_data
              .local_mass[d * n_dofs_1d * n_dofs_1d + tid + n_dofs_xd] =
              local_tid_x;
            shared_data.local_derivative[d * n_dofs_1d * n_dofs_1d + tid] =
              local_tid_x;
            shared_data
              .local_derivative[d * n_dofs_1d * n_dofs_1d + tid + n_dofs_xd] =
              local_tid_x;
#  elif GACCESS == 1
            shared_data.local_mass[d * n_dofs_1d * n_dofs_1d +
                                   (tid ^ Util::get_base<n_dofs_1d, Number>(
                                            threadIdx.y))] =
              gpu_data.laplace_mass_1d[gpu_data.patch_type[patch * dim + d] *
                                         n_dofs_1d * n_dofs_1d +
                                       tid];
            shared_data.local_mass[d * n_dofs_1d * n_dofs_1d +
                                   ((tid + n_dofs_xd) ^
                                    Util::get_base<n_dofs_1d, Number>(
                                      threadIdx.y + fe_degree + 1))] =
              gpu_data.laplace_mass_1d[gpu_data.patch_type[patch * dim + d] *
                                         n_dofs_1d * n_dofs_1d +
                                       tid + n_dofs_xd];

            shared_data
              .local_derivative[d * n_dofs_1d * n_dofs_1d +
                                (tid ^ Util::get_base<n_dofs_1d, Number>(
                                         threadIdx.y))] =
              gpu_data.laplace_stiff_1d[gpu_data.patch_type[patch * dim + d] *
                                          n_dofs_1d * n_dofs_1d +
                                        tid];
            shared_data.local_derivative[d * n_dofs_1d * n_dofs_1d +
                                         ((tid + n_dofs_xd) ^
                                          Util::get_base<n_dofs_1d, Number>(
                                            threadIdx.y + fe_degree + 1))] =
              gpu_data.laplace_stiff_1d[gpu_data.patch_type[patch * dim + d] *
                                          n_dofs_1d * n_dofs_1d +
                                        tid + n_dofs_xd];
#  elif GACCESS == 2
            shared_data.local_mass[d * n_dofs_1d * n_dofs_1d + tid] =
              gpu_data.laplace_mass_1d[gpu_data.patch_type[patch * dim + d] *
                                         n_dofs_1d * n_dofs_1d +
                                       tid];
            shared_data.local_derivative[d * n_dofs_1d * n_dofs_1d + tid] =
              gpu_data.laplace_stiff_1d[gpu_data.patch_type[patch * dim + d] *
                                          n_dofs_1d * n_dofs_1d +
                                        tid];
#  endif
          }

        for (int z = 0; z < n_dofs_z; ++z)
          {
#  if GACCESS == 0
            const int index  = z * n_dofs_1d * n_dofs_1d + tid;
            const int index2 = z * n_dofs_1d * n_dofs_1d + tid + n_dofs_xd;
            shared_data.local_src[index]  = local_tid_x;
            shared_data.local_src[index2] = local_tid_x;
            shared_data.local_dst[index]  = 0.;
            shared_data.local_dst[index2] = 0.;
#  elif GACCESS == 1
            const int index = (z * n_dofs_1d * n_dofs_1d + tid) ^
                              Util::get_base<n_dofs_1d, Number>(threadIdx.y, z);
            const int index2 =
              (z * n_dofs_1d * n_dofs_1d + tid + n_dofs_xd) ^
              Util::get_base<n_dofs_1d, Number>(threadIdx.y + fe_degree + 1, z);

            const int global_dof_indices =
              Util::compute_indices<dim, fe_degree>(
                &gpu_data.first_dof[patch * (1 << dim)],
                0,
                local_tid_x,
                threadIdx.y,
                z);
            const int global_dof_indices2 =
              Util::compute_indices<dim, fe_degree>(
                &gpu_data.first_dof[patch * (1 << dim)],
                0,
                local_tid_x,
                threadIdx.y + fe_degree + 1,
                z);

            shared_data.local_src[index]  = src[global_dof_indices];
            shared_data.local_src[index2] = src[global_dof_indices2];
            shared_data.local_dst[index]  = 0.;
            shared_data.local_dst[index2] = 0.;
#  elif GACCESS == 2
            const int index = z * n_dofs_1d * n_dofs_1d + tid;
            const int global_dof_indices =
              index + gpu_data.first_dof[patch * (1 << dim)];
            shared_data.local_src[index] = src[global_dof_indices];
            shared_data.local_dst[index] = 0.;
#  endif
          }

        evaluate_laplace<dim, fe_degree, Number, laplace>(0, &shared_data);

        for (int z = 0; z < n_dofs_z; ++z)
          {
#  if GACCESS == 1
            const int index =
              ((z * n_dofs_1d * n_dofs_1d + tid) ^
               Util::get_base<n_dofs_1d, Number>(threadIdx.y, z));
            const int index2 =
              ((z * n_dofs_1d * n_dofs_1d + tid + n_dofs_xd) ^
               Util::get_base<n_dofs_1d, Number>(threadIdx.y + fe_degree + 1,
                                                 z));

            const int global_dof_indices =
              Util::compute_indices<dim, fe_degree>(
                &gpu_data.first_dof[patch * (1 << dim)],
                0,
                local_tid_x,
                threadIdx.y,
                z);
            const int global_dof_indices2 =
              Util::compute_indices<dim, fe_degree>(
                &gpu_data.first_dof[patch * (1 << dim)],
                0,
                local_tid_x,
                threadIdx.y + fe_degree + 1,
                z);
            atomicAdd(&dst[global_dof_indices], shared_data.local_dst[index]);
            atomicAdd(&dst[global_dof_indices2], shared_data.local_dst[index2]);
#  elif GACCESS == 2
            const int index               = z * n_dofs_1d * n_dofs_1d + tid;
            const int global_dof_indices =
              index + gpu_data.first_dof[patch * (1 << dim)];
            atomicAdd(&dst[global_dof_indices], shared_data.local_dst[index]);
#  endif
          }
      }
  }
#endif

  template <int dim, int fe_degree, typename Number>
  __global__ void
  loop_kernel_seperate_inv(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d           = 2 * fe_degree;
    constexpr unsigned int local_dim           = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int regular_vpatch_size = Util::pow(2, dim);
    constexpr unsigned int n_dofs_z            = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedMemData<dim, Number, false> shared_data(get_shared_data_ptr<Number>(),
                                                  patch_per_block,
                                                  n_dofs_1d,
                                                  local_dim);

    if (patch < gpu_data.n_patches)
      {
        shared_data.local_mass[local_tid_x] = gpu_data.eigenvalues[local_tid_x];
        shared_data.local_derivative[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.eigenvectors[threadIdx.y * n_dofs_1d + local_tid_x];

        // #pragma unroll
        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index = local_patch * local_dim +
                                       z * n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x;

            const unsigned int global_dof_indices =
              Util::compute_indices<dim, fe_degree>(
                &gpu_data.first_dof[patch * regular_vpatch_size],
                local_patch,
                local_tid_x + 1,
                threadIdx.y + 1,
                z + dim - 2);

            shared_data.local_src[index] = src[global_dof_indices];

            shared_data.local_dst[index] = dst[global_dof_indices];
          }

        evaluate_smooth_inv<dim, fe_degree, Number, SmootherVariant::GLOBAL>(
          local_patch, &shared_data);

        // #pragma unroll
        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index = local_patch * local_dim +
                                       z * n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x;

            const unsigned int global_dof_indices =
              Util::compute_indices<dim, fe_degree>(
                &gpu_data.first_dof[patch * regular_vpatch_size],
                local_patch,
                local_tid_x + 1,
                threadIdx.y + 1,
                z + dim - 2);

            dst[global_dof_indices] =
              shared_data.local_dst[index] * gpu_data.relaxation;
          }
      }
  }

  template <int dim, int fe_degree, typename Number>
  __global__ void
  loop_kernel_seperate_all(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d           = 2 * fe_degree + 2;
    constexpr unsigned int local_dim           = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int regular_vpatch_size = Util::pow(2, dim);
    constexpr unsigned int n_dofs_z            = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedMemData<dim, Number, false> shared_data(get_shared_data_ptr<Number>(),
                                                  patch_per_block,
                                                  n_dofs_1d,
                                                  local_dim);

    if (patch < gpu_data.n_patches)
      {
        shared_data.local_mass[local_tid_x] =
          gpu_data.eigenvalues[n_dofs_1d * 2 + local_tid_x];
        shared_data.local_derivative[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.eigenvectors[n_dofs_1d * n_dofs_1d * 2 +
                                threadIdx.y * n_dofs_1d + local_tid_x];

        // #pragma unroll
        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index = local_patch * local_dim +
                                       z * n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x;

            const unsigned int global_dof_indices =
              Util::compute_indices<dim, fe_degree>(
                &gpu_data.first_dof[patch * regular_vpatch_size],
                local_patch,
                local_tid_x,
                threadIdx.y,
                z);

            shared_data.local_src[index] = src[global_dof_indices];

            shared_data.local_dst[index] = dst[global_dof_indices];
          }

        evaluate_smooth_inv<dim,
                            fe_degree + 1,
                            Number,
                            SmootherVariant::GLOBAL>(local_patch, &shared_data);

        // #pragma unroll
        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index = local_patch * local_dim +
                                       z * n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x;

            const unsigned int global_dof_indices =
              Util::compute_indices<dim, fe_degree>(
                &gpu_data.first_dof[patch * regular_vpatch_size],
                local_patch,
                local_tid_x,
                threadIdx.y,
                z);

            dst[global_dof_indices] =
              shared_data.local_dst[index] * gpu_data.relaxation;
          }
      }
  }

#if MMAKERNEL != 0
  template <int dim, int fe_degree, typename Number, LaplaceVariant laplace>
  __global__ typename std::enable_if<fe_degree != 3 && fe_degree != 7>::type
  laplace_kernel_tensorcoremma(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d   = 2 * fe_degree + 2;
    constexpr unsigned int n_dofs_1d_p = fe_degree <= 3 ? 8 : 16;
    constexpr unsigned int local_dim_p =
      Util::pow(n_dofs_1d_p, dim - 1) * n_dofs_1d;
    constexpr unsigned int n_dofs_z = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d_p;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d_p;

    SharedMemData<dim, Number, true> shared_data(get_shared_data_ptr<Number>(),
                                                 patch_per_block,
                                                 n_dofs_1d_p,
                                                 local_dim_p);

    if (patch < gpu_data.n_patches)
      {
#  if GACCESS == 1
#    if TIMING == 2
        auto start = clock64();
#    endif
        if (local_tid_x < n_dofs_1d && threadIdx.y < n_dofs_1d)
          for (unsigned int d = 0; d < dim; ++d)
            {
              shared_data.local_mass
                [(local_patch * dim + d) * n_dofs_1d_p * n_dofs_1d_p +
                 ((threadIdx.y * n_dofs_1d_p + local_tid_x) ^
                  Util::get_base<n_dofs_1d, Number>(threadIdx.y, 0))] =
                gpu_data.laplace_mass_1d[gpu_data.patch_type[patch * dim + d] *
                                           n_dofs_1d * n_dofs_1d +
                                         threadIdx.y * n_dofs_1d + local_tid_x];

              shared_data.local_derivative
                [(local_patch * dim + d) * n_dofs_1d_p * n_dofs_1d_p +
                 ((threadIdx.y * n_dofs_1d_p + local_tid_x) ^
                  Util::get_base<n_dofs_1d, Number>(threadIdx.y, 0))] =
                gpu_data
                  .laplace_stiff_1d[gpu_data.patch_type[patch * dim + d] *
                                      n_dofs_1d * n_dofs_1d +
                                    threadIdx.y * n_dofs_1d + local_tid_x];
            }
        else
          for (unsigned int d = 0; d < dim; ++d)
            {
              shared_data.local_mass
                [(local_patch * dim + d) * n_dofs_1d_p * n_dofs_1d_p +
                 ((threadIdx.y * n_dofs_1d_p + local_tid_x) ^
                  Util::get_base<n_dofs_1d, Number>(threadIdx.y, 0))] = 0.;

              shared_data.local_derivative
                [(local_patch * dim + d) * n_dofs_1d_p * n_dofs_1d_p +
                 ((threadIdx.y * n_dofs_1d_p + local_tid_x) ^
                  Util::get_base<n_dofs_1d, Number>(threadIdx.y, 0))] = 0.;
            }

        if (local_tid_x < n_dofs_1d && threadIdx.y < n_dofs_1d)
          for (unsigned int z = 0; z < n_dofs_z; ++z)
            {
              const unsigned int index =
                local_patch * local_dim_p +
                ((z * n_dofs_1d_p * n_dofs_1d_p + threadIdx.y * n_dofs_1d_p +
                  local_tid_x) ^
                 Util::get_base<n_dofs_1d, Number>(threadIdx.y, z));

              const unsigned int global_dof_indices =
                Util::compute_indices<dim, fe_degree>(
                  &gpu_data.first_dof[patch * (1 << dim)],
                  local_patch,
                  local_tid_x,
                  threadIdx.y,
                  z);

              shared_data.local_src[index] = src[global_dof_indices];

              shared_data.local_dst[index] = 0.;
            }
            // else
            //   for (unsigned int z = 0; z < n_dofs_z; ++z)
            //     {
            //       const unsigned int index =
            //         local_patch * local_dim_p +
            //         ((z * n_dofs_1d_p * n_dofs_1d_p + threadIdx.y *
            //         n_dofs_1d_p +
            //           local_tid_x) ^
            //          Util::get_base<n_dofs_1d, Number>(threadIdx.y, z));

            //       shared_data.local_src[index] = 0.;

            //       shared_data.local_dst[index] = 0.;
            //     }

#    if TIMING == 2
        __syncthreads();
        auto elapsed = clock64() - start;
        if (threadIdx.x == 0 && threadIdx.y == 0)
          printf("loading from global timing info: %ld cycles\n", elapsed);
#    endif
#  endif

#  if GACCESS == 0
        for (unsigned int d = 0; d < dim; ++d)
          {
            shared_data
              .local_mass[(local_patch * dim + d) * n_dofs_1d * n_dofs_1d_p +
                          (threadIdx.y * n_dofs_1d_p + local_tid_x)] =
              local_tid_x;

            shared_data
              .local_derivative[(local_patch * dim + d) * n_dofs_1d *
                                  n_dofs_1d_p +
                                (threadIdx.y * n_dofs_1d_p + local_tid_x)] =
              local_tid_x;
          }

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index =
              local_patch * local_dim_p +
              (z * n_dofs_1d * n_dofs_1d_p + threadIdx.y * n_dofs_1d_p +
               local_tid_x);

            shared_data.local_src[index] = index;

            shared_data.local_dst[index] = 0.;
          }
#  endif

#  if TIMING == 2
        __syncthreads();
        auto start_c = clock64();
#  endif
        evaluate_laplace<dim, fe_degree, Number, laplace>(local_patch,
                                                          &shared_data);
#  if TIMING == 2
        __syncthreads();
        auto elapsed_c = clock64() - start_c;
        if (threadIdx.x == 0 && threadIdx.y == 0)
          dst[patch] = elapsed_c;
#  endif

#  if GACCESS == 1
#    if TIMING == 2
        start = clock64();
#    endif
        if (local_tid_x < n_dofs_1d && threadIdx.y < n_dofs_1d)
          for (unsigned int z = 0; z < n_dofs_z; ++z)
            {
              const unsigned int index =
                local_patch * local_dim_p +
                ((z * n_dofs_1d_p * n_dofs_1d_p + threadIdx.y * n_dofs_1d_p +
                  local_tid_x) ^
                 Util::get_base<n_dofs_1d, Number>(threadIdx.y, z));

              const unsigned int global_dof_indices =
                Util::compute_indices<dim, fe_degree>(
                  &gpu_data.first_dof[patch * (1 << dim)],
                  local_patch,
                  local_tid_x,
                  threadIdx.y,
                  z);

              atomicAdd(&dst[global_dof_indices], shared_data.local_dst[index]);
            }
#    if TIMING == 2
        __syncthreads();
        elapsed = clock64() - start;
        if (threadIdx.x == 0 && threadIdx.y == 0)
          printf("Storing to global timing info: %ld cycles\n", elapsed);
#    endif
#  endif
      }
  }
#endif

  template <int dim, int fe_degree, typename Number, LaplaceVariant lapalace>
  __global__ void
  loop_kernel_fused_l(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d           = 2 * fe_degree + 2;
    constexpr unsigned int local_dim           = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int regular_vpatch_size = Util::pow(2, dim);
    constexpr unsigned int n_dofs_z            = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedMemData<dim, Number, false> shared_data(get_shared_data_ptr<Number>(),
                                                  patch_per_block,
                                                  n_dofs_1d,
                                                  local_dim);

    if (patch < gpu_data.n_patches)
      {
        shared_data.local_mass[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.smooth_mass_1d[threadIdx.y * n_dofs_1d + local_tid_x];
        shared_data.local_derivative[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.smooth_stiff_1d[threadIdx.y * n_dofs_1d + local_tid_x];

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index = local_patch * local_dim +
                                       z * n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x;

            const unsigned int global_dof_indices =
              Util::compute_indices<dim, fe_degree>(
                &gpu_data.first_dof[patch * regular_vpatch_size],
                local_patch,
                local_tid_x,
                threadIdx.y,
                z);

            shared_data.local_src[index] = src[global_dof_indices];

            shared_data.local_dst[index] = dst[global_dof_indices];
          }

        evaluate_smooth<dim,
                        fe_degree,
                        Number,
                        lapalace,
                        SmootherVariant::FUSED_L>(local_patch,
                                                  &shared_data,
                                                  &gpu_data);

        const unsigned int linear_tid = local_tid_x + threadIdx.y * n_dofs_1d;
        if (dim == 2)
          {
            if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
              {
                int row = linear_tid / (n_dofs_1d - 2) + 1;
                int col = linear_tid % (n_dofs_1d - 2) + 1;

                const unsigned int index =
                  local_patch * local_dim + row * n_dofs_1d + col;

                const unsigned int global_dof_indices =
                  Util::compute_indices<dim, fe_degree>(
                    &gpu_data.first_dof[patch * regular_vpatch_size],
                    local_patch,
                    col,
                    row,
                    0);

                dst[global_dof_indices] =
                  shared_data.local_dst[index] * gpu_data.relaxation;
              }
          }
        else if (dim == 3)
          {
            if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
              for (unsigned int z = 1; z < n_dofs_1d - 1; ++z)
                {
                  int row = linear_tid / (n_dofs_1d - 2) + 1;
                  int col = linear_tid % (n_dofs_1d - 2) + 1;

                  unsigned int index = local_patch * local_dim +
                                       z * n_dofs_1d * n_dofs_1d +
                                       row * n_dofs_1d + col;

                  const unsigned int global_dof_indices =
                    Util::compute_indices<dim, fe_degree>(
                      &gpu_data.first_dof[patch * regular_vpatch_size],
                      local_patch,
                      col,
                      row,
                      z);

                  dst[global_dof_indices] =
                    shared_data.local_dst[index] * gpu_data.relaxation;
                }
          }
      }
  }


  template <int dim, int fe_degree, typename Number, LaplaceVariant lapalace>
  __global__ void
  loop_kernel_fused_cf(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d           = 2 * fe_degree + 2;
    constexpr unsigned int local_dim           = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int regular_vpatch_size = Util::pow(2, dim);
    constexpr unsigned int n_dofs_z            = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedMemData<dim, Number, false> shared_data(get_shared_data_ptr<Number>(),
                                                  patch_per_block,
                                                  n_dofs_1d,
                                                  local_dim);

    if (patch < gpu_data.n_patches)
      {
        shared_data.local_mass[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.smooth_mass_1d[threadIdx.y * n_dofs_1d + local_tid_x];
        shared_data.local_derivative[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.smooth_stiff_1d[threadIdx.y * n_dofs_1d + local_tid_x];

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index = local_patch * local_dim +
                                       z * n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x;

            const unsigned int global_dof_indices =
              Util::compute_indices<dim, fe_degree>(
                &gpu_data.first_dof[patch * regular_vpatch_size],
                local_patch,
                local_tid_x,
                threadIdx.y,
                z);

            shared_data.local_src[index] = src[global_dof_indices];

            shared_data.local_dst[index] = dst[global_dof_indices];
          }

        evaluate_smooth_cf<dim,
                           fe_degree,
                           Number,
                           lapalace,
                           SmootherVariant::ConflictFree>(local_patch,
                                                          &shared_data,
                                                          &gpu_data);

        if (dim == 2)
          {
            unsigned int linear_tid = local_tid_x + threadIdx.y * n_dofs_1d;

            if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
              {
                int row = linear_tid / (n_dofs_1d - 2) + 1;
                int col = linear_tid % (n_dofs_1d - 2) + 1;

                const unsigned int index = 2 * local_patch * local_dim +
                                           (row - 1) * (n_dofs_1d - 2) + col -
                                           1;

                const unsigned int global_dof_indices =
                  Util::compute_indices<dim, fe_degree>(
                    &gpu_data.first_dof[patch * regular_vpatch_size],
                    local_patch,
                    col,
                    row,
                    0);

                dst[global_dof_indices] =
                  shared_data.tmp[index] * gpu_data.relaxation;
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

                    const unsigned int global_dof_indices =
                      Util::compute_indices<dim, fe_degree>(
                        &gpu_data.first_dof[patch * regular_vpatch_size],
                        local_patch,
                        col,
                        row,
                        z + 1);

                    dst[global_dof_indices] =
                      shared_data.tmp[index] * gpu_data.relaxation;
                  }
              }
          }
      }
  }

  template <int dim, int fe_degree, typename Number, LaplaceVariant lapalace>
  __global__ void
  loop_kernel_fused_exact(
    const Number                                                 *src,
    Number                                                       *dst,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data gpu_data)
  {
    constexpr unsigned int n_dofs_1d           = 2 * fe_degree + 2;
    constexpr unsigned int local_dim           = Util::pow(n_dofs_1d, dim);
    constexpr unsigned int regular_vpatch_size = Util::pow(2, dim);
    constexpr unsigned int n_dofs_z            = dim == 2 ? 1 : n_dofs_1d;

    const unsigned int patch_per_block = gpu_data.patch_per_block;
    const unsigned int local_patch     = threadIdx.x / n_dofs_1d;
    const unsigned int patch       = local_patch + patch_per_block * blockIdx.x;
    const unsigned int local_tid_x = threadIdx.x % n_dofs_1d;

    SharedMemData<dim, Number, false> shared_data(get_shared_data_ptr<Number>(),
                                                  patch_per_block,
                                                  n_dofs_1d,
                                                  local_dim);

    if (patch < gpu_data.n_patches)
      {
        shared_data.local_mass[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.smooth_mass_1d[threadIdx.y * n_dofs_1d + local_tid_x];
        shared_data.local_derivative[threadIdx.y * n_dofs_1d + local_tid_x] =
          gpu_data.smooth_stiff_1d[threadIdx.y * n_dofs_1d + local_tid_x];

        for (unsigned int z = 0; z < n_dofs_z; ++z)
          {
            const unsigned int index = local_patch * local_dim +
                                       z * n_dofs_1d * n_dofs_1d +
                                       threadIdx.y * n_dofs_1d + local_tid_x;

            const unsigned int global_dof_indices =
              Util::compute_indices<dim, fe_degree>(
                &gpu_data.first_dof[patch * regular_vpatch_size],
                local_patch,
                local_tid_x,
                threadIdx.y,
                z);

            shared_data.local_src[index] = src[global_dof_indices];

            shared_data.local_dst[index] = dst[global_dof_indices];
          }

        evaluate_smooth_exact<dim,
                              fe_degree,
                              Number,
                              lapalace,
                              SmootherVariant::ExactRes>(local_patch,
                                                         &shared_data,
                                                         &gpu_data);

        const unsigned int linear_tid = local_tid_x + threadIdx.y * n_dofs_1d;
        if (dim == 2)
          {
            if (linear_tid < (n_dofs_1d - 4) * (n_dofs_1d - 4))
              {
                int row = linear_tid / (n_dofs_1d - 4) + 2;
                int col = linear_tid % (n_dofs_1d - 4) + 2;

                const unsigned int index = 2 * local_patch * local_dim +
                                           (row - 2) * (n_dofs_1d - 4) + col -
                                           2;

                const unsigned int global_dof_indices =
                  Util::compute_indices<dim, fe_degree>(
                    &gpu_data.first_dof[patch * regular_vpatch_size],
                    local_patch,
                    col,
                    row,
                    0);

                dst[global_dof_indices] =
                  shared_data.tmp[index] * gpu_data.relaxation;
              }
          }
        else if (dim == 3)
          {
            if (linear_tid < (n_dofs_1d - 4) * (n_dofs_1d - 4))
              for (unsigned int z = 0; z < n_dofs_1d - 4; ++z)
                {
                  int row = linear_tid / (n_dofs_1d - 4) + 2;
                  int col = linear_tid % (n_dofs_1d - 4) + 2;

                  unsigned int index = 2 * local_patch * local_dim +
                                       z * n_dofs_1d * n_dofs_1d +
                                       (row - 2) * (n_dofs_1d - 4) + col - 2;

                  const unsigned int global_dof_indices =
                    Util::compute_indices<dim, fe_degree>(
                      &gpu_data.first_dof[patch * regular_vpatch_size],
                      local_patch,
                      col,
                      row,
                      z + 2);

                  dst[global_dof_indices] =
                    shared_data.tmp[index] * gpu_data.relaxation;
                }
          }
      }
  }

} // namespace PSMF

#endif // LOOP_KERNEL_CUH
