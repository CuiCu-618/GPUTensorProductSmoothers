/**
 * @file evaluate_kernel.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief collection of device functions
 * @version 1.0
 * @date 2022-12-26
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef CUDA_EVALUATE_CUH
#define CUDA_EVALUATE_CUH

#include <mma.h>

#include "patch_base.cuh"

using namespace nvcuda;


namespace PSMF
{

  template <int dim_m, int dim_n>
  struct Shape
  {
    static constexpr int m = dim_m;
    static constexpr int n = dim_n;
  };

  ////////////////////////////////////////////////////////////////////
  /////////////////////// TPEvaluatorBase ////////////////////////////
  ////////////////////////////////////////////////////////////////////
  /**
   * A base struct for the various TensorProduct Evaluator template
   * specializations, containing common functionalities.
   *
   * @tparam T Type of the actual vectorized array. We are using the
   *   Couriously Recurring Template Pattern (see
   *   https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern) in
   *   this struct to avoid having to resort to `virtual` member functions.
   */
  template <typename T,
            int n_dofs_1d,
            typename Number,
            LaplaceVariant laplace_type,
            int            dim,
            typename Number2 = Number>
  struct TPEvaluatorBase
  {
    __device__
    TPEvaluatorBase() = default;

    __device__ void
    vmult(Number        *dst,
          const Number  *src,
          const Number2 *mass_matrix,
          const Number2 *derivative_matrix,
          Number        *tmp)
    {}

    template <typename shapeA,
              typename shapeB,
              bool add,
              int  dir,
              int  g_row,
              int  g_col,
              int  cycle>
    __device__ void
    mma_op(const Number *shape_data, const Number *in, Number *out)
    {}

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number2 *shape_data, const Number *in, Number *out)
    {}
  };

  template <typename T, int n_dofs_1d, typename Number>
  struct TPEvaluatorBase<T, n_dofs_1d, Number, LaplaceVariant::Basic, 2>
  {
    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int row = threadIdx.y;
      const int col = threadIdx.x % n_dofs_1d;

      Number pval = 0;
      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      // #pragma unroll
      for (int k = 0; k < n_dofs_1d; ++k)
        {
          const int shape_idx = row * n_dofs_1d + k;

          const int source_idx =
            (direction == 0) ? (col * n_dofs_1d + k) : (k * n_dofs_1d + col);

          pval += shape_data[shape_idx] * in[source_idx];
        }


      const int destination_idx =
        (direction == 0) ? (col * n_dofs_1d + row) : (row * n_dofs_1d + col);

      if (add)
        out[destination_idx] += pval;
      else if (sub)
        out[destination_idx] -= pval;
      else
        out[destination_idx] = pval;
    }
  };

  template <typename T, int n_dofs_1d, typename Number>
  struct TPEvaluatorBase<T, n_dofs_1d, Number, LaplaceVariant::Basic, 3>
  {
    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr int stride = n_dofs_1d * n_dofs_1d;

      const int row = threadIdx.y;
      const int col = threadIdx.x % n_dofs_1d;

      Number pval[n_dofs_1d];
      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      for (int z = 0; z < n_dofs_1d; ++z)
        {
          pval[z] = 0;
          // #pragma unroll
          for (int k = 0; k < n_dofs_1d; ++k)
            {
              const int shape_idx = row * n_dofs_1d + k;

              const int source_idx =
                (direction == 0) ? (col * n_dofs_1d + k + z * stride) :
                (direction == 1) ? (k * n_dofs_1d + col + z * stride) :
                                   (z * n_dofs_1d + col + k * stride);

              pval[z] += shape_data[shape_idx] * in[source_idx];
            }
        }

      for (int z = 0; z < n_dofs_1d; ++z)
        {
          const int destination_idx =
            (direction == 0) ? (col * n_dofs_1d + row + z * stride) :
            (direction == 1) ? (row * n_dofs_1d + col + z * stride) :
                               (z * n_dofs_1d + col + row * stride);

          if (add)
            out[destination_idx] += pval[z];
          else if (sub)
            out[destination_idx] -= pval[z];
          else
            out[destination_idx] = pval[z];
        }
    }
  };



  template <typename T, int n_dofs_1d, typename Number>
  struct TPEvaluatorBase<T, n_dofs_1d, Number, LaplaceVariant::ConflictFree, 2>
  {
    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr int multiple = std::is_same<Number, double>::value ?
                                 Util::calculate_multiple<n_dofs_1d, 16>() :
                                 Util::calculate_multiple<n_dofs_1d, 32>();

      const int row = threadIdx.y;
      const int col = threadIdx.x % n_dofs_1d;

      Number pval = 0;
      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      // #pragma unroll
      for (int k = 0; k < n_dofs_1d; ++k)
        {
          const int shape_idx =
            (direction == 0) ?
              (col * n_dofs_1d + (k + col / multiple) % n_dofs_1d) :
              (row * n_dofs_1d + k);

          const int source_idx =
            (direction == 0) ?
              (row * n_dofs_1d + (k + col / multiple) % n_dofs_1d) :
              (k * n_dofs_1d + col);

          pval += shape_data[shape_idx] * in[source_idx];
        }

      const int destination_idx = row * n_dofs_1d + col;

      if (add)
        out[destination_idx] += pval;
      else if (sub)
        out[destination_idx] -= pval;
      else
        out[destination_idx] = pval;
    }
  };

  template <typename T, int n_dofs_1d, typename Number>
  struct TPEvaluatorBase<T, n_dofs_1d, Number, LaplaceVariant::ConflictFree, 3>
  {
    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr int multiple = std::is_same<Number, double>::value ?
                                 Util::calculate_multiple<n_dofs_1d, 16>() :
                                 Util::calculate_multiple<n_dofs_1d, 32>();

      constexpr int stride = n_dofs_1d * n_dofs_1d;

      const int row = threadIdx.y;
      const int col = threadIdx.x % n_dofs_1d;

      Number pval[n_dofs_1d];
      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      for (int z = 0; z < n_dofs_1d; ++z)
        {
          pval[z] = 0;
          // #pragma unroll
          for (int k = 0; k < n_dofs_1d; ++k)
            {
              const int shape_idx =
                (direction == 0) ?
                  col * n_dofs_1d + (k + col / multiple) % n_dofs_1d :
                (direction == 1) ? row * n_dofs_1d + k :
                                   z * n_dofs_1d + k;

              const int source_idx =
                (direction == 0) ?
                  (row * n_dofs_1d + (k + col / multiple) % n_dofs_1d +
                   z * stride) :
                (direction == 1) ? (k * n_dofs_1d + col + z * stride) :
                                   (row * n_dofs_1d + col + k * stride);

#ifdef USETEXTURE
              pval[z] += tex1Dfetch(stiff_data_d, shape_idx) * in[source_idx];
#else
              pval[z] += shape_data[shape_idx] * in[source_idx];
#endif
            }
        }

      for (int z = 0; z < n_dofs_1d; ++z)
        {
          const int destination_idx = row * n_dofs_1d + col + z * stride;

          if (add)
            out[destination_idx] += pval[z];
          else if (sub)
            out[destination_idx] -= pval[z];
          else
            out[destination_idx] = pval[z];
        }
    }
  };

  template <typename T, int n_dofs_1d, typename Number>
  struct TPEvaluatorBase<T,
                         n_dofs_1d,
                         Number,
                         LaplaceVariant::ConflictFreeMem,
                         2>
  {
    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr int multiple = std::is_same<Number, double>::value ?
                                 Util::calculate_multiple<n_dofs_1d, 16>() :
                                 Util::calculate_multiple<n_dofs_1d, 32>();

      const int row = threadIdx.y;
      const int col = threadIdx.x & (n_dofs_1d - 1);

      Number pval = 0;
      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      // #pragma unroll
      for (int k = 0; k < n_dofs_1d; ++k)
        {
          const int shape_idx = row * n_dofs_1d + k;

          const int source_idx =
            (direction == 0) ?
              col * n_dofs_1d + ((k + col / multiple) & (n_dofs_1d - 1)) :
              k * n_dofs_1d + ((col + k / multiple) & (n_dofs_1d - 1));

          pval += shape_data[shape_idx] * in[source_idx];
        }

      const int destination_idx =
        (direction == 0) ?
          col * n_dofs_1d + ((row + col / multiple) & (n_dofs_1d - 1)) :
          row * n_dofs_1d + ((col + row / multiple) & (n_dofs_1d - 1));

      if (add)
        out[destination_idx] += pval;
      else if (sub)
        out[destination_idx] -= pval;
      else
        out[destination_idx] = pval;
    }
  };

  template <typename T, int n_dofs_1d, typename Number>
  struct TPEvaluatorBase<T,
                         n_dofs_1d,
                         Number,
                         LaplaceVariant::ConflictFreeMem,
                         3>
  {
    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr int multiple = std::is_same<Number, double>::value ?
                                 Util::calculate_multiple<n_dofs_1d, 16>() :
                                 Util::calculate_multiple<n_dofs_1d, 32>();

      constexpr int stride = n_dofs_1d * n_dofs_1d;

      const int row = threadIdx.y;
      const int col = threadIdx.x & (n_dofs_1d - 1);

      Number pval[n_dofs_1d];
      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      for (int z = 0; z < n_dofs_1d; ++z)
        {
          pval[z] = 0;
          // #pragma unroll
          for (int k = 0; k < n_dofs_1d; ++k)
            {
              const int shape_idx = row * n_dofs_1d + k;

              const int source_idx =
                (direction == 0) ?
                  (((col + z) & (n_dofs_1d - 1)) * n_dofs_1d +
                   ((k + ((col + z) & (n_dofs_1d - 1)) / multiple) &
                    (n_dofs_1d - 1)) +
                   z * stride) :
                (direction == 1) ?
                  (((k + z) & (n_dofs_1d - 1)) * n_dofs_1d +
                   ((col + ((k + z) & (n_dofs_1d - 1)) / multiple) &
                    (n_dofs_1d - 1)) +
                   z * stride) :
                  (((z + k) & (n_dofs_1d - 1)) * n_dofs_1d +
                   ((col + ((k + z) & (n_dofs_1d - 1)) / multiple) &
                    (n_dofs_1d - 1)) +
                   k * stride);

              pval[z] += shape_data[shape_idx] * in[source_idx];
            }
        }

      for (int z = 0; z < n_dofs_1d; ++z)
        {
          const int destination_idx =
            (direction == 0) ?
              ((col + z) & (n_dofs_1d - 1)) * n_dofs_1d +
                ((row + ((col + z) & (n_dofs_1d - 1)) / multiple) &
                 (n_dofs_1d - 1)) +
                z * stride :
            (direction == 1) ?
              ((row + z) & (n_dofs_1d - 1)) * n_dofs_1d +
                ((col + ((row + z) & (n_dofs_1d - 1)) / multiple) &
                 (n_dofs_1d - 1)) +
                z * stride :
              ((z + row) & (n_dofs_1d - 1)) * n_dofs_1d +
                ((col + ((z + row) & (n_dofs_1d - 1)) / multiple) &
                 (n_dofs_1d - 1)) +
                row * stride;

          if (add)
            out[destination_idx] += pval[z];
          else if (sub)
            out[destination_idx] -= pval[z];
          else
            out[destination_idx] = pval[z];
        }
    }
  };

  // template <typename T, int n_dofs_1d, typename Number>
  // struct TPEvaluatorBase<T, n_dofs_1d, Number, LaplaceVariant::ConflictFree,
  // 3>
  // {
  //   /**
  //    * Default constructor.
  //    */
  //   __device__
  //   TPEvaluatorBase() = default;

  //   /**
  //    * Implements a matrix-vector product for Laplacian.
  //    */
  //   __device__ void
  //   vmult(Number       *dst,
  //         const Number *src,
  //         const Number *mass_matrix,
  //         const Number *derivative_matrix,
  //         Number       *tmp)
  //   {
  //     static_cast<T *>(this)->vmult_impl(
  //       dst, src, mass_matrix, derivative_matrix, tmp);
  //   }

  //   template <int direction, bool add, bool sub = false>
  //   __device__ void
  //   apply(const Number *shape_data, const Number *in, Number *out)
  //   {
  //     // constexpr int multiple = std::is_same<Number, double>::value ?
  //     //                            Util::calculate_multiple<n_dofs_1d, 16>()
  //     :
  //     //                            Util::calculate_multiple<n_dofs_1d,
  //     32>();

  //     constexpr int stride = n_dofs_1d * n_dofs_1d;

  //     const int row = threadIdx.y;
  //     const int col = threadIdx.x % n_dofs_1d;

  //     Number rc_sh[n_dofs_1d];
  //     // rc_sh[0] = shape_data[row * n_dofs_1d + col];
  //     // rc_sh[1] = shape_data[((row + 4) & 7) * n_dofs_1d + col];

  //     auto src_ind = direction == 0 ? col * n_dofs_1d : row * n_dofs_1d;
  //     for (int i = 0; i < n_dofs_1d; ++i)
  //       rc_sh[i] = shape_data[src_ind + i];

  //     Number pval[n_dofs_1d];
  //     // kernel product: A kdot src, [N x N] * [N^dim, 1]
  //     for (int z = 0; z < n_dofs_1d; ++z)
  //       {
  //         pval[z] = 0;
  //         // #pragma unroll
  //         for (int k = 0; k < n_dofs_1d; ++k)
  //           {
  //             const int shape_idx =
  //               (direction == 0) ? col * n_dofs_1d + k :
  //               (direction == 1) ? row * n_dofs_1d + k :
  //                                  z * n_dofs_1d + k;

  //             const int source_idx =
  //               (direction == 0) ? (row * n_dofs_1d + k + z * stride) :
  //               (direction == 1) ? (k * n_dofs_1d + col + z * stride) :
  //                                  (row * n_dofs_1d + col + k * stride);

  //             if (direction == 3)
  //               {
  //                 int src_lane_sh = (col * n_dofs_1d + k) % 32;
  //                 int bk          = (col / 4 + row / 4) % 2;

  //                 pval[z] += __shfl_sync(0xffffffff, rc_sh[bk], src_lane_sh)
  //                 *
  //                            in[source_idx];
  //               }
  //             else if (direction == 1 || direction == 0)
  //               {
  //                 pval[z] += rc_sh[k] * in[source_idx];

  //                 // int src_lane_sh = (row * n_dofs_1d + k) & 31;

  //                 // pval[z] += __shfl_sync(0xffffffff, rc_sh[0],
  //                 src_lane_sh) *
  //                 //            in[source_idx];
  //               }
  //             else if (direction == 3)
  //               {
  //                 int src_lane_sh = (z * n_dofs_1d + k) % 32;
  //                 int bk          = (z / 4 + row / 4) % 2;

  //                 pval[z] += __shfl_sync(0xffffffff, rc_sh[bk], src_lane_sh)
  //                 *
  //                            in[source_idx];
  //               }
  //             else
  //               {
  //                 pval[z] += shape_data[shape_idx] * in[source_idx];
  //               }

  //             // Number rc_in[2];
  //             // rc_in[0] = in[row * n_dofs_1d + col + z * stride];
  //             // rc_in[1] =
  //             //   in[((row + 4) % n_dofs_1d) * n_dofs_1d + col + z *
  //             stride];

  //             // if (direction == 1)
  //             //   {
  //             //     int src_lane_sh = (row * n_dofs_1d + k) % 32;
  //             //     int src_lane_in = (k * n_dofs_1d + col) % 32;

  //             //     int bk = (row / 4 + k / 4) % 2;

  //             //     pval[z] += __shfl_sync(0xffffffff, rc_sh, src_lane_sh) *
  //             //                __shfl_sync(0xffffffff, rc_in[bk],
  //             src_lane_in);
  //             //   }
  //             // else
  //             //   pval[z] += shape_data[shape_idx] * in[source_idx];
  //           }
  //       }

  //     for (int z = 0; z < n_dofs_1d; ++z)
  //       {
  //         const int destination_idx =
  //           row * n_dofs_1d + col + z * stride;

  //         if (add)
  //           out[destination_idx] += pval[z];
  //         else if (sub)
  //           out[destination_idx] -= pval[z];
  //         else
  //           out[destination_idx] = pval[z];
  //       }
  //   }
  // };


  template <typename T>
  struct TPEvaluatorBase<T, 8, double, LaplaceVariant::TensorCoreMMA, 2>
  {
    using Number = double;

    static constexpr int n_dofs_1d = 8;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int tid = (threadIdx.y * 8 + threadIdx.x);

      const int row = tid / 4;
      const int col = tid & 3;

      if (tid > 31)
        return;

      if constexpr (direction == 0)
        {
          double2 c = {0, 0};

          const int cd_idx0 =
            (row * n_dofs_1d + 2 * col) ^ Util::get_base<n_dofs_1d>(row);

          if constexpr (add)
            c = *((double2 *)(out + cd_idx0));

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx = (row * n_dofs_1d + col + cycle * 4) ^
                                Util::get_base<n_dofs_1d>(row);
              const int b_idx = ((col + cycle * 4) * n_dofs_1d + row) ^
                                Util::get_base<n_dofs_1d>(col + cycle * 4);

              auto a0 = (direction == 0) ? in[a_idx] : shape_data[a_idx];
              auto b0 = (direction == 0) ? shape_data[b_idx] : in[b_idx];

              asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                           "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                           : "=d"(c.x), "=d"(c.y)
                           : "d"(a0), "d"(b0), "d"(c.x), "d"(c.y));
            }

          *((double2 *)(out + cd_idx0)) = c;
        }
      else
        {
          double2 c = {0, 0};

          const int cd_idx0 =
            (row * n_dofs_1d + 2 * col) ^ Util::get_base<n_dofs_1d>(row);

          if constexpr (add)
            c = *((double2 *)(out + cd_idx0));

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx = (row * n_dofs_1d + col + cycle * 4) ^
                                Util::get_base<n_dofs_1d>(row);
              const int b_idx = ((col + cycle * 4) * n_dofs_1d + row) ^
                                Util::get_base<n_dofs_1d>(col + cycle * 4);

              auto a0 = (direction == 0) ? in[a_idx] : shape_data[a_idx];
              auto b0 = (direction == 0) ? shape_data[b_idx] : in[b_idx];

              asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                           "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                           : "=d"(c.x), "=d"(c.y)
                           : "d"(a0), "d"(b0), "d"(c.x), "d"(c.y));
            }

          *((double2 *)(out + cd_idx0)) = c;
        }
    }
  };

  template <typename T>
  struct TPEvaluatorBase<T, 16, double, LaplaceVariant::TensorCoreMMA, 2>
  {
    using Number = double;

    static constexpr int n_dofs_1d = 16;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int warpId = (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;
      const int rowId  = warpId / 2;
      const int colId  = warpId & 1;

      const int tid = (threadIdx.y * n_dofs_1d + threadIdx.x) & 31;

      const int row = tid / 4;
      const int col = tid & 3;

      if (warpId > 3)
        return;

      if constexpr (direction == 0)
        {
          double2 c = {0, 0};

          const int cd_idx0 =
            ((rowId * 8 + row) * n_dofs_1d + 2 * col + colId * 8) ^
            Util::get_base<n_dofs_1d>(rowId * 8 + row);

          if constexpr (add)
            c = *((double2 *)(out + cd_idx0));

          for (int cycle = 0; cycle < 4; ++cycle)
            {
              const int a_idx =
                ((rowId * 8 + row) * n_dofs_1d + col + cycle * 4) ^
                Util::get_base<n_dofs_1d>(rowId * 8 + row);
              const int b_idx =
                ((col + cycle * 4) * n_dofs_1d + row + colId * 8) ^
                Util::get_base<n_dofs_1d>(col + cycle * 4);

              auto a0 = (direction == 0) ? in[a_idx] : shape_data[a_idx];
              auto b0 = (direction == 0) ? shape_data[b_idx] : in[b_idx];

              asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                           "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                           : "=d"(c.x), "=d"(c.y)
                           : "d"(a0), "d"(b0), "d"(c.x), "d"(c.y));
            }

          *((double2 *)(out + cd_idx0)) = c;
        }
      else
        {
          double2 c = {0, 0};

          const int cd_idx0 =
            ((rowId * 8 + row) * n_dofs_1d + 2 * col + colId * 8) ^
            Util::get_base<n_dofs_1d>(rowId * 8 + row);

          if constexpr (add)
            c = *((double2 *)(out + cd_idx0));

          for (int cycle = 0; cycle < 4; ++cycle)
            {
              const int a_idx =
                ((rowId * 8 + row) * n_dofs_1d + col + cycle * 4) ^
                Util::get_base<n_dofs_1d>(rowId * 8 + row);
              const int b_idx =
                ((col + cycle * 4) * n_dofs_1d + row + colId * 8) ^
                Util::get_base<n_dofs_1d>(col + cycle * 4);

              auto a0 = (direction == 0) ? in[a_idx] : shape_data[a_idx];
              auto b0 = (direction == 0) ? shape_data[b_idx] : in[b_idx];

              asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                           "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                           : "=d"(c.x), "=d"(c.y)
                           : "d"(a0), "d"(b0), "d"(c.x), "d"(c.y));
            }

          *((double2 *)(out + cd_idx0)) = c;
        }
    }
  };

  template <typename T>
  struct TPEvaluatorBase<T, 16, float, LaplaceVariant::TensorCoreMMA, 2>
  {
    using Number = float;

    static constexpr int n_dofs_1d = 16;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int warpId = (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;

      const int tid   = (threadIdx.y * n_dofs_1d + threadIdx.x) & 31;
      const int rowId = tid / 8;
      const int colId = tid & 7;

      const int row = tid / 4;
      const int col = tid & 3;

      if (warpId > 1)
        return;

      if constexpr (direction == 0)
        {
          float2 c0 = {0, 0};
          float2 c1 = {0, 0};

          float a[4];
          float b[2];

          const int c_idx0 = (row * n_dofs_1d + 2 * col + warpId * 8) ^
                             Util::get_base<n_dofs_1d, float>(row);
          const int c_idx1 = ((row + 8) * n_dofs_1d + 2 * col + warpId * 8) ^
                             Util::get_base<n_dofs_1d, float>(row + 8);

          if constexpr (add)
            {
              c0 = *((float2 *)(out + c_idx0));
              c1 = *((float2 *)(out + c_idx1));
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx =
                ((colId + (rowId & 1) * 8) * n_dofs_1d + (rowId / 2) * 4 +
                 cycle * 8) ^
                Util::get_base<n_dofs_1d, float>(colId + (rowId & 1) * 8);

              const int b_idx0 =
                ((col + cycle * 8) * n_dofs_1d + row + warpId * 8) ^
                Util::get_base<n_dofs_1d, float>(col + cycle * 8);
              const int b_idx1 =
                ((col + cycle * 8 + 4) * n_dofs_1d + row + warpId * 8) ^
                Util::get_base<n_dofs_1d, float>(col + cycle * 8 + 4);

              auto smem_ptr =
                static_cast<uint32_t>(__cvta_generic_to_shared(&in[a_idx]));

              asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                           "{%0, %1, %2, %3}, [%4]; "
                           : "=f"(a[0]), "=f"(a[1]), "=f"(a[2]), "=f"(a[3])
                           : "r"(smem_ptr));

              b[0] = shape_data[b_idx0];
              b[1] = shape_data[b_idx1];

              uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
              uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

              asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(c0.x), "=f"(c0.y), "=f"(c1.x), "=f"(c1.y)
                : "r"(A[0]),
                  "r"(A[1]),
                  "r"(A[2]),
                  "r"(A[3]),
                  "r"(B[0]),
                  "r"(B[1]),
                  "f"(c0.x),
                  "f"(c0.y),
                  "f"(c1.x),
                  "f"(c1.y));
            }

          *((float2 *)(out + c_idx0)) = c0;
          *((float2 *)(out + c_idx1)) = c1;
        }
      else
        {
          float2 c0 = {0, 0};
          float2 c1 = {0, 0};

          float a[4];
          float b[2];

          const int c_idx0 = (row * n_dofs_1d + 2 * col + warpId * 8) ^
                             Util::get_base<n_dofs_1d, float>(row);
          const int c_idx1 = ((row + 8) * n_dofs_1d + 2 * col + warpId * 8) ^
                             Util::get_base<n_dofs_1d, float>(row + 8);

          if constexpr (add)
            {
              c0 = *((float2 *)(out + c_idx0));
              c1 = *((float2 *)(out + c_idx1));
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx =
                ((colId + (rowId & 1) * 8) * n_dofs_1d + (rowId / 2) * 4 +
                 cycle * 8) ^
                Util::get_base<n_dofs_1d, float>(colId + (rowId & 1) * 8);

              const int b_idx0 =
                ((col + cycle * 8) * n_dofs_1d + row + warpId * 8) ^
                Util::get_base<n_dofs_1d, float>(col + cycle * 8);
              const int b_idx1 =
                ((col + cycle * 8 + 4) * n_dofs_1d + row + warpId * 8) ^
                Util::get_base<n_dofs_1d, float>(col + cycle * 8 + 4);
              auto smem_ptr = static_cast<uint32_t>(
                __cvta_generic_to_shared(&shape_data[a_idx]));

              asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                           "{%0, %1, %2, %3}, [%4]; "
                           : "=f"(a[0]), "=f"(a[1]), "=f"(a[2]), "=f"(a[3])
                           : "r"(smem_ptr));

              b[0] = in[b_idx0];
              b[1] = in[b_idx1];

              uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
              uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

              asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(c0.x), "=f"(c0.y), "=f"(c1.x), "=f"(c1.y)
                : "r"(A[0]),
                  "r"(A[1]),
                  "r"(A[2]),
                  "r"(A[3]),
                  "r"(B[0]),
                  "r"(B[1]),
                  "f"(c0.x),
                  "f"(c0.y),
                  "f"(c1.x),
                  "f"(c1.y));
            }

          *((float2 *)(out + c_idx0)) = c0;
          *((float2 *)(out + c_idx1)) = c1;
        }
    }
  };

#if MMAKERNEL == 1
  template <typename T>
  struct TPEvaluatorBase<T, 6, double, LaplaceVariant::TensorCoreMMA, 3>
  {
    using Number = double;

    static constexpr int n_dofs_1d = 6;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int tid    = (threadIdx.y * 8 + threadIdx.x) & 31;
      const int warpId = (threadIdx.y * 8 + threadIdx.x) / 32;

      const int row = tid / 4;
      const int col = tid & 3;

      constexpr int offset = n_dofs_1d * n_dofs_1d;

      if (direction == 0)
        {
          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                (row * n_dofs_1d + 2 * col + (z * 2 + warpId) * offset) ^
                Util::get_base<n_dofs_1d>(row, z * 2 + warpId);

              if (add && row < 6 && col < 3)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
#  if TIMING == 1
              auto start = clock64();
#  endif
              const int b_idx = ((col + cycle * 4) * n_dofs_1d + row) ^
                                Util::get_base<n_dofs_1d>(col + cycle * 4);
              auto b0 = cycle == 0 ?
                          (row < 6 ? shape_data[b_idx] : 0) :
                          ((row < 6 && col < 2) ? shape_data[b_idx] : 0);
#  if TIMING == 1
              auto elapsed = clock64() - start;
              if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
                printf(
                  "mma load frag b dir-%d loop-%d timing info: %ld cycles\n",
                  direction,
                  cycle,
                  elapsed);
#  endif
#  pragma unroll
              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
#  if TIMING == 1
                  start = clock64();
#  endif
                  const int a_idx =
                    (row * n_dofs_1d + col + cycle * 4 +
                     (z * 2 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d>(row, z * 2 + warpId);

                  auto a0 = cycle == 0 ? (row < 6 ? in[a_idx] : 0) :
                                         ((row < 6 && col < 2) ? in[a_idx] : 0);
#  if TIMING == 1
                  elapsed = clock64() - start;
                  if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
                    printf(
                      "mma load frag a dir-%d loop-(%d,%d) timing info: %ld cycles\n",
                      direction,
                      cycle,
                      z,
                      elapsed);
#  endif

#  if TIMING == 1
                  start = clock64();
#  endif
                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
#  if TIMING == 1
                  elapsed = clock64() - start;
                  if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
                    printf(
                      "mma.sync.aligned.m8n8k4 dir-%d loop-(%d,%d) timing info: %ld cycles\n",
                      direction,
                      cycle,
                      z,
                      elapsed);
#  endif
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
#  if TIMING == 1
              auto start = clock64();
#  endif
              const int c_idx =
                (row * n_dofs_1d + 2 * col + (z * 2 + warpId) * offset) ^
                Util::get_base<n_dofs_1d>(row, z * 2 + warpId);

              if (row < 6 && col < 3)
                *((double2 *)(out + c_idx)) = c[z];
#  if TIMING == 1
              auto elapsed = clock64() - start;
              if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
                printf(
                  "mma store frag c dir-%d loop-%d timing info: %ld cycles\n",
                  direction,
                  z,
                  elapsed);
#  endif
            }
        }
      else if (direction == 1)
        {
          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                (row * n_dofs_1d + 2 * col + (z * 2 + warpId) * offset) ^
                Util::get_base<n_dofs_1d>(row, z * 2 + warpId);

              if (add && row < 6 && col < 3)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx = (row * n_dofs_1d + col + cycle * 4) ^
                                Util::get_base<n_dofs_1d>(row, 0);
              auto a0 = cycle == 0 ?
                          (row < 6 ? shape_data[a_idx] : 0) :
                          ((row < 6 && col < 2) ? shape_data[a_idx] : 0);

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int b_idx =
                    ((col + cycle * 4) * n_dofs_1d + row +
                     (z * 2 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d>(col + cycle * 4, z * 2 + warpId);

                  auto b0 = cycle == 0 ? (row < 6 ? in[b_idx] : 0) :
                                         ((row < 6 && col < 2) ? in[b_idx] : 0);

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                (row * n_dofs_1d + 2 * col + (z * 2 + warpId) * offset) ^
                Util::get_base<n_dofs_1d>(row, z * 2 + warpId);

              if (row < 6 && col < 3)
                *((double2 *)(out + c_idx)) = c[z];
            }
        }
      else
        {
          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((z * 2 + warpId) * n_dofs_1d + 2 * col + row * offset) ^
                Util::get_base<n_dofs_1d>(z * 2 + warpId, row);

              if (add && row < 6 && col < 3)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx = (row * n_dofs_1d + col + cycle * 4) ^
                                Util::get_base<n_dofs_1d>(row, 0);
              auto a0 = cycle == 0 ?
                          (row < 6 ? shape_data[a_idx] : 0) :
                          ((row < 6 && col < 2) ? shape_data[a_idx] : 0);

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int b_idx =
                    ((z * 2 + warpId) * n_dofs_1d + row +
                     (col + cycle * 4) * offset) ^
                    Util::get_base<n_dofs_1d>(z * 2 + warpId, col + cycle * 4);

                  auto b0 = cycle == 0 ? (row < 6 ? in[b_idx] : 0) :
                                         ((row < 6 && col < 2) ? in[b_idx] : 0);

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((z * 2 + warpId) * n_dofs_1d + 2 * col + row * offset) ^
                Util::get_base<n_dofs_1d>(z * 2 + warpId, row);

              if (row < 6 && col < 3)
                *((double2 *)(out + c_idx)) = c[z];
            }
        }
    }
  };
#endif

#if MMAKERNEL == 0
  template <typename T>
  struct TPEvaluatorBase<T, 8, double, LaplaceVariant::TensorCoreMMA, 3>
  {
    using Number = double;

    static constexpr int n_dofs_1d = 8;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int tid    = (threadIdx.y * 8 + threadIdx.x) & 31;
      const int warpId = (threadIdx.y * 8 + threadIdx.x) / 32;

      const int row = tid / 4;
      const int col = tid & 3;

      constexpr int offset = n_dofs_1d * n_dofs_1d;

      if (direction == 0)
        {
          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                (row * n_dofs_1d + 2 * col + (z * 2 + warpId) * offset);

              if constexpr (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
#  if TIMING == 1
              auto start = clock64();
#  endif
              const int b_idx = ((col + cycle * 4) * n_dofs_1d + row);
              auto      b0    = shape_data[b_idx];
#  if TIMING == 1
              auto elapsed = clock64() - start;
              if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
                printf(
                  "mma load frag b dir-%d loop-%d timing info: %ld cycles\n",
                  direction,
                  cycle,
                  elapsed);
#  endif
              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
#  if TIMING == 1
                  start = clock64();
#  endif
                  const int a_idx = (row * n_dofs_1d + col + cycle * 4 +
                                     (z * 2 + warpId) * offset);
                  auto      a0    = in[a_idx];
#  if TIMING == 1
                  elapsed = clock64() - start;
                  if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
                    printf(
                      "mma load frag a dir-%d loop-(%d,%d) timing info: %ld cycles\n",
                      direction,
                      cycle,
                      z,
                      elapsed);
#  endif

#  if TIMING == 1
                  start = clock64();
#  endif
                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
#  if TIMING == 1
                  elapsed = clock64() - start;
                  if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
                    printf(
                      "mma.sync.aligned.m8n8k4 dir-%d loop-(%d,%d) timing info: %ld cycles\n",
                      direction,
                      cycle,
                      z,
                      elapsed);
#  endif
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
#  if TIMING == 1
              auto start = clock64();
#  endif
              const int c_idx =
                (row * n_dofs_1d + 2 * col + (z * 2 + warpId) * offset);
              *((double2 *)(out + c_idx)) = c[z];
#  if TIMING == 1
              auto elapsed = clock64() - start;
              if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
                printf(
                  "mma store frag c dir-%d loop-%d timing info: %ld cycles\n",
                  direction,
                  z,
                  elapsed);
#  endif
            }
        }
      else if (direction == 1)
        {
          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                (row * n_dofs_1d + 2 * col + (z * 2 + warpId) * offset);

              if constexpr (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx = (row * n_dofs_1d + col + cycle * 4);
              auto      a0    = shape_data[a_idx];

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int b_idx = ((col + cycle * 4) * n_dofs_1d + row +
                                     (z * 2 + warpId) * offset);

                  auto b0 = in[b_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                (row * n_dofs_1d + 2 * col + (z * 2 + warpId) * offset);

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
      else
        {
          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((z * 2 + warpId) * n_dofs_1d + 2 * col + row * offset);

              if constexpr (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx = (row * n_dofs_1d + col + cycle * 4);
              auto      a0    = shape_data[a_idx];

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int b_idx = ((z * 2 + warpId) * n_dofs_1d + row +
                                     (col + cycle * 4) * offset);

                  auto b0 = in[b_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((z * 2 + warpId) * n_dofs_1d + 2 * col + row * offset);

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
    }
  };
#endif

#if MMAKERNEL == 1
  template <typename T>
  struct TPEvaluatorBase<T, 8, double, LaplaceVariant::TensorCoreMMA, 3>
  {
    using Number = double;

    static constexpr int n_dofs_1d = 8;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int tid    = (threadIdx.y * 8 + threadIdx.x) & 31;
      const int warpId = threadIdx.y / 4;

      const int row = tid / 4;
      const int col = tid & 3;

      constexpr int offset = n_dofs_1d * n_dofs_1d;

      if (direction == 0)
        {
          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
#  if TIMING == 1
              auto start = clock64();
#  endif
              const int b_idx = ((col + cycle * 4) * n_dofs_1d + row) ^
                                Util::get_base<n_dofs_1d>(col + cycle * 4);
              auto b0 = shape_data[b_idx];
#  if TIMING == 1
              auto elapsed = clock64() - start;
              if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
                printf(
                  "mma load frag b dir-%d loop-%d timing info: %ld cycles\n",
                  direction,
                  cycle,
                  elapsed);
#  endif

#  ifdef SKIPZERO
              if (abs(shape_data[cycle * (n_dofs_1d * 4 + 4)]) < 1e-10)
                continue;
#  endif

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
#  if TIMING == 1
                  start = clock64();
#  endif
                  const int a_idx =
                    (row * n_dofs_1d + col + cycle * 4 +
                     (z * 2 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d>(row, z * 2 + warpId);

                  auto a0 = in[a_idx];
#  if TIMING == 1
                  elapsed = clock64() - start;
                  if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
                    printf(
                      "mma load frag a dir-%d loop-(%d,%d) timing info: %ld cycles\n",
                      direction,
                      cycle,
                      z,
                      elapsed);
#  endif

#  if TIMING == 1
                  start = clock64();
#  endif
                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
#  if TIMING == 1
                  elapsed = clock64() - start;
                  if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
                    printf(
                      "mma.sync.aligned.m8n8k4 dir-%d loop-(%d,%d) timing info: %ld cycles\n",
                      direction,
                      cycle,
                      z,
                      elapsed);
#  endif
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
#  if TIMING == 1
              auto start = clock64();
#  endif
              const int c_idx =
                (row * n_dofs_1d + 2 * col + (z * 2 + warpId) * offset) ^
                Util::get_base<n_dofs_1d>(row, z * 2 + warpId);

              *((double2 *)(out + c_idx)) = c[z];
#  if TIMING == 1
              auto elapsed = clock64() - start;
              if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
                printf(
                  "mma store frag c dir-%d loop-%d timing info: %ld cycles\n",
                  direction,
                  z,
                  elapsed);
#  endif
            }
        }
      else if (direction == 1)
        {
          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                (row * n_dofs_1d + 2 * col + (z * 2 + warpId) * offset) ^
                Util::get_base<n_dofs_1d>(row, z * 2 + warpId);

              if constexpr (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx = (row * n_dofs_1d + col + cycle * 4) ^
                                Util::get_base<n_dofs_1d>(row, 0);
              auto a0 = shape_data[a_idx];

#  ifdef SKIPZERO
              if (abs(shape_data[cycle * (n_dofs_1d * 4 + 4)]) < 1e-10)
                continue;
#  endif

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int b_idx =
                    ((col + cycle * 4) * n_dofs_1d + row +
                     (z * 2 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d>(col + cycle * 4, z * 2 + warpId);

                  auto b0 = in[b_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                (row * n_dofs_1d + 2 * col + (z * 2 + warpId) * offset) ^
                Util::get_base<n_dofs_1d>(row, z * 2 + warpId);

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
      else
        {
          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((z * 2 + warpId) * n_dofs_1d + 2 * col + row * offset) ^
                Util::get_base<n_dofs_1d>(z * 2 + warpId, row);

              if constexpr (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx = (row * n_dofs_1d + col + cycle * 4) ^
                                Util::get_base<n_dofs_1d>(row, 0);
              auto a0 = shape_data[a_idx];

#  ifdef SKIPZERO
              if (abs(shape_data[cycle * (n_dofs_1d * 4 + 4)]) < 1e-10)
                continue;
#  endif

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int b_idx =
                    ((z * 2 + warpId) * n_dofs_1d + row +
                     (col + cycle * 4) * offset) ^
                    Util::get_base<n_dofs_1d>(z * 2 + warpId, col + cycle * 4);

                  auto b0 = in[b_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((z * 2 + warpId) * n_dofs_1d + 2 * col + row * offset) ^
                Util::get_base<n_dofs_1d>(z * 2 + warpId, row);

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
    }
  };
#endif

#if MMAKERNEL == 2
  template <typename T>
  struct TPEvaluatorBase<T, 8, double, LaplaceVariant::TensorCoreMMA, 3>
  {
    using Number = double;

    static constexpr int n_dofs_1d = 8;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int tid    = (threadIdx.y * 8 + threadIdx.x) & 31;
      const int warpId = threadIdx.y / 4;


      const int row = tid / 4;
      const int col = tid & 3;

      constexpr int offset = n_dofs_1d * n_dofs_1d;

      if (direction == 0)
        {
          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int b_idx = ((col + cycle * 4) * n_dofs_1d + row) ^
                                Util::get_base<n_dofs_1d>(col + cycle * 4);
              auto b0 = shape_data[b_idx];

              for (int z = 0; z < n_dofs_1d / 4; ++z)
                {
                  const int a_idx =
                    (row * n_dofs_1d + col + cycle * 4 +
                     (z * 4 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d>(row, z * 4 + warpId);
                  const int a_idx2 =
                    (row * n_dofs_1d + col + cycle * 4 +
                     (z * 4 + warpId + 2) * offset) ^
                    Util::get_base<n_dofs_1d>(row, z * 4 + warpId + 2);

                  auto a0 = in[a_idx];
                  auto a2 = in[a_idx2];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z * 2].x), "=d"(c[z * 2].y)
                    : "d"(a0), "d"(b0), "d"(c[z * 2].x), "d"(c[z * 2].y));
                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z * 2 + 1].x), "=d"(c[z * 2 + 1].y)
                    : "d"(a2),
                      "d"(b0),
                      "d"(c[z * 2 + 1].x),
                      "d"(c[z * 2 + 1].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                (row * n_dofs_1d + 2 * col + (z * 2 + warpId) * offset) ^
                Util::get_base<n_dofs_1d>(row, z * 2 + warpId);

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
      else if (direction == 1)
        {
          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                (row * n_dofs_1d + 2 * col + (z * 2 + warpId) * offset) ^
                Util::get_base<n_dofs_1d>(row, z * 2 + warpId);

              if constexpr (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx = (row * n_dofs_1d + col + cycle * 4) ^
                                Util::get_base<n_dofs_1d>(row);
              auto a0 = shape_data[a_idx];

              for (int z = 0; z < n_dofs_1d / 4; ++z)
                {
                  const int b_idx =
                    ((col + cycle * 4) * n_dofs_1d + row +
                     (z * 4 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d>(col + cycle * 4, z * 4 + warpId);
                  const int b_idx2 =
                    ((col + cycle * 4) * n_dofs_1d + row +
                     (z * 4 + warpId + 2) * offset) ^
                    Util::get_base<n_dofs_1d>(col + cycle * 4,
                                              z * 4 + warpId + 2);

                  auto b0 = in[b_idx];
                  auto b2 = in[b_idx2];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z * 2].x), "=d"(c[z * 2].y)
                    : "d"(a0), "d"(b0), "d"(c[z * 2].x), "d"(c[z * 2].y));
                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z * 2 + 1].x), "=d"(c[z * 2 + 1].y)
                    : "d"(a0),
                      "d"(b2),
                      "d"(c[z * 2 + 1].x),
                      "d"(c[z * 2 + 1].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                (row * n_dofs_1d + 2 * col + (z * 2 + warpId) * offset) ^
                Util::get_base<n_dofs_1d>(row, z * 2 + warpId);

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
      else
        {
          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((z * 2 + warpId) * n_dofs_1d + 2 * col + row * offset) ^
                Util::get_base<n_dofs_1d>(z * 2 + warpId, row);

              if constexpr (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx = (row * n_dofs_1d + col + cycle * 4) ^
                                Util::get_base<n_dofs_1d>(row);
              auto a0 = shape_data[a_idx];

              for (int z = 0; z < n_dofs_1d / 4; ++z)
                {
                  const int b_idx =
                    ((z * 4 + warpId) * n_dofs_1d + row +
                     (col + cycle * 4) * offset) ^
                    Util::get_base<n_dofs_1d>(z * 4 + warpId, col + cycle * 4);
                  const int b_idx2 =
                    ((z * 4 + warpId + 2) * n_dofs_1d + row +
                     (col + cycle * 4) * offset) ^
                    Util::get_base<n_dofs_1d>(z * 4 + warpId + 2,
                                              col + cycle * 4);

                  auto b0 = in[b_idx];
                  auto b2 = in[b_idx2];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z * 2].x), "=d"(c[z * 2].y)
                    : "d"(a0), "d"(b0), "d"(c[z * 2].x), "d"(c[z * 2].y));
                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z * 2 + 1].x), "=d"(c[z * 2 + 1].y)
                    : "d"(a0),
                      "d"(b2),
                      "d"(c[z * 2 + 1].x),
                      "d"(c[z * 2 + 1].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((z * 2 + warpId) * n_dofs_1d + 2 * col + row * offset) ^
                Util::get_base<n_dofs_1d>(z * 2 + warpId, row);

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
    }
  };
#endif

#if MMAKERNEL == 3
  template <typename T>
  struct TPEvaluatorBase<T, 8, double, LaplaceVariant::TensorCoreMMA, 3>
  {
    using Number = double;

    static constexpr int n_dofs_1d = 8;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int tid    = (threadIdx.y * 8 + threadIdx.x) & 31;
      const int warpId = threadIdx.y / 4;

      const int row = tid / 4;
      const int col = tid & 3;

      constexpr int offset = n_dofs_1d * n_dofs_1d;

      if (direction == 0)
        {
          double2 c[2];

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              const int c_idx =
                (row * n_dofs_1d + 2 * col + (z * 4 + warpId) * offset) ^
                Util::get_base<n_dofs_1d>(row, z * 4 + warpId);

              c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
#  if TIMING == 1
              auto start = clock64();
#  endif
              const int b_idx = ((col + cycle * 4) * n_dofs_1d + row) ^
                                Util::get_base<n_dofs_1d>(col + cycle * 4);
              auto b0 = shape_data[b_idx];
#  if TIMING == 1
              auto elapsed = clock64() - start;
              if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
                printf(
                  "mma load frag b dir-%d loop-%d timing info: %ld cycles\n",
                  direction,
                  cycle,
                  elapsed);
#  endif

              for (int z = 0; z < n_dofs_1d / 4; ++z)
                {
#  if TIMING == 1
                  start = clock64();
#  endif
                  const int a_idx =
                    (row * n_dofs_1d + col + cycle * 4 +
                     (z * 4 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d>(row, z * 4 + warpId);

                  auto a0 = in[a_idx];
#  if TIMING == 1
                  elapsed = clock64() - start;
                  if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
                    printf(
                      "mma load frag a dir-%d loop-(%d,%d) timing info: %ld cycles\n",
                      direction,
                      cycle,
                      z,
                      elapsed);
#  endif

#  if TIMING == 1
                  start = clock64();
#  endif
                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
#  if TIMING == 1
                  elapsed = clock64() - start;
                  if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
                    printf(
                      "mma.sync.aligned.m8n8k4 dir-%d loop-(%d,%d) timing info: %ld cycles\n",
                      direction,
                      cycle,
                      z,
                      elapsed);
#  endif
                }
            }

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
#  if TIMING == 1
              auto start = clock64();
#  endif
              const int c_idx =
                (row * n_dofs_1d + 2 * col + (z * 4 + warpId) * offset) ^
                Util::get_base<n_dofs_1d>(row, z * 4 + warpId);

              *((double2 *)(out + c_idx)) = c[z];
#  if TIMING == 1
              auto elapsed = clock64() - start;
              if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
                printf(
                  "mma store frag c dir-%d loop-%d timing info: %ld cycles\n",
                  direction,
                  z,
                  elapsed);
#  endif
            }
        }
      else if (direction == 1)
        {
          double2 c[2];
          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              const int c_idx =
                (row * n_dofs_1d + 2 * col + (z * 4 + warpId) * offset) ^
                Util::get_base<n_dofs_1d>(row, z * 4 + warpId);

              if constexpr (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx = (row * n_dofs_1d + col + cycle * 4) ^
                                Util::get_base<n_dofs_1d>(row, 0);
              auto a0 = shape_data[a_idx];

              for (int z = 0; z < n_dofs_1d / 4; ++z)
                {
                  const int b_idx =
                    ((col + cycle * 4) * n_dofs_1d + row +
                     (z * 4 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d>(col + cycle * 4, z * 4 + warpId);

                  auto b0 = in[b_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              const int c_idx =
                (row * n_dofs_1d + 2 * col + (z * 4 + warpId) * offset) ^
                Util::get_base<n_dofs_1d>(row, z * 4 + warpId);

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
      else
        {
          double2 c[2];
          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              const int c_idx =
                ((z * 4 + warpId) * n_dofs_1d + 2 * col + row * offset) ^
                Util::get_base<n_dofs_1d>(z * 4 + warpId, row);

              if constexpr (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx = (row * n_dofs_1d + col + cycle * 4) ^
                                Util::get_base<n_dofs_1d>(row, 0);
              auto a0 = shape_data[a_idx];

              for (int z = 0; z < n_dofs_1d / 4; ++z)
                {
                  const int b_idx =
                    ((z * 4 + warpId) * n_dofs_1d + row +
                     (col + cycle * 4) * offset) ^
                    Util::get_base<n_dofs_1d>(z * 4 + warpId, col + cycle * 4);

                  auto b0 = in[b_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              const int c_idx =
                ((z * 4 + warpId) * n_dofs_1d + 2 * col + row * offset) ^
                Util::get_base<n_dofs_1d>(z * 4 + warpId, row);

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
    }
  };
#endif

#if MMAKERNEL == 4
  template <typename T>
  struct TPEvaluatorBase<T, 8, double, LaplaceVariant::TensorCoreMMA, 3>
  {
    using Number = double;

    static constexpr int n_dofs_1d = 8;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int tid    = (threadIdx.y * 8 + threadIdx.x) & 31;
      const int warpId = (threadIdx.y * 8 + threadIdx.x) / 32;

      const int row = tid / 4;
      const int col = tid & 3;

      constexpr int offset = n_dofs_1d * n_dofs_1d;

      if (direction == 0)
        {
          double2 c[2];

          const int c_idx =
            (row * n_dofs_1d + 2 * col + (2 * warpId) * offset) ^
            Util::get_base<n_dofs_1d>(row, 2 * warpId);
          const int c_idx2 =
            (row * n_dofs_1d + 2 * col + (2 * warpId + 1) * offset) ^
            Util::get_base<n_dofs_1d>(row, 2 * warpId + 1);

          if constexpr (add)
            {
              c[0] = *((double2 *)(out + c_idx));
              c[1] = *((double2 *)(out + c_idx2));
            }
          else
            {
              c[0] = {0, 0};
              c[1] = {0, 0};
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
#  if TIMING == 1
              auto start = clock64();
#  endif
              const int b_idx = ((col + cycle * 4) * n_dofs_1d + row) ^
                                Util::get_base<n_dofs_1d>(col + cycle * 4);
              auto b0 = shape_data[b_idx];
#  if TIMING == 1
              auto elapsed = clock64() - start;
              if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
                printf(
                  "mma load frag b dir-%d loop-%d timing info: %ld cycles\n",
                  direction,
                  cycle,
                  elapsed);
#  endif

#  if TIMING == 1
              start = clock64();
#  endif
              const int a_idx =
                (row * n_dofs_1d + col + cycle * 4 + (2 * warpId) * offset) ^
                Util::get_base<n_dofs_1d>(row, 2 * warpId);
              const int a_idx2 = (row * n_dofs_1d + col + cycle * 4 +
                                  (2 * warpId + 1) * offset) ^
                                 Util::get_base<n_dofs_1d>(row, 2 * warpId + 1);

              auto a0 = in[a_idx];
              auto a2 = in[a_idx2];
#  if TIMING == 1
              elapsed = clock64() - start;
              if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
                printf(
                  "mma load frag a dir-%d loop-(%d,%d) timing info: %ld cycles\n",
                  direction,
                  cycle,
                  0,
                  elapsed);
#  endif

#  if TIMING == 1
              start = clock64();
#  endif
              asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                           "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                           : "=d"(c[0].x), "=d"(c[0].y)
                           : "d"(a0), "d"(b0), "d"(c[0].x), "d"(c[0].y));

              asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                           "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                           : "=d"(c[1].x), "=d"(c[1].y)
                           : "d"(a2), "d"(b0), "d"(c[1].x), "d"(c[1].y));
#  if TIMING == 1
              elapsed = clock64() - start;
              if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
                printf(
                  "mma.sync.aligned.m8n8k4 dir-%d loop-(%d,%d) timing info: %ld cycles\n",
                  direction,
                  cycle,
                  0,
                  elapsed);
#  endif
            }

#  if TIMING == 1
          auto start = clock64();
#  endif
          *((double2 *)(out + c_idx))  = c[0];
          *((double2 *)(out + c_idx2)) = c[1];
#  if TIMING == 1
          auto elapsed = clock64() - start;
          if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
            printf("mma store frag c dir-%d loop-%d timing info: %ld cycles\n",
                   direction,
                   0,
                   elapsed);
#  endif
        }
      else if (direction == 1)
        {
          double2 c[2];

          const int c_idx =
            (row * n_dofs_1d + 2 * col + (2 * warpId) * offset) ^
            Util::get_base<n_dofs_1d>(row, 2 * warpId);
          const int c_idx2 =
            (row * n_dofs_1d + 2 * col + (2 * warpId + 1) * offset) ^
            Util::get_base<n_dofs_1d>(row, 2 * warpId + 1);

          if constexpr (add)
            {
              c[0] = *((double2 *)(out + c_idx));
              c[1] = *((double2 *)(out + c_idx2));
            }
          else
            {
              c[0] = {0, 0};
              c[1] = {0, 0};
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx = (row * n_dofs_1d + col + cycle * 4) ^
                                Util::get_base<n_dofs_1d>(row, 0);
              auto a0 = shape_data[a_idx];


              const int b_idx =
                ((col + cycle * 4) * n_dofs_1d + row + (2 * warpId) * offset) ^
                Util::get_base<n_dofs_1d>(col + cycle * 4, 2 * warpId);
              const int b_idx2 =
                ((col + cycle * 4) * n_dofs_1d + row +
                 (2 * warpId + 1) * offset) ^
                Util::get_base<n_dofs_1d>(col + cycle * 4, 2 * warpId + 1);

              auto b0 = in[b_idx];
              auto b2 = in[b_idx2];

              asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                           "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                           : "=d"(c[0].x), "=d"(c[0].y)
                           : "d"(a0), "d"(b0), "d"(c[0].x), "d"(c[0].y));

              asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                           "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                           : "=d"(c[1].x), "=d"(c[1].y)
                           : "d"(a0), "d"(b2), "d"(c[1].x), "d"(c[1].y));
            }

          *((double2 *)(out + c_idx))  = c[0];
          *((double2 *)(out + c_idx2)) = c[1];
        }
      else
        {
          double2 c[2];

          const int c_idx =
            ((2 * warpId) * n_dofs_1d + 2 * col + row * offset) ^
            Util::get_base<n_dofs_1d>(2 * warpId, row);
          const int c_idx2 =
            ((2 * warpId + 1) * n_dofs_1d + 2 * col + row * offset) ^
            Util::get_base<n_dofs_1d>(2 * warpId + 1, row);

          if constexpr (add)
            {
              c[0] = *((double2 *)(out + c_idx));
              c[1] = *((double2 *)(out + c_idx2));
            }
          else
            {
              c[0] = {0, 0};
              c[1] = {0, 0};
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx = (row * n_dofs_1d + col + cycle * 4) ^
                                Util::get_base<n_dofs_1d>(row, 0);
              auto a0 = shape_data[a_idx];

              const int b_idx =
                ((2 * warpId) * n_dofs_1d + row + (col + cycle * 4) * offset) ^
                Util::get_base<n_dofs_1d>(2 * warpId, col + cycle * 4);
              const int b_idx2 =
                ((2 * warpId + 1) * n_dofs_1d + row +
                 (col + cycle * 4) * offset) ^
                Util::get_base<n_dofs_1d>(2 * warpId + 1, col + cycle * 4);

              auto b0 = in[b_idx];
              auto b2 = in[b_idx2];

              asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                           "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                           : "=d"(c[0].x), "=d"(c[0].y)
                           : "d"(a0), "d"(b0), "d"(c[0].x), "d"(c[0].y));

              asm volatile("mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                           "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                           : "=d"(c[1].x), "=d"(c[1].y)
                           : "d"(a0), "d"(b2), "d"(c[1].x), "d"(c[1].y));
            }

          *((double2 *)(out + c_idx))  = c[0];
          *((double2 *)(out + c_idx2)) = c[1];
        }
    }
  };
#endif

#if MMAKERNEL == 5
  template <typename T>
  struct TPEvaluatorBase<T, 8, double, LaplaceVariant::TensorCoreMMA, 3>
  {
    using Number = double;

    static constexpr int n_dofs_1d = 8;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int tid = (threadIdx.y * 8 + threadIdx.x) & 31;

      const int row = tid / 4;
      const int col = tid & 3;

      constexpr int offset = n_dofs_1d * n_dofs_1d;

      if (direction == 0)
        {
          double2 c[n_dofs_1d];
          for (int z = 0; z < n_dofs_1d; ++z)
            c[z] = {0, 0};

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int b_idx = ((col + cycle * 4) * n_dofs_1d + row) ^
                                Util::get_base<n_dofs_1d>(col + cycle * 4);
              auto b0 = shape_data[b_idx];

              for (int z = 0; z < n_dofs_1d; ++z)
                {
                  const int a_idx =
                    (row * n_dofs_1d + col + cycle * 4 + z * offset) ^
                    Util::get_base<n_dofs_1d>(row, z);

                  auto a0 = in[a_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d; ++z)
            {
              const int c_idx = (row * n_dofs_1d + 2 * col + z * offset) ^
                                Util::get_base<n_dofs_1d>(row, z);

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
      else if (direction == 1)
        {
          double2 c[n_dofs_1d];
          for (int z = 0; z < n_dofs_1d; ++z)
            {
              if constexpr (add)
                {
                  const int c_idx = (row * n_dofs_1d + 2 * col + z * offset) ^
                                    Util::get_base<n_dofs_1d>(row, z);

                  c[z] = *((double2 *)(out + c_idx));
                }
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx = (row * n_dofs_1d + col + cycle * 4) ^
                                Util::get_base<n_dofs_1d>(row, 0);
              auto a0 = shape_data[a_idx];

              for (int z = 0; z < n_dofs_1d; ++z)
                {
                  const int b_idx =
                    ((col + cycle * 4) * n_dofs_1d + row + z * offset) ^
                    Util::get_base<n_dofs_1d>(col + cycle * 4, z);

                  auto b0 = in[b_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d; ++z)
            {
              const int c_idx = (row * n_dofs_1d + 2 * col + z * offset) ^
                                Util::get_base<n_dofs_1d>(row, z);

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
      else
        {
          double2 c[n_dofs_1d];
          for (int z = 0; z < n_dofs_1d; ++z)
            {
              if constexpr (add)
                {
                  const int c_idx = (z * n_dofs_1d + 2 * col + row * offset) ^
                                    Util::get_base<n_dofs_1d>(z, row);

                  c[z] = *((double2 *)(out + c_idx));
                }
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx = (row * n_dofs_1d + col + cycle * 4) ^
                                Util::get_base<n_dofs_1d>(row, 0);
              auto a0 = shape_data[a_idx];

              for (int z = 0; z < n_dofs_1d; ++z)
                {
                  const int b_idx =
                    (z * n_dofs_1d + row + (col + cycle * 4) * offset) ^
                    Util::get_base<n_dofs_1d>(z, col + cycle * 4);

                  auto b0 = in[b_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d; ++z)
            {
              const int c_idx = (z * n_dofs_1d + 2 * col + row * offset) ^
                                Util::get_base<n_dofs_1d>(z, row);

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
    }
  };
#endif

  template <typename T>
  struct TPEvaluatorBase<T, 10, double, LaplaceVariant::TensorCoreMMA, 3>
  {
    using Number = double;

    static constexpr int n_dofs_1d   = 10;
    static constexpr int n_dofs_1d_p = 16;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int warpId = (threadIdx.y * 16 + threadIdx.x) / 32;
      const int subId  = warpId & 3;
      const int rowId  = subId / 2;
      const int colId  = subId & 1;

      const int tid = (threadIdx.y * 16 + threadIdx.x) & 31;

      const int row = tid / 4;
      const int col = tid & 3;

      constexpr int offset = n_dofs_1d_p * n_dofs_1d_p;

      if (direction == 0)
        {
          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((rowId * 8 + row) * n_dofs_1d_p + 2 * col + colId * 8 +
                 (z * 2 + warpId / 4) * offset) ^
                Util::get_base<n_dofs_1d>(rowId * 8 + row, z * 2 + warpId / 4);

              if (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 3; ++cycle)
            {
              const int b_idx =
                ((col + cycle * 4) * n_dofs_1d_p + row + colId * 8) ^
                Util::get_base<n_dofs_1d>(col + cycle * 4);
              auto b0 = shape_data[b_idx];

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int a_idx =
                    ((rowId * 8 + row) * n_dofs_1d_p + col + cycle * 4 +
                     (z * 2 + warpId / 4) * offset) ^
                    Util::get_base<n_dofs_1d>(rowId * 8 + row,
                                              z * 2 + warpId / 4);

                  auto a0 = in[a_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((rowId * 8 + row) * n_dofs_1d_p + 2 * col + colId * 8 +
                 (z * 2 + warpId / 4) * offset) ^
                Util::get_base<n_dofs_1d>(rowId * 8 + row, z * 2 + warpId / 4);

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
      else if (direction == 1)
        {
          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((rowId * 8 + row) * n_dofs_1d_p + 2 * col + colId * 8 +
                 (z * 2 + warpId / 4) * offset) ^
                Util::get_base<n_dofs_1d>(rowId * 8 + row, z * 2 + warpId / 4);

              if (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 3; ++cycle)
            {
              const int a_idx =
                ((rowId * 8 + row) * n_dofs_1d_p + col + cycle * 4) ^
                Util::get_base<n_dofs_1d>(rowId * 8 + row);
              auto a0 = shape_data[a_idx];

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int b_idx =
                    ((col + cycle * 4) * n_dofs_1d_p + row + colId * 8 +
                     (z * 2 + warpId / 4) * offset) ^
                    Util::get_base<n_dofs_1d>(col + cycle * 4,
                                              z * 2 + warpId / 4);

                  auto b0 = in[b_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((rowId * 8 + row) * n_dofs_1d_p + 2 * col + colId * 8 +
                 (z * 2 + warpId / 4) * offset) ^
                Util::get_base<n_dofs_1d>(rowId * 8 + row, z * 2 + warpId / 4);

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
      else
        {
          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((z * 2 + warpId / 4) * n_dofs_1d_p + 2 * col + colId * 8 +
                 (rowId * 8 + row) * offset) ^
                Util::get_base<n_dofs_1d>(z * 2 + warpId / 4, rowId * 8 + row);

              if (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 3; ++cycle)
            {
              const int a_idx =
                ((rowId * 8 + row) * n_dofs_1d_p + col + cycle * 4) ^
                Util::get_base<n_dofs_1d>(rowId * 8 + row);
              auto a0 = shape_data[a_idx];

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int b_idx =
                    ((z * 2 + warpId / 4) * n_dofs_1d_p + colId * 8 + row +
                     (col + cycle * 4) * offset) ^
                    Util::get_base<n_dofs_1d>(z * 2 + warpId / 4,
                                              col + cycle * 4);

                  auto b0 = in[b_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((z * 2 + warpId / 4) * n_dofs_1d_p + 2 * col + colId * 8 +
                 (rowId * 8 + row) * offset) ^
                Util::get_base<n_dofs_1d>(z * 2 + warpId / 4, rowId * 8 + row);

              if (rowId == 0 || row < 2)
                *((double2 *)(out + c_idx)) = c[z];
            }
        }
    }
  };

  template <typename T>
  struct TPEvaluatorBase<T, 12, double, LaplaceVariant::TensorCoreMMA, 3>
  {
    using Number = double;

    static constexpr int n_dofs_1d   = 12;
    static constexpr int n_dofs_1d_p = 16;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int warpId = (threadIdx.y * 16 + threadIdx.x) / 32;
      const int subId  = warpId & 3;
      const int rowId  = subId / 2;
      const int colId  = subId & 1;

      const int tid = (threadIdx.y * 16 + threadIdx.x) & 31;

      const int row = tid / 4;
      const int col = tid & 3;

      constexpr int offset = n_dofs_1d_p * n_dofs_1d_p;

      if (direction == 0)
        {
          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((rowId * 8 + row) * n_dofs_1d_p + 2 * col + colId * 8 +
                 (z * 2 + warpId / 4) * offset) ^
                Util::get_base<n_dofs_1d>(rowId * 8 + row, z * 2 + warpId / 4);

              c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 3; ++cycle)
            {
              const int b_idx =
                ((col + cycle * 4) * n_dofs_1d_p + row + colId * 8) ^
                Util::get_base<n_dofs_1d>(col + cycle * 4);
              auto b0 = shape_data[b_idx];

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int a_idx =
                    ((rowId * 8 + row) * n_dofs_1d_p + col + cycle * 4 +
                     (z * 2 + warpId / 4) * offset) ^
                    Util::get_base<n_dofs_1d>(rowId * 8 + row,
                                              z * 2 + warpId / 4);

                  auto a0 = in[a_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((rowId * 8 + row) * n_dofs_1d_p + 2 * col + colId * 8 +
                 (z * 2 + warpId / 4) * offset) ^
                Util::get_base<n_dofs_1d>(rowId * 8 + row, z * 2 + warpId / 4);

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
      else if (direction == 1)
        {
          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((rowId * 8 + row) * n_dofs_1d_p + 2 * col + colId * 8 +
                 (z * 2 + warpId / 4) * offset) ^
                Util::get_base<n_dofs_1d>(rowId * 8 + row, z * 2 + warpId / 4);

              if (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 3; ++cycle)
            {
              const int a_idx =
                ((rowId * 8 + row) * n_dofs_1d_p + col + cycle * 4) ^
                Util::get_base<n_dofs_1d>(rowId * 8 + row);
              auto a0 = shape_data[a_idx];

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int b_idx =
                    ((col + cycle * 4) * n_dofs_1d_p + row + colId * 8 +
                     (z * 2 + warpId / 4) * offset) ^
                    Util::get_base<n_dofs_1d>(col + cycle * 4,
                                              z * 2 + warpId / 4);

                  auto b0 = in[b_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((rowId * 8 + row) * n_dofs_1d_p + 2 * col + colId * 8 +
                 (z * 2 + warpId / 4) * offset) ^
                Util::get_base<n_dofs_1d>(rowId * 8 + row, z * 2 + warpId / 4);

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
      else
        {
          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((z * 2 + warpId / 4) * n_dofs_1d_p + 2 * col + colId * 8 +
                 (rowId * 8 + row) * offset) ^
                Util::get_base<n_dofs_1d>(z * 2 + warpId / 4, rowId * 8 + row);

              if (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 3; ++cycle)
            {
              const int a_idx =
                ((rowId * 8 + row) * n_dofs_1d_p + col + cycle * 4) ^
                Util::get_base<n_dofs_1d>(rowId * 8 + row);
              auto a0 = shape_data[a_idx];

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int b_idx =
                    ((z * 2 + warpId / 4) * n_dofs_1d_p + colId * 8 + row +
                     (col + cycle * 4) * offset) ^
                    Util::get_base<n_dofs_1d>(z * 2 + warpId / 4,
                                              col + cycle * 4);

                  auto b0 = in[b_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((z * 2 + warpId / 4) * n_dofs_1d_p + 2 * col + colId * 8 +
                 (rowId * 8 + row) * offset) ^
                Util::get_base<n_dofs_1d>(z * 2 + warpId / 4, rowId * 8 + row);

              if (rowId == 0 || row < 4)
                *((double2 *)(out + c_idx)) = c[z];
            }
        }
    }
  };

  template <typename T>
  struct TPEvaluatorBase<T, 14, double, LaplaceVariant::TensorCoreMMA, 3>
  {
    using Number = double;

    static constexpr int n_dofs_1d   = 14;
    static constexpr int n_dofs_1d_p = 16;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int warpId = (threadIdx.y * 16 + threadIdx.x) / 32;
      const int subId  = warpId & 3;
      const int rowId  = subId / 2;
      const int colId  = subId & 1;

      const int tid = (threadIdx.y * 16 + threadIdx.x) & 31;

      const int row = tid / 4;
      const int col = tid & 3;

      constexpr int offset = n_dofs_1d_p * n_dofs_1d_p;

      if (direction == 0)
        {
          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((rowId * 8 + row) * n_dofs_1d_p + 2 * col + colId * 8 +
                 (z * 2 + warpId / 4) * offset) ^
                Util::get_base<n_dofs_1d>(rowId * 8 + row, z * 2 + warpId / 4);

              if (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 4; ++cycle)
            {
              const int b_idx =
                ((col + cycle * 4) * n_dofs_1d_p + row + colId * 8) ^
                Util::get_base<n_dofs_1d>(col + cycle * 4);
              auto b0 = shape_data[b_idx];

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int a_idx =
                    ((rowId * 8 + row) * n_dofs_1d_p + col + cycle * 4 +
                     (z * 2 + warpId / 4) * offset) ^
                    Util::get_base<n_dofs_1d>(rowId * 8 + row,
                                              z * 2 + warpId / 4);

                  auto a0 = in[a_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((rowId * 8 + row) * n_dofs_1d_p + 2 * col + colId * 8 +
                 (z * 2 + warpId / 4) * offset) ^
                Util::get_base<n_dofs_1d>(rowId * 8 + row, z * 2 + warpId / 4);

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
      else if (direction == 1)
        {
          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((rowId * 8 + row) * n_dofs_1d_p + 2 * col + colId * 8 +
                 (z * 2 + warpId / 4) * offset) ^
                Util::get_base<n_dofs_1d>(rowId * 8 + row, z * 2 + warpId / 4);

              if (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 4; ++cycle)
            {
              const int a_idx =
                ((rowId * 8 + row) * n_dofs_1d_p + col + cycle * 4) ^
                Util::get_base<n_dofs_1d>(rowId * 8 + row);
              auto a0 = shape_data[a_idx];

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int b_idx =
                    ((col + cycle * 4) * n_dofs_1d_p + row + colId * 8 +
                     (z * 2 + warpId / 4) * offset) ^
                    Util::get_base<n_dofs_1d>(col + cycle * 4,
                                              z * 2 + warpId / 4);

                  auto b0 = in[b_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((rowId * 8 + row) * n_dofs_1d_p + 2 * col + colId * 8 +
                 (z * 2 + warpId / 4) * offset) ^
                Util::get_base<n_dofs_1d>(rowId * 8 + row, z * 2 + warpId / 4);

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
      else
        {
          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((z * 2 + warpId / 4) * n_dofs_1d_p + 2 * col + colId * 8 +
                 (rowId * 8 + row) * offset) ^
                Util::get_base<n_dofs_1d>(z * 2 + warpId / 4, rowId * 8 + row);

              if (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 4; ++cycle)
            {
              const int a_idx =
                ((rowId * 8 + row) * n_dofs_1d_p + col + cycle * 4) ^
                Util::get_base<n_dofs_1d>(rowId * 8 + row);
              auto a0 = shape_data[a_idx];

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int b_idx =
                    ((z * 2 + warpId / 4) * n_dofs_1d_p + colId * 8 + row +
                     (col + cycle * 4) * offset) ^
                    Util::get_base<n_dofs_1d>(z * 2 + warpId / 4,
                                              col + cycle * 4);

                  auto b0 = in[b_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((z * 2 + warpId / 4) * n_dofs_1d_p + 2 * col + colId * 8 +
                 (rowId * 8 + row) * offset) ^
                Util::get_base<n_dofs_1d>(z * 2 + warpId / 4, rowId * 8 + row);

              if (rowId == 0 || row < 6)
                *((double2 *)(out + c_idx)) = c[z];
            }
        }
    }
  };


#if MMAKERNEL == 0
  template <typename T>
  struct TPEvaluatorBase<T, 16, double, LaplaceVariant::TensorCoreMMA, 3>
  {
    using Number = double;

    static constexpr int n_dofs_1d = 16;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int warpId = (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;
      const int subId  = warpId & 3;
      const int rowId  = subId / 2;
      const int colId  = subId & 1;

      const int tid = (threadIdx.y * n_dofs_1d + threadIdx.x) & 31;

      const int row = tid / 4;
      const int col = tid & 3;

      constexpr int offset = n_dofs_1d * n_dofs_1d;

      if (direction == 0)
        {
          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx = (rowId * 8 + row) * n_dofs_1d + 2 * col +
                                colId * 8 + (z * 2 + warpId / 4) * offset;

              if constexpr (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 4; ++cycle)
            {
              const int b_idx = (col + cycle * 4) * n_dofs_1d + row + colId * 8;
              auto      b0    = shape_data[b_idx];

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int a_idx = (rowId * 8 + row) * n_dofs_1d + col +
                                    cycle * 4 + (z * 2 + warpId / 4) * offset;
                  auto a0 = in[a_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx = (rowId * 8 + row) * n_dofs_1d + 2 * col +
                                colId * 8 + (z * 2 + warpId / 4) * offset;

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
      else if (direction == 1)
        {
          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx = (rowId * 8 + row) * n_dofs_1d + 2 * col +
                                colId * 8 + (z * 2 + warpId / 4) * offset;

              if constexpr (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 4; ++cycle)
            {
              const int a_idx = (rowId * 8 + row) * n_dofs_1d + col + cycle * 4;
              auto      a0    = shape_data[a_idx];

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int b_idx = (col + cycle * 4) * n_dofs_1d + row +
                                    colId * 8 + (z * 2 + warpId / 4) * offset;

                  auto b0 = in[b_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx = (rowId * 8 + row) * n_dofs_1d + 2 * col +
                                colId * 8 + (z * 2 + warpId / 4) * offset;

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
      else
        {
          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx = (z * 2 + warpId / 4) * n_dofs_1d + 2 * col +
                                colId * 8 + (rowId * 8 + row) * offset;

              if constexpr (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 4; ++cycle)
            {
              const int a_idx = (rowId * 8 + row) * n_dofs_1d + col + cycle * 4;
              auto      a0    = shape_data[a_idx];

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int b_idx = (z * 2 + warpId / 4) * n_dofs_1d +
                                    colId * 8 + row +
                                    (col + cycle * 4) * offset;

                  auto b0 = in[b_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx = (z * 2 + warpId / 4) * n_dofs_1d + 2 * col +
                                colId * 8 + (rowId * 8 + row) * offset;

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
    }
  };
#endif



#if MMAKERNEL != 0
  template <typename T>
  struct TPEvaluatorBase<T, 16, double, LaplaceVariant::TensorCoreMMA, 3>
  {
    using Number = double;

    static constexpr int n_dofs_1d = 16;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int warpId = (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;
      const int subId  = warpId & 3;
      const int rowId  = subId / 2;
      const int colId  = subId & 1;

      const int tid = (threadIdx.y * n_dofs_1d + threadIdx.x) & 31;

      const int row = tid / 4;
      const int col = tid & 3;

      constexpr int offset = n_dofs_1d * n_dofs_1d;

      if (direction == 0)
        {
          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((rowId * 8 + row) * n_dofs_1d + 2 * col + colId * 8 +
                 (z * 2 + warpId / 4) * offset) ^
                Util::get_base<n_dofs_1d>(rowId * 8 + row, z * 2 + warpId / 4);

              if constexpr (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 4; ++cycle)
            {
#  ifdef SKIPZERO
              if ((shape_data[(cycle / 2) * (n_dofs_1d * 8 + 8)]) < 1e-10)
                continue;
#  endif
              const int b_idx =
                ((col + cycle * 4) * n_dofs_1d + row + colId * 8);
              // ^
              //   Util::get_base<n_dofs_1d>(col + cycle * 4);
#  ifdef USETEXTURE
              double b0 = tex1Dfetch(stiff_data_d, b_idx);
#  else
              auto b0 = shape_data[b_idx];
#  endif

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int a_idx =
                    ((rowId * 8 + row) * n_dofs_1d + col + cycle * 4 +
                     (z * 2 + warpId / 4) * offset) ^
                    Util::get_base<n_dofs_1d>(rowId * 8 + row,
                                              z * 2 + warpId / 4);

                  auto a0 = in[a_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((rowId * 8 + row) * n_dofs_1d + 2 * col + colId * 8 +
                 (z * 2 + warpId / 4) * offset) ^
                Util::get_base<n_dofs_1d>(rowId * 8 + row, z * 2 + warpId / 4);

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
      else if (direction == 1)
        {
          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((rowId * 8 + row) * n_dofs_1d + 2 * col + colId * 8 +
                 (z * 2 + warpId / 4) * offset) ^
                Util::get_base<n_dofs_1d>(rowId * 8 + row, z * 2 + warpId / 4);

              if constexpr (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 4; ++cycle)
            {
#  ifdef SKIPZERO
              if ((shape_data[(cycle / 2) * (n_dofs_1d * 8 + 8)]) < 1e-10)
                continue;
#  endif

              const int a_idx =
                ((rowId * 8 + row) * n_dofs_1d + col + cycle * 4);
              // ^
              //   Util::get_base<n_dofs_1d>(rowId * 8 + row);

#  ifdef USETEXTURE
              double a0 = tex1Dfetch(stiff_data_d, a_idx);
#  else
              auto a0 = shape_data[a_idx];
#  endif

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int b_idx =
                    ((col + cycle * 4) * n_dofs_1d + row + colId * 8 +
                     (z * 2 + warpId / 4) * offset) ^
                    Util::get_base<n_dofs_1d>(col + cycle * 4,
                                              z * 2 + warpId / 4);

                  auto b0 = in[b_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((rowId * 8 + row) * n_dofs_1d + 2 * col + colId * 8 +
                 (z * 2 + warpId / 4) * offset) ^
                Util::get_base<n_dofs_1d>(rowId * 8 + row, z * 2 + warpId / 4);

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
      else
        {
          double2 c[n_dofs_1d / 2];
          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((z * 2 + warpId / 4) * n_dofs_1d + 2 * col + colId * 8 +
                 (rowId * 8 + row) * offset) ^
                Util::get_base<n_dofs_1d>(z * 2 + warpId / 4, rowId * 8 + row);

              if constexpr (add)
                c[z] = *((double2 *)(out + c_idx));
              else
                c[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 4; ++cycle)
            {
#  ifdef SKIPZERO
              if ((shape_data[(cycle / 2) * (n_dofs_1d * 8 + 8)]) < 1e-10)
                continue;
#  endif

              const int a_idx =
                ((rowId * 8 + row) * n_dofs_1d + col + cycle * 4);
              // ^
              //   Util::get_base<n_dofs_1d>(rowId * 8 + row);

#  ifdef USETEXTURE
              double a0 = tex1Dfetch(stiff_data_d, a_idx);
#  else
              auto a0 = shape_data[a_idx];
#  endif

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int b_idx =
                    ((z * 2 + warpId / 4) * n_dofs_1d + colId * 8 + row +
                     (col + cycle * 4) * offset) ^
                    Util::get_base<n_dofs_1d>(z * 2 + warpId / 4,
                                              col + cycle * 4);

                  auto b0 = in[b_idx];

                  asm volatile(
                    "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
                    "{%0,%1}, {%2}, {%3}, {%4,%5};\n"
                    : "=d"(c[z].x), "=d"(c[z].y)
                    : "d"(a0), "d"(b0), "d"(c[z].x), "d"(c[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx =
                ((z * 2 + warpId / 4) * n_dofs_1d + 2 * col + colId * 8 +
                 (rowId * 8 + row) * offset) ^
                Util::get_base<n_dofs_1d>(z * 2 + warpId / 4, rowId * 8 + row);

              *((double2 *)(out + c_idx)) = c[z];
            }
        }
    }
  };
#endif


#if MMAKERNEL == 6
  template <typename T>
  struct TPEvaluatorBase<T, 16, float, LaplaceVariant::TensorCoreMMA, 3>
  {
    using Number = float;

    static constexpr int n_dofs_1d = 16;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int warpId = threadIdx.y / 2;

      const int tid = (threadIdx.y * n_dofs_1d + threadIdx.x) & 31;

      const int row = tid / 4;
      const int col = tid & 3;

      constexpr int offset = n_dofs_1d * n_dofs_1d;
      constexpr int scale  = 1 << 11;

      if constexpr (direction == 0)
        {
          for (int cycle = 0; cycle < 2; ++cycle)
            {
              float2 c0[2];
              float2 c1[2];

              half a[8];
              half b[4];

              for (int z = 0; z < 2; ++z)
                {
                  c0[z] = {0, 0};
                  c1[z] = {0, 0};
                }

              const int b_idx0 = (col * 2 * n_dofs_1d + row + cycle * 8) ^
                                 Util::get_base<n_dofs_1d, float>(col * 2);
              const int b_idx1 = ((col * 2 + 1) * n_dofs_1d + row + cycle * 8) ^
                                 Util::get_base<n_dofs_1d, float>(col * 2 + 1);
              const int b_idx2 = ((col * 2 + 8) * n_dofs_1d + row + cycle * 8) ^
                                 Util::get_base<n_dofs_1d, float>(col * 2 + 8);
              const int b_idx3 = ((col * 2 + 9) * n_dofs_1d + row + cycle * 8) ^
                                 Util::get_base<n_dofs_1d, float>(col * 2 + 9);
#  if ERRCOR == 0
              b[0] = __float2half(shape_data[b_idx0]);
              b[1] = __float2half(shape_data[b_idx1]);
              b[2] = __float2half(shape_data[b_idx2]);
              b[3] = __float2half(shape_data[b_idx3]);
#  elif ERRCOR == 1
              float fb[4];
              half  db[4];
              fb[0] = shape_data[b_idx0];
              fb[1] = shape_data[b_idx1];
              fb[2] = shape_data[b_idx2];
              fb[3] = shape_data[b_idx3];

              for (int i = 0; i < 4; ++i)
                {
                  b[i]  = __float2half(fb[i]);
                  db[i] = __float2half((fb[i] - __half2float(b[i])) * scale);
                }
#  endif
              uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
#  if ERRCOR == 1
              uint32_t const *dB = reinterpret_cast<uint32_t const *>(&db);
#  endif
              for (int z = 0; z < 2; ++z)
                {
                  const int a_idx =
                    (row * n_dofs_1d + col * 2 + (z * 8 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row, z * 8 + warpId);
                  const int a_idx1 =
                    ((row + 8) * n_dofs_1d + col * 2 +
                     (z * 8 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row + 8, z * 8 + warpId);
                  const int a_idx2 =
                    (row * n_dofs_1d + col * 2 + 8 +
                     (z * 8 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row, z * 8 + warpId);
                  const int a_idx3 =
                    ((row + 8) * n_dofs_1d + col * 2 + 8 +
                     (z * 8 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row + 8, z * 8 + warpId);
#  if ERRCOR == 0
                  a[0] = __float2half(in[a_idx]);
                  a[1] = __float2half(in[a_idx + 1]);
                  a[2] = __float2half(in[a_idx1]);
                  a[3] = __float2half(in[a_idx1 + 1]);
                  a[4] = __float2half(in[a_idx2]);
                  a[5] = __float2half(in[a_idx2 + 1]);
                  a[6] = __float2half(in[a_idx3]);
                  a[7] = __float2half(in[a_idx3 + 1]);
#  elif ERRCOR == 1
                  float fa[8];
                  half  da[8];
                  fa[0] = in[a_idx];
                  fa[1] = in[a_idx + 1];
                  fa[2] = in[a_idx1];
                  fa[3] = in[a_idx1 + 1];
                  fa[4] = in[a_idx2];
                  fa[5] = in[a_idx2 + 1];
                  fa[6] = in[a_idx3];
                  fa[7] = in[a_idx3 + 1];

                  for (int i = 0; i < 8; ++i)
                    {
                      a[i] = __float2half(fa[i]);
                      da[i] =
                        __float2half((fa[i] - __half2float(a[i])) * scale);
                    }
#  endif
                  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
#  if ERRCOR == 1
                  uint32_t const *dA = reinterpret_cast<uint32_t const *>(&da);
#  endif
                  asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(c0[z].x), "=f"(c0[z].y), "=f"(c1[z].x), "=f"(c1[z].y)
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(c0[z].x),
                      "f"(c0[z].y),
                      "f"(c1[z].x),
                      "f"(c1[z].y));
#  if ERRCOR == 1
                  float buf[4];
                  buf[0] = 0;
                  buf[1] = 0;
                  buf[2] = 0;
                  buf[3] = 0;
                  asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                    : "r"(dA[0]),
                      "r"(dA[1]),
                      "r"(dA[2]),
                      "r"(dA[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(buf[0]),
                      "f"(buf[1]),
                      "f"(buf[2]),
                      "f"(buf[3]));
                  asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(dB[0]),
                      "r"(dB[1]),
                      "f"(buf[0]),
                      "f"(buf[1]),
                      "f"(buf[2]),
                      "f"(buf[3]));
                  c0[z].x += buf[0] / scale;
                  c0[z].y += buf[1] / scale;
                  c1[z].x += buf[2] / scale;
                  c1[z].y += buf[3] / scale;
#  endif
                }

              for (int z = 0; z < 2; ++z)
                {
                  const int c_idx0 =
                    (row * n_dofs_1d + 2 * col + cycle * 8 +
                     (z * 8 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row, z * 8 + warpId);
                  const int c_idx1 =
                    ((row + 8) * n_dofs_1d + 2 * col + cycle * 8 +
                     (z * 8 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row + 8, z * 8 + warpId);

                  *((float2 *)(out + c_idx0)) = c0[z];
                  *((float2 *)(out + c_idx1)) = c1[z];
                }
            }
        }
      else if (direction == 1)
        {
          half a[8];

          const int a_idx =
            (row * n_dofs_1d + col * 2) ^ Util::get_base<n_dofs_1d, float>(row);
          const int a_idx1 = ((row + 8) * n_dofs_1d + col * 2) ^
                             Util::get_base<n_dofs_1d, float>(row + 8);
          const int a_idx2 = (row * n_dofs_1d + col * 2 + 8) ^
                             Util::get_base<n_dofs_1d, float>(row);
          const int a_idx3 = ((row + 8) * n_dofs_1d + col * 2 + 8) ^
                             Util::get_base<n_dofs_1d, float>(row + 8);
#  if ERRCOR == 0
          a[0] = __float2half(shape_data[a_idx]);
          a[1] = __float2half(shape_data[a_idx + 1]);
          a[2] = __float2half(shape_data[a_idx1]);
          a[3] = __float2half(shape_data[a_idx1 + 1]);
          a[4] = __float2half(shape_data[a_idx2]);
          a[5] = __float2half(shape_data[a_idx2 + 1]);
          a[6] = __float2half(shape_data[a_idx3]);
          a[7] = __float2half(shape_data[a_idx3 + 1]);
#  elif ERRCOR == 1
          float fa[8];
          half  da[8];
          fa[0] = shape_data[a_idx];
          fa[1] = shape_data[a_idx + 1];
          fa[2] = shape_data[a_idx1];
          fa[3] = shape_data[a_idx1 + 1];
          fa[4] = shape_data[a_idx2];
          fa[5] = shape_data[a_idx2 + 1];
          fa[6] = shape_data[a_idx3];
          fa[7] = shape_data[a_idx3 + 1];

          for (int i = 0; i < 8; ++i)
            {
              a[i]  = __float2half(fa[i]);
              da[i] = __float2half((fa[i] - __half2float(a[i])) * scale);
            }
#  endif
          uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
#  if ERRCOR == 1
          uint32_t const *dA = reinterpret_cast<uint32_t const *>(&da);
#  endif
          for (int cycle = 0; cycle < 2; ++cycle)
            {
              float2 c0[2];
              float2 c1[2];

              half b[4];

              for (int z = 0; z < 2; ++z)
                {
                  if constexpr (add)
                    {
                      const int c_idx0 =
                        (row * n_dofs_1d + 2 * col + cycle * 8 +
                         (z * 8 + warpId) * offset) ^
                        Util::get_base<n_dofs_1d, float>(row, z * 8 + warpId);
                      const int c_idx1 =
                        ((row + 8) * n_dofs_1d + 2 * col + cycle * 8 +
                         (z * 8 + warpId) * offset) ^
                        Util::get_base<n_dofs_1d, float>(row + 8,
                                                         z * 8 + warpId);

                      c0[z] = *((float2 *)(out + c_idx0));
                      c1[z] = *((float2 *)(out + c_idx1));
                    }
                  else
                    {
                      c0[z] = {0, 0};
                      c1[z] = {0, 0};
                    }
                }

              for (int z = 0; z < 2; ++z)
                {
                  const int b_idx0 =
                    ((col * 2) * n_dofs_1d + row + cycle * 8 +
                     (z * 8 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d, float>(col * 2, z * 8 + warpId);
                  const int b_idx1 =
                    ((col * 2 + 1) * n_dofs_1d + row + cycle * 8 +
                     (z * 8 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d, float>(col * 2 + 1,
                                                     z * 8 + warpId);
                  const int b_idx2 =
                    ((col * 2 + 8) * n_dofs_1d + row + cycle * 8 +
                     (z * 8 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d, float>(col * 2 + 8,
                                                     z * 8 + warpId);
                  const int b_idx3 =
                    ((col * 2 + 9) * n_dofs_1d + row + cycle * 8 +
                     (z * 8 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d, float>(col * 2 + 9,
                                                     z * 8 + warpId);
#  if ERRCOR == 0
                  b[0] = __float2half(in[b_idx0]);
                  b[1] = __float2half(in[b_idx1]);
                  b[2] = __float2half(in[b_idx2]);
                  b[3] = __float2half(in[b_idx3]);
#  elif ERRCOR == 1
                  float fb[4];
                  half  db[4];
                  fb[0] = in[b_idx0];
                  fb[1] = in[b_idx1];
                  fb[2] = in[b_idx2];
                  fb[3] = in[b_idx3];

                  for (int i = 0; i < 4; ++i)
                    {
                      b[i] = __float2half(fb[i]);
                      db[i] =
                        __float2half((fb[i] - __half2float(b[i])) * scale);
                    }
#  endif
                  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
#  if ERRCOR == 1
                  uint32_t const *dB = reinterpret_cast<uint32_t const *>(&db);
#  endif
                  asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(c0[z].x), "=f"(c0[z].y), "=f"(c1[z].x), "=f"(c1[z].y)
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(c0[z].x),
                      "f"(c0[z].y),
                      "f"(c1[z].x),
                      "f"(c1[z].y));
#  if ERRCOR == 1
                  float buf[4];
                  buf[0] = 0;
                  buf[1] = 0;
                  buf[2] = 0;
                  buf[3] = 0;
                  asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                    : "r"(dA[0]),
                      "r"(dA[1]),
                      "r"(dA[2]),
                      "r"(dA[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(buf[0]),
                      "f"(buf[1]),
                      "f"(buf[2]),
                      "f"(buf[3]));
                  asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(dB[0]),
                      "r"(dB[1]),
                      "f"(buf[0]),
                      "f"(buf[1]),
                      "f"(buf[2]),
                      "f"(buf[3]));
                  c0[z].x += buf[0] / scale;
                  c0[z].y += buf[1] / scale;
                  c1[z].x += buf[2] / scale;
                  c1[z].y += buf[3] / scale;
#  endif
                }

              for (int z = 0; z < 2; ++z)
                {
                  const int c_idx0 =
                    (row * n_dofs_1d + 2 * col + cycle * 8 +
                     (z * 8 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row, z * 8 + warpId);
                  const int c_idx1 =
                    ((row + 8) * n_dofs_1d + 2 * col + cycle * 8 +
                     (z * 8 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row + 8, z * 8 + warpId);

                  *((float2 *)(out + c_idx0)) = c0[z];
                  *((float2 *)(out + c_idx1)) = c1[z];
                }
            }
        }
      else
        {
          half a[8];

          const int a_idx =
            (row * n_dofs_1d + col * 2) ^ Util::get_base<n_dofs_1d, float>(row);
          const int a_idx1 = ((row + 8) * n_dofs_1d + col * 2) ^
                             Util::get_base<n_dofs_1d, float>(row + 8);
          const int a_idx2 = (row * n_dofs_1d + col * 2 + 8) ^
                             Util::get_base<n_dofs_1d, float>(row);
          const int a_idx3 = ((row + 8) * n_dofs_1d + col * 2 + 8) ^
                             Util::get_base<n_dofs_1d, float>(row + 8);

#  if ERRCOR == 0
          a[0] = __float2half(shape_data[a_idx]);
          a[1] = __float2half(shape_data[a_idx + 1]);
          a[2] = __float2half(shape_data[a_idx1]);
          a[3] = __float2half(shape_data[a_idx1 + 1]);
          a[4] = __float2half(shape_data[a_idx2]);
          a[5] = __float2half(shape_data[a_idx2 + 1]);
          a[6] = __float2half(shape_data[a_idx3]);
          a[7] = __float2half(shape_data[a_idx3 + 1]);
#  elif ERRCOR == 1
          float fa[8];
          half  da[8];
          fa[0] = shape_data[a_idx];
          fa[1] = shape_data[a_idx + 1];
          fa[2] = shape_data[a_idx1];
          fa[3] = shape_data[a_idx1 + 1];
          fa[4] = shape_data[a_idx2];
          fa[5] = shape_data[a_idx2 + 1];
          fa[6] = shape_data[a_idx3];
          fa[7] = shape_data[a_idx3 + 1];

          for (int i = 0; i < 8; ++i)
            {
              a[i]  = __float2half(fa[i]);
              da[i] = __float2half((fa[i] - __half2float(a[i])) * scale);
            }
#  endif
          uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
#  if ERRCOR == 1
          uint32_t const *dA = reinterpret_cast<uint32_t const *>(&da);
#  endif
          for (int cycle = 0; cycle < 2; ++cycle)
            {
              float2 c0[2];
              float2 c1[2];

              half b[4];

              for (int z = 0; z < 2; ++z)
                {
                  if constexpr (add)
                    {
                      const int c_idx0 =
                        ((z * 8 + warpId) * n_dofs_1d + 2 * col + cycle * 8 +
                         row * offset) ^
                        Util::get_base<n_dofs_1d, float>((z * 8 + warpId), row);
                      const int c_idx1 =
                        ((z * 8 + warpId) * n_dofs_1d + 2 * col + cycle * 8 +
                         (row + 8) * offset) ^
                        Util::get_base<n_dofs_1d, float>(z * 8 + warpId,
                                                         row + 8);

                      c0[z] = *((float2 *)(out + c_idx0));
                      c1[z] = *((float2 *)(out + c_idx1));
                    }
                  else
                    {
                      c0[z] = {0, 0};
                      c1[z] = {0, 0};
                    }
                }

              for (int z = 0; z < 2; ++z)
                {
                  const int b_idx0 =
                    ((z * 8 + warpId) * n_dofs_1d + row + cycle * 8 +
                     (col * 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(z * 8 + warpId, col * 2);
                  const int b_idx1 =
                    ((z * 8 + warpId) * n_dofs_1d + row + cycle * 8 +
                     (col * 2 + 1) * offset) ^
                    Util::get_base<n_dofs_1d, float>(z * 8 + warpId,
                                                     col * 2 + 1);
                  const int b_idx2 =
                    ((z * 8 + warpId) * n_dofs_1d + row + cycle * 8 +
                     (col * 2 + 8) * offset) ^
                    Util::get_base<n_dofs_1d, float>(z * 8 + warpId,
                                                     col * 2 + 8);
                  const int b_idx3 =
                    ((z * 8 + warpId) * n_dofs_1d + row + cycle * 8 +
                     (col * 2 + 9) * offset) ^
                    Util::get_base<n_dofs_1d, float>(z * 8 + warpId,
                                                     col * 2 + 9);
#  if ERRCOR == 0
                  b[0] = __float2half(in[b_idx0]);
                  b[1] = __float2half(in[b_idx1]);
                  b[2] = __float2half(in[b_idx2]);
                  b[3] = __float2half(in[b_idx3]);
#  elif ERRCOR == 1
                  float fb[4];
                  half  db[4];
                  fb[0] = in[b_idx0];
                  fb[1] = in[b_idx1];
                  fb[2] = in[b_idx2];
                  fb[3] = in[b_idx3];

                  for (int i = 0; i < 4; ++i)
                    {
                      b[i] = __float2half(fb[i]);
                      db[i] =
                        __float2half((fb[i] - __half2float(b[i])) * scale);
                    }
#  endif
                  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
#  if ERRCOR == 1
                  uint32_t const *dB = reinterpret_cast<uint32_t const *>(&db);
#  endif
                  asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(c0[z].x), "=f"(c0[z].y), "=f"(c1[z].x), "=f"(c1[z].y)
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(c0[z].x),
                      "f"(c0[z].y),
                      "f"(c1[z].x),
                      "f"(c1[z].y));
#  if ERRCOR == 1
                  float buf[4];
                  buf[0] = 0;
                  buf[1] = 0;
                  buf[2] = 0;
                  buf[3] = 0;
                  asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                    : "r"(dA[0]),
                      "r"(dA[1]),
                      "r"(dA[2]),
                      "r"(dA[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(buf[0]),
                      "f"(buf[1]),
                      "f"(buf[2]),
                      "f"(buf[3]));
                  asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(dB[0]),
                      "r"(dB[1]),
                      "f"(buf[0]),
                      "f"(buf[1]),
                      "f"(buf[2]),
                      "f"(buf[3]));
                  c0[z].x += buf[0] / scale;
                  c0[z].y += buf[1] / scale;
                  c1[z].x += buf[2] / scale;
                  c1[z].y += buf[3] / scale;
#  endif
                }

              for (int z = 0; z < 2; ++z)
                {
                  const int c_idx0 =
                    ((z * 8 + warpId) * n_dofs_1d + 2 * col + cycle * 8 +
                     row * offset) ^
                    Util::get_base<n_dofs_1d, float>((z * 8 + warpId), row);
                  const int c_idx1 =
                    ((z * 8 + warpId) * n_dofs_1d + 2 * col + cycle * 8 +
                     (row + 8) * offset) ^
                    Util::get_base<n_dofs_1d, float>(z * 8 + warpId, row + 8);

                  *((float2 *)(out + c_idx0)) = c0[z];
                  *((float2 *)(out + c_idx1)) = c1[z];
                }
            }
        }
    }
  };
#endif


#if MMAKERNEL == 7
  template <typename T>
  struct TPEvaluatorBase<T, 16, float, LaplaceVariant::TensorCoreMMA, 3, half>
  {
    using Number  = float;
    using Number2 = half;

    static constexpr int n_dofs_1d = 16;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number        *dst,
          const Number  *src,
          const Number2 *mass_matrix,
          const Number2 *derivative_matrix,
          Number        *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number2 *shape_data, const Number *in, Number *out)
    {
      const int warpId = threadIdx.y / 2;

      const int tid   = (threadIdx.y * n_dofs_1d + threadIdx.x) & 31;
      const int rowId = tid / 8;
      const int colId = tid & 7;

      const int row = tid / 4;
      const int col = tid & 3;

      constexpr int offset = n_dofs_1d * n_dofs_1d;

#  if ERRCOR == 1
      constexpr int shift = n_dofs_1d * n_dofs_1d * 3;
      constexpr int scale = 1 << 11;
#  endif

      if constexpr (direction == 0)
        {
          for (int cycle = 0; cycle < 2; ++cycle)
            {
              float2 c0[2];
              float2 c1[2];

              half a[8];
              half b[4];

              for (int z = 0; z < 2; ++z)
                {
                  c0[z] = {0, 0};
                  c1[z] = {0, 0};
                }

              const int bb_idx =
                ((colId + cycle * 8) * n_dofs_1d + (rowId & 1) * 8) ^
                Util::get_base<n_dofs_1d, half>(colId + cycle * 8);

              auto smem_ptr = static_cast<uint32_t>(
                __cvta_generic_to_shared(&shape_data[bb_idx]));

              float *B_ptr = reinterpret_cast<float *>(&b);
              asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 "
                           "{%0, %1}, [%2]; "
                           : "=f"(B_ptr[0]), "=f"(B_ptr[1])
                           : "r"(smem_ptr));
#  if ERRCOR == 1
              half db[4];

              auto smem_dptr = static_cast<uint32_t>(
                __cvta_generic_to_shared(&shape_data[bb_idx + shift]));

              float *dB_ptr = reinterpret_cast<float *>(&db);
              asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 "
                           "{%0, %1}, [%2]; "
                           : "=f"(dB_ptr[0]), "=f"(dB_ptr[1])
                           : "r"(smem_dptr));
#  endif
              uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
#  if ERRCOR == 1
              uint32_t const *dB = reinterpret_cast<uint32_t const *>(&db);
#  endif
              for (int z = 0; z < 2; ++z)
                {
                  const int a_idx =
                    (row * n_dofs_1d + col * 2 + (z * 8 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row, z * 8 + warpId);
                  const int a_idx1 =
                    ((row + 8) * n_dofs_1d + col * 2 +
                     (z * 8 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row + 8, z * 8 + warpId);
                  const int a_idx2 =
                    (row * n_dofs_1d + col * 2 + 8 +
                     (z * 8 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row, z * 8 + warpId);
                  const int a_idx3 =
                    ((row + 8) * n_dofs_1d + col * 2 + 8 +
                     (z * 8 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row + 8, z * 8 + warpId);
#  if ERRCOR == 0
                  a[0] = __float2half(in[a_idx]);
                  a[1] = __float2half(in[a_idx + 1]);
                  a[2] = __float2half(in[a_idx1]);
                  a[3] = __float2half(in[a_idx1 + 1]);
                  a[4] = __float2half(in[a_idx2]);
                  a[5] = __float2half(in[a_idx2 + 1]);
                  a[6] = __float2half(in[a_idx3]);
                  a[7] = __float2half(in[a_idx3 + 1]);
#  elif ERRCOR == 1
                  float2 fa[4];
                  half   da[8];
                  fa[0] = *((float2 *)(in + a_idx));
                  fa[1] = *((float2 *)(in + a_idx1));
                  fa[2] = *((float2 *)(in + a_idx2));
                  fa[3] = *((float2 *)(in + a_idx3));

                  for (int i = 0; i < 4; ++i)
                    {
                      a[i * 2]     = __float2half(fa[i].x);
                      a[i * 2 + 1] = __float2half(fa[i].y);
                      da[i * 2]    = __float2half(
                        (fa[i].x - __half2float(a[i * 2])) * scale);
                      da[i * 2 + 1] = __float2half(
                        (fa[i].y - __half2float(a[i * 2 + 1])) * scale);
                    }
#  endif
                  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
#  if ERRCOR == 1
                  uint32_t const *dA = reinterpret_cast<uint32_t const *>(&da);
#  endif

#  if ERRCOR == 1
                  float buf[4];
                  buf[0] = 0;
                  buf[1] = 0;
                  buf[2] = 0;
                  buf[3] = 0;
                  asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                    : "r"(dA[0]),
                      "r"(dA[1]),
                      "r"(dA[2]),
                      "r"(dA[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(buf[0]),
                      "f"(buf[1]),
                      "f"(buf[2]),
                      "f"(buf[3]));
                  asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(dB[0]),
                      "r"(dB[1]),
                      "f"(buf[0]),
                      "f"(buf[1]),
                      "f"(buf[2]),
                      "f"(buf[3]));
                  c0[z].x += buf[0] / scale;
                  c0[z].y += buf[1] / scale;
                  c1[z].x += buf[2] / scale;
                  c1[z].y += buf[3] / scale;
#  endif
                  asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(c0[z].x), "=f"(c0[z].y), "=f"(c1[z].x), "=f"(c1[z].y)
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(c0[z].x),
                      "f"(c0[z].y),
                      "f"(c1[z].x),
                      "f"(c1[z].y));
                }

              for (int z = 0; z < 2; ++z)
                {
                  const int c_idx0 =
                    (row * n_dofs_1d + 2 * col + cycle * 8 +
                     (z * 8 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row, z * 8 + warpId);
                  const int c_idx1 =
                    ((row + 8) * n_dofs_1d + 2 * col + cycle * 8 +
                     (z * 8 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row + 8, z * 8 + warpId);

                  *((float2 *)(out + c_idx0)) = c0[z];
                  *((float2 *)(out + c_idx1)) = c1[z];
                }
            }
        }
      else if (direction == 1)
        {
          half a[8];

          const int aa_idx =
            ((colId + (rowId & 1) * 8) * n_dofs_1d + (rowId / 2) * 8) ^
            Util::get_base<n_dofs_1d, half>(colId + (rowId & 1) * 8);

          auto smem_ptr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&shape_data[aa_idx]));

          float *A_ptr = reinterpret_cast<float *>(&a);
          asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                       "{%0, %1, %2, %3}, [%4]; "
                       : "=f"(A_ptr[0]),
                         "=f"(A_ptr[1]),
                         "=f"(A_ptr[2]),
                         "=f"(A_ptr[3])
                       : "r"(smem_ptr));
#  if ERRCOR == 1
          half da[8];

          auto smem_dptr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&shape_data[aa_idx + shift]));

          float *dA_ptr = reinterpret_cast<float *>(&da);
          asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                       "{%0, %1, %2, %3}, [%4]; "
                       : "=f"(dA_ptr[0]),
                         "=f"(dA_ptr[1]),
                         "=f"(dA_ptr[2]),
                         "=f"(dA_ptr[3])
                       : "r"(smem_dptr));
#  endif
          uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
#  if ERRCOR == 1
          uint32_t const *dA = reinterpret_cast<uint32_t const *>(&da);
#  endif
          for (int cycle = 0; cycle < 2; ++cycle)
            {
              float2 c0[2];
              float2 c1[2];

              half b[4];

              for (int z = 0; z < 2; ++z)
                {
                  if constexpr (add)
                    {
                      const int c_idx0 =
                        (row * n_dofs_1d + 2 * col + cycle * 8 +
                         (z * 8 + warpId) * offset) ^
                        Util::get_base<n_dofs_1d, float>(row, z * 8 + warpId);
                      const int c_idx1 =
                        ((row + 8) * n_dofs_1d + 2 * col + cycle * 8 +
                         (z * 8 + warpId) * offset) ^
                        Util::get_base<n_dofs_1d, float>(row + 8,
                                                         z * 8 + warpId);

                      c0[z] = *((float2 *)(out + c_idx0));
                      c1[z] = *((float2 *)(out + c_idx1));
                    }
                  else
                    {
                      c0[z] = {0, 0};
                      c1[z] = {0, 0};
                    }
                }

              for (int z = 0; z < 2; ++z)
                {
                  const int b_idx0 =
                    ((col * 2) * n_dofs_1d + row + cycle * 8 +
                     (z * 8 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d, float>(col * 2, z * 8 + warpId);
                  const int b_idx1 =
                    ((col * 2 + 1) * n_dofs_1d + row + cycle * 8 +
                     (z * 8 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d, float>(col * 2 + 1,
                                                     z * 8 + warpId);
                  const int b_idx2 =
                    ((col * 2 + 8) * n_dofs_1d + row + cycle * 8 +
                     (z * 8 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d, float>(col * 2 + 8,
                                                     z * 8 + warpId);
                  const int b_idx3 =
                    ((col * 2 + 9) * n_dofs_1d + row + cycle * 8 +
                     (z * 8 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d, float>(col * 2 + 9,
                                                     z * 8 + warpId);
#  if ERRCOR == 0
                  b[0] = __float2half(in[b_idx0]);
                  b[1] = __float2half(in[b_idx1]);
                  b[2] = __float2half(in[b_idx2]);
                  b[3] = __float2half(in[b_idx3]);
#  elif ERRCOR == 1
                  float fb[4];
                  half  db[4];
                  fb[0] = in[b_idx0];
                  fb[1] = in[b_idx1];
                  fb[2] = in[b_idx2];
                  fb[3] = in[b_idx3];

                  for (int i = 0; i < 4; ++i)
                    {
                      b[i] = __float2half(fb[i]);
                      db[i] =
                        __float2half((fb[i] - __half2float(b[i])) * scale);
                    }
#  endif
                  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
#  if ERRCOR == 1
                  uint32_t const *dB = reinterpret_cast<uint32_t const *>(&db);
#  endif

#  if ERRCOR == 1
                  float buf[4];
                  buf[0] = 0;
                  buf[1] = 0;
                  buf[2] = 0;
                  buf[3] = 0;
                  asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                    : "r"(dA[0]),
                      "r"(dA[1]),
                      "r"(dA[2]),
                      "r"(dA[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(buf[0]),
                      "f"(buf[1]),
                      "f"(buf[2]),
                      "f"(buf[3]));
                  asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(dB[0]),
                      "r"(dB[1]),
                      "f"(buf[0]),
                      "f"(buf[1]),
                      "f"(buf[2]),
                      "f"(buf[3]));
                  c0[z].x += buf[0] / scale;
                  c0[z].y += buf[1] / scale;
                  c1[z].x += buf[2] / scale;
                  c1[z].y += buf[3] / scale;
#  endif
                  asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(c0[z].x), "=f"(c0[z].y), "=f"(c1[z].x), "=f"(c1[z].y)
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(c0[z].x),
                      "f"(c0[z].y),
                      "f"(c1[z].x),
                      "f"(c1[z].y));
                }

              for (int z = 0; z < 2; ++z)
                {
                  const int c_idx0 =
                    (row * n_dofs_1d + 2 * col + cycle * 8 +
                     (z * 8 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row, z * 8 + warpId);
                  const int c_idx1 =
                    ((row + 8) * n_dofs_1d + 2 * col + cycle * 8 +
                     (z * 8 + warpId) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row + 8, z * 8 + warpId);

                  *((float2 *)(out + c_idx0)) = c0[z];
                  *((float2 *)(out + c_idx1)) = c1[z];
                }
            }
        }
      else
        {
          half a[8];

          const int aa_idx =
            ((colId + (rowId & 1) * 8) * n_dofs_1d + (rowId / 2) * 8) ^
            Util::get_base<n_dofs_1d, half>(colId + (rowId & 1) * 8);

          auto smem_ptr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&shape_data[aa_idx]));

          float *A_ptr = reinterpret_cast<float *>(&a);
          asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                       "{%0, %1, %2, %3}, [%4]; "
                       : "=f"(A_ptr[0]),
                         "=f"(A_ptr[1]),
                         "=f"(A_ptr[2]),
                         "=f"(A_ptr[3])
                       : "r"(smem_ptr));
#  if ERRCOR == 1
          half da[8];

          auto smem_dptr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&shape_data[aa_idx + shift]));

          float *dA_ptr = reinterpret_cast<float *>(&da);
          asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                       "{%0, %1, %2, %3}, [%4]; "
                       : "=f"(dA_ptr[0]),
                         "=f"(dA_ptr[1]),
                         "=f"(dA_ptr[2]),
                         "=f"(dA_ptr[3])
                       : "r"(smem_dptr));
#  endif
          uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
#  if ERRCOR == 1
          uint32_t const *dA = reinterpret_cast<uint32_t const *>(&da);
#  endif
          for (int cycle = 0; cycle < 2; ++cycle)
            {
              float2 c0[2];
              float2 c1[2];

              half b[4];

              for (int z = 0; z < 2; ++z)
                {
                  if constexpr (add)
                    {
                      const int c_idx0 =
                        ((z * 8 + warpId) * n_dofs_1d + 2 * col + cycle * 8 +
                         row * offset) ^
                        Util::get_base<n_dofs_1d, float>((z * 8 + warpId), row);
                      const int c_idx1 =
                        ((z * 8 + warpId) * n_dofs_1d + 2 * col + cycle * 8 +
                         (row + 8) * offset) ^
                        Util::get_base<n_dofs_1d, float>(z * 8 + warpId,
                                                         row + 8);

                      c0[z] = *((float2 *)(out + c_idx0));
                      c1[z] = *((float2 *)(out + c_idx1));
                    }
                  else
                    {
                      c0[z] = {0, 0};
                      c1[z] = {0, 0};
                    }
                }

              for (int z = 0; z < 2; ++z)
                {
                  const int b_idx0 =
                    ((z * 8 + warpId) * n_dofs_1d + row + cycle * 8 +
                     (col * 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(z * 8 + warpId, col * 2);
                  const int b_idx1 =
                    ((z * 8 + warpId) * n_dofs_1d + row + cycle * 8 +
                     (col * 2 + 1) * offset) ^
                    Util::get_base<n_dofs_1d, float>(z * 8 + warpId,
                                                     col * 2 + 1);
                  const int b_idx2 =
                    ((z * 8 + warpId) * n_dofs_1d + row + cycle * 8 +
                     (col * 2 + 8) * offset) ^
                    Util::get_base<n_dofs_1d, float>(z * 8 + warpId,
                                                     col * 2 + 8);
                  const int b_idx3 =
                    ((z * 8 + warpId) * n_dofs_1d + row + cycle * 8 +
                     (col * 2 + 9) * offset) ^
                    Util::get_base<n_dofs_1d, float>(z * 8 + warpId,
                                                     col * 2 + 9);
#  if ERRCOR == 0
                  b[0] = __float2half(in[b_idx0]);
                  b[1] = __float2half(in[b_idx1]);
                  b[2] = __float2half(in[b_idx2]);
                  b[3] = __float2half(in[b_idx3]);
#  elif ERRCOR == 1
                  float fb[4];
                  half  db[4];
                  fb[0] = in[b_idx0];
                  fb[1] = in[b_idx1];
                  fb[2] = in[b_idx2];
                  fb[3] = in[b_idx3];

                  for (int i = 0; i < 4; ++i)
                    {
                      b[i] = __float2half(fb[i]);
                      db[i] =
                        __float2half((fb[i] - __half2float(b[i])) * scale);
                    }
#  endif
                  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
#  if ERRCOR == 1
                  uint32_t const *dB = reinterpret_cast<uint32_t const *>(&db);
#  endif

#  if ERRCOR == 1
                  float buf[4];
                  buf[0] = 0;
                  buf[1] = 0;
                  buf[2] = 0;
                  buf[3] = 0;
                  asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                    : "r"(dA[0]),
                      "r"(dA[1]),
                      "r"(dA[2]),
                      "r"(dA[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(buf[0]),
                      "f"(buf[1]),
                      "f"(buf[2]),
                      "f"(buf[3]));
                  asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(dB[0]),
                      "r"(dB[1]),
                      "f"(buf[0]),
                      "f"(buf[1]),
                      "f"(buf[2]),
                      "f"(buf[3]));
                  c0[z].x += buf[0] / scale;
                  c0[z].y += buf[1] / scale;
                  c1[z].x += buf[2] / scale;
                  c1[z].y += buf[3] / scale;
#  endif
                  asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(c0[z].x), "=f"(c0[z].y), "=f"(c1[z].x), "=f"(c1[z].y)
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(c0[z].x),
                      "f"(c0[z].y),
                      "f"(c1[z].x),
                      "f"(c1[z].y));
                }

              for (int z = 0; z < 2; ++z)
                {
                  const int c_idx0 =
                    ((z * 8 + warpId) * n_dofs_1d + 2 * col + cycle * 8 +
                     row * offset) ^
                    Util::get_base<n_dofs_1d, float>((z * 8 + warpId), row);
                  const int c_idx1 =
                    ((z * 8 + warpId) * n_dofs_1d + 2 * col + cycle * 8 +
                     (row + 8) * offset) ^
                    Util::get_base<n_dofs_1d, float>(z * 8 + warpId, row + 8);

                  *((float2 *)(out + c_idx0)) = c0[z];
                  *((float2 *)(out + c_idx1)) = c1[z];
                }
            }
        }
    }
  };
#endif



#if MMAKERNEL == 8
  template <typename T>
  struct TPEvaluatorBase<T, 16, float, LaplaceVariant::TensorCoreMMA, 3, half>
  {
    using Number  = float;
    using Number2 = half;

    static constexpr int n_dofs_1d = 16;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number        *dst,
          const Number  *src,
          const Number2 *mass_matrix,
          const Number2 *derivative_matrix,
          Number        *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number2 *shape_data, const Number *in, Number *out)
    {
      const int warpId = threadIdx.y / 2;

      const int tid   = (threadIdx.y * n_dofs_1d + threadIdx.x) & 31;
      const int rowId = tid / 8;
      const int colId = tid & 7;

      const int row = tid / 4;
      const int col = tid & 3;

      constexpr int offset = n_dofs_1d * n_dofs_1d;

#  if ERRCOR == 1
      constexpr int shift = n_dofs_1d * n_dofs_1d * 3;
      constexpr int scale = 1 << 11;
#  endif

      if constexpr (direction == 0)
        {
          for (int cycle = 0; cycle < 2; ++cycle)
            {
              float2 c0 = {0, 0};
              float2 c1 = {0, 0};

              half a[8];
              half b[4];

              const int bb_idx =
                ((colId + cycle * 8) * n_dofs_1d + (rowId & 1) * 8) ^
                Util::get_base<n_dofs_1d, half>(colId + cycle * 8);

              auto smem_ptr = static_cast<uint32_t>(
                __cvta_generic_to_shared(&shape_data[bb_idx]));

              float *B_ptr = reinterpret_cast<float *>(&b);
              asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 "
                           "{%0, %1}, [%2]; "
                           : "=f"(B_ptr[0]), "=f"(B_ptr[1])
                           : "r"(smem_ptr));
#  if ERRCOR == 1
              half db[4];

              auto smem_dptr = static_cast<uint32_t>(
                __cvta_generic_to_shared(&shape_data[bb_idx + shift]));

              float *dB_ptr = reinterpret_cast<float *>(&db);
              asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 "
                           "{%0, %1}, [%2]; "
                           : "=f"(dB_ptr[0]), "=f"(dB_ptr[1])
                           : "r"(smem_dptr));
#  endif
              uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
#  if ERRCOR == 1
              uint32_t const *dB = reinterpret_cast<uint32_t const *>(&db);
#  endif

              const int a_idx = (row * n_dofs_1d + col * 2 + warpId * offset) ^
                                Util::get_base<n_dofs_1d, float>(row, warpId);
              const int a_idx1 =
                ((row + 8) * n_dofs_1d + col * 2 + warpId * offset) ^
                Util::get_base<n_dofs_1d, float>(row + 8, warpId);
              const int a_idx2 =
                (row * n_dofs_1d + col * 2 + 8 + warpId * offset) ^
                Util::get_base<n_dofs_1d, float>(row, warpId);
              const int a_idx3 =
                ((row + 8) * n_dofs_1d + col * 2 + 8 + warpId * offset) ^
                Util::get_base<n_dofs_1d, float>(row + 8, warpId);
#  if ERRCOR == 0
              a[0] = __float2half(in[a_idx]);
              a[1] = __float2half(in[a_idx + 1]);
              a[2] = __float2half(in[a_idx1]);
              a[3] = __float2half(in[a_idx1 + 1]);
              a[4] = __float2half(in[a_idx2]);
              a[5] = __float2half(in[a_idx2 + 1]);
              a[6] = __float2half(in[a_idx3]);
              a[7] = __float2half(in[a_idx3 + 1]);
#  elif ERRCOR == 1
              float2 fa[4];
              half   da[8];
              fa[0] = *((float2 *)(in + a_idx));
              fa[1] = *((float2 *)(in + a_idx1));
              fa[2] = *((float2 *)(in + a_idx2));
              fa[3] = *((float2 *)(in + a_idx3));

              for (int i = 0; i < 4; ++i)
                {
                  a[i * 2]     = __float2half(fa[i].x);
                  a[i * 2 + 1] = __float2half(fa[i].y);
                  da[i * 2] =
                    __float2half((fa[i].x - __half2float(a[i * 2])) * scale);
                  da[i * 2 + 1] = __float2half(
                    (fa[i].y - __half2float(a[i * 2 + 1])) * scale);
                }
#  endif
              uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
#  if ERRCOR == 1
              uint32_t const *dA = reinterpret_cast<uint32_t const *>(&da);
#  endif

#  if ERRCOR == 1
              float buf[4];
              buf[0] = 0;
              buf[1] = 0;
              buf[2] = 0;
              buf[3] = 0;
              asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                : "r"(dA[0]),
                  "r"(dA[1]),
                  "r"(dA[2]),
                  "r"(dA[3]),
                  "r"(B[0]),
                  "r"(B[1]),
                  "f"(buf[0]),
                  "f"(buf[1]),
                  "f"(buf[2]),
                  "f"(buf[3]));
              asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                : "r"(A[0]),
                  "r"(A[1]),
                  "r"(A[2]),
                  "r"(A[3]),
                  "r"(dB[0]),
                  "r"(dB[1]),
                  "f"(buf[0]),
                  "f"(buf[1]),
                  "f"(buf[2]),
                  "f"(buf[3]));
              c0.x += buf[0] / scale;
              c0.y += buf[1] / scale;
              c1.x += buf[2] / scale;
              c1.y += buf[3] / scale;
#  endif
              asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(c0.x), "=f"(c0.y), "=f"(c1.x), "=f"(c1.y)
                : "r"(A[0]),
                  "r"(A[1]),
                  "r"(A[2]),
                  "r"(A[3]),
                  "r"(B[0]),
                  "r"(B[1]),
                  "f"(c0.x),
                  "f"(c0.y),
                  "f"(c1.x),
                  "f"(c1.y));

              const int c_idx0 =
                (row * n_dofs_1d + 2 * col + cycle * 8 + warpId * offset) ^
                Util::get_base<n_dofs_1d, float>(row, warpId);
              const int c_idx1 =
                ((row + 8) * n_dofs_1d + 2 * col + cycle * 8 +
                 warpId * offset) ^
                Util::get_base<n_dofs_1d, float>(row + 8, warpId);

              *((float2 *)(out + c_idx0)) = c0;
              *((float2 *)(out + c_idx1)) = c1;
            }
        }
      else if (direction == 1)
        {
          half a[8];

          const int aa_idx =
            ((colId + (rowId & 1) * 8) * n_dofs_1d + (rowId / 2) * 8) ^
            Util::get_base<n_dofs_1d, half>(colId + (rowId & 1) * 8);

          auto smem_ptr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&shape_data[aa_idx]));

          float *A_ptr = reinterpret_cast<float *>(&a);
          asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                       "{%0, %1, %2, %3}, [%4]; "
                       : "=f"(A_ptr[0]),
                         "=f"(A_ptr[1]),
                         "=f"(A_ptr[2]),
                         "=f"(A_ptr[3])
                       : "r"(smem_ptr));
#  if ERRCOR == 1
          half da[8];

          auto smem_dptr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&shape_data[aa_idx + shift]));

          float *dA_ptr = reinterpret_cast<float *>(&da);
          asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                       "{%0, %1, %2, %3}, [%4]; "
                       : "=f"(dA_ptr[0]),
                         "=f"(dA_ptr[1]),
                         "=f"(dA_ptr[2]),
                         "=f"(dA_ptr[3])
                       : "r"(smem_dptr));
#  endif
          uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
#  if ERRCOR == 1
          uint32_t const *dA = reinterpret_cast<uint32_t const *>(&da);
#  endif
          for (int cycle = 0; cycle < 2; ++cycle)
            {
              float2 c0 = {0, 0};
              float2 c1 = {0, 0};

              half b[4];
              if constexpr (add)
                {
                  const int c_idx0 =
                    (row * n_dofs_1d + 2 * col + cycle * 8 + warpId * offset) ^
                    Util::get_base<n_dofs_1d, float>(row, warpId);
                  const int c_idx1 =
                    ((row + 8) * n_dofs_1d + 2 * col + cycle * 8 +
                     warpId * offset) ^
                    Util::get_base<n_dofs_1d, float>(row + 8, warpId);

                  c0 = *((float2 *)(out + c_idx0));
                  c1 = *((float2 *)(out + c_idx1));
                }

              const int b_idx0 =
                ((col * 2) * n_dofs_1d + row + cycle * 8 + warpId * offset) ^
                Util::get_base<n_dofs_1d, float>(col * 2, warpId);
              const int b_idx1 =
                ((col * 2 + 1) * n_dofs_1d + row + cycle * 8 +
                 warpId * offset) ^
                Util::get_base<n_dofs_1d, float>(col * 2 + 1, warpId);
              const int b_idx2 =
                ((col * 2 + 8) * n_dofs_1d + row + cycle * 8 +
                 warpId * offset) ^
                Util::get_base<n_dofs_1d, float>(col * 2 + 8, warpId);
              const int b_idx3 =
                ((col * 2 + 9) * n_dofs_1d + row + cycle * 8 +
                 warpId * offset) ^
                Util::get_base<n_dofs_1d, float>(col * 2 + 9, warpId);
#  if ERRCOR == 0
              b[0] = __float2half(in[b_idx0]);
              b[1] = __float2half(in[b_idx1]);
              b[2] = __float2half(in[b_idx2]);
              b[3] = __float2half(in[b_idx3]);
#  elif ERRCOR == 1
              float fb[4];
              half  db[4];
              fb[0] = in[b_idx0];
              fb[1] = in[b_idx1];
              fb[2] = in[b_idx2];
              fb[3] = in[b_idx3];

              for (int i = 0; i < 4; ++i)
                {
                  b[i]  = __float2half(fb[i]);
                  db[i] = __float2half((fb[i] - __half2float(b[i])) * scale);
                }
#  endif
              uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
#  if ERRCOR == 1
              uint32_t const *dB = reinterpret_cast<uint32_t const *>(&db);
#  endif

#  if ERRCOR == 1
              float buf[4];
              buf[0] = 0;
              buf[1] = 0;
              buf[2] = 0;
              buf[3] = 0;
              asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                : "r"(dA[0]),
                  "r"(dA[1]),
                  "r"(dA[2]),
                  "r"(dA[3]),
                  "r"(B[0]),
                  "r"(B[1]),
                  "f"(buf[0]),
                  "f"(buf[1]),
                  "f"(buf[2]),
                  "f"(buf[3]));
              asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                : "r"(A[0]),
                  "r"(A[1]),
                  "r"(A[2]),
                  "r"(A[3]),
                  "r"(dB[0]),
                  "r"(dB[1]),
                  "f"(buf[0]),
                  "f"(buf[1]),
                  "f"(buf[2]),
                  "f"(buf[3]));
              c0.x += buf[0] / scale;
              c0.y += buf[1] / scale;
              c1.x += buf[2] / scale;
              c1.y += buf[3] / scale;
#  endif
              asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(c0.x), "=f"(c0.y), "=f"(c1.x), "=f"(c1.y)
                : "r"(A[0]),
                  "r"(A[1]),
                  "r"(A[2]),
                  "r"(A[3]),
                  "r"(B[0]),
                  "r"(B[1]),
                  "f"(c0.x),
                  "f"(c0.y),
                  "f"(c1.x),
                  "f"(c1.y));

              const int c_idx0 =
                (row * n_dofs_1d + 2 * col + cycle * 8 + warpId * offset) ^
                Util::get_base<n_dofs_1d, float>(row, warpId);
              const int c_idx1 =
                ((row + 8) * n_dofs_1d + 2 * col + cycle * 8 +
                 warpId * offset) ^
                Util::get_base<n_dofs_1d, float>(row + 8, warpId);

              *((float2 *)(out + c_idx0)) = c0;
              *((float2 *)(out + c_idx1)) = c1;
            }
        }
      else
        {
          half a[8];

          const int aa_idx =
            ((colId + (rowId & 1) * 8) * n_dofs_1d + (rowId / 2) * 8) ^
            Util::get_base<n_dofs_1d, half>(colId + (rowId & 1) * 8);

          auto smem_ptr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&shape_data[aa_idx]));

          float *A_ptr = reinterpret_cast<float *>(&a);
          asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                       "{%0, %1, %2, %3}, [%4]; "
                       : "=f"(A_ptr[0]),
                         "=f"(A_ptr[1]),
                         "=f"(A_ptr[2]),
                         "=f"(A_ptr[3])
                       : "r"(smem_ptr));
#  if ERRCOR == 1
          half da[8];

          auto smem_dptr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&shape_data[aa_idx + shift]));

          float *dA_ptr = reinterpret_cast<float *>(&da);
          asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                       "{%0, %1, %2, %3}, [%4]; "
                       : "=f"(dA_ptr[0]),
                         "=f"(dA_ptr[1]),
                         "=f"(dA_ptr[2]),
                         "=f"(dA_ptr[3])
                       : "r"(smem_dptr));
#  endif
          uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
#  if ERRCOR == 1
          uint32_t const *dA = reinterpret_cast<uint32_t const *>(&da);
#  endif
          for (int cycle = 0; cycle < 2; ++cycle)
            {
              float2 c0 = {0, 0};
              float2 c1 = {0, 0};

              half b[4];

              if constexpr (add)
                {
                  const int c_idx0 =
                    (warpId * n_dofs_1d + 2 * col + cycle * 8 + row * offset) ^
                    Util::get_base<n_dofs_1d, float>(warpId, row);
                  const int c_idx1 =
                    (warpId * n_dofs_1d + 2 * col + cycle * 8 +
                     (row + 8) * offset) ^
                    Util::get_base<n_dofs_1d, float>(warpId, row + 8);

                  c0 = *((float2 *)(out + c_idx0));
                  c1 = *((float2 *)(out + c_idx1));
                }

              const int b_idx0 =
                (warpId * n_dofs_1d + row + cycle * 8 + (col * 2) * offset) ^
                Util::get_base<n_dofs_1d, float>(warpId, col * 2);
              const int b_idx1 =
                (warpId * n_dofs_1d + row + cycle * 8 +
                 (col * 2 + 1) * offset) ^
                Util::get_base<n_dofs_1d, float>(warpId, col * 2 + 1);
              const int b_idx2 =
                (warpId * n_dofs_1d + row + cycle * 8 +
                 (col * 2 + 8) * offset) ^
                Util::get_base<n_dofs_1d, float>(warpId, col * 2 + 8);
              const int b_idx3 =
                (warpId * n_dofs_1d + row + cycle * 8 +
                 (col * 2 + 9) * offset) ^
                Util::get_base<n_dofs_1d, float>(warpId, col * 2 + 9);
#  if ERRCOR == 0
              b[0] = __float2half(in[b_idx0]);
              b[1] = __float2half(in[b_idx1]);
              b[2] = __float2half(in[b_idx2]);
              b[3] = __float2half(in[b_idx3]);
#  elif ERRCOR == 1
              float fb[4];
              half  db[4];
              fb[0] = in[b_idx0];
              fb[1] = in[b_idx1];
              fb[2] = in[b_idx2];
              fb[3] = in[b_idx3];

              for (int i = 0; i < 4; ++i)
                {
                  b[i]  = __float2half(fb[i]);
                  db[i] = __float2half((fb[i] - __half2float(b[i])) * scale);
                }
#  endif
              uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
#  if ERRCOR == 1
              uint32_t const *dB = reinterpret_cast<uint32_t const *>(&db);
#  endif

#  if ERRCOR == 1
              float buf[4];
              buf[0] = 0;
              buf[1] = 0;
              buf[2] = 0;
              buf[3] = 0;
              asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                : "r"(dA[0]),
                  "r"(dA[1]),
                  "r"(dA[2]),
                  "r"(dA[3]),
                  "r"(B[0]),
                  "r"(B[1]),
                  "f"(buf[0]),
                  "f"(buf[1]),
                  "f"(buf[2]),
                  "f"(buf[3]));
              asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                : "r"(A[0]),
                  "r"(A[1]),
                  "r"(A[2]),
                  "r"(A[3]),
                  "r"(dB[0]),
                  "r"(dB[1]),
                  "f"(buf[0]),
                  "f"(buf[1]),
                  "f"(buf[2]),
                  "f"(buf[3]));
              c0.x += buf[0] / scale;
              c0.y += buf[1] / scale;
              c1.x += buf[2] / scale;
              c1.y += buf[3] / scale;
#  endif
              asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(c0.x), "=f"(c0.y), "=f"(c1.x), "=f"(c1.y)
                : "r"(A[0]),
                  "r"(A[1]),
                  "r"(A[2]),
                  "r"(A[3]),
                  "r"(B[0]),
                  "r"(B[1]),
                  "f"(c0.x),
                  "f"(c0.y),
                  "f"(c1.x),
                  "f"(c1.y));

              const int c_idx0 =
                (warpId * n_dofs_1d + 2 * col + cycle * 8 + row * offset) ^
                Util::get_base<n_dofs_1d, float>(warpId, row);
              const int c_idx1 =
                (warpId * n_dofs_1d + 2 * col + cycle * 8 +
                 (row + 8) * offset) ^
                Util::get_base<n_dofs_1d, float>(warpId, row + 8);

              *((float2 *)(out + c_idx0)) = c0;
              *((float2 *)(out + c_idx1)) = c1;
            }
        }
    }
  };
#endif

#if MMAKERNEL == 0
  template <typename T>
  struct TPEvaluatorBase<T, 16, float, LaplaceVariant::TensorCoreMMA, 3>
  {
    using Number = float;

    static constexpr int n_dofs_1d = 16;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int warpId = threadIdx.y / 2;

      const int tid = (threadIdx.y * n_dofs_1d + threadIdx.x) & 31;

      const int row = tid / 4;
      const int col = tid & 3;

      constexpr int offset = n_dofs_1d * n_dofs_1d;

      if constexpr (direction == 0)
        {
          float2 c0[n_dofs_1d / 4];
          float2 c1[n_dofs_1d / 4];

          float a[4];
          float b[2];

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              c0[z] = {0, 0};
              c1[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int b_idx0 =
                (col + cycle * 8) * n_dofs_1d + row + (warpId & 1) * 8;
              const int b_idx1 =
                (col + cycle * 8 + 4) * n_dofs_1d + row + (warpId & 1) * 8;

              b[0] = shape_data[b_idx0];
              b[1] = shape_data[b_idx1];

              uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

              for (int z = 0; z < n_dofs_1d / 4; ++z)
                {
                  const int a_idx = row * n_dofs_1d + col + cycle * 8 +
                                    (z * 4 + warpId / 2) * offset;
                  const int a_idx1 = (row + 8) * n_dofs_1d + col + cycle * 8 +
                                     (z * 4 + warpId / 2) * offset;
                  const int a_idx2 = row * n_dofs_1d + col + 4 + cycle * 8 +
                                     (z * 4 + warpId / 2) * offset;
                  const int a_idx3 = (row + 8) * n_dofs_1d + col + 4 +
                                     cycle * 8 + (z * 4 + warpId / 2) * offset;

                  a[0] = in[a_idx];
                  a[1] = in[a_idx1];
                  a[2] = in[a_idx2];
                  a[3] = in[a_idx3];

                  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);

                  asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(c0[z].x), "=f"(c0[z].y), "=f"(c1[z].x), "=f"(c1[z].y)
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(c0[z].x),
                      "f"(c0[z].y),
                      "f"(c1[z].x),
                      "f"(c1[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              const int c_idx0 = row * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                                 (z * 4 + warpId / 2) * offset;
              const int c_idx1 = (row + 8) * n_dofs_1d + 2 * col +
                                 (warpId & 1) * 8 +
                                 (z * 4 + warpId / 2) * offset;

              *((float2 *)(out + c_idx0)) = c0[z];
              *((float2 *)(out + c_idx1)) = c1[z];
            }
        }
      else if (direction == 1)
        {
          float2 c0[n_dofs_1d / 4];
          float2 c1[n_dofs_1d / 4];

          float a[4];
          float b[2];

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              const int c_idx0 = row * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                                 (z * 4 + warpId / 2) * offset;
              const int c_idx1 = (row + 8) * n_dofs_1d + 2 * col +
                                 (warpId & 1) * 8 +
                                 (z * 4 + warpId / 2) * offset;

              if constexpr (add)
                {
                  c0[z] = *((float2 *)(out + c_idx0));
                  c1[z] = *((float2 *)(out + c_idx1));
                }
              else
                {
                  c0[z] = {0, 0};
                  c1[z] = {0, 0};
                }
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx  = row * n_dofs_1d + col + cycle * 8;
              const int a_idx1 = (row + 8) * n_dofs_1d + col + cycle * 8;
              const int a_idx2 = row * n_dofs_1d + col + 4 + cycle * 8;
              const int a_idx3 = (row + 8) * n_dofs_1d + col + 4 + cycle * 8;

              a[0] = shape_data[a_idx];
              a[1] = shape_data[a_idx1];
              a[2] = shape_data[a_idx2];
              a[3] = shape_data[a_idx3];

              uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);

              for (int z = 0; z < n_dofs_1d / 4; ++z)
                {
                  const int b_idx0 = (col + cycle * 8) * n_dofs_1d + row +
                                     (warpId & 1) * 8 +
                                     (z * 4 + warpId / 2) * offset;
                  const int b_idx1 = (col + cycle * 8 + 4) * n_dofs_1d + row +
                                     (warpId & 1) * 8 +
                                     (z * 4 + warpId / 2) * offset;

                  b[0] = in[b_idx0];
                  b[1] = in[b_idx1];

                  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

                  asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(c0[z].x), "=f"(c0[z].y), "=f"(c1[z].x), "=f"(c1[z].y)
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(c0[z].x),
                      "f"(c0[z].y),
                      "f"(c1[z].x),
                      "f"(c1[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              const int c_idx0 = row * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                                 (z * 4 + warpId / 2) * offset;
              const int c_idx1 = (row + 8) * n_dofs_1d + 2 * col +
                                 (warpId & 1) * 8 +
                                 (z * 4 + warpId / 2) * offset;

              *((float2 *)(out + c_idx0)) = c0[z];
              *((float2 *)(out + c_idx1)) = c1[z];
            }
        }
      else
        {
          float2 c0[n_dofs_1d / 4];
          float2 c1[n_dofs_1d / 4];

          float a[4];
          float b[2];

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              const int c_idx0 = (z * 4 + warpId / 2) * n_dofs_1d + 2 * col +
                                 (warpId & 1) * 8 + row * offset;
              const int c_idx1 = (z * 4 + warpId / 2) * n_dofs_1d + 2 * col +
                                 (warpId & 1) * 8 + (row + 8) * offset;

              if constexpr (add)
                {
                  c0[z] = *((float2 *)(out + c_idx0));
                  c1[z] = *((float2 *)(out + c_idx1));
                }
              else
                {
                  c0[z] = {0, 0};
                  c1[z] = {0, 0};
                }
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx  = row * n_dofs_1d + col + cycle * 8;
              const int a_idx1 = (row + 8) * n_dofs_1d + col + cycle * 8;
              const int a_idx2 = row * n_dofs_1d + col + 4 + cycle * 8;
              const int a_idx3 = (row + 8) * n_dofs_1d + col + 4 + cycle * 8;

              a[0] = shape_data[a_idx];
              a[1] = shape_data[a_idx1];
              a[2] = shape_data[a_idx2];
              a[3] = shape_data[a_idx3];

              uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);

              for (int z = 0; z < n_dofs_1d / 4; ++z)
                {
                  const int b_idx0 = (z * 4 + warpId / 2) * n_dofs_1d + row +
                                     (warpId & 1) * 8 +
                                     (col + cycle * 8) * offset;
                  const int b_idx1 = (z * 4 + warpId / 2) * n_dofs_1d + row +
                                     (warpId & 1) * 8 +
                                     (col + cycle * 8 + 4) * offset;

                  b[0] = in[b_idx0];
                  b[1] = in[b_idx1];

                  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

                  asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(c0[z].x), "=f"(c0[z].y), "=f"(c1[z].x), "=f"(c1[z].y)
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(c0[z].x),
                      "f"(c0[z].y),
                      "f"(c1[z].x),
                      "f"(c1[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              const int c_idx0 = (z * 4 + warpId / 2) * n_dofs_1d + 2 * col +
                                 (warpId & 1) * 8 + row * offset;
              const int c_idx1 = (z * 4 + warpId / 2) * n_dofs_1d + 2 * col +
                                 (warpId & 1) * 8 + (row + 8) * offset;

              *((float2 *)(out + c_idx0)) = c0[z];
              *((float2 *)(out + c_idx1)) = c1[z];
            }
        }
    }
  };
#endif



#if MMAKERNEL == 0
  template <typename T>
  struct TPEvaluatorBase<T, 16, float, LaplaceVariant::TensorCoreMMA, 3, half>
  {
    using Number  = float;
    using Number2 = half;

    static constexpr int n_dofs_1d = 16;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number        *dst,
          const Number  *src,
          const Number2 *mass_matrix,
          const Number2 *derivative_matrix,
          Number        *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number2 *shape_data, const Number *in, Number *out)
    {
      const int warpId = threadIdx.y / 2;

      const int tid   = (threadIdx.y * n_dofs_1d + threadIdx.x) & 31;
      const int rowId = tid / 8;
      const int colId = tid & 7;

      const int row = tid / 4;
      const int col = tid & 3;

      constexpr int offset = n_dofs_1d * n_dofs_1d;

#  if ERRCOR == 1
      constexpr int shift = n_dofs_1d * n_dofs_1d * 3;
      constexpr int scale = 1 << 11;
#  endif

      if constexpr (direction == 0)
        {
          for (int cycle = 0; cycle < 2; ++cycle)
            {
              float2 c0 = {0, 0};
              float2 c1 = {0, 0};

              half a[8];
              half b[4];

              const int bb_idx =
                (colId + cycle * 8) * n_dofs_1d + (rowId & 1) * 8;

              auto smem_ptr = static_cast<uint32_t>(
                __cvta_generic_to_shared(&shape_data[bb_idx]));

              float *B_ptr = reinterpret_cast<float *>(&b);
              asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 "
                           "{%0, %1}, [%2]; "
                           : "=f"(B_ptr[0]), "=f"(B_ptr[1])
                           : "r"(smem_ptr));
#  if ERRCOR == 1
              half db[4];

              auto smem_dptr = static_cast<uint32_t>(
                __cvta_generic_to_shared(&shape_data[bb_idx + shift]));

              float *dB_ptr = reinterpret_cast<float *>(&db);
              asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 "
                           "{%0, %1}, [%2]; "
                           : "=f"(dB_ptr[0]), "=f"(dB_ptr[1])
                           : "r"(smem_dptr));
#  endif
              uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
#  if ERRCOR == 1
              uint32_t const *dB = reinterpret_cast<uint32_t const *>(&db);
#  endif

              const int a_idx = (row * n_dofs_1d + col * 2 + warpId * offset) ^
                                Util::get_base<n_dofs_1d, float>(row, warpId);
              const int a_idx1 =
                (row + 8) * n_dofs_1d + col * 2 + warpId * offset;
              const int a_idx2 =
                row * n_dofs_1d + col * 2 + 8 + warpId * offset;
              const int a_idx3 =
                (row + 8) * n_dofs_1d + col * 2 + 8 + warpId * offset;
#  if ERRCOR == 0
              a[0] = __float2half(in[a_idx]);
              a[1] = __float2half(in[a_idx + 1]);
              a[2] = __float2half(in[a_idx1]);
              a[3] = __float2half(in[a_idx1 + 1]);
              a[4] = __float2half(in[a_idx2]);
              a[5] = __float2half(in[a_idx2 + 1]);
              a[6] = __float2half(in[a_idx3]);
              a[7] = __float2half(in[a_idx3 + 1]);
#  elif ERRCOR == 1
              float2 fa[4];
              half   da[8];
              fa[0] = *((float2 *)(in + a_idx));
              fa[1] = *((float2 *)(in + a_idx1));
              fa[2] = *((float2 *)(in + a_idx2));
              fa[3] = *((float2 *)(in + a_idx3));

              for (int i = 0; i < 4; ++i)
                {
                  a[i * 2]     = __float2half(fa[i].x);
                  a[i * 2 + 1] = __float2half(fa[i].y);
                  da[i * 2] =
                    __float2half((fa[i].x - __half2float(a[i * 2])) * scale);
                  da[i * 2 + 1] = __float2half(
                    (fa[i].y - __half2float(a[i * 2 + 1])) * scale);
                }
#  endif
              uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
#  if ERRCOR == 1
              uint32_t const *dA = reinterpret_cast<uint32_t const *>(&da);
#  endif

#  if ERRCOR == 1
              float buf[4];
              buf[0] = 0;
              buf[1] = 0;
              buf[2] = 0;
              buf[3] = 0;
              asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                : "r"(dA[0]),
                  "r"(dA[1]),
                  "r"(dA[2]),
                  "r"(dA[3]),
                  "r"(B[0]),
                  "r"(B[1]),
                  "f"(buf[0]),
                  "f"(buf[1]),
                  "f"(buf[2]),
                  "f"(buf[3]));
              asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                : "r"(A[0]),
                  "r"(A[1]),
                  "r"(A[2]),
                  "r"(A[3]),
                  "r"(dB[0]),
                  "r"(dB[1]),
                  "f"(buf[0]),
                  "f"(buf[1]),
                  "f"(buf[2]),
                  "f"(buf[3]));
              c0.x += buf[0] / scale;
              c0.y += buf[1] / scale;
              c1.x += buf[2] / scale;
              c1.y += buf[3] / scale;
#  endif
              asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(c0.x), "=f"(c0.y), "=f"(c1.x), "=f"(c1.y)
                : "r"(A[0]),
                  "r"(A[1]),
                  "r"(A[2]),
                  "r"(A[3]),
                  "r"(B[0]),
                  "r"(B[1]),
                  "f"(c0.x),
                  "f"(c0.y),
                  "f"(c1.x),
                  "f"(c1.y));

              const int c_idx0 =
                row * n_dofs_1d + 2 * col + cycle * 8 + warpId * offset;
              const int c_idx1 =
                (row + 8) * n_dofs_1d + 2 * col + cycle * 8 + warpId * offset;

              *((float2 *)(out + c_idx0)) = c0;
              *((float2 *)(out + c_idx1)) = c1;
            }
        }
      else if (direction == 1)
        {
          half a[8];

          const int aa_idx =
            (colId + (rowId & 1) * 8) * n_dofs_1d + (rowId / 2) * 8;

          auto smem_ptr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&shape_data[aa_idx]));

          float *A_ptr = reinterpret_cast<float *>(&a);
          asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                       "{%0, %1, %2, %3}, [%4]; "
                       : "=f"(A_ptr[0]),
                         "=f"(A_ptr[1]),
                         "=f"(A_ptr[2]),
                         "=f"(A_ptr[3])
                       : "r"(smem_ptr));
#  if ERRCOR == 1
          half da[8];

          auto smem_dptr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&shape_data[aa_idx + shift]));

          float *dA_ptr = reinterpret_cast<float *>(&da);
          asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                       "{%0, %1, %2, %3}, [%4]; "
                       : "=f"(dA_ptr[0]),
                         "=f"(dA_ptr[1]),
                         "=f"(dA_ptr[2]),
                         "=f"(dA_ptr[3])
                       : "r"(smem_dptr));
#  endif
          uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
#  if ERRCOR == 1
          uint32_t const *dA = reinterpret_cast<uint32_t const *>(&da);
#  endif
          for (int cycle = 0; cycle < 2; ++cycle)
            {
              float2 c0 = {0, 0};
              float2 c1 = {0, 0};

              half b[4];
              if constexpr (add)
                {
                  const int c_idx0 =
                    row * n_dofs_1d + 2 * col + cycle * 8 + warpId * offset;
                  const int c_idx1 = (row + 8) * n_dofs_1d + 2 * col +
                                     cycle * 8 + warpId * offset;

                  c0 = *((float2 *)(out + c_idx0));
                  c1 = *((float2 *)(out + c_idx1));
                }

              const int b_idx0 =
                (col * 2) * n_dofs_1d + row + cycle * 8 + warpId * offset;
              const int b_idx1 =
                (col * 2 + 1) * n_dofs_1d + row + cycle * 8 + warpId * offset;
              const int b_idx2 =
                (col * 2 + 8) * n_dofs_1d + row + cycle * 8 + warpId * offset;
              const int b_idx3 =
                (col * 2 + 9) * n_dofs_1d + row + cycle * 8 + warpId * offset;
#  if ERRCOR == 0
              b[0] = __float2half(in[b_idx0]);
              b[1] = __float2half(in[b_idx1]);
              b[2] = __float2half(in[b_idx2]);
              b[3] = __float2half(in[b_idx3]);
#  elif ERRCOR == 1
              float fb[4];
              half  db[4];
              fb[0] = in[b_idx0];
              fb[1] = in[b_idx1];
              fb[2] = in[b_idx2];
              fb[3] = in[b_idx3];

              for (int i = 0; i < 4; ++i)
                {
                  b[i]  = __float2half(fb[i]);
                  db[i] = __float2half((fb[i] - __half2float(b[i])) * scale);
                }
#  endif
              uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
#  if ERRCOR == 1
              uint32_t const *dB = reinterpret_cast<uint32_t const *>(&db);
#  endif

#  if ERRCOR == 1
              float buf[4];
              buf[0] = 0;
              buf[1] = 0;
              buf[2] = 0;
              buf[3] = 0;
              asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                : "r"(dA[0]),
                  "r"(dA[1]),
                  "r"(dA[2]),
                  "r"(dA[3]),
                  "r"(B[0]),
                  "r"(B[1]),
                  "f"(buf[0]),
                  "f"(buf[1]),
                  "f"(buf[2]),
                  "f"(buf[3]));
              asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                : "r"(A[0]),
                  "r"(A[1]),
                  "r"(A[2]),
                  "r"(A[3]),
                  "r"(dB[0]),
                  "r"(dB[1]),
                  "f"(buf[0]),
                  "f"(buf[1]),
                  "f"(buf[2]),
                  "f"(buf[3]));
              c0.x += buf[0] / scale;
              c0.y += buf[1] / scale;
              c1.x += buf[2] / scale;
              c1.y += buf[3] / scale;
#  endif
              asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(c0.x), "=f"(c0.y), "=f"(c1.x), "=f"(c1.y)
                : "r"(A[0]),
                  "r"(A[1]),
                  "r"(A[2]),
                  "r"(A[3]),
                  "r"(B[0]),
                  "r"(B[1]),
                  "f"(c0.x),
                  "f"(c0.y),
                  "f"(c1.x),
                  "f"(c1.y));

              const int c_idx0 =
                row * n_dofs_1d + 2 * col + cycle * 8 + warpId * offset;
              const int c_idx1 =
                (row + 8) * n_dofs_1d + 2 * col + cycle * 8 + warpId * offset;

              *((float2 *)(out + c_idx0)) = c0;
              *((float2 *)(out + c_idx1)) = c1;
            }
        }
      else
        {
          half a[8];

          const int aa_idx =
            (colId + (rowId & 1) * 8) * n_dofs_1d + (rowId / 2) * 8;

          auto smem_ptr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&shape_data[aa_idx]));

          float *A_ptr = reinterpret_cast<float *>(&a);
          asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                       "{%0, %1, %2, %3}, [%4]; "
                       : "=f"(A_ptr[0]),
                         "=f"(A_ptr[1]),
                         "=f"(A_ptr[2]),
                         "=f"(A_ptr[3])
                       : "r"(smem_ptr));
#  if ERRCOR == 1
          half da[8];

          auto smem_dptr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&shape_data[aa_idx + shift]));

          float *dA_ptr = reinterpret_cast<float *>(&da);
          asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                       "{%0, %1, %2, %3}, [%4]; "
                       : "=f"(dA_ptr[0]),
                         "=f"(dA_ptr[1]),
                         "=f"(dA_ptr[2]),
                         "=f"(dA_ptr[3])
                       : "r"(smem_dptr));
#  endif
          uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
#  if ERRCOR == 1
          uint32_t const *dA = reinterpret_cast<uint32_t const *>(&da);
#  endif
          for (int cycle = 0; cycle < 2; ++cycle)
            {
              float2 c0 = {0, 0};
              float2 c1 = {0, 0};

              half b[4];

              if constexpr (add)
                {
                  const int c_idx0 =
                    warpId * n_dofs_1d + 2 * col + cycle * 8 + row * offset;
                  const int c_idx1 = warpId * n_dofs_1d + 2 * col + cycle * 8 +
                                     (row + 8) * offset;

                  c0 = *((float2 *)(out + c_idx0));
                  c1 = *((float2 *)(out + c_idx1));
                }

              const int b_idx0 =
                warpId * n_dofs_1d + row + cycle * 8 + (col * 2) * offset;
              const int b_idx1 =
                warpId * n_dofs_1d + row + cycle * 8 + (col * 2 + 1) * offset;
              const int b_idx2 =
                warpId * n_dofs_1d + row + cycle * 8 + (col * 2 + 8) * offset;
              const int b_idx3 =
                warpId * n_dofs_1d + row + cycle * 8 + (col * 2 + 9) * offset;
#  if ERRCOR == 0
              b[0] = __float2half(in[b_idx0]);
              b[1] = __float2half(in[b_idx1]);
              b[2] = __float2half(in[b_idx2]);
              b[3] = __float2half(in[b_idx3]);
#  elif ERRCOR == 1
              float fb[4];
              half  db[4];
              fb[0] = in[b_idx0];
              fb[1] = in[b_idx1];
              fb[2] = in[b_idx2];
              fb[3] = in[b_idx3];

              for (int i = 0; i < 4; ++i)
                {
                  b[i]  = __float2half(fb[i]);
                  db[i] = __float2half((fb[i] - __half2float(b[i])) * scale);
                }
#  endif
              uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
#  if ERRCOR == 1
              uint32_t const *dB = reinterpret_cast<uint32_t const *>(&db);
#  endif

#  if ERRCOR == 1
              float buf[4];
              buf[0] = 0;
              buf[1] = 0;
              buf[2] = 0;
              buf[3] = 0;
              asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                : "r"(dA[0]),
                  "r"(dA[1]),
                  "r"(dA[2]),
                  "r"(dA[3]),
                  "r"(B[0]),
                  "r"(B[1]),
                  "f"(buf[0]),
                  "f"(buf[1]),
                  "f"(buf[2]),
                  "f"(buf[3]));
              asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                : "r"(A[0]),
                  "r"(A[1]),
                  "r"(A[2]),
                  "r"(A[3]),
                  "r"(dB[0]),
                  "r"(dB[1]),
                  "f"(buf[0]),
                  "f"(buf[1]),
                  "f"(buf[2]),
                  "f"(buf[3]));
              c0.x += buf[0] / scale;
              c0.y += buf[1] / scale;
              c1.x += buf[2] / scale;
              c1.y += buf[3] / scale;
#  endif
              asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(c0.x), "=f"(c0.y), "=f"(c1.x), "=f"(c1.y)
                : "r"(A[0]),
                  "r"(A[1]),
                  "r"(A[2]),
                  "r"(A[3]),
                  "r"(B[0]),
                  "r"(B[1]),
                  "f"(c0.x),
                  "f"(c0.y),
                  "f"(c1.x),
                  "f"(c1.y));

              const int c_idx0 =
                warpId * n_dofs_1d + 2 * col + cycle * 8 + row * offset;
              const int c_idx1 =
                warpId * n_dofs_1d + 2 * col + cycle * 8 + (row + 8) * offset;

              *((float2 *)(out + c_idx0)) = c0;
              *((float2 *)(out + c_idx1)) = c1;
            }
        }
    }
  };
#endif


#if MMAKERNEL == 1
  template <typename T>
  struct TPEvaluatorBase<T, 16, float, LaplaceVariant::TensorCoreMMA, 3>
  {
    using Number = float;

    static constexpr int n_dofs_1d = 16;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int warpId = threadIdx.y / 2;

      const int tid   = (threadIdx.y * n_dofs_1d + threadIdx.x) & 31;
      const int rowId = tid / 8;
      const int colId = tid & 7;

      const int row = tid / 4;
      const int col = tid & 3;

      constexpr int offset = n_dofs_1d * n_dofs_1d;

      if constexpr (direction == 0)
        {
          float2 c0[n_dofs_1d / 4];
          float2 c1[n_dofs_1d / 4];

          float a[4];
          float b[2];

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              c0[z] = {0, 0};
              c1[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int b_idx0 =
                ((col + cycle * 8) * n_dofs_1d + row + (warpId & 1) * 8) ^
                Util::get_base<n_dofs_1d, float>(col + cycle * 8);
              const int b_idx1 =
                ((col + cycle * 8 + 4) * n_dofs_1d + row + (warpId & 1) * 8) ^
                Util::get_base<n_dofs_1d, float>(col + cycle * 8 + 4);

              b[0] = shape_data[b_idx0];
              b[1] = shape_data[b_idx1];

              uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

              for (int z = 0; z < n_dofs_1d / 4; ++z)
                {
                  const int a_idx =
                    ((colId + (rowId & 1) * 8) * n_dofs_1d + (rowId / 2) * 4 +
                     cycle * 8 + (z * 4 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(colId + (rowId & 1) * 8,
                                                     z * 4 + warpId / 2);

                  auto smem_ptr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&in[a_idx]));
#  if TIMING == 1
                  auto start = clock64();
#  endif
                  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                               "{%0, %1, %2, %3}, [%4]; "
                               : "=f"(a[0]), "=f"(a[1]), "=f"(a[2]), "=f"(a[3])
                               : "r"(smem_ptr));
#  if TIMING == 1
                  auto elapsed = clock64() - start;
                  if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
                    printf(
                      "mma load frag a dir-%d loop-%d timing info: %ld cycles\n",
                      direction,
                      cycle,
                      elapsed);
#  endif
                  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
#  if TIMING == 1
                  start = clock64();
#  endif
                  asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(c0[z].x), "=f"(c0[z].y), "=f"(c1[z].x), "=f"(c1[z].y)
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(c0[z].x),
                      "f"(c0[z].y),
                      "f"(c1[z].x),
                      "f"(c1[z].y));
#  if TIMING == 1
                  elapsed = clock64() - start;
                  if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
                    printf(
                      "mma.sync.aligned.m16n8k8 dir-%d loop-(%d,%d) timing info: %ld cycles\n",
                      direction,
                      cycle,
                      z,
                      elapsed);
#  endif
                }
            }

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              const int c_idx0 =
                (row * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (z * 4 + warpId / 2) * offset) ^
                Util::get_base<n_dofs_1d, float>(row, z * 4 + warpId / 2);
              const int c_idx1 =
                ((row + 8) * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (z * 4 + warpId / 2) * offset) ^
                Util::get_base<n_dofs_1d, float>(row + 8, z * 4 + warpId / 2);

              *((float2 *)(out + c_idx0)) = c0[z];
              *((float2 *)(out + c_idx1)) = c1[z];
            }
        }
      else if (direction == 1)
        {
          float2 c0[n_dofs_1d / 4];
          float2 c1[n_dofs_1d / 4];

          float a[4];
          float b[2];

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              if constexpr (add)
                {
                  const int c_idx0 =
                    (row * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                     (z * 4 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row, z * 4 + warpId / 2);
                  const int c_idx1 =
                    ((row + 8) * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                     (z * 4 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row + 8,
                                                     z * 4 + warpId / 2);

                  c0[z] = *((float2 *)(out + c_idx0));
                  c1[z] = *((float2 *)(out + c_idx1));
                }
              else
                {
                  c0[z] = {0, 0};
                  c1[z] = {0, 0};
                }
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx =
                ((colId + (rowId & 1) * 8) * n_dofs_1d + (rowId / 2) * 4 +
                 cycle * 8) ^
                Util::get_base<n_dofs_1d, float>(colId + (rowId & 1) * 8);

              auto smem_ptr = static_cast<uint32_t>(
                __cvta_generic_to_shared(&shape_data[a_idx]));

              asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                           "{%0, %1, %2, %3}, [%4]; "
                           : "=f"(a[0]), "=f"(a[1]), "=f"(a[2]), "=f"(a[3])
                           : "r"(smem_ptr));

              uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);

              for (int z = 0; z < n_dofs_1d / 4; ++z)
                {
                  const int b_idx0 =
                    ((col + cycle * 8) * n_dofs_1d + row + (warpId & 1) * 8 +
                     (z * 4 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(col + cycle * 8,
                                                     z * 4 + warpId / 2);
                  const int b_idx1 =
                    ((col + cycle * 8 + 4) * n_dofs_1d + row +
                     (warpId & 1) * 8 + (z * 4 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(col + cycle * 8 + 4,
                                                     z * 4 + warpId / 2);

                  b[0] = in[b_idx0];
                  b[1] = in[b_idx1];

                  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

                  asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(c0[z].x), "=f"(c0[z].y), "=f"(c1[z].x), "=f"(c1[z].y)
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(c0[z].x),
                      "f"(c0[z].y),
                      "f"(c1[z].x),
                      "f"(c1[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              const int c_idx0 =
                (row * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (z * 4 + warpId / 2) * offset) ^
                Util::get_base<n_dofs_1d, float>(row, z * 4 + warpId / 2);
              const int c_idx1 =
                ((row + 8) * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (z * 4 + warpId / 2) * offset) ^
                Util::get_base<n_dofs_1d, float>(row + 8, z * 4 + warpId / 2);

              *((float2 *)(out + c_idx0)) = c0[z];
              *((float2 *)(out + c_idx1)) = c1[z];
            }
        }
      else
        {
          float2 c0[n_dofs_1d / 4];
          float2 c1[n_dofs_1d / 4];

          float a[4];
          float b[2];

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              if constexpr (add)
                {
                  const int c_idx0 =
                    ((z * 4 + warpId / 2) * n_dofs_1d + 2 * col +
                     (warpId & 1) * 8 + row * offset) ^
                    Util::get_base<n_dofs_1d, float>((z * 4 + warpId / 2), row);
                  const int c_idx1 =
                    ((z * 4 + warpId / 2) * n_dofs_1d + 2 * col +
                     (warpId & 1) * 8 + (row + 8) * offset) ^
                    Util::get_base<n_dofs_1d, float>(z * 4 + warpId / 2,
                                                     row + 8);

                  c0[z] = *((float2 *)(out + c_idx0));
                  c1[z] = *((float2 *)(out + c_idx1));
                }
              else
                {
                  c0[z] = {0, 0};
                  c1[z] = {0, 0};
                }
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx =
                ((colId + (rowId & 1) * 8) * n_dofs_1d + (rowId / 2) * 4 +
                 cycle * 8) ^
                Util::get_base<n_dofs_1d, float>(colId + (rowId & 1) * 8);

              auto smem_ptr = static_cast<uint32_t>(
                __cvta_generic_to_shared(&shape_data[a_idx]));

              asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                           "{%0, %1, %2, %3}, [%4]; "
                           : "=f"(a[0]), "=f"(a[1]), "=f"(a[2]), "=f"(a[3])
                           : "r"(smem_ptr));

              uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);

              for (int z = 0; z < n_dofs_1d / 4; ++z)
                {
                  const int b_idx0 =
                    ((z * 4 + warpId / 2) * n_dofs_1d + row + (warpId & 1) * 8 +
                     (col + cycle * 8) * offset) ^
                    Util::get_base<n_dofs_1d, float>(z * 4 + warpId / 2,
                                                     col + cycle * 8);
                  const int b_idx1 =
                    ((z * 4 + warpId / 2) * n_dofs_1d + row + (warpId & 1) * 8 +
                     (col + cycle * 8 + 4) * offset) ^
                    Util::get_base<n_dofs_1d, float>(z * 4 + warpId / 2,
                                                     col + cycle * 8 + 4);

                  b[0] = in[b_idx0];
                  b[1] = in[b_idx1];

                  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

                  asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(c0[z].x), "=f"(c0[z].y), "=f"(c1[z].x), "=f"(c1[z].y)
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(c0[z].x),
                      "f"(c0[z].y),
                      "f"(c1[z].x),
                      "f"(c1[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              const int c_idx0 =
                ((z * 4 + warpId / 2) * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 row * offset) ^
                Util::get_base<n_dofs_1d, float>((z * 4 + warpId / 2), row);
              const int c_idx1 =
                ((z * 4 + warpId / 2) * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (row + 8) * offset) ^
                Util::get_base<n_dofs_1d, float>(z * 4 + warpId / 2, row + 8);

              *((float2 *)(out + c_idx0)) = c0[z];
              *((float2 *)(out + c_idx1)) = c1[z];
            }
        }
    }
  };
#endif


#if MMAKERNEL == 2
  template <typename T>
  struct TPEvaluatorBase<T, 16, float, LaplaceVariant::TensorCoreMMA, 3>
  {
    using Number = float;

    static constexpr int n_dofs_1d = 16;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int warpId = threadIdx.y / 2;

      const int tid = (threadIdx.y * n_dofs_1d + threadIdx.x) & 31;

      const int row = tid / 4;
      const int col = tid & 3;

      constexpr int offset = n_dofs_1d * n_dofs_1d;
      constexpr int scale  = 1 << 11;

      if constexpr (direction == 0)
        {
          float2 c0[n_dofs_1d / 4];
          float2 c1[n_dofs_1d / 4];

          float a[4];
          float b[2];

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              c0[z] = {0, 0};
              c1[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int b_idx0 =
                ((col + cycle * 8) * n_dofs_1d + row + (warpId & 1) * 8) ^
                Util::get_base<n_dofs_1d, float>(col + cycle * 8);
              const int b_idx1 =
                ((col + cycle * 8 + 4) * n_dofs_1d + row + (warpId & 1) * 8) ^
                Util::get_base<n_dofs_1d, float>(col + cycle * 8 + 4);

              b[0] = shape_data[b_idx0];
              b[1] = shape_data[b_idx1];

              uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

#  if ERRCOR == 1
              float db[2];
              for (int i = 0; i < 2; ++i)
                {
                  db[i] = (b[i] - wmma::__float_to_tf32(b[i])) * scale;
                }
              uint32_t const *dB = reinterpret_cast<uint32_t const *>(&db);
#  endif
              for (int z = 0; z < n_dofs_1d / 4; ++z)
                {
                  const int a_idx =
                    (row * n_dofs_1d + col + cycle * 8 +
                     (z * 4 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row, z * 4 + warpId / 2);
                  const int a_idx1 =
                    ((row + 8) * n_dofs_1d + col + cycle * 8 +
                     (z * 4 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row + 8,
                                                     z * 4 + warpId / 2);
                  const int a_idx2 =
                    (row * n_dofs_1d + col + 4 + cycle * 8 +
                     (z * 4 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row, z * 4 + warpId / 2);
                  const int a_idx3 =
                    ((row + 8) * n_dofs_1d + col + 4 + cycle * 8 +
                     (z * 4 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row, z * 4 + warpId / 2);

                  a[0] = in[a_idx];
                  a[1] = in[a_idx1];
                  a[2] = in[a_idx2];
                  a[3] = in[a_idx3];

                  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
#  if ERRCOR == 1
                  float fa[4];
                  float da[4];
                  for (int i = 0; i < 4; ++i)
                    {
                      da[i] = (a[i] - wmma::__float_to_tf32(a[i])) * scale;
                    }
                  uint32_t const *dA = reinterpret_cast<uint32_t const *>(&da);
#  endif
                  asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(c0[z].x), "=f"(c0[z].y), "=f"(c1[z].x), "=f"(c1[z].y)
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(c0[z].x),
                      "f"(c0[z].y),
                      "f"(c1[z].x),
                      "f"(c1[z].y));
#  if ERRCOR == 1
                  float buf[4];
                  buf[0] = 0;
                  buf[1] = 0;
                  buf[2] = 0;
                  buf[3] = 0;
                  asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                    : "r"(dA[0]),
                      "r"(dA[1]),
                      "r"(dA[2]),
                      "r"(dA[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(buf[0]),
                      "f"(buf[1]),
                      "f"(buf[2]),
                      "f"(buf[3]));
                  asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(dB[0]),
                      "r"(dB[1]),
                      "f"(buf[0]),
                      "f"(buf[1]),
                      "f"(buf[2]),
                      "f"(buf[3]));
                  c0[z].x += buf[0] / scale;
                  c0[z].y += buf[1] / scale;
                  c1[z].x += buf[2] / scale;
                  c1[z].y += buf[3] / scale;
#  endif
                }
            }

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              const int c_idx0 =
                (row * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (z * 4 + warpId / 2) * offset) ^
                Util::get_base<n_dofs_1d, float>(row, z * 4 + warpId / 2);
              const int c_idx1 =
                ((row + 8) * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (z * 4 + warpId / 2) * offset) ^
                Util::get_base<n_dofs_1d, float>(row + 8, z * 4 + warpId / 2);

              *((float2 *)(out + c_idx0)) = c0[z];
              *((float2 *)(out + c_idx1)) = c1[z];
            }
        }
      else if (direction == 1)
        {
          float2 c0[n_dofs_1d / 4];
          float2 c1[n_dofs_1d / 4];

          float a[4];
          float b[2];

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              const int c_idx0 =
                (row * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (z * 4 + warpId / 2) * offset) ^
                Util::get_base<n_dofs_1d, float>(row, z * 4 + warpId / 2);
              const int c_idx1 =
                ((row + 8) * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (z * 4 + warpId / 2) * offset) ^
                Util::get_base<n_dofs_1d, float>(row + 8, z * 4 + warpId / 2);

              if constexpr (add)
                {
                  c0[z] = *((float2 *)(out + c_idx0));
                  c1[z] = *((float2 *)(out + c_idx1));
                }
              else
                {
                  c0[z] = {0, 0};
                  c1[z] = {0, 0};
                }
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx = (row * n_dofs_1d + col + cycle * 8) ^
                                Util::get_base<n_dofs_1d, float>(row);
              const int a_idx1 = ((row + 8) * n_dofs_1d + col + cycle * 8) ^
                                 Util::get_base<n_dofs_1d, float>(row + 8);
              const int a_idx2 = (row * n_dofs_1d + col + 4 + cycle * 8) ^
                                 Util::get_base<n_dofs_1d, float>(row);
              const int a_idx3 = ((row + 8) * n_dofs_1d + col + 4 + cycle * 8) ^
                                 Util::get_base<n_dofs_1d, float>(row);

              a[0] = shape_data[a_idx];
              a[1] = shape_data[a_idx1];
              a[2] = shape_data[a_idx2];
              a[3] = shape_data[a_idx3];

              uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
#  if ERRCOR == 1
              float fa[4];
              float da[4];
              for (int i = 0; i < 4; ++i)
                {
                  fa[i] = wmma::__float_to_tf32(a[i]);
                  da[i] = (a[i] - fa[i]) * scale;
                }
              uint32_t const *dA = reinterpret_cast<uint32_t const *>(&da);
#  endif
              for (int z = 0; z < n_dofs_1d / 4; ++z)
                {
                  const int b_idx0 =
                    ((col + cycle * 8) * n_dofs_1d + row + (warpId & 1) * 8 +
                     (z * 4 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(col + cycle * 8,
                                                     z * 4 + warpId / 2);
                  const int b_idx1 =
                    ((col + cycle * 8 + 4) * n_dofs_1d + row +
                     (warpId & 1) * 8 + (z * 4 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(col + cycle * 8 + 4,
                                                     z * 4 + warpId / 2);

                  b[0] = in[b_idx0];
                  b[1] = in[b_idx1];

                  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
#  if ERRCOR == 1
                  float fb[2];
                  float db[2];
                  for (int i = 0; i < 2; ++i)
                    {
                      fb[i] = wmma::__float_to_tf32(b[i]);
                      db[i] = (b[i] - fb[i]) * scale;
                    }
                  uint32_t const *dB = reinterpret_cast<uint32_t const *>(&db);
#  endif
                  asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(c0[z].x), "=f"(c0[z].y), "=f"(c1[z].x), "=f"(c1[z].y)
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(c0[z].x),
                      "f"(c0[z].y),
                      "f"(c1[z].x),
                      "f"(c1[z].y));
#  if ERRCOR == 1
                  float buf[4];
                  buf[0] = 0;
                  buf[1] = 0;
                  buf[2] = 0;
                  buf[3] = 0;
                  asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                    : "r"(dA[0]),
                      "r"(dA[1]),
                      "r"(dA[2]),
                      "r"(dA[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(buf[0]),
                      "f"(buf[1]),
                      "f"(buf[2]),
                      "f"(buf[3]));
                  asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(dB[0]),
                      "r"(dB[1]),
                      "f"(buf[0]),
                      "f"(buf[1]),
                      "f"(buf[2]),
                      "f"(buf[3]));
                  c0[z].x += buf[0] / scale;
                  c0[z].y += buf[1] / scale;
                  c1[z].x += buf[2] / scale;
                  c1[z].y += buf[3] / scale;
#  endif
                }
            }

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              const int c_idx0 =
                (row * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (z * 4 + warpId / 2) * offset) ^
                Util::get_base<n_dofs_1d, float>(row, z * 4 + warpId / 2);
              const int c_idx1 =
                ((row + 8) * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (z * 4 + warpId / 2) * offset) ^
                Util::get_base<n_dofs_1d, float>(row + 8, z * 4 + warpId / 2);

              *((float2 *)(out + c_idx0)) = c0[z];
              *((float2 *)(out + c_idx1)) = c1[z];
            }
        }
      else
        {
          float2 c0[n_dofs_1d / 4];
          float2 c1[n_dofs_1d / 4];

          float a[4];
          float b[2];

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              const int c_idx0 =
                ((z * 4 + warpId / 2) * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 row * offset) ^
                Util::get_base<n_dofs_1d, float>((z * 4 + warpId / 2), row);
              const int c_idx1 =
                ((z * 4 + warpId / 2) * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (row + 8) * offset) ^
                Util::get_base<n_dofs_1d, float>(z * 4 + warpId / 2, row + 8);

              if constexpr (add)
                {
                  c0[z] = *((float2 *)(out + c_idx0));
                  c1[z] = *((float2 *)(out + c_idx1));
                }
              else
                {
                  c0[z] = {0, 0};
                  c1[z] = {0, 0};
                }
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx = (row * n_dofs_1d + col + cycle * 8) ^
                                Util::get_base<n_dofs_1d, float>(row);
              const int a_idx1 = ((row + 8) * n_dofs_1d + col + cycle * 8) ^
                                 Util::get_base<n_dofs_1d, float>(row + 8);
              const int a_idx2 = (row * n_dofs_1d + col + 4 + cycle * 8) ^
                                 Util::get_base<n_dofs_1d, float>(row);
              const int a_idx3 = ((row + 8) * n_dofs_1d + col + 4 + cycle * 8) ^
                                 Util::get_base<n_dofs_1d, float>(row);

              a[0] = shape_data[a_idx];
              a[1] = shape_data[a_idx1];
              a[2] = shape_data[a_idx2];
              a[3] = shape_data[a_idx3];

              uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
#  if ERRCOR == 1
              float fa[4];
              float da[4];
              for (int i = 0; i < 4; ++i)
                {
                  fa[i] = wmma::__float_to_tf32(a[i]);
                  da[i] = (a[i] - fa[i]) * scale;
                }
              uint32_t const *dA = reinterpret_cast<uint32_t const *>(&da);
#  endif
              for (int z = 0; z < n_dofs_1d / 4; ++z)
                {
                  const int b_idx0 =
                    ((z * 4 + warpId / 2) * n_dofs_1d + row + (warpId & 1) * 8 +
                     (col + cycle * 8) * offset) ^
                    Util::get_base<n_dofs_1d, float>(z * 4 + warpId / 2,
                                                     col + cycle * 8);
                  const int b_idx1 =
                    ((z * 4 + warpId / 2) * n_dofs_1d + row + (warpId & 1) * 8 +
                     (col + cycle * 8 + 4) * offset) ^
                    Util::get_base<n_dofs_1d, float>(z * 4 + warpId / 2,
                                                     col + cycle * 8 + 4);

                  b[0] = in[b_idx0];
                  b[1] = in[b_idx1];

                  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
#  if ERRCOR == 1
                  float fb[2];
                  float db[2];
                  for (int i = 0; i < 2; ++i)
                    {
                      fb[i] = wmma::__float_to_tf32(b[i]);
                      db[i] = (b[i] - fb[i]) * scale;
                    }
                  uint32_t const *dB = reinterpret_cast<uint32_t const *>(&db);
#  endif
                  asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(c0[z].x), "=f"(c0[z].y), "=f"(c1[z].x), "=f"(c1[z].y)
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(c0[z].x),
                      "f"(c0[z].y),
                      "f"(c1[z].x),
                      "f"(c1[z].y));
#  if ERRCOR == 1
                  float buf[4];
                  buf[0] = 0;
                  buf[1] = 0;
                  buf[2] = 0;
                  buf[3] = 0;
                  asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                    : "r"(dA[0]),
                      "r"(dA[1]),
                      "r"(dA[2]),
                      "r"(dA[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(buf[0]),
                      "f"(buf[1]),
                      "f"(buf[2]),
                      "f"(buf[3]));
                  asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(dB[0]),
                      "r"(dB[1]),
                      "f"(buf[0]),
                      "f"(buf[1]),
                      "f"(buf[2]),
                      "f"(buf[3]));
                  c0[z].x += buf[0] / scale;
                  c0[z].y += buf[1] / scale;
                  c1[z].x += buf[2] / scale;
                  c1[z].y += buf[3] / scale;
#  endif
                }
            }

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              const int c_idx0 =
                ((z * 4 + warpId / 2) * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 row * offset) ^
                Util::get_base<n_dofs_1d, float>((z * 4 + warpId / 2), row);
              const int c_idx1 =
                ((z * 4 + warpId / 2) * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (row + 8) * offset) ^
                Util::get_base<n_dofs_1d, float>(z * 4 + warpId / 2, row + 8);

              *((float2 *)(out + c_idx0)) = c0[z];
              *((float2 *)(out + c_idx1)) = c1[z];
            }
        }
    }
  };
#endif



#if MMAKERNEL == 3
  template <typename T>
  struct TPEvaluatorBase<T, 16, float, LaplaceVariant::TensorCoreMMA, 3>
  {
    using Number = float;

    static constexpr int n_dofs_1d = 16;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int warpId = threadIdx.y / 2;

      const int tid   = (threadIdx.y * n_dofs_1d + threadIdx.x) & 31;
      const int rowId = tid / 8;
      const int colId = tid & 7;

      const int row = tid / 4;
      const int col = tid & 3;

      constexpr int offset = n_dofs_1d * n_dofs_1d;

      if constexpr (direction == 0)
        {
          float2 c0[n_dofs_1d / 4];
          float2 c1[n_dofs_1d / 4];

          float a[8];
          float b[2];

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              c0[z] = {0, 0};
              c1[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int b_idx0 =
                ((col + cycle * 8) * n_dofs_1d + row + (warpId & 1) * 8) ^
                Util::get_base<n_dofs_1d, float>(col + cycle * 8);
              const int b_idx1 =
                ((col + cycle * 8 + 4) * n_dofs_1d + row + (warpId & 1) * 8) ^
                Util::get_base<n_dofs_1d, float>(col + cycle * 8 + 4);

              b[0] = shape_data[b_idx0];
              b[1] = shape_data[b_idx1];

              uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

              for (int z = 0; z < n_dofs_1d / 8; ++z)
                {
                  const int a_idx =
                    ((colId + (rowId & 1) * 8) * n_dofs_1d + (rowId / 2) * 4 +
                     cycle * 8 + (z * 8 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(colId + (rowId & 1) * 8,
                                                     z * 8 + warpId / 2);
                  const int a_idx1 =
                    ((colId + (rowId & 1) * 8) * n_dofs_1d + (rowId / 2) * 4 +
                     cycle * 8 + (z * 8 + 4 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(colId + (rowId & 1) * 8,
                                                     z * 8 + 4 + warpId / 2);

                  auto smem_ptr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&in[a_idx]));
                  auto smem_ptr1 = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&in[a_idx1]));

                  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                               "{%0, %1, %2, %3}, [%4]; "
                               : "=f"(a[0]), "=f"(a[1]), "=f"(a[2]), "=f"(a[3])
                               : "r"(smem_ptr));
                  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                               "{%0, %1, %2, %3}, [%4]; "
                               : "=f"(a[4]), "=f"(a[5]), "=f"(a[6]), "=f"(a[7])
                               : "r"(smem_ptr1));

                  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);

                  asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(c0[z * 2].x),
                      "=f"(c0[z * 2].y),
                      "=f"(c1[z * 2].x),
                      "=f"(c1[z * 2].y)
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(c0[z * 2].x),
                      "f"(c0[z * 2].y),
                      "f"(c1[z * 2].x),
                      "f"(c1[z * 2].y));

                  asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(c0[z * 2 + 1].x),
                      "=f"(c0[z * 2 + 1].y),
                      "=f"(c1[z * 2 + 1].x),
                      "=f"(c1[z * 2 + 1].y)
                    : "r"(A[4]),
                      "r"(A[5]),
                      "r"(A[6]),
                      "r"(A[7]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(c0[z * 2 + 1].x),
                      "f"(c0[z * 2 + 1].y),
                      "f"(c1[z * 2 + 1].x),
                      "f"(c1[z * 2 + 1].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              const int c_idx0 =
                (row * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (z * 4 + warpId / 2) * offset) ^
                Util::get_base<n_dofs_1d, float>(row, z * 4 + warpId / 2);
              const int c_idx1 =
                ((row + 8) * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (z * 4 + warpId / 2) * offset) ^
                Util::get_base<n_dofs_1d, float>(row + 8, z * 4 + warpId / 2);

              *((float2 *)(out + c_idx0)) = c0[z];
              *((float2 *)(out + c_idx1)) = c1[z];
            }
        }
      else if (direction == 1)
        {
          float2 c0[n_dofs_1d / 4];
          float2 c1[n_dofs_1d / 4];

          float a[4];
          float b[4];

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              if constexpr (add)
                {
                  const int c_idx0 =
                    (row * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                     (z * 4 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row, z * 4 + warpId / 2);
                  const int c_idx1 =
                    ((row + 8) * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                     (z * 4 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row + 8,
                                                     z * 4 + warpId / 2);

                  c0[z] = *((float2 *)(out + c_idx0));
                  c1[z] = *((float2 *)(out + c_idx1));
                }
              else
                {
                  c0[z] = {0, 0};
                  c1[z] = {0, 0};
                }
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx =
                ((colId + (rowId & 1) * 8) * n_dofs_1d + (rowId / 2) * 4 +
                 cycle * 8) ^
                Util::get_base<n_dofs_1d, float>(colId + (rowId & 1) * 8);

              auto smem_ptr = static_cast<uint32_t>(
                __cvta_generic_to_shared(&shape_data[a_idx]));

              asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                           "{%0, %1, %2, %3}, [%4]; "
                           : "=f"(a[0]), "=f"(a[1]), "=f"(a[2]), "=f"(a[3])
                           : "r"(smem_ptr));

              uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);

              for (int z = 0; z < n_dofs_1d / 8; ++z)
                {
                  const int b_idx0 =
                    ((col + cycle * 8) * n_dofs_1d + row + (warpId & 1) * 8 +
                     (z * 8 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(col + cycle * 8,
                                                     z * 8 + warpId / 2);
                  const int b_idx1 =
                    ((col + cycle * 8 + 4) * n_dofs_1d + row +
                     (warpId & 1) * 8 + (z * 8 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(col + cycle * 8 + 4,
                                                     z * 8 + warpId / 2);

                  const int b_idx2 =
                    ((col + cycle * 8) * n_dofs_1d + row + (warpId & 1) * 8 +
                     (z * 8 + 4 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(col + cycle * 8,
                                                     z * 8 + 4 + warpId / 2);
                  const int b_idx3 =
                    ((col + cycle * 8 + 4) * n_dofs_1d + row +
                     (warpId & 1) * 8 + (z * 8 + 4 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(col + cycle * 8 + 4,
                                                     z * 8 + 4 + warpId / 2);

                  b[0] = in[b_idx0];
                  b[1] = in[b_idx1];

                  b[2] = in[b_idx2];
                  b[3] = in[b_idx3];

                  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

                  asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(c0[z * 2].x),
                      "=f"(c0[z * 2].y),
                      "=f"(c1[z * 2].x),
                      "=f"(c1[z * 2].y)
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(c0[z * 2].x),
                      "f"(c0[z * 2].y),
                      "f"(c1[z * 2].x),
                      "f"(c1[z * 2].y));

                  asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(c0[z * 2 + 1].x),
                      "=f"(c0[z * 2 + 1].y),
                      "=f"(c1[z * 2 + 1].x),
                      "=f"(c1[z * 2 + 1].y)
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(B[2]),
                      "r"(B[3]),
                      "f"(c0[z * 2 + 1].x),
                      "f"(c0[z * 2 + 1].y),
                      "f"(c1[z * 2 + 1].x),
                      "f"(c1[z * 2 + 1].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              const int c_idx0 =
                (row * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (z * 4 + warpId / 2) * offset) ^
                Util::get_base<n_dofs_1d, float>(row, z * 4 + warpId / 2);
              const int c_idx1 =
                ((row + 8) * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (z * 4 + warpId / 2) * offset) ^
                Util::get_base<n_dofs_1d, float>(row + 8, z * 4 + warpId / 2);

              *((float2 *)(out + c_idx0)) = c0[z];
              *((float2 *)(out + c_idx1)) = c1[z];
            }
        }
      else
        {
          float2 c0[n_dofs_1d / 4];
          float2 c1[n_dofs_1d / 4];

          float a[4];
          float b[4];

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              if constexpr (add)
                {
                  const int c_idx0 =
                    ((z * 4 + warpId / 2) * n_dofs_1d + 2 * col +
                     (warpId & 1) * 8 + row * offset) ^
                    Util::get_base<n_dofs_1d, float>((z * 4 + warpId / 2), row);
                  const int c_idx1 =
                    ((z * 4 + warpId / 2) * n_dofs_1d + 2 * col +
                     (warpId & 1) * 8 + (row + 8) * offset) ^
                    Util::get_base<n_dofs_1d, float>(z * 4 + warpId / 2,
                                                     row + 8);

                  c0[z] = *((float2 *)(out + c_idx0));
                  c1[z] = *((float2 *)(out + c_idx1));
                }
              else
                {
                  c0[z] = {0, 0};
                  c1[z] = {0, 0};
                }
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx =
                ((colId + (rowId & 1) * 8) * n_dofs_1d + (rowId / 2) * 4 +
                 cycle * 8) ^
                Util::get_base<n_dofs_1d, float>(colId + (rowId & 1) * 8);

              auto smem_ptr = static_cast<uint32_t>(
                __cvta_generic_to_shared(&shape_data[a_idx]));

              asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                           "{%0, %1, %2, %3}, [%4]; "
                           : "=f"(a[0]), "=f"(a[1]), "=f"(a[2]), "=f"(a[3])
                           : "r"(smem_ptr));

              uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);

              for (int z = 0; z < n_dofs_1d / 8; ++z)
                {
                  const int b_idx0 =
                    ((z * 8 + warpId / 2) * n_dofs_1d + row + (warpId & 1) * 8 +
                     (col + cycle * 8) * offset) ^
                    Util::get_base<n_dofs_1d, float>(z * 8 + warpId / 2,
                                                     col + cycle * 8);
                  const int b_idx1 =
                    ((z * 8 + warpId / 2) * n_dofs_1d + row + (warpId & 1) * 8 +
                     (col + cycle * 8 + 4) * offset) ^
                    Util::get_base<n_dofs_1d, float>(z * 8 + warpId / 2,
                                                     col + cycle * 8 + 4);
                  const int b_idx2 =
                    ((z * 8 + 4 + warpId / 2) * n_dofs_1d + row +
                     (warpId & 1) * 8 + (col + cycle * 8) * offset) ^
                    Util::get_base<n_dofs_1d, float>(z * 8 + 4 + warpId / 2,
                                                     col + cycle * 8);
                  const int b_idx3 =
                    ((z * 8 + 4 + warpId / 2) * n_dofs_1d + row +
                     (warpId & 1) * 8 + (col + cycle * 8 + 4) * offset) ^
                    Util::get_base<n_dofs_1d, float>(z * 8 + 4 + warpId / 2,
                                                     col + cycle * 8 + 4);

                  b[0] = in[b_idx0];
                  b[1] = in[b_idx1];
                  b[2] = in[b_idx2];
                  b[3] = in[b_idx3];

                  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

                  asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(c0[z * 2].x),
                      "=f"(c0[z * 2].y),
                      "=f"(c1[z * 2].x),
                      "=f"(c1[z * 2].y)
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(c0[z * 2].x),
                      "f"(c0[z * 2].y),
                      "f"(c1[z * 2].x),
                      "f"(c1[z * 2].y));

                  asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(c0[z * 2 + 1].x),
                      "=f"(c0[z * 2 + 1].y),
                      "=f"(c1[z * 2 + 1].x),
                      "=f"(c1[z * 2 + 1].y)
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(B[2]),
                      "r"(B[3]),
                      "f"(c0[z * 2 + 1].x),
                      "f"(c0[z * 2 + 1].y),
                      "f"(c1[z * 2 + 1].x),
                      "f"(c1[z * 2 + 1].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 4; ++z)
            {
              const int c_idx0 =
                ((z * 4 + warpId / 2) * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 row * offset) ^
                Util::get_base<n_dofs_1d, float>((z * 4 + warpId / 2), row);
              const int c_idx1 =
                ((z * 4 + warpId / 2) * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (row + 8) * offset) ^
                Util::get_base<n_dofs_1d, float>(z * 4 + warpId / 2, row + 8);

              *((float2 *)(out + c_idx0)) = c0[z];
              *((float2 *)(out + c_idx1)) = c1[z];
            }
        }
    }
  };
#endif


#if MMAKERNEL == 4
  template <typename T>
  struct TPEvaluatorBase<T, 16, float, LaplaceVariant::TensorCoreMMA, 3>
  {
    using Number = float;

    static constexpr int n_dofs_1d = 16;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int warpId = threadIdx.y / 2;

      const int tid   = (threadIdx.y * n_dofs_1d + threadIdx.x) & 31;
      const int rowId = tid / 8;
      const int colId = tid & 7;

      const int row = tid / 4;
      const int col = tid & 3;

      constexpr int offset = n_dofs_1d * n_dofs_1d;

      if constexpr (direction == 0)
        {
          float2 c0[2];
          float2 c1[2];

          float a[4];
          float b[2];

          for (int z = 0; z < 2; ++z)
            {
              c0[z] = {0, 0};
              c1[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int b_idx0 =
                ((col + cycle * 8) * n_dofs_1d + row + (warpId & 1) * 8) ^
                Util::get_base<n_dofs_1d, float>(col + cycle * 8);
              const int b_idx1 =
                ((col + cycle * 8 + 4) * n_dofs_1d + row + (warpId & 1) * 8) ^
                Util::get_base<n_dofs_1d, float>(col + cycle * 8 + 4);

              b[0] = shape_data[b_idx0];
              b[1] = shape_data[b_idx1];

              uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

              for (int z = 0; z < 2; ++z)
                {
                  const int a_idx =
                    ((colId + (rowId & 1) * 8) * n_dofs_1d + (rowId / 2) * 4 +
                     cycle * 8 + (z * 8 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(colId + (rowId & 1) * 8,
                                                     z * 8 + warpId / 2);

                  auto smem_ptr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&in[a_idx]));

                  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                               "{%0, %1, %2, %3}, [%4]; "
                               : "=f"(a[0]), "=f"(a[1]), "=f"(a[2]), "=f"(a[3])
                               : "r"(smem_ptr));

                  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);

                  asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(c0[z].x), "=f"(c0[z].y), "=f"(c1[z].x), "=f"(c1[z].y)
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(c0[z].x),
                      "f"(c0[z].y),
                      "f"(c1[z].x),
                      "f"(c1[z].y));
                }
            }

          for (int z = 0; z < 2; ++z)
            {
              const int c_idx0 =
                (row * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (z * 8 + warpId / 2) * offset) ^
                Util::get_base<n_dofs_1d, float>(row, z * 8 + warpId / 2);
              const int c_idx1 =
                ((row + 8) * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (z * 8 + warpId / 2) * offset) ^
                Util::get_base<n_dofs_1d, float>(row + 8, z * 8 + warpId / 2);

              *((float2 *)(out + c_idx0)) = c0[z];
              *((float2 *)(out + c_idx1)) = c1[z];
            }
        }
      else if (direction == 1)
        {
          float2 c0[2];
          float2 c1[2];

          float a[4];
          float b[2];

          for (int z = 0; z < 2; ++z)
            {
              if constexpr (add)
                {
                  const int c_idx0 =
                    (row * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                     (z * 8 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row, z * 8 + warpId / 2);
                  const int c_idx1 =
                    ((row + 8) * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                     (z * 8 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row + 8,
                                                     z * 8 + warpId / 2);

                  c0[z] = *((float2 *)(out + c_idx0));
                  c1[z] = *((float2 *)(out + c_idx1));
                }
              else
                {
                  c0[z] = {0, 0};
                  c1[z] = {0, 0};
                }
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx =
                ((colId + (rowId & 1) * 8) * n_dofs_1d + (rowId / 2) * 4 +
                 cycle * 8) ^
                Util::get_base<n_dofs_1d, float>(colId + (rowId & 1) * 8);

              auto smem_ptr = static_cast<uint32_t>(
                __cvta_generic_to_shared(&shape_data[a_idx]));

              asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                           "{%0, %1, %2, %3}, [%4]; "
                           : "=f"(a[0]), "=f"(a[1]), "=f"(a[2]), "=f"(a[3])
                           : "r"(smem_ptr));

              uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);

              for (int z = 0; z < 2; ++z)
                {
                  const int b_idx0 =
                    ((col + cycle * 8) * n_dofs_1d + row + (warpId & 1) * 8 +
                     (z * 8 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(col + cycle * 8,
                                                     z * 8 + warpId / 2);
                  const int b_idx1 =
                    ((col + cycle * 8 + 4) * n_dofs_1d + row +
                     (warpId & 1) * 8 + (z * 8 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(col + cycle * 8 + 4,
                                                     z * 8 + warpId / 2);

                  b[0] = in[b_idx0];
                  b[1] = in[b_idx1];

                  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

                  asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(c0[z].x), "=f"(c0[z].y), "=f"(c1[z].x), "=f"(c1[z].y)
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(c0[z].x),
                      "f"(c0[z].y),
                      "f"(c1[z].x),
                      "f"(c1[z].y));
                }
            }

          for (int z = 0; z < 2; ++z)
            {
              const int c_idx0 =
                (row * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (z * 8 + warpId / 2) * offset) ^
                Util::get_base<n_dofs_1d, float>(row, z * 8 + warpId / 2);
              const int c_idx1 =
                ((row + 8) * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (z * 8 + warpId / 2) * offset) ^
                Util::get_base<n_dofs_1d, float>(row + 8, z * 8 + warpId / 2);

              *((float2 *)(out + c_idx0)) = c0[z];
              *((float2 *)(out + c_idx1)) = c1[z];
            }
        }
      else
        {
          float2 c0[2];
          float2 c1[2];

          float a[4];
          float b[2];

          for (int z = 0; z < 2; ++z)
            {
              if constexpr (add)
                {
                  const int c_idx0 =
                    ((z * 8 + warpId / 2) * n_dofs_1d + 2 * col +
                     (warpId & 1) * 8 + row * offset) ^
                    Util::get_base<n_dofs_1d, float>(z * 8 + warpId / 2, row);
                  const int c_idx1 =
                    ((z * 8 + warpId / 2) * n_dofs_1d + 2 * col +
                     (warpId & 1) * 8 + (row + 8) * offset) ^
                    Util::get_base<n_dofs_1d, float>(z * 8 + warpId / 2,
                                                     row + 8);

                  c0[z] = *((float2 *)(out + c_idx0));
                  c1[z] = *((float2 *)(out + c_idx1));
                }
              else
                {
                  c0[z] = {0, 0};
                  c1[z] = {0, 0};
                }
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx =
                ((colId + (rowId & 1) * 8) * n_dofs_1d + (rowId / 2) * 4 +
                 cycle * 8) ^
                Util::get_base<n_dofs_1d, float>(colId + (rowId & 1) * 8);

              auto smem_ptr = static_cast<uint32_t>(
                __cvta_generic_to_shared(&shape_data[a_idx]));

              asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                           "{%0, %1, %2, %3}, [%4]; "
                           : "=f"(a[0]), "=f"(a[1]), "=f"(a[2]), "=f"(a[3])
                           : "r"(smem_ptr));

              uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);

              for (int z = 0; z < 2; ++z)
                {
                  const int b_idx0 =
                    ((z * 8 + warpId / 2) * n_dofs_1d + row + (warpId & 1) * 8 +
                     (col + cycle * 8) * offset) ^
                    Util::get_base<n_dofs_1d, float>(z * 8 + warpId / 2,
                                                     col + cycle * 8);
                  const int b_idx1 =
                    ((z * 8 + warpId / 2) * n_dofs_1d + row + (warpId & 1) * 8 +
                     (col + cycle * 8 + 4) * offset) ^
                    Util::get_base<n_dofs_1d, float>(z * 8 + warpId / 2,
                                                     col + cycle * 8 + 4);

                  b[0] = in[b_idx0];
                  b[1] = in[b_idx1];

                  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

                  asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(c0[z].x), "=f"(c0[z].y), "=f"(c1[z].x), "=f"(c1[z].y)
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(c0[z].x),
                      "f"(c0[z].y),
                      "f"(c1[z].x),
                      "f"(c1[z].y));
                }
            }

          for (int z = 0; z < 2; ++z)
            {
              const int c_idx0 =
                ((z * 8 + warpId / 2) * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 row * offset) ^
                Util::get_base<n_dofs_1d, float>(z * 8 + warpId / 2, row);
              const int c_idx1 =
                ((z * 8 + warpId / 2) * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (row + 8) * offset) ^
                Util::get_base<n_dofs_1d, float>(z * 8 + warpId / 2, row + 8);

              *((float2 *)(out + c_idx0)) = c0[z];
              *((float2 *)(out + c_idx1)) = c1[z];
            }
        }
    }
  };
#endif


#if MMAKERNEL == 5
  template <typename T>
  struct TPEvaluatorBase<T, 16, float, LaplaceVariant::TensorCoreMMA, 3>
  {
    using Number = float;

    static constexpr int n_dofs_1d = 16;

    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int warpId = threadIdx.y / 2;

      const int tid   = (threadIdx.y * n_dofs_1d + threadIdx.x) & 31;
      const int rowId = tid / 8;
      const int colId = tid & 7;

      const int row = tid / 4;
      const int col = tid & 3;

      constexpr int offset = n_dofs_1d * n_dofs_1d;

      if constexpr (direction == 0)
        {
          float2 c0[n_dofs_1d / 2];
          float2 c1[n_dofs_1d / 2];

          float a[4];
          float b[2];

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              c0[z] = {0, 0};
              c1[z] = {0, 0};
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int b_idx0 =
                ((col + cycle * 8) * n_dofs_1d + row + (warpId & 1) * 8) ^
                Util::get_base<n_dofs_1d, float>(col + cycle * 8);
              const int b_idx1 =
                ((col + cycle * 8 + 4) * n_dofs_1d + row + (warpId & 1) * 8) ^
                Util::get_base<n_dofs_1d, float>(col + cycle * 8 + 4);

              b[0] = shape_data[b_idx0];
              b[1] = shape_data[b_idx1];

              uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int a_idx =
                    ((colId + (rowId & 1) * 8) * n_dofs_1d + (rowId / 2) * 4 +
                     cycle * 8 + (z * 2 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(colId + (rowId & 1) * 8,
                                                     z * 2 + warpId / 2);

                  auto smem_ptr =
                    static_cast<uint32_t>(__cvta_generic_to_shared(&in[a_idx]));
#  if TIMING == 1
                  auto start = clock64();
#  endif
                  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                               "{%0, %1, %2, %3}, [%4]; "
                               : "=f"(a[0]), "=f"(a[1]), "=f"(a[2]), "=f"(a[3])
                               : "r"(smem_ptr));
#  if TIMING == 1
                  auto elapsed = clock64() - start;
                  if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
                    printf(
                      "mma load frag a dir-%d loop-%d timing info: %ld cycles\n",
                      direction,
                      cycle,
                      elapsed);
#  endif
                  uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
#  if TIMING == 1
                  start = clock64();
#  endif
                  asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(c0[z].x), "=f"(c0[z].y), "=f"(c1[z].x), "=f"(c1[z].y)
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(c0[z].x),
                      "f"(c0[z].y),
                      "f"(c1[z].x),
                      "f"(c1[z].y));
#  if TIMING == 1
                  elapsed = clock64() - start;
                  if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
                    printf(
                      "mma.sync.aligned.m16n8k8 dir-%d loop-(%d,%d) timing info: %ld cycles\n",
                      direction,
                      cycle,
                      z,
                      elapsed);
#  endif
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx0 =
                (row * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (z * 2 + warpId / 2) * offset) ^
                Util::get_base<n_dofs_1d, float>(row, z * 2 + warpId / 2);
              const int c_idx1 =
                ((row + 8) * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (z * 2 + warpId / 2) * offset) ^
                Util::get_base<n_dofs_1d, float>(row + 8, z * 2 + warpId / 2);

              *((float2 *)(out + c_idx0)) = c0[z];
              *((float2 *)(out + c_idx1)) = c1[z];
            }
        }
      else if (direction == 1)
        {
          float2 c0[n_dofs_1d / 2];
          float2 c1[n_dofs_1d / 2];

          float a[4];
          float b[2];

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              if constexpr (add)
                {
                  const int c_idx0 =
                    (row * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                     (z * 2 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row, z * 2 + warpId / 2);
                  const int c_idx1 =
                    ((row + 8) * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                     (z * 2 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(row + 8,
                                                     z * 2 + warpId / 2);

                  c0[z] = *((float2 *)(out + c_idx0));
                  c1[z] = *((float2 *)(out + c_idx1));
                }
              else
                {
                  c0[z] = {0, 0};
                  c1[z] = {0, 0};
                }
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx =
                ((colId + (rowId & 1) * 8) * n_dofs_1d + (rowId / 2) * 4 +
                 cycle * 8) ^
                Util::get_base<n_dofs_1d, float>(colId + (rowId & 1) * 8);

              auto smem_ptr = static_cast<uint32_t>(
                __cvta_generic_to_shared(&shape_data[a_idx]));

              asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                           "{%0, %1, %2, %3}, [%4]; "
                           : "=f"(a[0]), "=f"(a[1]), "=f"(a[2]), "=f"(a[3])
                           : "r"(smem_ptr));

              uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int b_idx0 =
                    ((col + cycle * 8) * n_dofs_1d + row + (warpId & 1) * 8 +
                     (z * 2 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(col + cycle * 8,
                                                     z * 2 + warpId / 2);
                  const int b_idx1 =
                    ((col + cycle * 8 + 4) * n_dofs_1d + row +
                     (warpId & 1) * 8 + (z * 2 + warpId / 2) * offset) ^
                    Util::get_base<n_dofs_1d, float>(col + cycle * 8 + 4,
                                                     z * 2 + warpId / 2);

                  b[0] = in[b_idx0];
                  b[1] = in[b_idx1];

                  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

                  asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(c0[z].x), "=f"(c0[z].y), "=f"(c1[z].x), "=f"(c1[z].y)
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(c0[z].x),
                      "f"(c0[z].y),
                      "f"(c1[z].x),
                      "f"(c1[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx0 =
                (row * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (z * 2 + warpId / 2) * offset) ^
                Util::get_base<n_dofs_1d, float>(row, z * 2 + warpId / 2);
              const int c_idx1 =
                ((row + 8) * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (z * 2 + warpId / 2) * offset) ^
                Util::get_base<n_dofs_1d, float>(row + 8, z * 2 + warpId / 2);

              *((float2 *)(out + c_idx0)) = c0[z];
              *((float2 *)(out + c_idx1)) = c1[z];
            }
        }
      else
        {
          float2 c0[n_dofs_1d / 2];
          float2 c1[n_dofs_1d / 2];

          float a[4];
          float b[2];

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              if constexpr (add)
                {
                  const int c_idx0 =
                    ((z * 2 + warpId / 2) * n_dofs_1d + 2 * col +
                     (warpId & 1) * 8 + row * offset) ^
                    Util::get_base<n_dofs_1d, float>((z * 2 + warpId / 2), row);
                  const int c_idx1 =
                    ((z * 2 + warpId / 2) * n_dofs_1d + 2 * col +
                     (warpId & 1) * 8 + (row + 8) * offset) ^
                    Util::get_base<n_dofs_1d, float>(z * 2 + warpId / 2,
                                                     row + 8);

                  c0[z] = *((float2 *)(out + c_idx0));
                  c1[z] = *((float2 *)(out + c_idx1));
                }
              else
                {
                  c0[z] = {0, 0};
                  c1[z] = {0, 0};
                }
            }

          for (int cycle = 0; cycle < 2; ++cycle)
            {
              const int a_idx =
                ((colId + (rowId & 1) * 8) * n_dofs_1d + (rowId / 2) * 4 +
                 cycle * 8) ^
                Util::get_base<n_dofs_1d, float>(colId + (rowId & 1) * 8);

              auto smem_ptr = static_cast<uint32_t>(
                __cvta_generic_to_shared(&shape_data[a_idx]));

              asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                           "{%0, %1, %2, %3}, [%4]; "
                           : "=f"(a[0]), "=f"(a[1]), "=f"(a[2]), "=f"(a[3])
                           : "r"(smem_ptr));

              uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  const int b_idx0 =
                    ((z * 2 + warpId / 2) * n_dofs_1d + row + (warpId & 1) * 8 +
                     (col + cycle * 8) * offset) ^
                    Util::get_base<n_dofs_1d, float>(z * 2 + warpId / 2,
                                                     col + cycle * 8);
                  const int b_idx1 =
                    ((z * 2 + warpId / 2) * n_dofs_1d + row + (warpId & 1) * 8 +
                     (col + cycle * 8 + 4) * offset) ^
                    Util::get_base<n_dofs_1d, float>(z * 2 + warpId / 2,
                                                     col + cycle * 8 + 4);

                  b[0] = in[b_idx0];
                  b[1] = in[b_idx1];

                  uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);

                  asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(c0[z].x), "=f"(c0[z].y), "=f"(c1[z].x), "=f"(c1[z].y)
                    : "r"(A[0]),
                      "r"(A[1]),
                      "r"(A[2]),
                      "r"(A[3]),
                      "r"(B[0]),
                      "r"(B[1]),
                      "f"(c0[z].x),
                      "f"(c0[z].y),
                      "f"(c1[z].x),
                      "f"(c1[z].y));
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            {
              const int c_idx0 =
                ((z * 2 + warpId / 2) * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 row * offset) ^
                Util::get_base<n_dofs_1d, float>((z * 2 + warpId / 2), row);
              const int c_idx1 =
                ((z * 2 + warpId / 2) * n_dofs_1d + 2 * col + (warpId & 1) * 8 +
                 (row + 8) * offset) ^
                Util::get_base<n_dofs_1d, float>(z * 2 + warpId / 2, row + 8);

              *((float2 *)(out + c_idx0)) = c0[z];
              *((float2 *)(out + c_idx1)) = c1[z];
            }
        }
    }
  };
#endif

  template <typename T>
  struct TPEvaluatorBase<T, 8, double, LaplaceVariant::TensorCore, 2>
  {
    using Number = double;
    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr int n_dofs_1d   = 8;
      constexpr int skew_double = Util::padding;

      if (direction == 0)
        {
          const int warpId = (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;

          wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major>
            a_frag;
          wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major>
                                                             b_frag;
          wmma::fragment<wmma::accumulator, 8, 8, 4, double> c_frag;

          wmma::fill_fragment(c_frag, 0.0f);

          if (warpId != 0)
            return;

          for (int i = 0; i < 2; ++i)
            {
              wmma::load_matrix_sync(b_frag,
                                     &shape_data[i * 4 * (8 + skew_double)],
                                     8 + skew_double);

              wmma::load_matrix_sync(
                a_frag,
                &in[warpId * 8 * (8 + skew_double) + i * 4],
                8 + skew_double);

              wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

          wmma::store_matrix_sync(&out[warpId * 8 * (8 + skew_double)],
                                  c_frag,
                                  8 + skew_double,
                                  wmma::mem_row_major);
        }
      else if (direction == 1)
        {
          const int warpId = (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;

          wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major>
            a_frag;
          wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major>
                                                             b_frag;
          wmma::fragment<wmma::accumulator, 8, 8, 4, double> c_frag;

          if (add)
            wmma::load_matrix_sync(c_frag,
                                   &out[warpId * 8 * (8 + skew_double)],
                                   8 + skew_double,
                                   wmma::mem_row_major);
          else
            wmma::fill_fragment(c_frag, 0.0f);

          if (add)
            __syncthreads();

          if (warpId != 0)
            return;

          for (int i = 0; i < 2; ++i)
            {
              wmma::load_matrix_sync(a_frag,
                                     &shape_data[i * 4],
                                     8 + skew_double);

              wmma::load_matrix_sync(
                b_frag,
                &in[warpId * 8 * (8 + skew_double) + i * 4 * (8 + skew_double)],
                8 + skew_double);

              wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

          wmma::store_matrix_sync(&out[warpId * 8 * (8 + skew_double)],
                                  c_frag,
                                  8 + skew_double,
                                  wmma::mem_row_major);
        }
    }
  };

  template <typename T>
  struct TPEvaluatorBase<T, 8, double, LaplaceVariant::TensorCore, 3>
  {
    using Number = double;
    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr int n_dofs_1d   = 8;
      constexpr int skew_double = Util::padding;

      if (direction == 0)
        {
          const int warpId = (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;
          wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major>
            a_frag;
          wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major>
            b_frag;
          wmma::fragment<wmma::accumulator, 8, 8, 4, double>
            c_frag[n_dofs_1d / 2];

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            wmma::fill_fragment(c_frag[z], 0.0f);

          for (int i = 0; i < 2; ++i)
            {
#if TIMING == 1
              auto start = clock64();
#endif
              wmma::load_matrix_sync(b_frag,
                                     &shape_data[i * 4 * (8 + skew_double)],
                                     8 + skew_double);
#if TIMING == 1
              auto elapsed = clock64() - start;
              if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
                printf(
                  "wmma load frag b dir-%d loop-%d timing info: %ld cycles\n",
                  direction,
                  i,
                  elapsed);
#endif
              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
#if TIMING == 1
                  start = clock64();
#endif
                  wmma::load_matrix_sync(
                    a_frag,
                    &in[(z * 2 + warpId) * 8 * (8 + skew_double) + i * 4],
                    8 + skew_double);
#if TIMING == 1
                  elapsed = clock64() - start;
                  if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
                    printf(
                      "wmma load frag a dir-%d loop-(%d,%d) timing info: %ld cycles\n",
                      direction,
                      i,
                      z,
                      elapsed);
#endif

#if TIMING == 1
                  start = clock64();
#endif
                  wmma::mma_sync(c_frag[z], a_frag, b_frag, c_frag[z]);
#if TIMING == 1
                  elapsed = clock64() - start;
                  if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
                    printf(
                      "wmma mma_sync dir-%d loop-(%d,%d) timing info: %ld cycles\n",
                      direction,
                      i,
                      z,
                      elapsed);
#endif
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
#if TIMING == 1
            {
              auto start = clock64();
#endif
              wmma::store_matrix_sync(
                &out[(z * 2 + warpId) * 8 * (8 + skew_double)],
                c_frag[z],
                8 + skew_double,
                wmma::mem_row_major);
#if TIMING == 1
              auto elapsed = clock64() - start;
              if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0)
                printf(
                  "wmma store frag c dir-%d loop-%d timing info: %ld cycles\n",
                  direction,
                  z,
                  elapsed);
            }
#endif
        }
      else if (direction == 1)
        {
          const int warpId = (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;

          wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major>
            a_frag;
          wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major>
            b_frag;
          wmma::fragment<wmma::accumulator, 8, 8, 4, double>
            c_frag[n_dofs_1d / 2];

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            if (add)
              wmma::load_matrix_sync(
                c_frag[z],
                &out[(z * 2 + warpId) * 8 * (8 + skew_double)],
                8 + skew_double,
                wmma::mem_row_major);
            else
              wmma::fill_fragment(c_frag[z], 0.0f);

          if (add)
            __syncthreads();

          for (int i = 0; i < 2; ++i)
            {
              wmma::load_matrix_sync(a_frag,
                                     &shape_data[i * 4],
                                     8 + skew_double);

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  wmma::load_matrix_sync(
                    b_frag,
                    &in[(z * 2 + warpId) * 8 * (8 + skew_double) +
                        i * 4 * (8 + skew_double)],
                    8 + skew_double);

                  wmma::mma_sync(c_frag[z], a_frag, b_frag, c_frag[z]);
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            wmma::store_matrix_sync(
              &out[(z * 2 + warpId) * 8 * (8 + skew_double)],
              c_frag[z],
              8 + skew_double,
              wmma::mem_row_major);
        }
      else
        {
          constexpr int n_dofs_1d_padding = n_dofs_1d + skew_double;
          constexpr int stride            = n_dofs_1d * n_dofs_1d_padding;

          const int row = threadIdx.y;
          const int col = threadIdx.x % n_dofs_1d;

          double pval[n_dofs_1d];

          // kernel product: A kdot src, [N x N] * [N^dim, 1]
          for (int z = 0; z < n_dofs_1d; ++z)
            {
              pval[z] = 0;
              // #pragma unroll
              for (int k = 0; k < n_dofs_1d; ++k)
                {
                  const int shape_idx = row * n_dofs_1d_padding + k;
                  const int source_idx =
                    z * n_dofs_1d_padding + col + k * stride;

                  pval[z] += shape_data[shape_idx] * in[source_idx];
                }
            }

          for (int z = 0; z < n_dofs_1d; ++z)
            {
              const int destination_idx =
                z * n_dofs_1d_padding + col + row * stride;

              if (add)
                out[destination_idx] += pval[z];
              else if (sub)
                out[destination_idx] -= pval[z];
              else
                out[destination_idx] = pval[z];
            }
        }
    }
  };

  template <typename T>
  struct TPEvaluatorBase<T, 16, double, LaplaceVariant::TensorCore, 2>
  {
    using Number = double;
    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr int n_dofs_1d   = 16;
      constexpr int skew_double = Util::padding;

      if (direction == 0)
        {
          const int warpId = (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;
          const int subId  = warpId % 4;

          wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major>
            a_frag;
          wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major>
                                                             b_frag;
          wmma::fragment<wmma::accumulator, 8, 8, 4, double> c_frag;

          wmma::fill_fragment(c_frag, 0.0f);

          if (warpId > 3)
            return;

          for (int i = 0; i < 4; ++i)
            {
              wmma::load_matrix_sync(
                b_frag,
                &shape_data[subId / 2 * 8 + i * 4 * (16 + skew_double)],
                16 + skew_double);

              wmma::load_matrix_sync(
                a_frag,
                &in[subId % 2 * (16 + skew_double) * 8 + i * 4],
                16 + skew_double);

              wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

          wmma::store_matrix_sync(
            &out[subId % 2 * (16 + skew_double) * 8 + subId / 2 * 8],
            c_frag,
            16 + skew_double,
            wmma::mem_row_major);
        }
      else if (direction == 1)
        {
          const int warpId = (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;
          const int subId  = warpId % 4;

          wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major>
            a_frag;
          wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major>
                                                             b_frag;
          wmma::fragment<wmma::accumulator, 8, 8, 4, double> c_frag;

          if (add)
            wmma::load_matrix_sync(
              c_frag,
              &out[subId / 2 * (16 + skew_double) * 8 + subId % 2 * 8],
              16 + skew_double,
              wmma::mem_row_major);
          else
            wmma::fill_fragment(c_frag, 0.0f);

          if (add)
            __syncthreads();

          if (warpId > 3)
            return;

          for (int i = 0; i < 4; ++i)
            {
              wmma::load_matrix_sync(
                a_frag,
                &shape_data[subId / 2 * (16 + skew_double) * 8 + i * 4],
                16 + skew_double);

              wmma::load_matrix_sync(
                b_frag,
                &in[subId % 2 * 8 + i * 4 * (16 + skew_double)],
                16 + skew_double);

              wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

          wmma::store_matrix_sync(
            &out[subId / 2 * (16 + skew_double) * 8 + subId % 2 * 8],
            c_frag,
            16 + skew_double,
            wmma::mem_row_major);
        }
    }
  };

  template <typename T>
  struct TPEvaluatorBase<T, 16, double, LaplaceVariant::TensorCore, 3>
  {
    using Number = double;
    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr int n_dofs_1d   = 16;
      constexpr int skew_double = Util::padding;

      if (direction == 0)
        {
          const int warpId = (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;
          // sub matrix id
          const int subId = warpId % 4;

          wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major>
            a_frag;
          wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major>
            b_frag;
          wmma::fragment<wmma::accumulator, 8, 8, 4, double>
            c_frag[n_dofs_1d / 2];

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            wmma::fill_fragment(c_frag[z], 0.0f);

          for (int i = 0; i < 4; ++i)
            {
#ifdef SKIPZERO
              if ((shape_data[(i / 2) * (n_dofs_1d * 8 + 8)]) < 1e-10)
                continue;
#endif

              wmma::load_matrix_sync(
                b_frag,
                &shape_data[subId / 2 * 8 + i * 4 * (16 + skew_double)],
                16 + skew_double);

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  wmma::load_matrix_sync(
                    a_frag,
                    &in[(z * 2 + warpId / 4) * (16 + skew_double) * 16 +
                        subId % 2 * (16 + skew_double) * 8 + i * 4],
                    16 + skew_double);

                  wmma::mma_sync(c_frag[z], a_frag, b_frag, c_frag[z]);
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            wmma::store_matrix_sync(
              &out[(z * 2 + warpId / 4) * (16 + skew_double) * 16 +
                   subId % 2 * (16 + skew_double) * 8 + subId / 2 * 8],
              c_frag[z],
              16 + skew_double,
              wmma::mem_row_major);
        }
      else if (direction == 1)
        {
          const int warpId = (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;
          // sub matrix id
          const int subId = warpId % 4;

          wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major>
            a_frag;
          wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major>
            b_frag;
          wmma::fragment<wmma::accumulator, 8, 8, 4, double>
            c_frag[n_dofs_1d / 2];

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            if (add)
              wmma::load_matrix_sync(
                c_frag[z],
                &out[(z * 2 + warpId / 4) * (16 + skew_double) * 16 +
                     subId / 2 * (16 + skew_double) * 8 + subId % 2 * 8],
                16 + skew_double,
                wmma::mem_row_major);
            else
              wmma::fill_fragment(c_frag[z], 0.0f);

          if (add)
            __syncthreads();

          for (int i = 0; i < 4; ++i)
            {
#ifdef SKIPZERO
              if ((shape_data[(i / 2) * (n_dofs_1d * 8 + 8)]) < 1e-10)
                continue;
#endif
              wmma::load_matrix_sync(
                a_frag,
                &shape_data[subId / 2 * (16 + skew_double) * 8 + i * 4],
                16 + skew_double);

              for (int z = 0; z < n_dofs_1d / 2; ++z)
                {
                  wmma::load_matrix_sync(
                    b_frag,
                    &in[(z * 2 + warpId / 4) * (16 + skew_double) * 16 +
                        subId % 2 * 8 + i * 4 * (16 + skew_double)],
                    16 + skew_double);

                  wmma::mma_sync(c_frag[z], a_frag, b_frag, c_frag[z]);
                }
            }

          for (int z = 0; z < n_dofs_1d / 2; ++z)
            wmma::store_matrix_sync(
              &out[(z * 2 + warpId / 4) * (16 + skew_double) * 16 +
                   subId / 2 * (16 + skew_double) * 8 + subId % 2 * 8],
              c_frag[z],
              16,
              wmma::mem_row_major);
        }
      else
        {
          constexpr int n_dofs_1d_padding = n_dofs_1d + skew_double;
          constexpr int stride            = n_dofs_1d * n_dofs_1d_padding;

          const int row = threadIdx.y;
          const int col = threadIdx.x % n_dofs_1d;

          double pval[n_dofs_1d];

          // kernel product: A kdot src, [N x N] * [N^dim, 1]
          for (int z = 0; z < n_dofs_1d; ++z)
            {
              pval[z] = 0;
              // #pragma unroll
              for (int k = 0; k < n_dofs_1d; ++k)
                {
                  const int shape_idx = row * n_dofs_1d_padding + k;
                  const int source_idx =
                    z * n_dofs_1d_padding + col + k * stride;

                  pval[z] += shape_data[shape_idx] * in[source_idx];
                }
            }

          for (int z = 0; z < n_dofs_1d; ++z)
            {
              const int destination_idx =
                z * n_dofs_1d_padding + col + row * stride;

              if (add)
                out[destination_idx] += pval[z];
              else if (sub)
                out[destination_idx] -= pval[z];
              else
                out[destination_idx] = pval[z];
            }
        }
    }
  };


  // TODO
  template <typename T>
  struct TPEvaluatorBase<T, 16, float, LaplaceVariant::TensorCore, 2>
  {
    using Number = float;
    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr int n_dofs_1d   = 16;
      constexpr int skew_double = Util::padding;

      if (direction == 0)
        {
          const int warpId = (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;

          wmma::fragment<wmma::matrix_a,
                         16,
                         16,
                         8,
                         wmma::precision::tf32,
                         wmma::row_major>
            a_frag;
          wmma::fragment<wmma::matrix_b,
                         16,
                         16,
                         8,
                         wmma::precision::tf32,
                         wmma::col_major>
                                                              b_frag;
          wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag;

          wmma::fill_fragment(c_frag, 0.0f);

          if (warpId != 0)
            return;

          for (int i = 0; i < 2; ++i)
            {
              wmma::load_matrix_sync(a_frag,
                                     &shape_data[i * 8],
                                     16 + skew_double);
              // #pragma unroll
              //               for (int t = 0; t < a_frag.num_elements; t++)
              //                 {
              //                   a_frag.x[t] =
              //                   wmma::__float_to_tf32(a_frag.x[t]);
              //                 }

              wmma::load_matrix_sync(
                b_frag,
                &in[warpId * 16 * (16 + skew_double) + i * 8],
                16 + skew_double);
              // #pragma unroll
              //               for (int t = 0; t < b_frag.num_elements; t++)
              //                 {
              //                   b_frag.x[t] =
              //                   wmma::__float_to_tf32(b_frag.x[t]);
              //                 }

              wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

          wmma::store_matrix_sync(&out[warpId * 16 * (16 + skew_double)],
                                  c_frag,
                                  16 + skew_double,
                                  wmma::mem_col_major);
        }
      else if (direction == 1)
        {
          const int warpId = (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;

          wmma::fragment<wmma::matrix_a,
                         16,
                         16,
                         8,
                         wmma::precision::tf32,
                         wmma::row_major>
            a_frag;
          wmma::fragment<wmma::matrix_b,
                         16,
                         16,
                         8,
                         wmma::precision::tf32,
                         wmma::row_major>
                                                              b_frag;
          wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag;

          if (add)
            wmma::load_matrix_sync(c_frag,
                                   &out[warpId * 16 * (16 + skew_double)],
                                   16 + skew_double,
                                   wmma::mem_row_major);
          else
            wmma::fill_fragment(c_frag, 0.0f);

          if (add)
            __syncthreads();

          if (warpId != 0)
            return;

          for (int i = 0; i < 2; ++i)
            {
              wmma::load_matrix_sync(a_frag,
                                     &shape_data[i * 8],
                                     16 + skew_double);
              // #pragma unroll
              //               for (int t = 0; t < a_frag.num_elements; t++)
              //                 {
              //                   a_frag.x[t] =
              //                   wmma::__float_to_tf32(a_frag.x[t]);
              //                 }

              wmma::load_matrix_sync(b_frag,
                                     &in[warpId * 16 * (16 + skew_double) +
                                         i * 8 * (16 + skew_double)],
                                     16 + skew_double);
              // #pragma unroll
              //               for (int t = 0; t < b_frag.num_elements; t++)
              //                 {
              //                   b_frag.x[t] =
              //                   wmma::__float_to_tf32(b_frag.x[t]);
              //                 }
              wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }

          wmma::store_matrix_sync(&out[warpId * 16 * (16 + skew_double)],
                                  c_frag,
                                  16 + skew_double,
                                  wmma::mem_row_major);
        }
    }
  };

  template <typename T>
  struct TPEvaluatorBase<T, 16, float, LaplaceVariant::TensorCore, 3>
  {
    using Number = float;
    /**
     * Default constructor.
     */
    __device__
    TPEvaluatorBase() = default;

    /**
     * Implements a matrix-vector product for Laplacian.
     */
    __device__ void
    vmult(Number       *dst,
          const Number *src,
          const Number *mass_matrix,
          const Number *derivative_matrix,
          Number       *tmp)
    {
      static_cast<T *>(this)->vmult_impl(
        dst, src, mass_matrix, derivative_matrix, tmp);
    }

    template <int direction, bool add, bool sub = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr int n_dofs_1d   = 16;
      constexpr int skew_double = Util::padding;

      if (direction == 0)
        {
          const int warpId = (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;

          wmma::fragment<wmma::matrix_a,
                         16,
                         16,
                         8,
                         wmma::precision::tf32,
                         wmma::row_major>
            a_frag;
          wmma::fragment<wmma::matrix_b,
                         16,
                         16,
                         8,
                         wmma::precision::tf32,
                         wmma::row_major>
            b_frag;
          wmma::fragment<wmma::accumulator, 16, 16, 8, float>
            c_frag[n_dofs_1d / 8];

          for (int z = 0; z < n_dofs_1d / 8; ++z)
            wmma::fill_fragment(c_frag[z], 0.0f);

          for (int i = 0; i < 2; ++i)
            {
              wmma::load_matrix_sync(b_frag,
                                     &shape_data[i * 8 * (16 + skew_double)],
                                     16 + skew_double);
              // #pragma unroll
              //               for (int t = 0; t < a_frag.num_elements; t++)
              //                 {
              //                   a_frag.x[t] =
              //                   wmma::__float_to_tf32(a_frag.x[t]);
              //                 }
              for (int z = 0; z < n_dofs_1d / 8; ++z)
                {
                  wmma::load_matrix_sync(
                    a_frag,
                    &in[(z * 8 + warpId) * 16 * (16 + skew_double) + i * 8],
                    16 + skew_double);
                  // #pragma unroll
                  //                   for (int t = 0; t < b_frag.num_elements;
                  //                   t++)
                  //                     {
                  //                       b_frag.x[t] =
                  //                       wmma::__float_to_tf32(b_frag.x[t]);
                  //                     }

                  wmma::mma_sync(c_frag[z], a_frag, b_frag, c_frag[z]);
                }
            }

          for (int z = 0; z < n_dofs_1d / 8; ++z)
            wmma::store_matrix_sync(
              &out[(z * 8 + warpId) * 16 * (16 + skew_double)],
              c_frag[z],
              16 + skew_double,
              wmma::mem_row_major);
        }
      else if (direction == 1)
        {
          const int warpId = (threadIdx.y * n_dofs_1d + threadIdx.x) / 32;

          wmma::fragment<wmma::matrix_a,
                         16,
                         16,
                         8,
                         wmma::precision::tf32,
                         wmma::row_major>
            a_frag;
          wmma::fragment<wmma::matrix_b,
                         16,
                         16,
                         8,
                         wmma::precision::tf32,
                         wmma::row_major>
            b_frag;
          wmma::fragment<wmma::accumulator, 16, 16, 8, float>
            c_frag[n_dofs_1d / 8];

          for (int z = 0; z < n_dofs_1d / 8; ++z)
            if (add)
              wmma::load_matrix_sync(
                c_frag[z],
                &out[(z * 8 + warpId) * 16 * (16 + skew_double)],
                16 + skew_double,
                wmma::mem_row_major);
            else
              wmma::fill_fragment(c_frag[z], 0.0f);

          if (add)
            __syncthreads();

          for (int i = 0; i < 2; ++i)
            {
              wmma::load_matrix_sync(a_frag,
                                     &shape_data[i * 8],
                                     16 + skew_double);
              // #pragma unroll
              //               for (int t = 0; t < a_frag.num_elements; t++)
              //                 {
              //                   a_frag.x[t] =
              //                   wmma::__float_to_tf32(a_frag.x[t]);
              //                 }
              for (int z = 0; z < n_dofs_1d / 8; ++z)
                {
                  wmma::load_matrix_sync(
                    b_frag,
                    &in[(z * 8 + warpId) * 16 * (16 + skew_double) +
                        i * 8 * (16 + skew_double)],
                    16 + skew_double);
                  // #pragma unroll
                  //                   for (int t = 0; t < b_frag.num_elements;
                  //                   t++)
                  //                     {
                  //                       b_frag.x[t] =
                  //                       wmma::__float_to_tf32(b_frag.x[t]);
                  //                     }
                  wmma::mma_sync(c_frag[z], a_frag, b_frag, c_frag[z]);
                }
            }

          for (int z = 0; z < n_dofs_1d / 8; ++z)
            wmma::store_matrix_sync(
              &out[(z * 8 + warpId) * 16 * (16 + skew_double)],
              c_frag[z],
              16 + skew_double,
              wmma::mem_row_major);
        }
      else
        {
          constexpr int n_dofs_1d_padding = n_dofs_1d + skew_double;
          constexpr int stride            = n_dofs_1d * n_dofs_1d_padding;

          const int row = threadIdx.y;
          const int col = threadIdx.x % n_dofs_1d;

          float pval[n_dofs_1d];

          // kernel product: A kdot src, [N x N] * [N^dim, 1]
          for (int z = 0; z < n_dofs_1d; ++z)
            {
              pval[z] = 0;
              // #pragma unroll
              for (int k = 0; k < n_dofs_1d; ++k)
                {
                  const int shape_idx = row * n_dofs_1d_padding + k;
                  const int source_idx =
                    z * n_dofs_1d_padding + col + k * stride;

                  pval[z] += shape_data[shape_idx] * in[source_idx];
                }
            }

          for (int z = 0; z < n_dofs_1d; ++z)
            {
              const int destination_idx =
                z * n_dofs_1d_padding + col + row * stride;

              if (add)
                out[destination_idx] += pval[z];
              else if (sub)
                out[destination_idx] -= pval[z];
              else
                out[destination_idx] = pval[z];
            }
        }
    }
  };



  ////////////////////////////////////////////////////////////////////
  //////////////////// TPEvaluatorLaplace ////////////////////////////
  ////////////////////////////////////////////////////////////////////
  template <LaplaceVariant laplace_type,
            typename Number,
            typename Number2,
            int n_dofs_1d,
            int dim>
  struct TPEvaluatorLaplace
    : TPEvaluatorBase<
        TPEvaluatorLaplace<laplace_type, Number, Number2, n_dofs_1d, dim>,
        n_dofs_1d,
        Number,
        laplace_type,
        dim,
        Number2>
  {
    using TPEvaluatorBase<
      TPEvaluatorLaplace<laplace_type, Number, Number2, n_dofs_1d, dim>,
      n_dofs_1d,
      Number,
      laplace_type,
      dim,
      Number2>::apply;
    __device__ void
    vmult()
    {}
  };

  template <LaplaceVariant laplace_type,
            typename Number,
            typename Number2,
            int n_dofs_1d>
  struct TPEvaluatorLaplace<laplace_type, Number, Number2, n_dofs_1d, 2>
    : TPEvaluatorBase<
        TPEvaluatorLaplace<laplace_type, Number, Number2, n_dofs_1d, 2>,
        n_dofs_1d,
        Number,
        laplace_type,
        2,
        Number2>
  {
    using TPEvaluatorBase<
      TPEvaluatorLaplace<laplace_type, Number, Number2, n_dofs_1d, 2>,
      n_dofs_1d,
      Number,
      laplace_type,
      2,
      Number2>::apply;

    __device__ void
    vmult_impl(Number        *dst,
               const Number  *src,
               const Number2 *mass_matrix,
               const Number2 *derivative_matrix,
               Number        *tmp)
    {
      constexpr int offset = n_dofs_1d * (n_dofs_1d + Util::padding);

      apply<0, false>(mass_matrix, src, tmp);
      __syncthreads();
      apply<1, false>(&derivative_matrix[offset], tmp, dst);
      __syncthreads();
      apply<0, false>(derivative_matrix, src, tmp);
      __syncthreads();
      apply<1, true>(&mass_matrix[offset], tmp, dst);
    }
  };

  template <LaplaceVariant laplace_type,
            typename Number,
            typename Number2,
            int n_dofs_1d>
  struct TPEvaluatorLaplace<laplace_type, Number, Number2, n_dofs_1d, 3>
    : TPEvaluatorBase<
        TPEvaluatorLaplace<laplace_type, Number, Number2, n_dofs_1d, 3>,
        n_dofs_1d,
        Number,
        laplace_type,
        3,
        Number2>
  {
    using TPEvaluatorBase<
      TPEvaluatorLaplace<laplace_type, Number, Number2, n_dofs_1d, 3>,
      n_dofs_1d,
      Number,
      laplace_type,
      3,
      Number2>::apply;

    __device__ void
    vmult_impl(Number        *dst,
               const Number  *src,
               const Number2 *mass_matrix,
               const Number2 *derivative_matrix,
               Number        *tmp)
    {
      constexpr int n_dofs_1d_p =
        (laplace_type == LaplaceVariant::TensorCoreMMA) ?
          (n_dofs_1d <= 8 ? 8 : 16) :
          n_dofs_1d;
      constexpr int local_dim = Util::pow(n_dofs_1d_p, 2) * n_dofs_1d;

#ifdef USECONSTMEM

      apply<0, false>(mass_matrix, src, &tmp[local_dim]);
      __syncthreads();
      apply<1, false>(mass_matrix, &tmp[local_dim], tmp);
      __syncthreads();
      apply<2, false>(derivative_matrix, tmp, dst);
      __syncthreads();
      apply<1, false>(derivative_matrix, &tmp[local_dim], tmp);
      __syncthreads();
      apply<0, false>(derivative_matrix, src, &tmp[local_dim]);
      __syncthreads();
      apply<1, true>(mass_matrix, &tmp[local_dim], tmp);
      __syncthreads();
      apply<2, true>(mass_matrix, tmp, dst);
#else
      constexpr int offset = n_dofs_1d_p * n_dofs_1d_p;

      apply<0, false>(mass_matrix, src, &tmp[local_dim]);
      __syncthreads();
      apply<1, false>(&mass_matrix[offset], &tmp[local_dim], tmp);
      __syncthreads();
      apply<2, false>(&derivative_matrix[offset * 2], tmp, dst);
      __syncthreads();
      apply<1, false>(&derivative_matrix[offset], &tmp[local_dim], tmp);
      __syncthreads();
      apply<0, false>(derivative_matrix, src, &tmp[local_dim]);
      __syncthreads();
      apply<1, true>(&mass_matrix[offset], &tmp[local_dim], tmp);
      __syncthreads();
      apply<2, true>(&mass_matrix[offset * 2], tmp, dst);
#endif
    }
  };


  ////////////////////////////////////////////////////////////////////
  //////////////////// TPEvaluatorSmoother ///////////////////////////
  ////////////////////////////////////////////////////////////////////
  template <typename Number,
            int            n_dofs_1d,
            LaplaceVariant laplace_type,
            int            dim>
  struct TPEvaluatorSmootherVmult
    : TPEvaluatorBase<
        TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace_type, dim>,
        n_dofs_1d,
        Number,
        laplace_type,
        dim>
  {
    using TPEvaluatorBase<
      TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace_type, dim>,
      n_dofs_1d,
      Number,
      laplace_type,
      dim>::apply;

    __device__ void
    vmult_impl(Number       *dst,
               const Number *src,
               const Number *mass_matrix,
               const Number *derivative_matrix,
               Number       *tmp)
    {}
  };


  template <typename Number, int n_dofs_1d, LaplaceVariant laplace_type>
  struct TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace_type, 2>
    : TPEvaluatorBase<
        TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace_type, 2>,
        n_dofs_1d,
        Number,
        laplace_type,
        2>
  {
    using TPEvaluatorBase<
      TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace_type, 2>,
      n_dofs_1d,
      Number,
      laplace_type,
      2>::apply;

    __device__ void
    vmult_impl(Number       *dst,
               const Number *src,
               const Number *mass_matrix,
               const Number *derivative_matrix,
               Number       *tmp)
    {
      apply<0, false>(mass_matrix, src, tmp);
      __syncthreads();
      apply<1, false, true>(derivative_matrix, tmp, dst);
      __syncthreads();
      apply<0, false>(derivative_matrix, src, tmp);
      __syncthreads();
      apply<1, false, true>(mass_matrix, tmp, dst);
    }
  };

  template <typename Number, int n_dofs_1d, LaplaceVariant laplace_type>
  struct TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace_type, 3>
    : TPEvaluatorBase<
        TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace_type, 3>,
        n_dofs_1d,
        Number,
        laplace_type,
        3>
  {
    using TPEvaluatorBase<
      TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace_type, 3>,
      n_dofs_1d,
      Number,
      laplace_type,
      3>::apply;

    __device__ void
    vmult_impl(Number       *dst,
               const Number *src,
               const Number *mass_matrix,
               const Number *derivative_matrix,
               Number       *tmp)
    {
      constexpr int local_dim = Util::pow(n_dofs_1d, 3);

      apply<0, false>(mass_matrix, src, &tmp[local_dim]);
      __syncthreads();
      apply<1, false>(mass_matrix, &tmp[local_dim], tmp);
      __syncthreads();
      apply<2, false, true>(derivative_matrix, tmp, dst);
      __syncthreads();
      apply<1, false>(derivative_matrix, &tmp[local_dim], tmp);
      __syncthreads();
      apply<0, false>(derivative_matrix, src, &tmp[local_dim]);
      __syncthreads();
      apply<1, true>(mass_matrix, &tmp[local_dim], tmp);
      __syncthreads();
      apply<2, false, true>(mass_matrix, tmp, dst);
    }
  };


  template <typename Number, int n_dofs_1d>
  struct TPEvaluatorSmootherVmult<Number,
                                  n_dofs_1d,
                                  LaplaceVariant::TensorCore,
                                  2>
    : TPEvaluatorBase<TPEvaluatorSmootherVmult<Number,
                                               n_dofs_1d,
                                               LaplaceVariant::TensorCore,
                                               2>,
                      n_dofs_1d,
                      Number,
                      LaplaceVariant::TensorCore,
                      2>
  {
    using TPEvaluatorBase<TPEvaluatorSmootherVmult<Number,
                                                   n_dofs_1d,
                                                   LaplaceVariant::TensorCore,
                                                   2>,
                          n_dofs_1d,
                          Number,
                          LaplaceVariant::TensorCore,
                          2>::apply;

    __device__ void
    vmult_impl(Number       *dst,
               const Number *src,
               const Number *mass_matrix,
               const Number *derivative_matrix,
               Number       *tmp)
    {
      apply<0, false>(mass_matrix, src, tmp);
      __syncthreads();
      apply<1, false, false>(derivative_matrix, tmp, tmp);
      __syncthreads();
      dst[threadIdx.y * n_dofs_1d + threadIdx.x] -=
        tmp[threadIdx.y * n_dofs_1d + threadIdx.x];
      __syncthreads();
      apply<0, false>(derivative_matrix, src, tmp);
      __syncthreads();
      apply<1, false, false>(mass_matrix, tmp, tmp);
      __syncthreads();
      dst[threadIdx.y * n_dofs_1d + threadIdx.x] -=
        tmp[threadIdx.y * n_dofs_1d + threadIdx.x];
    }
  };



  template <typename Number,
            int             n_dofs_1d,
            SmootherVariant smoother_type,
            int             dim>
  struct TPEvaluatorSmootherInv
  {
    __device__ void
    apply_inverse(Number       *dst,
                  Number       *src,
                  const Number *eigenvalues,
                  const Number *eigenvectors,
                  Number       *tmp)
    {}

    template <int direction, bool contract_over_rows, bool add = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {}
  };


  template <typename Number, int n_dofs_1d>
  struct TPEvaluatorSmootherInv<Number, n_dofs_1d, SmootherVariant::GLOBAL, 2>
  {
    __device__ void
    apply_inverse(Number       *dst,
                  Number       *src,
                  const Number *eigenvalues,
                  const Number *eigenvectors,
                  Number       *tmp)
    {
      apply<0, true>(eigenvectors, src, tmp);
      __syncthreads();
      apply<1, true>(eigenvectors, tmp, src);
      __syncthreads();
      src[threadIdx.y * n_dofs_1d + threadIdx.x % n_dofs_1d] /=
        (eigenvalues[threadIdx.y] + eigenvalues[threadIdx.x % n_dofs_1d]);
      __syncthreads();
      apply<0, false>(eigenvectors, src, tmp);
      __syncthreads();
      apply<1, false, true>(eigenvectors, tmp, dst);
    }

    template <int direction, bool contract_over_rows, bool add = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int row = threadIdx.y;
      const int col = threadIdx.x % n_dofs_1d;

      Number pval = 0;

      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      // #pragma unroll
      for (int k = 0; k < n_dofs_1d; ++k)
        {
          const int shape_idx =
            contract_over_rows ? k * n_dofs_1d + row : row * n_dofs_1d + k;

          const int source_idx =
            (direction == 0) ? (col * n_dofs_1d + k) : (k * n_dofs_1d + col);

          pval += shape_data[shape_idx] * in[source_idx];
        }


      const int destination_idx =
        (direction == 0) ? (col * n_dofs_1d + row) : (row * n_dofs_1d + col);
      if (add)
        out[destination_idx] += pval;
      else
        out[destination_idx] = pval;
    }
  };

  template <typename Number, int n_dofs_1d>
  struct TPEvaluatorSmootherInv<Number, n_dofs_1d, SmootherVariant::GLOBAL, 3>
  {
    __device__ void
    apply_inverse(Number       *dst,
                  Number       *src,
                  const Number *eigenvalues,
                  const Number *eigenvectors,
                  Number       *tmp)
    {
      constexpr int local_dim = Util::pow(n_dofs_1d, 3);

      apply<0, true>(eigenvectors, src, tmp);
      __syncthreads();
      apply<1, true>(eigenvectors, tmp, &tmp[local_dim]);
      __syncthreads();
      apply<2, true>(eigenvectors, &tmp[local_dim], tmp);
      __syncthreads();
      for (int z = 0; z < n_dofs_1d; ++z)
        {
          tmp[z * n_dofs_1d * n_dofs_1d + threadIdx.y * n_dofs_1d +
              threadIdx.x % n_dofs_1d] /=
            (eigenvalues[z] + eigenvalues[threadIdx.y] +
             eigenvalues[threadIdx.x % n_dofs_1d]);
        }
      __syncthreads();
      apply<0, false>(eigenvectors, tmp, &tmp[local_dim]);
      __syncthreads();
      apply<1, false>(eigenvectors, &tmp[local_dim], tmp);
      __syncthreads();
      apply<2, false, true>(eigenvectors, tmp, dst);
    }

    template <int direction, bool contract_over_rows, bool add = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr int stride = n_dofs_1d * n_dofs_1d;

      const int row = threadIdx.y;
      const int col = threadIdx.x % n_dofs_1d;

      Number pval[n_dofs_1d];

      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      for (int z = 0; z < n_dofs_1d; ++z)
        {
          pval[z] = 0;
          // #pragma unroll
          for (int k = 0; k < n_dofs_1d; ++k)
            {
              const int shape_idx =
                contract_over_rows ? k * n_dofs_1d + row : row * n_dofs_1d + k;

              const int source_idx =
                (direction == 0) ? (col * n_dofs_1d + k + z * stride) :
                (direction == 1) ? (k * n_dofs_1d + col + z * stride) :
                                   (z * n_dofs_1d + col + k * stride);

              pval[z] += shape_data[shape_idx] * in[source_idx];
            }
        }

      for (int z = 0; z < n_dofs_1d; ++z)
        {
          const int destination_idx =
            (direction == 0) ? (col * n_dofs_1d + row + z * stride) :
            (direction == 1) ? (row * n_dofs_1d + col + z * stride) :
                               (z * n_dofs_1d + col + row * stride);
          if (add)
            out[destination_idx] += pval[z];
          else
            out[destination_idx] = pval[z];
        }
    }
  };



  template <typename Number, int n_dofs_1d>
  struct TPEvaluatorSmootherInv<Number, n_dofs_1d, SmootherVariant::FUSED_L, 2>
  {
    __device__ void
    apply_inverse(Number       *dst,
                  Number       *src,
                  const Number *eigenvalues,
                  const Number *eigenvectors,
                  Number       *tmp)
    {
      const int linear_tid = threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

      int row = linear_tid / (n_dofs_1d - 2) + 1;
      int col = linear_tid % (n_dofs_1d - 2) + 1;

      apply<0, true>(eigenvectors, src, tmp);
      __syncthreads();
      apply<1, true>(eigenvectors, tmp, src);
      __syncthreads();
      if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
        src[row * n_dofs_1d + col] /= (eigenvalues[row] + eigenvalues[col]);
      __syncthreads();
      apply<0, false>(eigenvectors, src, tmp);
      __syncthreads();
      apply<1, false, true>(eigenvectors, tmp, dst);
    }

    template <int direction, bool contract_over_rows, bool add = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      const int linear_tid = threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

      const int row = linear_tid / (n_dofs_1d - 2) + 1;
      const int col = linear_tid % (n_dofs_1d - 2) + 1;

      Number pval;

      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
        {
          pval = 0;
          // #pragma unroll
          for (int k = 1; k < n_dofs_1d - 1; ++k)
            {
              const int shape_idx =
                contract_over_rows ? k * n_dofs_1d + row : row * n_dofs_1d + k;

              const int source_idx = (direction == 0) ? (col * n_dofs_1d + k) :
                                                        (k * n_dofs_1d + col);

              pval += shape_data[shape_idx] * in[source_idx];
            }
        }

      if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
        {
          const int destination_idx = (direction == 0) ?
                                        (col * n_dofs_1d + row) :
                                        (row * n_dofs_1d + col);
          if (add)
            out[destination_idx] += pval;
          else
            out[destination_idx] = pval;
        }
    }
  };

  template <typename Number, int n_dofs_1d>
  struct TPEvaluatorSmootherInv<Number, n_dofs_1d, SmootherVariant::FUSED_L, 3>
  {
    __device__ void
    apply_inverse(Number       *dst,
                  Number       *src,
                  const Number *eigenvalues,
                  const Number *eigenvectors,
                  Number       *tmp)
    {
      constexpr int local_dim = Util::pow(n_dofs_1d, 3);
      const int linear_tid = threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

      int row = linear_tid / (n_dofs_1d - 2) + 1;
      int col = linear_tid % (n_dofs_1d - 2) + 1;

      apply<0, true>(eigenvectors, src, tmp);
      __syncthreads();
      apply<1, true>(eigenvectors, tmp, &tmp[local_dim]);
      __syncthreads();
      apply<2, true>(eigenvectors, &tmp[local_dim], tmp);
      __syncthreads();
      if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
        for (int z = 1; z < n_dofs_1d - 1; ++z)
          {
            tmp[z * n_dofs_1d * n_dofs_1d + row * n_dofs_1d + col] /=
              (eigenvalues[z] + eigenvalues[row] + eigenvalues[col]);
          }
      __syncthreads();
      apply<0, false>(eigenvectors, tmp, &tmp[local_dim]);
      __syncthreads();
      apply<1, false>(eigenvectors, &tmp[local_dim], tmp);
      __syncthreads();
      apply<2, false, true>(eigenvectors, tmp, dst);
    }

    template <int direction, bool contract_over_rows, bool add = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr int stride = n_dofs_1d * n_dofs_1d;
      const int linear_tid = threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

      const int row = linear_tid / (n_dofs_1d - 2) + 1;
      const int col = linear_tid % (n_dofs_1d - 2) + 1;

      Number pval[n_dofs_1d - 2];

      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
        for (int z = 1; z < n_dofs_1d - 1; ++z)
          {
            pval[z - 1] = 0;
            // #pragma unroll
            for (int k = 1; k < n_dofs_1d - 1; ++k)
              {
                const int shape_idx = contract_over_rows ? k * n_dofs_1d + row :
                                                           row * n_dofs_1d + k;

                const int source_idx =
                  (direction == 0) ? (col * n_dofs_1d + k + z * stride) :
                  (direction == 1) ? (k * n_dofs_1d + col + z * stride) :
                                     (z * n_dofs_1d + col + k * stride);

                pval[z - 1] += shape_data[shape_idx] * in[source_idx];
              }
          }

      if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
        for (int z = 1; z < n_dofs_1d - 1; ++z)
          {
            const int destination_idx =
              (direction == 0) ? (col * n_dofs_1d + row + z * stride) :
              (direction == 1) ? (row * n_dofs_1d + col + z * stride) :
                                 (z * n_dofs_1d + col + row * stride);
            if (add)
              out[destination_idx] += pval[z - 1];
            else
              out[destination_idx] = pval[z - 1];
          }
    }
  };



  template <typename Number, int n_dofs_1d>
  struct TPEvaluatorSmootherInv<Number,
                                n_dofs_1d,
                                SmootherVariant::ConflictFree,
                                2>
  {
    __device__ void
    apply_inverse(Number       *dst,
                  Number       *src,
                  const Number *eigenvalues,
                  const Number *eigenvectors,
                  Number       *tmp)
    {
      const int linear_tid = threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

      int row = linear_tid / (n_dofs_1d - 2);
      int col = linear_tid % (n_dofs_1d - 2);

      apply<0, true>(eigenvectors, src, tmp);
      __syncthreads();
      apply<1, true>(eigenvectors, tmp, src);
      __syncthreads();
      if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
        src[row * (n_dofs_1d - 2) + col] /=
          (eigenvalues[row] + eigenvalues[col]);
      __syncthreads();
      apply<0, false>(eigenvectors, src, tmp);
      __syncthreads();
      apply<1, false, true>(eigenvectors, tmp, dst);
    }

    template <int direction, bool contract_over_rows, bool add = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr int n_dofs_1d_i = n_dofs_1d - 2;
      const int linear_tid = threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

      const int row = linear_tid / n_dofs_1d_i;
      const int col = linear_tid % n_dofs_1d_i;

      Number pval;

      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      if (linear_tid < n_dofs_1d_i * n_dofs_1d_i)
        {
          pval = 0;
          // #pragma unroll
          for (int k = 0; k < n_dofs_1d_i; ++k)
            {
              const int shape_idx =
                contract_over_rows ?
                  ((direction == 0) ? k * n_dofs_1d_i + col :
                                      k * n_dofs_1d_i + row) :
                  ((direction == 0) ? col * n_dofs_1d_i + k :
                                      row * n_dofs_1d_i + k);

              const int source_idx = (direction == 0) ?
                                       (row * n_dofs_1d_i + k) :
                                       (k * n_dofs_1d_i + col);

              pval += shape_data[shape_idx] * in[source_idx];
            }
        }

      if (linear_tid < n_dofs_1d_i * n_dofs_1d_i)
        {
          const int destination_idx = row * n_dofs_1d_i + col;

          if (add)
            out[destination_idx] += pval;
          else
            out[destination_idx] = pval;
        }
    }
  };

  template <typename Number, int n_dofs_1d>
  struct TPEvaluatorSmootherInv<Number,
                                n_dofs_1d,
                                SmootherVariant::ConflictFree,
                                3>
  {
    __device__ void
    apply_inverse(Number       *dst,
                  Number       *src,
                  const Number *eigenvalues,
                  const Number *eigenvectors,
                  Number       *tmp)
    {
      const int linear_tid = threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

      int row = linear_tid / (n_dofs_1d - 2);
      int col = linear_tid % (n_dofs_1d - 2);

      apply<0, true>(eigenvectors, src, tmp);
      __syncthreads();
      apply<1, true>(eigenvectors, tmp, src);
      __syncthreads();
      apply<2, true>(eigenvectors, src, tmp);
      __syncthreads();
      if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
        for (int z = 0; z < n_dofs_1d - 2; ++z)
          {
            tmp[z * n_dofs_1d * n_dofs_1d + row * (n_dofs_1d - 2) + col] /=
              (eigenvalues[z] + eigenvalues[row] + eigenvalues[col]);
          }
      __syncthreads();
      apply<0, false>(eigenvectors, tmp, src);
      __syncthreads();
      apply<1, false>(eigenvectors, src, tmp);
      __syncthreads();
      apply<2, false, true>(eigenvectors, tmp, dst);
    }

    template <int direction, bool contract_over_rows, bool add = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr int stride      = n_dofs_1d * n_dofs_1d;
      constexpr int n_dofs_1d_i = n_dofs_1d - 2;
      const int linear_tid = threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

      const int row = linear_tid / n_dofs_1d_i;
      const int col = linear_tid % n_dofs_1d_i;

      Number pval[n_dofs_1d_i];

      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      if (linear_tid < n_dofs_1d_i * n_dofs_1d_i)
        for (int z = 0; z < n_dofs_1d_i; ++z)
          {
            pval[z] = 0;
            // #pragma unroll
            for (int k = 0; k < n_dofs_1d_i; ++k)
              {
                const int shape_idx =
                  contract_over_rows ?
                    ((direction == 0) ? k * n_dofs_1d_i + col :
                     (direction == 1) ? k * n_dofs_1d_i + row :
                                        k * n_dofs_1d_i + z) :
                    ((direction == 0) ? col * n_dofs_1d_i + k :
                     (direction == 1) ? row * n_dofs_1d_i + k :
                                        z * n_dofs_1d_i + k);

                const int source_idx =
                  (direction == 0) ? (row * n_dofs_1d_i + k + z * stride) :
                  (direction == 1) ? (k * n_dofs_1d_i + col + z * stride) :
                                     (row * n_dofs_1d_i + col + k * stride);

                pval[z] += shape_data[shape_idx] * in[source_idx];
              }
          }

      if (linear_tid < n_dofs_1d_i * n_dofs_1d_i)
        for (int z = 0; z < n_dofs_1d_i; ++z)
          {
            const int destination_idx = row * n_dofs_1d_i + col + z * stride;

            if (add)
              out[destination_idx] += pval[z];
            else
              out[destination_idx] = pval[z];
          }
    }
  };

  template <typename Number, int n_dofs_1d>
  struct TPEvaluatorSmootherInv<Number, n_dofs_1d, SmootherVariant::ExactRes, 2>
  {
    __device__ void
    apply_inverse(Number       *dst,
                  Number       *src,
                  const Number *eigenvalues,
                  const Number *eigenvectors,
                  Number       *tmp)
    {
      const int linear_tid = threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

      int row = linear_tid / (n_dofs_1d - 4);
      int col = linear_tid % (n_dofs_1d - 4);

      apply<0, true>(eigenvectors, src, tmp);
      __syncthreads();
      apply<1, true>(eigenvectors, tmp, src);
      __syncthreads();
      if (linear_tid < (n_dofs_1d - 4) * (n_dofs_1d - 4))
        src[row * (n_dofs_1d - 4) + col] /=
          (eigenvalues[row] + eigenvalues[col]);
      __syncthreads();
      apply<0, false>(eigenvectors, src, tmp);
      __syncthreads();
      apply<1, false, true>(eigenvectors, tmp, dst);
    }

    template <int direction, bool contract_over_rows, bool add = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr int n_dofs_1d_i = n_dofs_1d - 4;
      const int linear_tid = threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

      const int row = linear_tid / n_dofs_1d_i;
      const int col = linear_tid % n_dofs_1d_i;

      Number pval;

      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      if (linear_tid < n_dofs_1d_i * n_dofs_1d_i)
        {
          pval = 0;
          // #pragma unroll
          for (int k = 0; k < n_dofs_1d_i; ++k)
            {
              const int shape_idx =
                contract_over_rows ?
                  ((direction == 0) ? k * n_dofs_1d_i + col :
                                      k * n_dofs_1d_i + row) :
                  ((direction == 0) ? col * n_dofs_1d_i + k :
                                      row * n_dofs_1d_i + k);

              const int source_idx = (direction == 0) ?
                                       (row * n_dofs_1d_i + k) :
                                       (k * n_dofs_1d_i + col);

              pval += shape_data[shape_idx] * in[source_idx];
            }
        }

      if (linear_tid < n_dofs_1d_i * n_dofs_1d_i)
        {
          const int destination_idx = row * n_dofs_1d_i + col;

          if (add)
            out[destination_idx] += pval;
          else
            out[destination_idx] = pval;
        }
    }
  };

  template <typename Number, int n_dofs_1d>
  struct TPEvaluatorSmootherInv<Number, n_dofs_1d, SmootherVariant::ExactRes, 3>
  {
    __device__ void
    apply_inverse(Number       *dst,
                  Number       *src,
                  const Number *eigenvalues,
                  const Number *eigenvectors,
                  Number       *tmp)
    {
      const int linear_tid = threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

      int row = linear_tid / (n_dofs_1d - 4);
      int col = linear_tid % (n_dofs_1d - 4);

      apply<0, true>(eigenvectors, src, tmp);
      __syncthreads();
      apply<1, true>(eigenvectors, tmp, src);
      __syncthreads();
      apply<2, true>(eigenvectors, src, tmp);
      __syncthreads();
      if (linear_tid < (n_dofs_1d - 4) * (n_dofs_1d - 4))
        for (int z = 0; z < n_dofs_1d - 4; ++z)
          {
            tmp[z * n_dofs_1d * n_dofs_1d + row * (n_dofs_1d - 4) + col] /=
              (eigenvalues[z] + eigenvalues[row] + eigenvalues[col]);
          }
      __syncthreads();
      apply<0, false>(eigenvectors, tmp, src);
      __syncthreads();
      apply<1, false>(eigenvectors, src, tmp);
      __syncthreads();
      apply<2, false, true>(eigenvectors, tmp, dst);
    }

    template <int direction, bool contract_over_rows, bool add = false>
    __device__ void
    apply(const Number *shape_data, const Number *in, Number *out)
    {
      constexpr int stride      = n_dofs_1d * n_dofs_1d;
      constexpr int n_dofs_1d_i = n_dofs_1d - 4;
      const int linear_tid = threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

      const int row = linear_tid / n_dofs_1d_i;
      const int col = linear_tid % n_dofs_1d_i;

      Number pval[n_dofs_1d_i];

      // kernel product: A kdot src, [N x N] * [N^dim, 1]
      if (linear_tid < n_dofs_1d_i * n_dofs_1d_i)
        for (int z = 0; z < n_dofs_1d_i; ++z)
          {
            pval[z] = 0;
            // #pragma unroll
            for (int k = 0; k < n_dofs_1d_i; ++k)
              {
                const int shape_idx =
                  contract_over_rows ?
                    ((direction == 0) ? k * n_dofs_1d_i + col :
                     (direction == 1) ? k * n_dofs_1d_i + row :
                                        k * n_dofs_1d_i + z) :
                    ((direction == 0) ? col * n_dofs_1d_i + k :
                     (direction == 1) ? row * n_dofs_1d_i + k :
                                        z * n_dofs_1d_i + k);

                const int source_idx =
                  (direction == 0) ? (row * n_dofs_1d_i + k + z * stride) :
                  (direction == 1) ? (k * n_dofs_1d_i + col + z * stride) :
                                     (row * n_dofs_1d_i + col + k * stride);

                pval[z] += shape_data[shape_idx] * in[source_idx];
              }
          }

      if (linear_tid < n_dofs_1d_i * n_dofs_1d_i)
        for (int z = 0; z < n_dofs_1d_i; ++z)
          {
            const int destination_idx = row * n_dofs_1d_i + col + z * stride;

            if (add)
              out[destination_idx] += pval[z];
            else
              out[destination_idx] = pval[z];
          }
    }
  };

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////

  template <int dim, int fe_degree, typename Number, LaplaceVariant laplace>
  __device__ void
  evaluate_laplace(const int                         local_patch,
                   SharedMemData<dim, Number, true> *shared_data)
  {
    constexpr int n_dofs_1d = 2 * fe_degree + 2;

    if constexpr (std::is_same_v<Number, double> ||
                  (MMAKERNEL != 0 && MMAKERNEL != 7 && MMAKERNEL != 8))
      {
        TPEvaluatorLaplace<laplace, Number, Number, n_dofs_1d, dim> eval;
        __syncthreads();
#ifdef USECONSTMEM
        eval.vmult(shared_data->local_dst,
                   shared_data->local_src,
                   shared_data->const_mass,
                   shared_data->const_stiff,
                   shared_data->tmp);
#else
        eval.vmult(shared_data->local_dst,
                   shared_data->local_src,
                   shared_data->local_mass,
                   shared_data->local_derivative,
                   shared_data->tmp);
#endif
        __syncthreads();
      }
    else
      {
        TPEvaluatorLaplace<laplace, Number, half, n_dofs_1d, dim> eval;
        __syncthreads();

        eval.vmult(shared_data->local_dst,
                   shared_data->local_src,
                   shared_data->mass_half,
                   shared_data->der_half,
                   shared_data->tmp);
      }
    __syncthreads();
  }

  template <int dim,
            int fe_degree,
            typename Number,
            LaplaceVariant  laplace,
            SmootherVariant smooth>
  __device__ void
  evaluate_smooth(
    const int                                                      local_patch,
    SharedMemData<dim, Number, false>                             *shared_data,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data *gpu_data)
  {
    constexpr int n_dofs_1d = 2 * fe_degree + 2;
    constexpr int local_dim = Util::pow(n_dofs_1d, dim);

    TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace, dim> eval_vmult;
    TPEvaluatorSmootherInv<Number, n_dofs_1d, smooth, dim>    eval_inverse;
    __syncthreads();

    eval_vmult.vmult(&shared_data->local_src[local_patch * local_dim],
                     &shared_data->local_dst[local_patch * local_dim],
                     shared_data->local_mass,
                     shared_data->local_derivative,
                     &shared_data->tmp[local_patch * local_dim * (dim - 1)]);
    __syncthreads();

    const int linear_tid = threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

    if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
      {
        const int row = linear_tid / (n_dofs_1d - 2);
        const int col = linear_tid % (n_dofs_1d - 2);

        shared_data->local_mass[col + 1] = gpu_data->eigenvalues[col];
        shared_data->local_derivative[(row + 1) * n_dofs_1d + col + 1] =
          gpu_data->eigenvectors[row * (n_dofs_1d - 2) + col];
      }
    __syncthreads();

    eval_inverse.apply_inverse(
      &shared_data->local_dst[local_patch * local_dim],
      &shared_data->local_src[local_patch * local_dim],
      shared_data->local_mass,
      shared_data->local_derivative,
      &shared_data->tmp[local_patch * local_dim * (dim - 1)]);
    __syncthreads();
  }


  template <int dim,
            int fe_degree,
            typename Number,
            LaplaceVariant  laplace,
            SmootherVariant smooth>
  __device__ void
  evaluate_smooth_cf(
    const int                                                      local_patch,
    SharedMemData<dim, Number, false>                             *shared_data,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data *gpu_data)
  {
    constexpr int n_dofs_1d   = 2 * fe_degree + 2;
    constexpr int n_dofs_1d_z = dim == 2 ? 1 : n_dofs_1d - 2;
    constexpr int local_dim   = Util::pow(n_dofs_1d, dim);

    TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace, dim> eval_vmult;
    TPEvaluatorSmootherInv<Number, n_dofs_1d, smooth, dim>    eval_inverse;
    __syncthreads();

    eval_vmult.vmult(&shared_data->local_src[local_patch * local_dim],
                     &shared_data->local_dst[local_patch * local_dim],
                     shared_data->local_mass,
                     shared_data->local_derivative,
                     &shared_data->tmp[local_patch * local_dim * (dim - 1)]);
    __syncthreads();

    const int linear_tid = threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

    if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
      {
        int row = linear_tid / (n_dofs_1d - 2);
        int col = linear_tid % (n_dofs_1d - 2);

        shared_data->local_mass[col] = gpu_data->eigenvalues[col];
        shared_data->local_derivative[row * (n_dofs_1d - 2) + col] =
          gpu_data->eigenvectors[row * (n_dofs_1d - 2) + col];
      }
    // __syncthreads();


    if (linear_tid < (n_dofs_1d - 2) * (n_dofs_1d - 2))
      {
        int row = linear_tid / (n_dofs_1d - 2) + 1;
        int col = linear_tid % (n_dofs_1d - 2) + 1;

        for (int z = 0; z < n_dofs_1d_z; ++z)
          {
            shared_data
              ->tmp[2 * local_patch * local_dim + z * n_dofs_1d * n_dofs_1d +
                    (row - 1) * (n_dofs_1d - 2) + col - 1] =
              shared_data->local_dst[local_patch * local_dim +
                                     (z + dim - 2) * n_dofs_1d * n_dofs_1d +
                                     row * n_dofs_1d + col];

            shared_data->tmp[2 * local_patch * local_dim + local_dim +
                             z * n_dofs_1d * n_dofs_1d +
                             (row - 1) * (n_dofs_1d - 2) + col - 1] =
              shared_data->local_src[local_patch * local_dim +
                                     (z + dim - 2) * n_dofs_1d * n_dofs_1d +
                                     row * n_dofs_1d + col];
          }
      }
    __syncthreads();

    eval_inverse.apply_inverse(
      &shared_data->tmp[local_patch * local_dim * 2],
      &shared_data->tmp[local_patch * local_dim * 2 + local_dim],
      shared_data->local_mass,
      shared_data->local_derivative,
      &shared_data->local_src[local_patch * local_dim]);
    __syncthreads();
  }


  template <int dim, int fe_degree, typename Number, SmootherVariant smooth>
  __device__ void
  evaluate_smooth_inv(const int                          local_patch,
                      SharedMemData<dim, Number, false> *shared_data)
  {
    constexpr int n_dofs_1d = 2 * fe_degree;
    constexpr int local_dim = Util::pow(n_dofs_1d, dim);

    TPEvaluatorSmootherInv<Number, n_dofs_1d, smooth, dim> eval;
    __syncthreads();

    eval.apply_inverse(&shared_data->local_dst[local_patch * local_dim],
                       &shared_data->local_src[local_patch * local_dim],
                       shared_data->local_mass,
                       shared_data->local_derivative,
                       &shared_data->tmp[local_patch * local_dim * (dim - 1)]);
    __syncthreads();
  }

  template <int dim,
            int fe_degree,
            typename Number,
            LaplaceVariant  laplace,
            SmootherVariant smooth>
  __device__ void
  evaluate_smooth_exact(
    const int                                                      local_patch,
    SharedMemData<dim, Number, false>                             *shared_data,
    const typename LevelVertexPatch<dim, fe_degree, Number>::Data *gpu_data)
  {
    constexpr int n_dofs_1d   = 2 * fe_degree + 2;
    constexpr int local_dim   = Util::pow(n_dofs_1d, dim);
    constexpr int n_dofs_1d_z = dim == 2 ? 1 : n_dofs_1d - 4;

    TPEvaluatorSmootherVmult<Number, n_dofs_1d, laplace, dim> eval_vmult;
    TPEvaluatorSmootherInv<Number, n_dofs_1d, smooth, dim>    eval_inverse;
    __syncthreads();

    eval_vmult.vmult(&shared_data->local_src[local_patch * local_dim],
                     &shared_data->local_dst[local_patch * local_dim],
                     shared_data->local_mass,
                     shared_data->local_derivative,
                     &shared_data->tmp[local_patch * local_dim * (dim - 1)]);
    __syncthreads();

    const int linear_tid = threadIdx.x % n_dofs_1d + threadIdx.y * n_dofs_1d;

    if (linear_tid < (n_dofs_1d - 4) * (n_dofs_1d - 4))
      {
        const int row = linear_tid / (n_dofs_1d - 4);
        const int col = linear_tid % (n_dofs_1d - 4);

        shared_data->local_mass[col] = gpu_data->eigenvalues[col + n_dofs_1d];
        shared_data->local_derivative[row * (n_dofs_1d - 4) + col] =
          gpu_data
            ->eigenvectors[row * (n_dofs_1d - 4) + col + n_dofs_1d * n_dofs_1d];
      }
    // __syncthreads();

    if (linear_tid < (n_dofs_1d - 4) * (n_dofs_1d - 4))
      {
        int row = linear_tid / (n_dofs_1d - 4) + 2;
        int col = linear_tid % (n_dofs_1d - 4) + 2;

        for (int z = 0; z < n_dofs_1d_z; ++z)
          {
            shared_data
              ->tmp[2 * local_patch * local_dim + z * n_dofs_1d * n_dofs_1d +
                    (row - 2) * (n_dofs_1d - 4) + col - 2] =
              shared_data->local_dst[local_patch * local_dim +
                                     (z + 2 * dim - 4) * n_dofs_1d * n_dofs_1d +
                                     row * n_dofs_1d + col];

            shared_data->tmp[2 * local_patch * local_dim + local_dim +
                             z * n_dofs_1d * n_dofs_1d +
                             (row - 2) * (n_dofs_1d - 4) + col - 2] =
              shared_data->local_src[local_patch * local_dim +
                                     (z + 2 * dim - 4) * n_dofs_1d * n_dofs_1d +
                                     row * n_dofs_1d + col];
          }
      }
    __syncthreads();

    eval_inverse.apply_inverse(
      &shared_data->tmp[local_patch * local_dim * 2],
      &shared_data->tmp[local_patch * local_dim * 2 + local_dim],
      shared_data->local_mass,
      shared_data->local_derivative,
      &shared_data->local_src[local_patch * local_dim]);
    __syncthreads();
  }

} // namespace PSMF


#endif // CUDA_EVALUATE_CUH
