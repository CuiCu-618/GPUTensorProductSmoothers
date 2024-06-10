/**
 * @file utilities.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief collection of helper functions
 * @version 1.0
 * @date 2022-12-26
 *
 * @copyright Copyright (c) 2022
 *
 */

#ifndef UTILITIES_CUH
#define UTILITIES_CUH

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_values.h>

#include <deal.II/hp/fe_values.h>

#include <vector>

using namespace dealii;

namespace Util
{

  constexpr unsigned int padding = 0;

  template <typename T>
  __host__ __device__ constexpr T
  pow(const T base, const int iexp)
  {
    return iexp <= 0 ?
             1 :
             (iexp == 1 ?
                base :
                (((iexp % 2 == 1) ? base : 1) * pow(base * base, iexp / 2)));
  }

  /**
   * Compute dofs in a patch based on first_dof.
   * Data layout for local vectors:
   * 12 13 | 14 15
   *  8  9 | 10 11
   * ------|------
   *  4  5 |  6  7
   *  0  1 |  2  3
   */
  template <int dim, int fe_degree>
  __device__ unsigned int
  compute_indices(unsigned int *first_dofs,
                  unsigned int  local_patch,
                  unsigned int  local_tid_x,
                  unsigned int  tid_y,
                  unsigned int  tid_z)
  {
    const unsigned int z_off = tid_z / (fe_degree + 1);
    const unsigned int y_off = tid_y / (fe_degree + 1);
    const unsigned int x_off = local_tid_x / (fe_degree + 1);
    const unsigned int z     = tid_z % (fe_degree + 1);
    const unsigned int y     = tid_y % (fe_degree + 1);
    const unsigned int x     = local_tid_x % (fe_degree + 1);

    return first_dofs[z_off * 4 + y_off * 2 + x_off] +
           z * (fe_degree + 1) * (fe_degree + 1) + y * (fe_degree + 1) + x;
  }


  /**
   * Compute dofs in a patch based on first_dof.
   * Data layout for local vectors:
   * 10 11 | 14 15
   *  8  9 | 12 13
   * ------|------
   *  2  3 |  6  7
   *  0  1 |  4  5
   */
  template <int dim, int fe_degree>
  __device__ unsigned int
  compute_indices_cell(unsigned int *first_dofs, unsigned int linear_tid)
  {
    constexpr unsigned int cell_dofs = pow(fe_degree + 1, dim);

    const unsigned int cell           = linear_tid / cell_dofs;
    const unsigned int local_cell_tid = linear_tid % cell_dofs;

    return first_dofs[cell] + local_cell_tid;
  }

  // Function to calculate GCD of two numbers
  __device__ constexpr unsigned int
  gcd(unsigned int a, unsigned int b)
  {
    if (b == 0)
      return a;
    return gcd(b, a % b);
  }

  // Recursive template function to calculate LCM of two numbers
  template <int a, int b>
  struct LCM
  {
    static constexpr unsigned int value = (a * b) / gcd(a, b);
  };

  // Function to calculate the multiple of a number
  template <int n, int constant>
  __device__ constexpr unsigned int
  calculate_multiple()
  {
    // Calculate the multiple of n
    constexpr unsigned int multiple = LCM<n, constant>::value / n;

    return multiple;
  }

  template <int n_dofs_1d, typename Number = double>
  __host__ __device__ inline unsigned int
  get_base(const unsigned int row, const unsigned int z = 0)
  {
    return 0;
  }

  template <>
  __host__ __device__ inline unsigned int
  get_base<8, double>(const unsigned int row, const unsigned int z)
  {
    auto base1 = (row & 3) < 2 ? 0 : 4;
    auto base2 = (z & 1) << 3;
    auto base3 = (z & 3) < 2 ? 0 : 4;

    return base1 ^ base2 ^ base3;
  }

  template <>
  __host__ __device__ inline unsigned int
  get_base<10, double>(const unsigned int row, const unsigned int z)
  {
    auto base1 = (row & 1) < 1 ? 0 : 8;
    auto base2 = (row & 3) < 2 ? 0 : 4;
    auto base3 = (z & 1) < 1 ? 0 : 8;
    auto base4 = (z & 3) < 2 ? 0 : 4;

    return base1 ^ base2 ^ base3 ^ base4;
  }

  template <>
  __host__ __device__ inline unsigned int
  get_base<12, double>(const unsigned int row, const unsigned int z)
  {
    auto base1 = (row & 1) < 1 ? 0 : 8;
    auto base2 = (row & 3) < 2 ? 0 : 4;
    auto base3 = (z & 1) < 1 ? 0 : 8;
    auto base4 = (z & 3) < 2 ? 0 : 4;

    return base1 ^ base2 ^ base3 ^ base4;
  }

  template <>
  __host__ __device__ inline unsigned int
  get_base<14, double>(const unsigned int row, const unsigned int z)
  {
    auto base1 = (row & 1) < 1 ? 0 : 8;
    auto base2 = (row & 3) < 2 ? 0 : 4;
    auto base3 = (z & 1) < 1 ? 0 : 8;
    auto base4 = (z & 3) < 2 ? 0 : 4;

    return base1 ^ base2 ^ base3 ^ base4;
  }

  template <>
  __host__ __device__ inline unsigned int
  get_base<16, double>(const unsigned int row, const unsigned int z)
  {
    // return 0;

    auto base1 = (row & 1) < 1 ? 0 : 8;
    auto base2 = (row & 3) < 2 ? 0 : 4;
    auto base3 = (z & 1) < 1 ? 0 : 8;
    auto base4 = (z & 3) < 2 ? 0 : 4;

    return base1 ^ base2 ^ base3 ^ base4;
  }

  // template <>
  // __host__ __device__ inline unsigned int
  // get_base<16, float>(const unsigned int row, const unsigned int z)
  // {
  //   auto base1 = (row & 3) < 2 ? 0 : 8;
  //   auto base2 = (row & 7) < 4 ? 0 : 4;
  //   auto base3 = (z & 1) < 1 ? 0 : 16;
  //   auto base4 = (z & 3) < 2 ? 0 : 8;
  //   auto base5 = (z & 7) < 4 ? 0 : 4;

  //   return base1 ^ base2 ^ base3 ^ base4 ^ base5;
  // }

  template <>
  __host__ __device__ inline unsigned int
  get_base<16, float>(const unsigned int row, const unsigned int z)
  {
    // return 0;

    auto base1 = (row & 3) < 2 ? 0 : 8;
    auto base2 = (row & 7) < 4 ? 0 : 16;
    auto base3 = (z & 1) < 1 ? 0 : 16;
    auto base4 = (z & 3) < 2 ? 0 : 8;
    auto base5 = (z & 7) < 4 ? 0 : 16;

    return base1 ^ base2 ^ base3 ^ base4 ^ base5;
  }

  template <>
  __host__ __device__ inline unsigned int
  get_base<16, half>(const unsigned int row, const unsigned int)
  {
    return (row & 7) < 4 ? 0 : 8;
  }
} // namespace Util

#endif // UTILITIES_CUH
