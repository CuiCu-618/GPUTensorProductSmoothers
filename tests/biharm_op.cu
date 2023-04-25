/**
 * Created by Cu Cui on 2023/4/17.
 */

// Testing biharm operator

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/fe/fe_dgq.h>

#include <iostream>

#include "TPSS/tensor_product_matrix.h"

using namespace dealii;

template <int dim, int degree, typename Number = double>
void
test()
{
  FE_DGQ<1> fe(degree);
  QGauss<1> quadrature(degree + 1);

  constexpr int n_cell_dofs  = degree + 1;
  constexpr int n_patch_dofs = 2 * n_cell_dofs - 1;

  const Number h              = 4.0;
  const Number penalty_factor = h * degree * (degree + 1);

  std::array<Table<2, VectorizedArray<Number>>, dim> mass_matrices;
  std::array<Table<2, VectorizedArray<Number>>, dim> laplace_matrices;

  for (unsigned int d = 0; d < dim; ++d)
    {
      mass_matrices[d].reinit(n_patch_dofs, n_patch_dofs);
      laplace_matrices[d].reinit(n_patch_dofs, n_patch_dofs);
    }

  for (unsigned int i = 0; i < n_cell_dofs; ++i)
    for (unsigned int j = 0; j < n_cell_dofs; ++j)
      {
        Number sum_mass = 0, sum_laplace = 0;
        for (unsigned int q = 0; q < quadrature.size(); ++q)
          {
            sum_mass += (fe.shape_value(i, quadrature.point(q)) *
                         fe.shape_value(j, quadrature.point(q))) *
                        quadrature.weight(q) / h;

            sum_laplace += (fe.shape_grad(i, quadrature.point(q))[0] *
                            fe.shape_grad(j, quadrature.point(q))[0]) *
                           quadrature.weight(q) * h;
          }
        for (unsigned int d = 0; d < dim; ++d)
          {
            mass_matrices[d](i, j) += sum_mass;
            mass_matrices[d](i + n_cell_dofs - 1, j + n_cell_dofs - 1) +=
              sum_mass;

            laplace_matrices[d](i, j) += sum_laplace;
            laplace_matrices[d](i + n_cell_dofs - 1, j + n_cell_dofs - 1) +=
              sum_laplace;
          }
      }

  auto cell_bilaplace = [&](unsigned int type) {
    Number boundary_factor_left  = 1.;
    Number boundary_factor_right = 1.;

    if (type == 0)
      boundary_factor_left = 2.;
    else if (type == 2)
      boundary_factor_right = 2.;

    FullMatrix<Number> cell_laplace(n_cell_dofs, n_cell_dofs);

    for (unsigned int i = 0; i < n_cell_dofs; ++i)
      for (unsigned int j = 0; j < n_cell_dofs; ++j)
        {
          Number sum_laplace = 0;
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            {
              sum_laplace += (fe.shape_grad_grad(i, quadrature.point(q))[0] *
                              fe.shape_grad_grad(j, quadrature.point(q))[0]) *
                             quadrature.weight(q) * h * h * h;
            }

          sum_laplace +=
            boundary_factor_left *
            (1. * fe.shape_grad(i, Point<1>()) * fe.shape_grad(j, Point<1>()) *
               penalty_factor * h * h +
             0.5 * fe.shape_grad_grad(i, Point<1>())[0] *
               fe.shape_grad(j, Point<1>()) * h * h * h +
             0.5 * fe.shape_grad_grad(j, Point<1>())[0] *
               fe.shape_grad(i, Point<1>()) * h * h * h);

          sum_laplace +=
            boundary_factor_right *
            (1. * fe.shape_grad(i, Point<1>(1.0)) *
               fe.shape_grad(j, Point<1>(1.0)) * penalty_factor * h * h -
             0.5 * fe.shape_grad_grad(i, Point<1>(1.0))[0] *
               fe.shape_grad(j, Point<1>(1.0)) * h * h * h -
             0.5 * fe.shape_grad_grad(j, Point<1>(1.0))[0] *
               fe.shape_grad(i, Point<1>(1.0)) * h * h * h);

          // scaling to real cells
          cell_laplace(i, j) = sum_laplace;
        }

    return cell_laplace;
  };

  auto cell0 = cell_bilaplace(0);
  auto cell1 = cell_bilaplace(1);

  FullMatrix<Number> laplace_interface_mixed(n_cell_dofs, n_cell_dofs);
  FullMatrix<Number> laplace_interface_penalty(n_cell_dofs, n_cell_dofs);

  for (unsigned int i = 0; i < n_cell_dofs; ++i)
    for (unsigned int j = 0; j < n_cell_dofs; ++j)
      {
        Number sum_mixed = 0, sum_penalty = 0;
        sum_mixed += (-0.5 * fe.shape_grad_grad(i, Point<1>())[0] *
                      fe.shape_grad(j, Point<1>(1.0)) * h * h * h);

        sum_penalty +=
          (-1. * fe.shape_grad(i, Point<1>()) *
           fe.shape_grad(j, Point<1>(1.0)) * penalty_factor * h * h);

        laplace_interface_mixed(n_cell_dofs - 1 - i, n_cell_dofs - 1 - j) =
          sum_mixed;
        laplace_interface_penalty(n_cell_dofs - 1 - i, n_cell_dofs - 1 - j) =
          sum_penalty;
      }

  auto patch_bilaplace = [&](auto left, auto right) {
    std::array<Table<2, Number>, dim> patch_laplace;
    for (unsigned int d = 0; d < dim; ++d)
      {
        patch_laplace[d].reinit(n_patch_dofs, n_patch_dofs);
      }

    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int i = 0; i < n_cell_dofs; ++i)
        for (unsigned int j = 0; j < n_cell_dofs; ++j)
          {
            patch_laplace[d](i, j) += left(i, j);
            patch_laplace[d](i + n_cell_dofs - 1, j + n_cell_dofs - 1) +=
              right(i, j);

            patch_laplace[d](i, j + n_cell_dofs - 1) +=
              laplace_interface_mixed(i, j);
            patch_laplace[d](i, j + n_cell_dofs - 1) +=
              laplace_interface_mixed(n_cell_dofs - 1 - j, n_cell_dofs - 1 - i);
            patch_laplace[d](i, j + n_cell_dofs - 1) +=
              laplace_interface_penalty(n_cell_dofs - 1 - j,
                                        n_cell_dofs - 1 - i);

            patch_laplace[d](i + n_cell_dofs - 1, j) +=
              laplace_interface_mixed(j, i);
            patch_laplace[d](i + n_cell_dofs - 1, j) +=
              laplace_interface_mixed(n_cell_dofs - 1 - i, n_cell_dofs - 1 - j);
            patch_laplace[d](i + n_cell_dofs - 1, j) +=
              laplace_interface_penalty(n_cell_dofs - 1 - i,
                                        n_cell_dofs - 1 - j);
          }

    return patch_laplace;
  };
  std::cout << std::endl;
  auto patch0 = patch_bilaplace(cell0, cell1);

  std::array<Table<2, VectorizedArray<Number>>, dim> bilaplace_matrices;
  for (unsigned int d = 0; d < dim; ++d)
    {
      bilaplace_matrices[d].reinit(n_patch_dofs, n_patch_dofs);
      std::transform(patch0[d].begin(),
                     patch0[d].end(),
                     bilaplace_matrices[d].begin(),
                     [](Number i) -> VectorizedArray<Number> {
                       return make_vectorized_array(i);
                     });
    }

  auto *mass = new Number[n_patch_dofs * n_patch_dofs];
  std::transform(bilaplace_matrices[0].begin(),
                 bilaplace_matrices[0].end(),
                 mass,
                 [](auto m) -> Number { return m.value()[0]; });

  delete[] mass;

  auto print_matrices = [](auto matrix) {
    for (auto i = 0U; i < matrix.size(); ++i)
      {
        for (auto m = 0U; m < matrix[i].size(1); ++m)
          {
            for (auto n = 0U; n < matrix[i].size(0); ++n)
              std::cout << matrix[i](m, n) << " ";
            std::cout << std::endl;
          }
        std::cout << std::endl;
      }
    std::cout << std::endl;
  };

  print_matrices(mass_matrices);
  print_matrices(laplace_matrices);
  print_matrices(bilaplace_matrices);

  auto interior = [](auto matrix) {
    std::array<Table<2, VectorizedArray<Number>>, dim> dst;
    for (unsigned int d = 0; d < dim; ++d)
      {
        dst[d].reinit(matrix[d].n_rows() - 2, matrix[d].n_cols() - 2);

        for (unsigned int i = 0; i < matrix[d].n_rows() - 2; ++i)
          for (unsigned int j = 0; j < matrix[d].n_cols() - 2; ++j)
            dst[d](i, j) = matrix[d](i + 1, j + 1);
      }

    return dst;
  };

  auto mass_matrices_inv      = interior(mass_matrices);
  auto laplace_matrices_inv   = interior(laplace_matrices);
  auto bilaplace_matrices_inv = interior(bilaplace_matrices);

  print_matrices(mass_matrices_inv);
  print_matrices(laplace_matrices_inv);
  print_matrices(bilaplace_matrices_inv);

  Tensors::TensorProductMatrix<dim, VectorizedArray<Number>, n_patch_dofs>
    local_matrices;
  std::vector<std::array<Table<2, VectorizedArray<Number>>, dim>> rank1_tensors;

  /// store rank1 tensors of separable Kronecker representation
  /// BxMxM + MxBxM + MxMxB
  const auto &BxMxM = [&](const int direction) {
    std::array<Table<2, VectorizedArray<Number>>, dim> kronecker_tensor;
    for (auto d = 0; d < dim; ++d)
      kronecker_tensor[d] =
        d == direction ? bilaplace_matrices[direction] : mass_matrices[d];
    return kronecker_tensor;
  };
  for (auto direction = 0; direction < dim; ++direction)
    rank1_tensors.emplace_back(BxMxM(direction));

  /// store rank1 tensors of mixed derivatives
  /// 2(LxLxM + LxMxL + MxLxL)
  const auto &LxLxM = [&](const int direction1, const int direction2) {
    std::array<Table<2, VectorizedArray<Number>>, dim> kronecker_tensor;
    for (auto d = 0; d < dim; ++d)
      kronecker_tensor[d] = (d == direction1 || d == direction2) ?
                              laplace_matrices[d] :
                              mass_matrices[d];
    return kronecker_tensor;
  };
  for (auto direction1 = 0; direction1 < dim; ++direction1)
    for (auto direction2 = 0; direction2 < dim; ++direction2)
      if (direction1 != direction2)
        rank1_tensors.emplace_back(LxLxM(direction1, direction2));

  AssertDimension(rank1_tensors.size(), 2 * dim);

  for (auto &t : rank1_tensors)
    print_matrices(t);

  local_matrices.reinit(rank1_tensors);

  // auto eigenvalue_tensor  = local_matrices.get_eigenvalue_tensor();

  // {
  //     unsigned int n_dofs_2d = std::pow(2 * degree - 1, 2);

  //     auto mat = new Number[n_dofs_2d * dim];
  //     for (unsigned int i = 0; i < dim; ++i)
  //       std::transform(eigenvalue_tensor[i].begin(),
  //                      eigenvalue_tensor[i].end(),
  //                      &mat[n_dofs_2d * i],
  //                      [](auto m) -> Number { return m[0]; });
  // }
}

int
main()
{
  test<2, 2>();
  //   test<2, 3>();
  // test<2, 5>();
}