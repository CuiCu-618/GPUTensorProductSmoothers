/**
 * Created by Cu Cui on 2023/4/17.
 */

// Testing local patch dofs numbering

#include <deal.II/base/polynomials_raviart_thomas.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_raviart_thomas_new.h>
#include <deal.II/fe/fe_tools.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/matrix_free/shape_info.h>

#include <iostream>

#include "TPSS/tensor_product_matrix.h"
#include "utilities.cuh"

using namespace dealii;

template <int dim>
std::vector<unsigned int>
get_lexicographic_numbering(const unsigned int normal_degree,
                            const unsigned int tangential_degree)
{
  const unsigned int n_dofs_face =
    Utilities::pow(tangential_degree + 1, dim - 1);
  std::vector<unsigned int> lexicographic_numbering;
  // component 1
  for (unsigned int j = 0; j < n_dofs_face; ++j)
    {
      lexicographic_numbering.push_back(j);
      if (normal_degree > 1)
        for (unsigned int i = n_dofs_face * 2 * dim;
             i < n_dofs_face * 2 * dim + normal_degree - 1;
             ++i)
          lexicographic_numbering.push_back(i + j * (normal_degree - 1));
      lexicographic_numbering.push_back(n_dofs_face + j);
    }

  // component 2
  unsigned int layers = (dim == 3) ? tangential_degree + 1 : 1;
  for (unsigned int k = 0; k < layers; ++k)
    for (unsigned int j = 0; j < tangential_degree + 1; ++j)
      {
        unsigned int s = j + n_dofs_face * 2;

        unsigned int k_add = k * (tangential_degree + 1);

        lexicographic_numbering.push_back(s + k_add);

        if (normal_degree > 1)
          for (unsigned int i = n_dofs_face * (2 * dim + (normal_degree - 1));
               i < n_dofs_face * (2 * dim + (normal_degree - 1)) +
                     (normal_degree - 1) * (tangential_degree + 1);
               i += tangential_degree + 1)
            {
              lexicographic_numbering.push_back(i + j +
                                                k_add * tangential_degree);
            }
        unsigned int e = j + n_dofs_face * 3;
        lexicographic_numbering.push_back(e + k_add);
      }
  // unsigned int layers = (dim == 3) ? tangential_degree + 1 : 1;
  // for (unsigned int k = 0; k < layers; ++k)
  //   {
  //     unsigned int k_add = k * (tangential_degree + 1);
  //     for (unsigned int j = n_dofs_face * 2;
  //          j < n_dofs_face * 2 + tangential_degree + 1;
  //          ++j)
  //       lexicographic_numbering.push_back(j + k_add);

  //     if (normal_degree > 1)
  //       for (unsigned int i = n_dofs_face * (2 * dim + (normal_degree - 1));
  //            i < n_dofs_face * (2 * dim + (normal_degree - 1)) +
  //                  (normal_degree - 1) * (tangential_degree + 1);
  //            ++i)
  //         {
  //           lexicographic_numbering.push_back(i + k_add * tangential_degree);
  //         }
  //     for (unsigned int j = n_dofs_face * 3;
  //          j < n_dofs_face * 3 + tangential_degree + 1;
  //          ++j)
  //       lexicographic_numbering.push_back(j + k_add);
  //   }

  // component 3
  if (dim == 3)
    {
      for (unsigned int k = 0; k < layers; ++k)
        for (unsigned int j = 0; j < tangential_degree + 1; ++j)
          {
            unsigned int k_add = k * (tangential_degree + 1);

            unsigned int s = j + 4 * n_dofs_face;
            lexicographic_numbering.push_back(s + k_add);

            if (normal_degree > 1)
              {
                unsigned int m =
                  6 * n_dofs_face + n_dofs_face * 2 * (normal_degree - 1);
                lexicographic_numbering.push_back(m + j + k_add);
              }

            unsigned int e = j + 5 * n_dofs_face;
            lexicographic_numbering.push_back(e + k_add);
          }
      // for (unsigned int i = 4 * n_dofs_face; i < 5 * n_dofs_face; ++i)
      //   lexicographic_numbering.push_back(i);
      // if (normal_degree > 1)
      //   for (unsigned int i =
      //          6 * n_dofs_face + n_dofs_face * 2 * (normal_degree - 1);
      //        i < 6 * n_dofs_face + n_dofs_face * 3 * (normal_degree - 1);
      //        ++i)
      //     lexicographic_numbering.push_back(i);
      // for (unsigned int i = 5 * n_dofs_face; i < 6 * n_dofs_face; ++i)
      //   lexicographic_numbering.push_back(i);
    }

  return lexicographic_numbering;
}

template <int dim, int degree>
std::vector<types::global_dof_index>
patch_dofs_numbering(std::array<std::vector<unsigned int>, 1 << dim> &cell_dofs)
{
  constexpr unsigned int n_cells = 1 << dim;

  std::array<std::vector<unsigned int>, dim> cell_number;

  if (dim == 2)
    {
      cell_number[0] = {{0, 1, 2, 3}};
      cell_number[1] = {{0, 2, 1, 3}};
    }
  else if (dim == 3)
    {
      cell_number[0] = {{0, 1, 2, 3, 4, 5, 6, 7}};
      cell_number[1] = {{0, 2, 1, 3, 4, 6, 5, 7}};
      cell_number[2] = {{0, 4, 1, 5, 2, 6, 3, 7}};
    }

  std::vector<types::global_dof_index> local_dof_indices;

  const unsigned int layer = dim == 2 ? 1 : degree + 1;
  const unsigned int n_z   = dim == 2 ? 1 : 2;

  for (auto d = 0U; d < dim; ++d)
    for (auto z = 0U; z < n_z; ++z)
      for (auto l = 0U; l < layer; ++l)
        for (auto row = 0U; row < 2; ++row)
          for (auto i = 0U; i < degree + 1; ++i)
            for (auto col = 0U; col < 2; ++col)
              for (auto k = 0U; k < degree + 2; ++k)
                {
                  if (k == 0 && col == 1)
                    continue;

                  const unsigned int cell =
                    cell_number[d][z * 4 + row * 2 + col];

                  local_dof_indices.push_back(
                    cell_dofs[cell][d * Util::pow(degree + 1, dim - 1) *
                                      (degree + 2) +
                                    l * (degree + 1) * (degree + 2) +
                                    i * (degree + 2) + k]);
                }

  return local_dof_indices;
}

template <int dim, int degree, typename Number = double>
void
test()
{
  FE_RaviartThomas_new<dim> fe_v(degree);
  FE_DGQLegendre<dim>       fe_p(degree);
  QGauss<1>                 quadrature(degree + 2);

  auto numbering = get_lexicographic_numbering<dim>(degree + 1, degree);

  for (auto ind : numbering)
    std::cout << ind << ", ";
  std::cout << std::endl;

  Triangulation<dim> triangulation(
    Triangulation<dim>::limit_level_difference_at_vertices);
  GridGenerator::hyper_cube(triangulation, 0., 1.);
  triangulation.refine_global(1);

  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe_v);
  dof_handler.distribute_mg_dofs();

  const unsigned int                   dofs_per_cell = fe_v.n_dofs_per_cell();
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::array<std::vector<unsigned int>, 1 << dim> cell_dofs;

  unsigned int c = 0;

  std::cout << std::endl;
  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      cell->get_dof_indices(local_dof_indices);

      for (auto ind : local_dof_indices)
        std::cout << ind << ", ";
      std::cout << std::endl;

      for (auto i = 0U; i < dofs_per_cell; ++i)
        std::cout << local_dof_indices[numbering[i]] << ", ";
      std::cout << std::endl;

      std::cout << std::endl;

      cell_dofs[c].resize(dofs_per_cell);
      for (auto i = 0U; i < dofs_per_cell; ++i)
        cell_dofs[c][i] = local_dof_indices[numbering[i]];

      c++;
    }

  std::cout << std::endl;


  auto patch_numbering = patch_dofs_numbering<dim, degree>(cell_dofs);

  for (auto i : patch_numbering)
    std::cout << i << " ";
  std::cout << std::endl;
}

int
main()
{
  test<2, 1>();
  test<2, 2>();
  test<3, 1>();

  // test<2, 5>();
}