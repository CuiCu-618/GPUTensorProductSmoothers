
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/affine_constraints.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <helper_cuda.h>

#include <iostream>

#include "tensor_product.h"
#include "utilities.cuh"

// Matrix-free implementation with continous elements.

using namespace dealii;

template <int dim, int fe_degree>
void
test()
{
  Triangulation<dim> triangulation(
    Triangulation<dim>::limit_level_difference_at_vertices);
  GridGenerator::hyper_cube(triangulation, 0., 1.);

  triangulation.refine_global(1);

  FE_Q<dim>       fe(fe_degree);
  DoFHandler<dim> dof_handler(triangulation);

  dof_handler.distribute_dofs(fe);
  dof_handler.distribute_mg_dofs();

  const unsigned int nlevels = triangulation.n_global_levels();
  for (unsigned int level = 0; level < nlevels; ++level)
    Util::Lexicographic(dof_handler, level);
  Util::Lexicographic(dof_handler);


  AffineConstraints<double> constraints;
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  VectorTools::interpolate_boundary_values(dof_handler,
                                           0,
                                           Functions::ZeroFunction<dim>(),
                                           constraints);
  constraints.close();

  {
    auto print_matrices = [](auto matrix) {
      for (auto m = 0U; m < matrix.size(1); ++m)
        {
          for (auto n = 0U; n < matrix.size(0); ++n)
            std::cout << matrix(m, n) << " ";
          std::cout << std::endl;
        }
      std::cout << std::endl;
    };


    FE_DGQ<1> fe_1d(fe_degree);

    constexpr unsigned int N              = fe_degree + 1;
    const double           tau            = 0.1;
    const double           h              = 1. / 2;
    const double           scaling_factor = dim == 2 ? 1 : 1. / Util::pow(2, 1);

    QGauss<1> quadrature(N);

    std::array<Table<2, double>, dim> patch_mass;

    for (unsigned int d = 0; d < dim; ++d)
      {
        patch_mass[d].reinit(2 * N - 1, 2 * N - 1);
      }

    auto get_cell_laplace = [&]() {
      FullMatrix<double> cell_laplace(N, N);

      for (unsigned int i = 0; i < N; ++i)
        for (unsigned int j = 0; j < N; ++j)
          {
            double sum_laplace = 0;
            for (unsigned int q = 0; q < quadrature.size(); ++q)
              {
                sum_laplace += tau *
                               (fe_1d.shape_grad(i, quadrature.point(q))[0] *
                                fe_1d.shape_grad(j, quadrature.point(q))[0]) *
                               quadrature.weight(q);

                sum_laplace += (fe_1d.shape_value(i, quadrature.point(q)) *
                                fe_1d.shape_value(j, quadrature.point(q))) *
                               quadrature.weight(q) * h * h / dim;
              }

            // scaling to real cells
            cell_laplace(i, j) = sum_laplace * scaling_factor;
          }

      return cell_laplace;
    };

    for (unsigned int i = 0; i < N; ++i)
      for (unsigned int j = 0; j < N; ++j)
        {
          double sum_mass = 0;
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            {
              sum_mass += (fe_1d.shape_value(i, quadrature.point(q)) *
                           fe_1d.shape_value(j, quadrature.point(q))) *
                          quadrature.weight(q);
            }
          for (unsigned int d = 0; d < dim; ++d)
            {
              patch_mass[d](i, j) += sum_mass;
              patch_mass[d](i + N - 1, j + N - 1) += sum_mass;
            }
        }

    auto laplace_middle = get_cell_laplace();

    // mass, laplace
    auto get_patch_laplace = [&](auto left, auto right) {
      std::array<Table<2, double>, dim> patch_laplace;

      for (unsigned int d = 0; d < dim; ++d)
        {
          patch_laplace[d].reinit(2 * N - 1, 2 * N - 1);
        }

      for (unsigned int d = 0; d < dim; ++d)
        for (unsigned int i = 0; i < N; ++i)
          for (unsigned int j = 0; j < N; ++j)
            {
              patch_laplace[d](i, j) += left(i, j);
              patch_laplace[d](i + N - 1, j + N - 1) += right(i, j);
            }

      return patch_laplace;
    };

    auto patch_laplace = get_patch_laplace(laplace_middle, laplace_middle);

    print_matrices(patch_laplace[0]);
    print_matrices(patch_mass[0]);

    // eigenvalue, eigenvector
    std::array<Table<2, double>, dim> patch_mass_inv;
    std::array<Table<2, double>, dim> patch_laplace_inv;

    for (unsigned int d = 0; d < dim; ++d)
      {
        patch_mass_inv[d].reinit(2 * N - 3, 2 * N - 3);
        patch_laplace_inv[d].reinit(2 * N - 3, 2 * N - 3);
      }

    for (unsigned int d = 0; d < dim; ++d)
      for (unsigned int i = 0; i < 2 * N - 3; ++i)
        for (unsigned int j = 0; j < 2 * N - 3; ++j)
          {
            patch_mass_inv[d](i, j)    = patch_mass[d](i + 1, j + 1);
            patch_laplace_inv[d](i, j) = patch_laplace[d](i + 1, j + 1);
          }

    print_matrices(patch_laplace_inv[0]);
    print_matrices(patch_mass_inv[0]);
    print_matrices(patch_laplace_inv[1]);
    print_matrices(patch_mass_inv[1]);

    PSMF::TensorProductData<dim, fe_degree, double> tensor_product;
    tensor_product.reinit(patch_mass_inv, patch_laplace_inv);

    std::array<AlignedVector<double>, dim> eigenval;
    std::array<Table<2, double>, dim>      eigenvec;
    tensor_product.get_eigenvalues(eigenval);
    tensor_product.get_eigenvectors(eigenvec);

    print_matrices(eigenvec[0]);
    print_matrices(eigenvec[1]);
    for (auto e : eigenval[0])
      std::cout << e << " ";
    std::cout << "\n";
    for (auto e : eigenval[1])
      std::cout << e << " ";
    std::cout << "\n";
  }
}

int
main(int argc, char *argv[])
{
  int device_id = findCudaDevice(argc, (const char **)argv);
  AssertCuda(cudaSetDevice(device_id));

  test<2, 1>();

  return 0;
}
