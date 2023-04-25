/**
 * @file patch_base.template.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief This class collects all the data that is stored for the matrix free implementation.
 * @version 1.0
 * @date 2023-02-02
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "loop_kernel.cuh"

namespace PSMF
{

  template <int dim, int fe_degree, typename Number>
  LevelVertexPatch<dim, fe_degree, Number>::LevelVertexPatch()
  {}

  template <int dim, int fe_degree, typename Number>
  LevelVertexPatch<dim, fe_degree, Number>::~LevelVertexPatch()
  {
    free();
  }

  template <int dim, int fe_degree, typename Number>
  void
  LevelVertexPatch<dim, fe_degree, Number>::free()
  {
    for (auto &first_dof_color_ptr : first_dof_laplace)
      Utilities::CUDA::free(first_dof_color_ptr);
    first_dof_laplace.clear();

    for (auto &first_dof_color_ptr : first_dof_smooth)
      Utilities::CUDA::free(first_dof_color_ptr);
    first_dof_smooth.clear();

    for (auto &patch_id_color_ptr : patch_id)
      Utilities::CUDA::free(patch_id_color_ptr);
    patch_id.clear();

    for (auto &patch_type_color_ptr : patch_type)
      Utilities::CUDA::free(patch_type_color_ptr);
    patch_type.clear();

    Utilities::CUDA::free(laplace_mass_1d);
    Utilities::CUDA::free(laplace_stiff_1d);
    Utilities::CUDA::free(bilaplace_stiff_1d);
    Utilities::CUDA::free(smooth_mass_1d);
    Utilities::CUDA::free(smooth_stiff_1d);
    Utilities::CUDA::free(smooth_bilaplace_1d);
    Utilities::CUDA::free(eigenvalues);
    Utilities::CUDA::free(eigenvectors);


    ordering_to_type.clear();
    patch_id_host.clear();
    patch_type_host.clear();
    first_dof_host.clear();
    h_to_l_host.clear();
    l_to_h_host.clear();
  }

  template <int dim, int fe_degree, typename Number>
  std::size_t
  LevelVertexPatch<dim, fe_degree, Number>::memory_consumption() const
  {
    const unsigned int n_dofs_1d = 2 * fe_degree + 1;

    std::size_t result = 0;

    // For each color, add first_dof, patch_id, {mass,derivative}_matrix,
    // and eigen{values,vectors}.
    for (unsigned int i = 0; i < n_colors; ++i)
      {
        result += 2 * n_patches_laplace[i] * sizeof(unsigned int) +
                  2 * n_dofs_1d * n_dofs_1d * (1 << level) * sizeof(Number) +
                  2 * n_dofs_1d * dim * sizeof(Number);
      }
    return result;
  }

  template <int dim, int fe_degree, typename Number>
  std::vector<std::vector<
    typename LevelVertexPatch<dim, fe_degree, Number>::CellIterator>>
  LevelVertexPatch<dim, fe_degree, Number>::gather_vertex_patches(
    const DoFHandler<dim> &dof_handler,
    const unsigned int     level) const
  {
    // LAMBDA checks if a vertex is at the physical boundary
    auto &&is_boundary_vertex = [](const CellIterator &cell,
                                   const unsigned int  vertex_id) {
      return std::any_of(
        std::begin(GeometryInfo<dim>::vertex_to_face[vertex_id]),
        std::end(GeometryInfo<dim>::vertex_to_face[vertex_id]),
        [&cell](const auto &face_no) { return cell->at_boundary(face_no); });
    };

    const auto locally_owned_range_mg =
      filter_iterators(dof_handler.mg_cell_iterators_on_level(level),
                       IteratorFilters::LocallyOwnedLevelCell());
    /**
     * A mapping @p global_to_local_map between the global vertex and
     * the pair containing the number of locally owned cells and the
     * number of all cells (including ghosts) is constructed
     */
    std::map<unsigned int, std::pair<unsigned int, unsigned int>>
      global_to_local_map;
    for (const auto &cell : locally_owned_range_mg)
      {
        for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          if (!is_boundary_vertex(cell, v))
            {
              const unsigned int global_index = cell->vertex_index(v);
              const auto element = global_to_local_map.find(global_index);
              if (element != global_to_local_map.cend())
                {
                  ++(element->second.first);
                  ++(element->second.second);
                }
              else
                {
                  const auto n_cells_pair = std::pair<unsigned, unsigned>{1, 1};
                  const auto status       = global_to_local_map.insert(
                    std::make_pair(global_index, n_cells_pair));
                  (void)status;
                  Assert(status.second,
                         ExcMessage("failed to insert key-value-pair"))
                }
            }
      }

    /**
     * Enumerate the patches contained in @p global_to_local_map by
     * replacing the former number of locally owned cells in terms of a
     * consecutive numbering. The local numbering is required for
     * gathering the level cell iterators into a collection @
     * cell_collections according to the global vertex index.
     */
    unsigned int local_index = 0;
    for (auto &key_value : global_to_local_map)
      {
        key_value.second.first = local_index++;
      }
    const unsigned n_subdomains = global_to_local_map.size();
    AssertDimension(n_subdomains, local_index);
    std::vector<std::vector<CellIterator>> cell_collections;
    cell_collections.resize(n_subdomains);
    for (auto &cell : dof_handler.mg_cell_iterators_on_level(level))
      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
        {
          const unsigned int global_index = cell->vertex_index(v);
          const auto         element = global_to_local_map.find(global_index);
          if (element != global_to_local_map.cend())
            {
              const unsigned int local_index = element->second.first;
              const unsigned int patch_size  = element->second.second;
              auto              &collection  = cell_collections[local_index];
              if (collection.empty())
                collection.resize(patch_size);
              if (patch_size == regular_vpatch_size) // regular patch
                collection[regular_vpatch_size - 1 - v] = cell;
              else                                   // irregular patch
                AssertThrow(false, ExcMessage("TODO irregular vertex patches"));
            }
        }
    return cell_collections;
  }

  template <int dim, int fe_degree, typename Number>
  void
  LevelVertexPatch<dim, fe_degree, Number>::get_patch_data(
    const PatchIterator &patch,
    const unsigned int   patch_id)
  {
    std::vector<unsigned int> local_dof_indices(Util::pow(fe_degree + 1, dim));
    std::vector<unsigned int> numbering(regular_vpatch_size);
    std::iota(numbering.begin(), numbering.end(), 0);

    // first_dof
    for (unsigned int cell = 0; cell < 1; ++cell)
      {
        auto cell_ptr = (*patch)[cell];
        cell_ptr->get_mg_dof_indices(local_dof_indices);
        first_dof_host[patch_id * 1 + cell] = local_dof_indices[0];
      }

    // patch_type. TODO: Fix: only works on [0,1]^d
    // TODO: level == 1, one patch only.
    const double h            = 1. / Util::pow(2, level);
    auto         first_center = (*patch)[0]->center();

    if (level == 1)
      for (unsigned int d = 0; d < dim; ++d)
        patch_type_host[patch_id * dim + d] = 2;
    else
      for (unsigned int d = 0; d < dim; ++d)
        {
          auto pos = std::floor(first_center[d] / h + 1 / 3);
          patch_type_host[patch_id * dim + d] =
            (pos > 0) + (pos == (Util::pow(2, level) - 2));
        }


    // patch_id
    // std::sort(numbering.begin(),
    //           numbering.end(),
    //           [&](unsigned lhs, unsigned rhs) {
    //             return first_dof_host[patch_id * 1 + lhs] <
    //                    first_dof_host[patch_id * 1 + rhs];
    //           });

    // auto encode = [&](unsigned int sum, int val) { return sum * 10 + val; };
    // unsigned int label =
    //   std::accumulate(numbering.begin(), numbering.end(), 0, encode);

    // const auto element = ordering_to_type.find(label);
    // if (element != ordering_to_type.end()) // Fouond
    //   {
    //     patch_id_host[patch_id] = element->second;
    //   }
    // else // Not found
    //   {
    //     ordering_to_type.insert({label, ordering_types++});
    //     patch_id_host[patch_id] = ordering_to_type[label];
    //   }
  }

  template <int dim, int fe_degree, typename Number>
  void
  LevelVertexPatch<dim, fe_degree, Number>::reinit(
    const DoFHandler<dim>   &mg_dof,
    const MGConstrainedDoFs &mg_constrained_dofs,
    const unsigned int       mg_level,
    const AdditionalData    &additional_data)
  {
    if (typeid(Number) == typeid(double))
      cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

    this->relaxation         = additional_data.relaxation;
    this->use_coloring       = additional_data.use_coloring;
    this->granularity_scheme = additional_data.granularity_scheme;

    dof_handler = &mg_dof;
    level       = mg_level;

    if (use_coloring)
      n_colors = regular_vpatch_size;
    else
      n_colors = 1;

    switch (granularity_scheme)
      {
        case GranularityScheme::none:
          patch_per_block = 1;
          break;
        case GranularityScheme::user_define:
          patch_per_block = additional_data.patch_per_block;
          break;
        case GranularityScheme::multiple:
          patch_per_block = granularity_shmem<dim, fe_degree>();
          break;
        default:
          AssertThrow(false, ExcMessage("Invalid granularity scheme."));
          break;
      }

    // create patches
    std::vector<std::vector<CellIterator>> cell_collections;
    cell_collections = std::move(gather_vertex_patches(*dof_handler, level));

    graph_ptr_raw.clear();
    graph_ptr_raw.resize(1);
    for (auto patch = cell_collections.begin(); patch != cell_collections.end();
         ++patch)
      graph_ptr_raw[0].push_back(patch);

    // coloring
    graph_ptr_colored.clear();
    graph_ptr_colored.resize(regular_vpatch_size);
    for (auto patch = cell_collections.begin(); patch != cell_collections.end();
         ++patch)
      {
        auto first_cell = (*patch)[0];

        graph_ptr_colored[first_cell->parent()->child_iterator_to_index(
                            first_cell)]
          .push_back(patch);
      }


    setup_color_arrays(n_colors);

    for (unsigned int i = 0; i < regular_vpatch_size; ++i)
      {
        auto n_patches      = graph_ptr_colored[i].size();
        n_patches_smooth[i] = n_patches;

        patch_type_host.clear();
        patch_id_host.clear();
        first_dof_host.clear();
        patch_id_host.resize(n_patches);
        patch_type_host.resize(n_patches * dim);
        first_dof_host.resize(n_patches * 1);

        auto patch     = graph_ptr_colored[i].begin(),
             end_patch = graph_ptr_colored[i].end();
        for (unsigned int p_id = 0; patch != end_patch; ++patch, ++p_id)
          get_patch_data(*patch, p_id);

        alloc_arrays(&first_dof_smooth[i], n_patches * 1);

        cudaError_t error_code =
          cudaMemcpy(first_dof_smooth[i],
                     first_dof_host.data(),
                     1 * n_patches * sizeof(unsigned int),
                     cudaMemcpyHostToDevice);
        AssertCuda(error_code);
      }

    std::vector<std::vector<PatchIterator>> tmp_ptr;
    tmp_ptr = use_coloring ? graph_ptr_colored : graph_ptr_raw;

    ordering_to_type.clear();
    ordering_types = 0;
    for (unsigned int i = 0; i < n_colors; ++i)
      {
        auto n_patches       = tmp_ptr[i].size();
        n_patches_laplace[i] = n_patches;

        patch_type_host.clear();
        patch_id_host.clear();
        first_dof_host.clear();
        patch_id_host.resize(n_patches);
        patch_type_host.resize(n_patches * dim);
        first_dof_host.resize(n_patches * 1);

        auto patch = tmp_ptr[i].begin(), end_patch = tmp_ptr[i].end();
        for (unsigned int p_id = 0; patch != end_patch; ++patch, ++p_id)
          get_patch_data(*patch, p_id);

        // alloc_and_copy_arrays(i);
        alloc_arrays(&first_dof_laplace[i], n_patches * 1);
        alloc_arrays(&patch_id[i], n_patches);
        alloc_arrays(&patch_type[i], n_patches * dim);

        cudaError_t error_code = cudaMemcpy(patch_id[i],
                                            patch_id_host.data(),
                                            n_patches * sizeof(unsigned int),
                                            cudaMemcpyHostToDevice);
        AssertCuda(error_code);

        error_code = cudaMemcpy(first_dof_laplace[i],
                                first_dof_host.data(),
                                1 * n_patches * sizeof(unsigned int),
                                cudaMemcpyHostToDevice);
        AssertCuda(error_code);

        error_code = cudaMemcpy(patch_type[i],
                                patch_type_host.data(),
                                dim * n_patches * sizeof(unsigned int),
                                cudaMemcpyHostToDevice);
        AssertCuda(error_code);
      }

    setup_configuration(n_colors);

    // Mapping
    if (dim == 2)
      {
        lookup_table.insert({123, {{0, 1}}}); // x-y
        lookup_table.insert({213, {{1, 0}}}); // y-x
      }
    else if (dim == 3)
      {
        lookup_table.insert({1234567, {{0, 1, 2}}}); // x-y-z
        lookup_table.insert({1452367, {{0, 2, 1}}}); // x-z-y
        lookup_table.insert({2134657, {{1, 0, 2}}}); // y-x-z
        lookup_table.insert({2461357, {{1, 2, 0}}}); // y-z-x
        lookup_table.insert({4152637, {{2, 0, 1}}}); // z-x-y
        lookup_table.insert({4261537, {{2, 1, 0}}}); // z-y-x
      }

    constexpr unsigned int n_dofs_1d = 2 * fe_degree + 1;
    constexpr unsigned int N         = fe_degree + 1;
    constexpr unsigned int z         = dim == 2 ? 1 : fe_degree + 1;
    h_to_l_host.resize(Util::pow(n_dofs_1d, dim) * dim * (dim - 1));
    l_to_h_host.resize(Util::pow(n_dofs_1d, dim) * dim * (dim - 1));

    auto generate_indices = [&](unsigned int label, unsigned int type) {
      const unsigned int          offset = type * Util::pow(n_dofs_1d, dim);
      std::array<unsigned int, 3> strides;
      for (unsigned int i = 0; i < 3; ++i)
        strides[i] = Util::pow(n_dofs_1d, lookup_table[label][i]);

      unsigned int count = 0;

      for (unsigned i = 0; i < dim - 1; ++i)
        for (unsigned int j = 0; j < 2; ++j)
          for (unsigned int k = 0; k < 2; ++k)
            for (unsigned int l = 0; l < z; ++l)
              for (unsigned int m = 0; m < fe_degree + 1; ++m)
                for (unsigned int n = 0; n < fe_degree + 1; ++n)
                  {
                    h_to_l_host[offset + (i * N) * strides[2] +
                                l * n_dofs_1d * n_dofs_1d +
                                (j * N) * strides[1] + m * n_dofs_1d +
                                (k * N) * strides[0] + n] = count;
                    l_to_h_host[offset + count++] =
                      (i * N) * strides[2] + l * n_dofs_1d * n_dofs_1d +
                      (j * N) * strides[1] + m * n_dofs_1d +
                      (k * N) * strides[0] + n;
                  }
    };
    for (auto &el : ordering_to_type)
      generate_indices(el.first, el.second);

    alloc_arrays(&l_to_h, Util::pow(n_dofs_1d, dim) * dim * (dim - 1));
    alloc_arrays(&h_to_l, Util::pow(n_dofs_1d, dim) * dim * (dim - 1));

    cudaError_t error_code = cudaMemcpy(l_to_h,
                                        l_to_h_host.data(),
                                        Util::pow(n_dofs_1d, dim) * dim *
                                          (dim - 1) * sizeof(unsigned int),
                                        cudaMemcpyHostToDevice);
    AssertCuda(error_code);

    error_code = cudaMemcpy(h_to_l,
                            h_to_l_host.data(),
                            Util::pow(n_dofs_1d, dim) * dim * (dim - 1) *
                              sizeof(unsigned int),
                            cudaMemcpyHostToDevice);
    AssertCuda(error_code);

    auto copy_to_device = [](auto &device, const auto &host) {
      LinearAlgebra::ReadWriteVector<unsigned int> rw_vector(host.size());
      device.reinit(host.size());
      for (unsigned int i = 0; i < host.size(); ++i)
        rw_vector[i] = host[i];
      device.import(rw_vector, VectorOperation::insert);
    };

    std::vector<unsigned int> dirichlet_index_vector;
    if (mg_constrained_dofs.have_boundary_indices())
      {
        mg_constrained_dofs.get_boundary_indices(level).fill_index_vector(
          dirichlet_index_vector);
        copy_to_device(dirichlet_indices, dirichlet_index_vector);
      }

    constexpr unsigned n_dofs_2d = n_dofs_1d * n_dofs_1d;

    alloc_arrays(&eigenvalues, n_dofs_1d * dim);
    alloc_arrays(&eigenvectors, n_dofs_2d * dim);
    alloc_arrays(&smooth_mass_1d, n_dofs_2d);
    alloc_arrays(&smooth_stiff_1d, n_dofs_2d);
    alloc_arrays(&smooth_bilaplace_1d, n_dofs_2d);
    alloc_arrays(&laplace_mass_1d, n_dofs_2d * 3);
    alloc_arrays(&laplace_stiff_1d, n_dofs_2d * 3);
    alloc_arrays(&bilaplace_stiff_1d, n_dofs_2d * 3);

    reinit_tensor_product_laplace();
    reinit_tensor_product_smoother();
  }

  template <int dim, int fe_degree, typename Number>
  LevelVertexPatch<dim, fe_degree, Number>::Data
  LevelVertexPatch<dim, fe_degree, Number>::get_laplace_data(
    unsigned int color) const
  {
    Data data_copy;

    data_copy.n_dofs_per_dim     = (1 << level) * fe_degree + 1;
    data_copy.n_patches          = n_patches_laplace[color];
    data_copy.patch_per_block    = patch_per_block;
    data_copy.first_dof          = first_dof_laplace[color];
    data_copy.patch_id           = patch_id[color];
    data_copy.patch_type         = patch_type[color];
    data_copy.l_to_h             = l_to_h;
    data_copy.h_to_l             = h_to_l;
    data_copy.laplace_mass_1d    = laplace_mass_1d;
    data_copy.laplace_stiff_1d   = laplace_stiff_1d;
    data_copy.bilaplace_stiff_1d = bilaplace_stiff_1d;

    return data_copy;
  }

  template <int dim, int fe_degree, typename Number>
  LevelVertexPatch<dim, fe_degree, Number>::Data
  LevelVertexPatch<dim, fe_degree, Number>::get_smooth_data(
    unsigned int color) const
  {
    Data data_copy;

    data_copy.n_dofs_per_dim      = (1 << level) * fe_degree + 1;
    data_copy.n_patches           = n_patches_smooth[color];
    data_copy.patch_per_block     = patch_per_block;
    data_copy.relaxation          = relaxation;
    data_copy.first_dof           = first_dof_smooth[color];
    data_copy.l_to_h              = l_to_h;
    data_copy.h_to_l              = h_to_l;
    data_copy.eigenvalues         = eigenvalues;
    data_copy.eigenvectors        = eigenvectors;
    data_copy.smooth_mass_1d      = smooth_mass_1d;
    data_copy.smooth_stiff_1d     = smooth_stiff_1d;
    data_copy.smooth_bilaplace_1d = smooth_bilaplace_1d;

    return data_copy;
  }

  template <int dim, int fe_degree, typename Number>
  template <typename Operator, typename VectorType>
  void
  LevelVertexPatch<dim, fe_degree, Number>::patch_loop(const Operator   &op,
                                                       const VectorType &src,
                                                       VectorType &dst) const
  {
    op.setup_kernel(patch_per_block);

    for (unsigned int i = 0; i < regular_vpatch_size; ++i)
      if (n_patches_smooth[i] > 0)
        {
          op.loop_kernel(src,
                         dst,
                         get_smooth_data(i),
                         grid_dim_smooth[i],
                         block_dim_smooth[i]);

          AssertCudaKernel();
        }
  }

  template <int dim, int fe_degree, typename Number>
  template <typename Operator, typename VectorType>
  void
  LevelVertexPatch<dim, fe_degree, Number>::cell_loop(const Operator   &op,
                                                      const VectorType &src,
                                                      VectorType &dst) const
  {
    op.setup_kernel(patch_per_block);

    for (unsigned int i = 0; i < n_colors; ++i)
      if (n_patches_laplace[i] > 0)
        {
          op.loop_kernel(src,
                         dst,
                         get_laplace_data(i),
                         grid_dim_lapalce[i],
                         block_dim_laplace[i]);

          AssertCudaKernel();
        }
  }

  template <int dim, int fe_degree, typename Number>
  void
  LevelVertexPatch<dim, fe_degree, Number>::reinit_tensor_product_smoother()
    const
  {
    auto mass_tensor      = assemble_mass_tensor();
    auto laplace_tensor   = assemble_laplace_tensor();
    auto bilaplace_tensor = assemble_bilaplace_tensor();

    auto copy_mats = [](auto tensor, auto dst) {
      constexpr unsigned int n_dofs_2d = Util::pow(2 * fe_degree + 1, 2);

      auto mat = new Number[n_dofs_2d];

      std::transform(tensor.begin(), tensor.end(), mat, [](auto m) -> Number {
        return m.value()[0];
      });

      cudaError_t error_code = cudaMemcpy(dst,
                                          mat,
                                          n_dofs_2d * sizeof(Number),
                                          cudaMemcpyHostToDevice);
      AssertCuda(error_code);

      delete[] mat;
    };

    copy_mats(mass_tensor.back(), smooth_mass_1d);
    copy_mats(laplace_tensor.back(), smooth_stiff_1d);
    copy_mats(bilaplace_tensor.back(), smooth_bilaplace_1d);

    auto interior = [](auto matrix) {
      Table<2, VectorizedArray<Number>> dst(matrix.back().n_rows() - 2,
                                            matrix.back().n_cols() - 2);

      for (unsigned int i = 0; i < matrix.back().n_rows() - 2; ++i)
        for (unsigned int j = 0; j < matrix.back().n_cols() - 2; ++j)
          dst(i, j) = matrix.back()(i + 1, j + 1);
      return dst;
    };

    auto mass_tensor_inv      = interior(mass_tensor);
    auto laplace_tensor_inv   = interior(laplace_tensor);
    auto bilaplace_tensor_inv = interior(bilaplace_tensor);


    // auto print_matrices = [](auto matrix) {
    //   for (auto m = 0U; m < matrix.size(1); ++m)
    //     {
    //       for (auto n = 0U; n < matrix.size(0); ++n)
    //         std::cout << matrix(m, n) << " ";
    //       std::cout << std::endl;
    //     }
    //   std::cout << std::endl;
    // };

    // print_matrices(mass_tensor_inv);
    // print_matrices(laplace_tensor_inv);
    // print_matrices(bilaplace_tensor_inv);

    auto copy_vals = [](auto tensor, auto dst) {
      constexpr unsigned int n_dofs_1d = Util::pow(2 * fe_degree - 1, 1);

      auto mat = new Number[n_dofs_1d * dim];
      for (unsigned int i = 0; i < dim; ++i)
        std::transform(tensor[i].begin(),
                       tensor[i].end(),
                       &mat[n_dofs_1d * i],
                       [](auto m) -> Number { return m[0]; });

      cudaError_t error_code = cudaMemcpy(dst,
                                          mat,
                                          dim * n_dofs_1d * sizeof(Number),
                                          cudaMemcpyHostToDevice);
      AssertCuda(error_code);

      delete[] mat;
    };

    auto copy_vecs = [](auto tensor, auto dst) {
      constexpr unsigned int n_dofs_2d = Util::pow(2 * fe_degree - 1, 2);

      auto mat = new Number[n_dofs_2d * dim];
      for (unsigned int i = 0; i < dim; ++i)
        std::transform(tensor[i].begin(),
                       tensor[i].end(),
                       &mat[n_dofs_2d * i],
                       [](auto m) -> Number { return m.value()[0]; });

      cudaError_t error_code = cudaMemcpy(dst,
                                          mat,
                                          dim * n_dofs_2d * sizeof(Number),
                                          cudaMemcpyHostToDevice);
      AssertCuda(error_code);

      delete[] mat;
    };

    /// store rank1 tensors of separable Kronecker representation
    /// BxMxM + MxBxM + MxMxB
    const auto &BxMxM = [&](const int direction) {
      std::array<Table<2, VectorizedArray<Number>>, dim> kronecker_tensor;
      for (auto d = 0; d < dim; ++d)
        kronecker_tensor[d] =
          d == direction ? bilaplace_tensor_inv : mass_tensor_inv;
      return kronecker_tensor;
    };

    /// store rank1 tensors of mixed derivatives
    /// 2(LxLxM + LxMxL + MxLxL)
    const auto &LxLxM = [&](const int direction1, const int direction2) {
      std::array<Table<2, VectorizedArray<Number>>, dim> kronecker_tensor;
      for (auto d = 0; d < dim; ++d)
        kronecker_tensor[d] = (d == direction1 || d == direction2) ?
                                laplace_tensor_inv :
                                mass_tensor_inv;
      return kronecker_tensor;
    };


    // KSVD
    std::set<unsigned int> ksvd_tensor_indices = {0U, 1U};

    {
      std::vector<std::array<Table<2, VectorizedArray<Number>>, dim>>
        rank1_tensors;

      for (auto direction = 0; direction < dim; ++direction)
        rank1_tensors.emplace_back(BxMxM(direction));

      for (auto direction1 = 0; direction1 < dim; ++direction1)
        for (auto direction2 = 0; direction2 < dim; ++direction2)
          if (direction1 != direction2)
            rank1_tensors.emplace_back(LxLxM(direction1, direction2));

      AssertDimension(rank1_tensors.size(), dim * dim);

      // KSVD
      std::array<std::size_t, dim> rows, columns;
      for (auto d = 0U; d < dim; ++d)
        {
          const auto &A_d = rank1_tensors.front()[d];
          rows[d]         = A_d.size(0);
          columns[d]      = A_d.size(1);
        }

      const auto ksvd_rank = *(ksvd_tensor_indices.rbegin()) + 1;
      AssertIndexRange(ksvd_rank, rank1_tensors.size() + 1);
      auto ksvd_tensors =
        Tensors::make_zero_rank1_tensors<dim, VectorizedArray<Number>>(
          ksvd_rank, rows, columns);

      const auto &ksvd_singular_values =
        compute_ksvd(rank1_tensors, ksvd_tensors, 5);

      std::vector<std::array<Table<2, VectorizedArray<Number>>, dim>>
        approximation;
      for (auto i = 0U; i < ksvd_tensors.size(); ++i)
        if (ksvd_tensor_indices.find(i) != ksvd_tensor_indices.cend())
          approximation.emplace_back(ksvd_tensors[i]);

      AssertDimension(ksvd_tensor_indices.size(), approximation.size());

      if (approximation.size() == 2U)
        {
          using matrix_type =
            Tensors::TensorProductMatrix<dim, VectorizedArray<Number>>;
          using matrix_state = typename matrix_type::State;

          Number addition_to_min_eigenvalue = 0.025;

          matrix_type local_matrices;

          /// first tensor must contain s.p.d. matrices ("mass matrices")
          typename matrix_type::AdditionalData additional_data;
          additional_data.state = matrix_state::ranktwo;

          local_matrices.reinit(approximation, additional_data);

          const auto &tensor_of_eigenvalues =
            local_matrices.get_eigenvalue_tensor();
          const auto eigenvalues_ksvd1 =
            Tensors::kronecker_product<dim, VectorizedArray<Number>>(
              tensor_of_eigenvalues);

          /// if the rank-2 KSVD isn't positive definite we scale the
          /// second tensor of matrices by a factor \alpha (with 0 <
          /// \alpha < 1), thus obtaing an approximation that is better
          /// than the best rank-1 approximation but worse than the best
          /// rank-2 approximation. \alpha is computed at negligible costs
          /// due to the specific eigendecomposition with tensor structure
          if (ksvd_tensor_indices == std::set<unsigned int>{0U, 1U})
            {
              VectorizedArray<Number> alpha(1.);
              for (auto lane = 0U; lane < VectorizedArray<Number>::size();
                   ++lane)
                {
                  // std::cout << "eigenvalues of KSVD[1]:\n"
                  //           <<
                  //           vector_to_string(alignedvector_to_vector(eigenvalues_ksvd1,
                  //           lane))
                  //           << std::endl;
                  const auto min_elem =
                    std::min_element(eigenvalues_ksvd1.begin(),
                                     eigenvalues_ksvd1.end(),
                                     [&](const auto &lhs, const auto &rhs) {
                                       return lhs[lane] < rhs[lane];
                                     });
                  const Number lambda_min = (*min_elem)[lane];

                  /// \alpha = -1 / ((1 + \epsilon) * \lambda_{min})
                  if (lambda_min < -0.99) // KSVD isn't positive definite
                    alpha[lane] /=
                      -(1. + addition_to_min_eigenvalue) * lambda_min;
                  if (alpha[lane] > 1.)
                    alpha[lane] = 0.99;
                }

              // std::cout << "alpha: " << varray_to_string(alpha) <<
              // std::endl;
              Tensors::scaling<dim>(alpha, approximation.at(1U));
              local_matrices.reinit(approximation, additional_data);

              auto eigenvalue_tensor  = local_matrices.get_eigenvalue_tensor();
              auto eigenvector_tensor = local_matrices.get_eigenvector_tensor();

              auto eigenvalues_ = local_matrices.get_eigenvalues();
              auto eigenvector_ = local_matrices.get_eigenvectors();

              copy_vals(eigenvalue_tensor, eigenvalues);
              copy_vecs(eigenvector_tensor, eigenvectors);

              // print_matrices(eigenvector_);

              // print_matrices(eigenvector_tensor[0]);
              // print_matrices(eigenvector_tensor[1]);

              // for (unsigned int j = 0; j < eigenvalues_.size(); ++j)
              //   std::cout << eigenvalues_[j] << " ";
              // std::cout << std::endl;

              // for (unsigned int i = 0; i < dim; ++i)
              //   for (unsigned int j = 0; j < eigenvalue_tensor[i].size();
              //   ++j)
              //     std::cout << eigenvalue_tensor[i][j] << " ";
              // std::cout << std::endl;
            }
        }
    }
  }


  template <int dim, int fe_degree, typename Number>
  void
  LevelVertexPatch<dim, fe_degree, Number>::reinit_tensor_product_laplace()
    const
  {
    auto mass_tensor      = assemble_mass_tensor();
    auto laplace_tensor   = assemble_laplace_tensor();
    auto bilaplace_tensor = assemble_bilaplace_tensor();

    auto copy_to_device = [](auto tensor, auto dst) {
      constexpr unsigned int n_dofs_2d = Util::pow(2 * fe_degree + 1, 2);

      auto mat = new Number[n_dofs_2d * 3];
      for (unsigned int i = 0; i < 3; ++i)
        std::transform(tensor[i].begin(),
                       tensor[i].end(),
                       &mat[n_dofs_2d * i],
                       [](auto m) -> Number { return m.value()[0]; });

      cudaError_t error_code = cudaMemcpy(dst,
                                          mat,
                                          3 * n_dofs_2d * sizeof(Number),
                                          cudaMemcpyHostToDevice);
      AssertCuda(error_code);

      delete[] mat;
    };

    copy_to_device(mass_tensor, laplace_mass_1d);
    copy_to_device(laplace_tensor, laplace_stiff_1d);
    copy_to_device(bilaplace_tensor, bilaplace_stiff_1d);
  }

  template <int dim, int fe_degree, typename Number>
  std::array<Table<2, VectorizedArray<Number>>, 3>
  LevelVertexPatch<dim, fe_degree, Number>::assemble_mass_tensor() const
  {
    constexpr int n_cell_dofs  = fe_degree + 1;
    constexpr int n_patch_dofs = 2 * n_cell_dofs - 1;

    const Number h              = Util::pow(2, level);
    const Number penalty_factor = h * fe_degree * (fe_degree + 1);

    FE_DGQ<1> fe(fe_degree);
    QGauss<1> quadrature(fe_degree + 1);

    Table<2, Number> patch_mass_01(n_patch_dofs, n_patch_dofs);
    Table<2, Number> patch_mass_2(n_patch_dofs, n_patch_dofs);

    for (unsigned int i = 0; i < n_cell_dofs; ++i)
      for (unsigned int j = 0; j < n_cell_dofs; ++j)
        {
          double sum_mass = 0;
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            {
              sum_mass += (fe.shape_value(i, quadrature.point(q)) *
                           fe.shape_value(j, quadrature.point(q))) *
                          quadrature.weight(q) / h;
            }

          patch_mass_01(i, j) = sum_mass;
          patch_mass_2(i, j) += sum_mass;
          patch_mass_2(i + n_cell_dofs - 1, j + n_cell_dofs - 1) += sum_mass;
        }

    std::array<Table<2, VectorizedArray<Number>>, 3> mass_matrices;
    for (unsigned int d = 0; d < 3; ++d)
      {
        mass_matrices[d].reinit(n_patch_dofs, n_patch_dofs);
        if (d == 2)
          std::transform(patch_mass_2.begin(),
                         patch_mass_2.end(),
                         mass_matrices[d].begin(),
                         [](Number i) -> VectorizedArray<Number> {
                           return make_vectorized_array(i);
                         });
        else
          std::transform(patch_mass_01.begin(),
                         patch_mass_01.end(),
                         mass_matrices[d].begin(),
                         [](Number i) -> VectorizedArray<Number> {
                           return make_vectorized_array(i);
                         });
      }

    return mass_matrices;
  }

  template <int dim, int fe_degree, typename Number>
  std::array<Table<2, VectorizedArray<Number>>, 3>
  LevelVertexPatch<dim, fe_degree, Number>::assemble_laplace_tensor() const
  {
    constexpr int n_cell_dofs  = fe_degree + 1;
    constexpr int n_patch_dofs = 2 * n_cell_dofs - 1;

    const Number h              = Util::pow(2, level);
    const Number penalty_factor = h * fe_degree * (fe_degree + 1);

    FE_DGQ<1> fe(fe_degree);
    QGauss<1> quadrature(fe_degree + 1);

    Table<2, Number> patch_laplace_01(n_patch_dofs, n_patch_dofs);
    Table<2, Number> patch_laplace_2(n_patch_dofs, n_patch_dofs);

    for (unsigned int i = 0; i < n_cell_dofs; ++i)
      for (unsigned int j = 0; j < n_cell_dofs; ++j)
        {
          double sum_laplace = 0;
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            {
              sum_laplace += (fe.shape_grad(i, quadrature.point(q))[0] *
                              fe.shape_grad(j, quadrature.point(q))[0]) *
                             quadrature.weight(q);
            }

          patch_laplace_01(i, j) = sum_laplace * h;
          patch_laplace_2(i, j) += sum_laplace * h;
          patch_laplace_2(i + n_cell_dofs - 1, j + n_cell_dofs - 1) +=
            sum_laplace * h;
        }

    std::array<Table<2, VectorizedArray<Number>>, 3> laplace_matrices;
    for (unsigned int d = 0; d < 3; ++d)
      {
        laplace_matrices[d].reinit(n_patch_dofs, n_patch_dofs);
        if (d == 2)
          std::transform(patch_laplace_2.begin(),
                         patch_laplace_2.end(),
                         laplace_matrices[d].begin(),
                         [](Number i) -> VectorizedArray<Number> {
                           return make_vectorized_array(i);
                         });
        else
          std::transform(patch_laplace_01.begin(),
                         patch_laplace_01.end(),
                         laplace_matrices[d].begin(),
                         [](Number i) -> VectorizedArray<Number> {
                           return make_vectorized_array(i);
                         });
      }

    return laplace_matrices;
  }

  template <int dim, int fe_degree, typename Number>
  std::array<Table<2, VectorizedArray<Number>>, 4>
  LevelVertexPatch<dim, fe_degree, Number>::assemble_bilaplace_tensor() const
  {
    constexpr int n_cell_dofs  = fe_degree + 1;
    constexpr int n_patch_dofs = 2 * n_cell_dofs - 1;

    const Number h              = Util::pow(2, level);
    const Number penalty_factor = h * fe_degree * (fe_degree + 1);

    FE_DGQ<1> fe(fe_degree);
    QGauss<1> quadrature(fe_degree + 1);

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

    auto cell_bilaplace = [&](unsigned int type, unsigned int pos) {
      Number boundary_factor_left  = 1.;
      Number boundary_factor_right = 1.;

      unsigned int is_first = pos == 0 ? 1 : 0;

      if (type == 0)
        boundary_factor_left = 2.;
      else if (type == 1 && pos == 0)
        boundary_factor_left = 0.;
      else if (type == 1 && pos == 1)
        boundary_factor_right = 0.;
      else if (type == 2)
        boundary_factor_right = 2.;
      else if (type == 3)
        is_first = 1;

      FullMatrix<Number> cell(n_cell_dofs, n_cell_dofs);

      for (unsigned int i = 0; i < n_cell_dofs; ++i)
        for (unsigned int j = 0; j < n_cell_dofs; ++j)
          {
            Number sum_laplace = 0;
            for (unsigned int q = 0; q < quadrature.size(); ++q)
              {
                sum_laplace += (fe.shape_grad_grad(i, quadrature.point(q))[0] *
                                fe.shape_grad_grad(j, quadrature.point(q))[0]) *
                               quadrature.weight(q) * h * h * h * is_first;
              }

            sum_laplace +=
              boundary_factor_left *
              (1. * fe.shape_grad(i, Point<1>()) *
                 fe.shape_grad(j, Point<1>()) * penalty_factor * h * h +
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
            cell(i, j) = sum_laplace;
          }

      return cell;
    };

    auto cell_left     = cell_bilaplace(0, 0);
    auto cell_middle_0 = cell_bilaplace(1, 0);
    auto cell_middle_1 = cell_bilaplace(1, 1);
    auto cell_right    = cell_bilaplace(2, 0);
    auto cell          = cell_bilaplace(3, 0);

    auto patch_bilaplace = [&](auto left, auto right) {
      Table<2, Number> patch_bi(n_patch_dofs, n_patch_dofs);

      for (unsigned int i = 0; i < n_cell_dofs; ++i)
        for (unsigned int j = 0; j < n_cell_dofs; ++j)
          {
            patch_bi(i, j) += left(i, j);
            patch_bi(i + n_cell_dofs - 1, j + n_cell_dofs - 1) += right(i, j);

            patch_bi(i, j + n_cell_dofs - 1) += laplace_interface_mixed(i, j);
            patch_bi(i, j + n_cell_dofs - 1) +=
              laplace_interface_mixed(n_cell_dofs - 1 - j, n_cell_dofs - 1 - i);
            patch_bi(i, j + n_cell_dofs - 1) +=
              laplace_interface_penalty(n_cell_dofs - 1 - j,
                                        n_cell_dofs - 1 - i);

            patch_bi(i + n_cell_dofs - 1, j) += laplace_interface_mixed(j, i);
            patch_bi(i + n_cell_dofs - 1, j) +=
              laplace_interface_mixed(n_cell_dofs - 1 - i, n_cell_dofs - 1 - j);
            patch_bi(i + n_cell_dofs - 1, j) +=
              laplace_interface_penalty(n_cell_dofs - 1 - i,
                                        n_cell_dofs - 1 - j);
          }

      return patch_bi;
    };

    auto patch0 = patch_bilaplace(cell_left, cell_middle_1);
    auto patch1 = patch_bilaplace(cell_middle_0, cell_middle_1);
    auto patch2 = patch_bilaplace(cell_middle_0, cell_right);
    auto patch3 = patch_bilaplace(cell, cell);

    if (level == 1)
      patch2 = patch_bilaplace(cell_left, cell_right);

    std::array<Table<2, VectorizedArray<Number>>, 4> bilaplace_matrices;
    for (unsigned int d = 0; d < 4; ++d)
      {
        bilaplace_matrices[d].reinit(n_patch_dofs, n_patch_dofs);
        if (d == 0)
          std::transform(patch0.begin(),
                         patch0.end(),
                         bilaplace_matrices[d].begin(),
                         [](Number i) -> VectorizedArray<Number> {
                           return make_vectorized_array(i);
                         });
        else if (d == 1)
          std::transform(patch1.begin(),
                         patch1.end(),
                         bilaplace_matrices[d].begin(),
                         [](Number i) -> VectorizedArray<Number> {
                           return make_vectorized_array(i);
                         });
        else if (d == 2)
          std::transform(patch2.begin(),
                         patch2.end(),
                         bilaplace_matrices[d].begin(),
                         [](Number i) -> VectorizedArray<Number> {
                           return make_vectorized_array(i);
                         });
        else if (d == 3)
          std::transform(patch3.begin(),
                         patch3.end(),
                         bilaplace_matrices[d].begin(),
                         [](Number i) -> VectorizedArray<Number> {
                           return make_vectorized_array(i);
                         });
      }

    return bilaplace_matrices;
  }

  template <int dim, int fe_degree, typename Number>
  void
  LevelVertexPatch<dim, fe_degree, Number>::setup_color_arrays(
    const unsigned int n_colors)
  {
    this->n_patches_laplace.resize(n_colors);
    this->grid_dim_lapalce.resize(n_colors);
    this->block_dim_laplace.resize(n_colors);
    this->first_dof_laplace.resize(n_colors);
    this->patch_id.resize(n_colors);
    this->patch_type.resize(n_colors);

    this->n_patches_smooth.resize(regular_vpatch_size);
    this->grid_dim_smooth.resize(regular_vpatch_size);
    this->block_dim_smooth.resize(regular_vpatch_size);
    this->first_dof_smooth.resize(regular_vpatch_size);
  }

  template <int dim, int fe_degree, typename Number>
  void
  LevelVertexPatch<dim, fe_degree, Number>::setup_configuration(
    const unsigned int n_colors)
  {
    constexpr unsigned int n_dofs_1d = 2 * fe_degree + 1;

    for (unsigned int i = 0; i < n_colors; ++i)
      {
        auto         n_patches = n_patches_laplace[i];
        const double apply_n_blocks =
          std::ceil(static_cast<double>(n_patches) /
                    static_cast<double>(patch_per_block));

        grid_dim_lapalce[i]  = dim3(apply_n_blocks);
        block_dim_laplace[i] = dim3(patch_per_block * n_dofs_1d, n_dofs_1d);
      }

    for (unsigned int i = 0; i < regular_vpatch_size; ++i)
      {
        auto         n_patches = n_patches_smooth[i];
        const double apply_n_blocks =
          std::ceil(static_cast<double>(n_patches) /
                    static_cast<double>(patch_per_block));

        grid_dim_smooth[i]  = dim3(apply_n_blocks);
        block_dim_smooth[i] = dim3(patch_per_block * n_dofs_1d, n_dofs_1d);
      }
  }

  template <int dim, int fe_degree, typename Number>
  template <typename Number1>
  void
  LevelVertexPatch<dim, fe_degree, Number>::alloc_arrays(Number1 **array_device,
                                                         const unsigned int n)
  {
    cudaError_t error_code = cudaMalloc(array_device, n * sizeof(Number1));
    AssertCuda(error_code);
  }

  template <typename Number>
  __global__ void
  copy_constrained_values_kernel(const Number       *src,
                                 Number             *dst,
                                 const unsigned int *indices,
                                 const unsigned int  len)
  {
    const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len)
      {
        dst[indices[idx]] = src[indices[idx]];
      }
  }

  template <int dim, int fe_degree, typename Number>
  template <typename VectorType>
  void
  LevelVertexPatch<dim, fe_degree, Number>::copy_constrained_values(
    const VectorType &src,
    VectorType       &dst) const
  {
    const unsigned int len = dirichlet_indices.size();
    if (len > 0)
      {
        const unsigned int bksize  = 256;
        const unsigned int nblocks = (len - 1) / bksize + 1;
        dim3               bk_dim(bksize);
        dim3               gd_dim(nblocks);

        copy_constrained_values_kernel<<<gd_dim, bk_dim>>>(
          src.get_values(),
          dst.get_values(),
          dirichlet_indices.get_values(),
          len);
        AssertCudaKernel();
      }
  }

} // namespace PSMF

  /**
   * \page patch_base.template
   * \include patch_base.template.cuh
   */