/**
 * @file patch_base.cuh
 * @author Cu Cui (cu.cui@iwr.uni-heidelberg.de)
 * @brief This class collects all the data that is stored for the matrix free implementation.
 * @version 1.0
 * @date 2023-02-02
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef PATCH_BASE_CUH
#define PATCH_BASE_CUH

#include <deal.II/grid/filtered_iterator.h>

#include "tensor_product.h"
#include "utilities.cuh"

#define GACCESS 1
// 0 - no global memory access
// 1 - load/store global memeory
// 2 - ideal global memeory access

#define TIMING 0
// 0 - No timing
// 1 - Instruction level
// 2 - Component level, e.g. load, store, vmult.
#define MMAKERNEL 1
// mma.m8n8k4.f64
// 0 - Basic, without permutation_d
// 1 - Conflict Free, 2 warps ILP = 1
// 2 - Conflict Free, 2 warps ILP = 2
// 3 - 4 warps ILP = 1
// 4 - 4 warps ILP = 2
// 5 - Conflict Free, 1 warps ILP = 1
// mma.m16n8k8.tf32
// 0 - Basic, without permutation_d, tbd
// 1 - Conflict Free, 8 warps ILP = 1
// 2 - 8 warps ILP = 1, ld
// 3 - Conflict Free, 8 warps ILP = 2
// 4 - Conflict Free, 16 warps ILP = 1
// 5 - Conflict Free, 4 warps ILP = 1
// mma.m16n8k16.f16
// 0 - Basic, without permutation_d
// 6 - Conflict Free, 8 warps ILP = 1, ld
// 7 - Conflict Free, 8 warps ILP = 1
// 8 - Conflict Free, 16 warps ILP = 1
// TODO: wmma api Q7 half

#define ERRCOR 0
// 0 - basic
// 1 - error correction

// #define DUPLICATE
// root cell = 2 x 2 in 2D and 2 x 2 x 1 in 3D

// #define LOOPUNROLL
// loop unroll, WIP

// #define SKIPZERO
// ignore mat mul with zeros

// #define USECONSTMEM
// stote shape data into constant memory

#define USETEXTURE
// stote shape data into texture memory

using namespace dealii;

/**
 * Namespace for the Patch Smoother Matrix-Free
 */
namespace PSMF
{

  /**
   * @brief Laplace Variant: kernel type for Laplace operator.
   */
  enum class LaplaceVariant
  {
    /**
     * Basic implementation.
     */
    Basic,

    /**
     * Basic implementation. Load data cell by cell.
     */
    BasicCell,

    /**
     * A conflict-free implementation by restructuring shared memory access.
     */
    ConflictFree,

    /**
     * A conflict-free implementation by restructuring data layout in shared
     * memory.
     */
    ConflictFreeMem,

    /**
     * Using the Warp Matrix Multiply and Accumulate (WMMA) API introduced in
     * CUDA 11.0.
     */
    TensorCore,

    /**
     * Using the Matrix Multiply and Accumulate ISA with inline PTX.
     */
    TensorCoreMMA
  };


  /**
   * @brief Smoother Variant: kernel type for
   * Multiplicative Schwarz Smoother.
   */
  enum class SmootherVariant
  {
    AllPatch,

    /**
     * Compute the residual globally, i.e.
     * r = b - Ax, where A is the system matrix.
     */
    GLOBAL,

    /**
     * Same as above, but with linear thread indicex instead of tmasking
     * boundary threads for local solver.
     */
    FUSED_L,

    /**
     * A conflict-free implementation by restructuring shared memory access.
     */
    ConflictFree,

    /**
     * Using the Warp Matrix Multiply and Accumulate (WMMA) API introduced in
     * CUDA 11.0.
     */
    TensorCore,

    /**
     * Compute the residual b - Ax exactly with FE_DGQHermite element.
     */
    ExactRes
  };


  enum class DoFLayout
  {
    DGQ,
    Q,
    RT
  };

  enum class SolverVariant
  {
    GMRES,
    PCG,
    FMG,
    Linear_FMG,
    Vcycle
  };

  enum class ExpementsSets
  {
    none,
    kernel,
    error_analysis,
    solvers,
    vnum
  };

  /**
   * Granularity Scheme: number of patches per thread-block
   */
  enum class GranularityScheme
  {
    none,
    user_define,
    multiple
  };


  /**
   * @brief Implementation for Discontinuous Galerkin(DG) element
   *
   * @tparam dim
   * @tparam fe_degree
   * @tparam Number
   */
  template <int dim, int fe_degree, typename Number>
  class LevelVertexPatch : public Subscriptor
  {
  public:
    using CellIterator = typename DoFHandler<dim>::level_cell_iterator;
    using PatchIterator =
      typename std::vector<std::vector<CellIterator>>::const_iterator;

    static constexpr unsigned int regular_vpatch_size = 1 << dim;

    /**
     * Standardized data struct to pipe additional data to LevelVertexPatch.
     */
    struct AdditionalData
    {
      /**
       * Constructor.
       */
      AdditionalData(
        const Number            relaxation         = 1.,
        const bool              use_coloring       = true,
        const unsigned int      patch_per_block    = 1,
        const GranularityScheme granularity_scheme = GranularityScheme::none)
        : relaxation(relaxation)
        , use_coloring(use_coloring)
        , patch_per_block(patch_per_block)
        , granularity_scheme(granularity_scheme)
      {}

      /**
       * Relaxation parameter.
       */
      Number relaxation;

      /**
       * If true, use coloring. Otherwise, use atomic operations.
       * Coloring ensures bitwise reproducibility but is slower on Pascal and
       * newer architectures (need to check).
       */
      bool use_coloring;

      /**
       * Number of patches per thread block.
       */
      unsigned int patch_per_block;

      GranularityScheme granularity_scheme;
    };


    /**
     * Structure which is passed to the kernel.
     * It is used to pass all the necessary information from the CPU to the
     * GPU.
     */
    struct Data
    {
      /**
       * Number of patches for each color.
       */
      unsigned int n_patches;

      /**
       * Number of patches per thread block.
       */
      unsigned int patch_per_block;

      /**
       * Relaxation parameter.
       */
      Number relaxation;

      /**
       * Pointer to the the first degree of freedom in each patch.
       * @note Need Lexicographic ordering degree of freedoms.
       * @note For DG case, the first degree of freedom index of
       *       four cells in a patch is stored consecutively.
       */
      unsigned int *first_dof;

      /**
       * Pointer to the patch cell ordering type.
       */
      unsigned int *patch_id;

      /**
       * Pointer to the patch type. left, middle, right
       */
      unsigned int *patch_type;

      /**
       * Pointer to mapping from l to h
       */
      unsigned int *l_to_h;

      /**
       * Pointer to mapping from l to h
       */
      unsigned int *h_to_l;

      /**
       * Pointer to 1D mass matrix for lapalace operator.
       */
      Number *laplace_mass_1d;

      /**
       * Pointer to 1D stiffness matrix for lapalace operator.
       */
      Number *laplace_stiff_1d;

      /**
       * Pointer to 1D mass matrix for smoothing operator.
       */
      Number *smooth_mass_1d;

      /**
       * Pointer to 1D stiffness matrix for smoothing operator.
       */
      Number *smooth_stiff_1d;

      /**
       * Pointer to 1D eigenvalues for smoothing operator.
       */
      Number *eigenvalues;

      /**
       * Pointer to 1D eigenvectors for smoothing operator.
       */
      Number *eigenvectors;
    };

    /**
     * Default constructor.
     */
    LevelVertexPatch();

    /**
     * Destructor.
     */
    ~LevelVertexPatch();

    /**
     * Return the Data structure associated with @p color for lapalce operator.
     */
    Data
    get_laplace_data(unsigned int color) const;

    /**
     * Return the Data structure associated with @p color for smoothing operator.
     */
    Data
    get_smooth_data(unsigned int color) const;

    /**
     * Extracts the information needed to perform loops over cells.
     */
    void
    reinit(const DoFHandler<dim> &dof_handler,
           const unsigned int     mg_level,
           const AdditionalData  &additional_data = AdditionalData());

    /**
     * @brief This method runs the loop over all patches and apply the local operation on
     * each element in parallel.
     *
     * @tparam Operator a operator which is applied on each patch
     * @tparam VectorType
     * @param func
     * @param src
     * @param dst
     */
    template <typename Operator, typename VectorType>
    void
    patch_loop(const Operator   &op,
               const VectorType &src,
               VectorType       &dst) const;

    /**
     * This method runs the loop over all cells and compute tensor product on
     * each element in parallel for global mat-vec operation.
     * We call it 'cell loop' because although patches are used as a unit, only
     * some of the cells are updated, thus avoiding the duplication of face
     * integrals.
     */
    template <typename Operator, typename VectorType>
    void
    cell_loop(const Operator &op, const VectorType &src, VectorType &dst) const;

    /**
     * @brief Initializes the tensor product matrix for local smoothing.
     */
    void
    reinit_tensor_product_smoother() const;

    /**
     * @brief Initializes the tensor product matrix for local laplace.
     */
    void
    reinit_tensor_product_laplace() const;

    /**
     * Free all the memory allocated.
     */
    void
    free();

    /**
     * Return an approximation of the memory consumption of this class in
     * bytes.
     */
    std::size_t
    memory_consumption() const;

  private:
    /**
     * Helper function. Setup color arrays for collecting data.
     */
    void
    setup_color_arrays(const unsigned int n_colors);

    /**
     * Helper function. Setup color arrays for collecting data.
     */
    void
    setup_configuration(const unsigned int n_colors);

    /**
     * Helper function. Get tensor product data for each patch.
     */
    void
    get_patch_data(const PatchIterator &patch, const unsigned int patch_id);

    /**
     * Gathering the locally owned and ghost cells attached to a common
     * vertex as the collection of cell iterators (patch). The
     * successive distribution of the collections with ghost cells
     * follows the logic:
     *
     * 1.) if one mpi-proc owns more than half of the cells (locally owned)
     * of the vertex patch the mpi-proc takes ownership
     *
     * 2.) for the remaining ghost patches the mpi-proc with the cell of
     * the lowest CellId (see dealii::Triangulation) takes ownership
     */
    std::vector<std::vector<CellIterator>>
    gather_vertex_patches(const DoFHandler<dim> &dof_handler,
                          const unsigned int     level) const;

    /**
     * Allocate an array to the device.
     */
    template <typename Number1>
    void
    alloc_arrays(Number1 **array_device, const unsigned int n);

    /**
     * Number of global refinments.
     */
    unsigned int level;

    /**
     * Number of colors produced by the coloring algorithm.
     */
    unsigned int n_colors;

    /**
     * Relaxation parameter.
     */
    Number relaxation;

    /**
     * If true, use coloring. Otherwise, use atomic operations.
     * Coloring ensures bitwise reproducibility but is slower on Pascal and
     * newer architectures.
     */
    bool use_coloring;


    GranularityScheme granularity_scheme;

    /**
     * Grid dimensions associated to the different colors. The grid dimensions
     * are used to launch the CUDA kernels.
     */
    std::vector<dim3> grid_dim_lapalce;
    std::vector<dim3> grid_dim_smooth;

    /**
     * Block dimensions associated to the different colors. The block
     * dimensions are used to launch the CUDA kernels.
     */
    std::vector<dim3> block_dim_laplace;
    std::vector<dim3> block_dim_smooth;

    /**
     * Number of patches per thread block.
     */
    unsigned int patch_per_block;

    /**
     * Auxiliary vector.
     */
    mutable LinearAlgebra::distributed::Vector<Number, MemorySpace::CUDA> tmp;

    /**
     * Raw graphed of locally owned active patches.
     */
    std::vector<std::vector<PatchIterator>> graph_ptr_raw;

    /**
     * Colored graphed of locally owned active patches.
     */
    std::vector<std::vector<PatchIterator>> graph_ptr_colored;

    /**
     * Number of patches in each color.
     */
    std::vector<unsigned int> n_patches_laplace;
    std::vector<unsigned int> n_patches_smooth;

    /**
     * Pointer to the DoFHandler associated with the object.
     */
    const DoFHandler<dim> *dof_handler;

    /**
     * Vector of pointer to the the first degree of freedom
     * in each patch of each color.
     * @note Need Lexicographic ordering degree of freedoms.
     * @note For DG case, the first degree of freedom index of
     *       four cells in a patch is stored consecutively.
     */
    std::vector<unsigned int *> first_dof_laplace;
    std::vector<unsigned int *> first_dof_smooth;

    /**
     * Vector of the the first degree of freedom
     * in each patch of a single color.
     * Initialize on host and copy to device later.
     */
    std::vector<unsigned int> first_dof_host;

    /**
     * Vector of pointer to patch type: left, middle, right.
     */
    std::vector<unsigned int *> patch_type;

    /**
     * Vector of patch type: left, middle, right.
     * Initialize on host and copy to device later.
     */
    std::vector<unsigned int> patch_type_host;

    /**
     * Vector of pointer to the local cell ordering type for each patch.
     */
    std::vector<unsigned int *> patch_id;

    /**
     * Vector of patch cell ordering type.
     * Initialize on host and copy to device later.
     */
    std::vector<unsigned int> patch_id_host;

    /**
     * Mapping from cell ordering to type id.
     */
    std::unordered_map<unsigned int, unsigned int> ordering_to_type;

    /**
     * Categories of ordering types.
     */
    unsigned int ordering_types;

    /**
     * Lookup table
     */
    std::unordered_map<unsigned int, std::array<unsigned int, 3>> lookup_table;

    /**
     * Pointer to mapping from l to h
     */
    unsigned int *l_to_h;

    std::vector<unsigned int> l_to_h_host;

    /**
     * Pointer to mapping from l to h
     */
    unsigned int *h_to_l;

    std::vector<unsigned int> h_to_l_host;

    /**
     * Pointer to 1D mass matrix for lapalace operator.
     */
    Number *laplace_mass_1d;

    /**
     * Pointer to 1D stiffness matrix for lapalace operator.
     */
    Number *laplace_stiff_1d;

    /**
     * Pointer to 1D mass matrix for smoothing operator.
     */
    Number *smooth_mass_1d;

    /**
     * Pointer to 1D stiffness matrix for smoothing operator.
     */
    Number *smooth_stiff_1d;

    /**
     * Pointer to 1D eigenvalues for smoothing operator.
     */
    Number *eigenvalues;

    /**
     * Pointer to 1D eigenvectors for smoothing operator.
     */
    Number *eigenvectors;
  };

  // __constant__ unsigned int permutation_d[16 * 16 * 16];
  // __constant__ unsigned int permutation_f[16 * 16 * 16];

#ifdef USECONSTMEM
  __constant__ double mass_data_d[16 * 16 * 4];
  __constant__ double stiff_data_d[16 * 16 * 4];

  __constant__ float mass_data_f[16 * 16 * 4];
  __constant__ float stiff_data_f[16 * 16 * 4];

  template <typename NumberType>
  using DataArray = NumberType[16 * 16 * 4];

  template <typename Number>
  __host__ __device__ inline DataArray<Number>          &
  get_mass_data();

  template <>
  __host__ __device__ inline DataArray<double>          &
  get_mass_data<double>()
  {
    return mass_data_d;
  }

  template <>
  __host__ __device__ inline DataArray<float>          &
  get_mass_data<float>()
  {
    return mass_data_f;
  }

  template <typename Number>
  __host__ __device__ inline DataArray<Number>          &
  get_stiff_data();

  template <>
  __host__ __device__ inline DataArray<double>          &
  get_stiff_data<double>()
  {
    return stiff_data_d;
  }

  template <>
  __host__ __device__ inline DataArray<float>          &
  get_stiff_data<float>()
  {
    return stiff_data_f;
  }
#endif

#ifdef USETEXTURE
  texture<float, 1, cudaReadModeElementType> mass_data_d;

  texture<float, 1, cudaReadModeElementType> stiff_data_d;
#endif

  /**
   * Structure to pass the shared memory into a general user function.
   * TODO: specialize for cell loop and patch loop
   */
  template <int dim, typename Number, bool is_laplace>
  struct SharedMemData
  {
    /**
     * Constructor.
     */
    __device__
    SharedMemData(Number      *data,
                  unsigned int n_buff,
                  unsigned int n_dofs_1d,
                  unsigned int local_dim)
    {
      constexpr unsigned int n = is_laplace ? 3 : 1;

      if constexpr (n == 1 || std::is_same_v<Number, double> ||
                    (MMAKERNEL != 0 && MMAKERNEL != 7 && MMAKERNEL != 8))
        {
          local_src = data;
          local_dst = local_src + n_buff * local_dim;

          local_mass       = local_dst + n_buff * local_dim;
          local_derivative = local_mass + n_buff * n_dofs_1d * n_dofs_1d * n;
          tmp = local_derivative + n_buff * n_dofs_1d * n_dofs_1d * n;
        }
      else
        {
          local_src = data;
          local_dst = local_src + n_buff * local_dim;

          tmp = local_dst + n_buff * local_dim;

          mass_half = (half *)(tmp + n_buff * local_dim * 2);
          der_half  = mass_half + 2 * n_buff * n_dofs_1d * n_dofs_1d * n;
        }
    }


    /**
     * Shared memory for local and interior src.
     */
    Number *local_src;

    /**
     * Shared memory for local and interior dst.
     */
    Number *local_dst;

    /**
     * Shared memory for computed 1D mass matrix.
     */
    Number *local_mass;

    /**
     * Shared memory for computed 1D Laplace matrix.
     */
    Number *local_derivative;

    /**
     * Shared memory for internal buffer.
     */
    Number *tmp;

    half *mass_half;
    half *der_half;

#ifdef USECONSTMEM
    Number *const_mass;
    Number *const_stiff;
#endif
  };

  /**
   * This function determines number of patches per block at compile time.
   */
  template <int dim, int fe_degree>
  __host__ __device__ constexpr unsigned int
  granularity_shmem()
  {
    return dim == 2 ? (fe_degree == 1 ? 32 :
                       fe_degree == 2 ? 16 :
                       fe_degree == 3 ? 4 :
                       fe_degree == 4 ? 2 :
                                        1) :
           dim == 3 ? (fe_degree == 1 ? 16 :
                       fe_degree == 2 ? 2 :
                                        1) :
                      1;
  }

} // namespace PSMF

#include "patch_base.template.cuh"

/**
 * \page patch_base
 * \include patch_base.cuh
 */

#endif // PATCH_BASE_CUH
