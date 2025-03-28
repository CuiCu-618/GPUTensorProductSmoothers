/**
 * Created by Cu Cui on 2022/4/15.
 */

// test coloring patches with mpi
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <nvToolsExt.h>

#include <fstream>
#include <iostream>

const uint32_t colors[]   = {0x0000ff00,
                             0x000000ff,
                             0x00ffff00,
                             0x00ff00ff,
                             0x0000ffff,
                             0x00ff0000,
                             0x00ffffff};
const int      num_colors = sizeof(colors) / sizeof(uint32_t);

#define PUSH_RANGE(name, cid)                                          \
  {                                                                    \
    int color_id                      = cid;                           \
    color_id                          = color_id % num_colors;         \
    nvtxEventAttributes_t eventAttrib = {0};                           \
    eventAttrib.version               = NVTX_VERSION;                  \
    eventAttrib.size                  = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
    eventAttrib.colorType             = NVTX_COLOR_ARGB;               \
    eventAttrib.color                 = colors[color_id];              \
    eventAttrib.messageType           = NVTX_MESSAGE_TYPE_ASCII;       \
    eventAttrib.message.ascii         = name;                          \
    nvtxRangePushEx(&eventAttrib);                                     \
  }
#define POP_RANGE nvtxRangePop();


using namespace dealii;

__global__ void
write_ghost(double *vec, int idx)
{
  vec[idx] += 6000000;
}

int
main(int argc, char *argv[])
{
  try
    {
      Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

      {
        int         n_devices       = 0;
        cudaError_t cuda_error_code = cudaGetDeviceCount(&n_devices);
        AssertCuda(cuda_error_code);
        const unsigned int my_mpi_id =
          Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
        const int device_id = my_mpi_id % n_devices;
        cuda_error_code     = cudaSetDevice(device_id);
        AssertCuda(cuda_error_code);
      }


      int world_rank;
      MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

      int         send_data, recv_data;
      MPI_Request request[2];
      MPI_Status  status[2];

      if (world_rank == 0)
        {
          send_data = 123;

          MPI_Isend(&send_data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &request[0]);

          MPI_Irecv(&recv_data, 1, MPI_INT, 1, 0, MPI_COMM_WORLD, &request[1]);

          printf("Rank 0 is doing computation...\n");

          MPI_Waitall(2, request, status);

          printf("Rank 0 received data from Rank 1: %d\n", recv_data);
        }
      else if (world_rank == 1)
        {
          send_data = 456;

          MPI_Isend(&send_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &request[0]);

          MPI_Irecv(&recv_data, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &request[1]);

          printf("Rank 1 is doing computation...\n");

          MPI_Waitall(2, request, status);

          printf("Rank 1 received data from Rank 0: %d\n", recv_data);
        }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
