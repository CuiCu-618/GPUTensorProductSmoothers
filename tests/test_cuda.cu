#include <iostream>

__device__ void
device_func()
{
  printf("%d ", threadIdx.x);
  __syncthreads();

  if (threadIdx.x == 0)
    printf("\n\n");

  if (threadIdx.x < 45)
    {
      printf("%d ", threadIdx.x);
      __syncthreads();
    }
}

__global__ void
global_func()
{
  printf("%d ", threadIdx.x);
  __syncthreads();
  if (threadIdx.x == 0)
    printf("\n\n");

  if (threadIdx.x < 55)
    {
      printf("%d ", threadIdx.x);
      __syncthreads();

      if (threadIdx.x == 0)
        printf("\n\n");
      device_func();
    }
}

__global__ void
test_ldmatrix()
{
  __shared__ float buf[32];

  int tid = threadIdx.x;

  buf[tid] = 100.1 + tid;
  __syncthreads();

  uint32_t reg;
  unsigned smem_ptr =
    static_cast<unsigned>(__cvta_generic_to_shared(buf + tid * 4));

  asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];"
               : "=r"(reg)
               : "r"(smem_ptr));

  __syncthreads();
  printf("[%d, %d, %.15f]\n", tid, reg, buf[tid]);
}

__global__ void
test_wideload()
{
  __shared__ double buf[32];

  int tid = threadIdx.x;

  buf[tid] = 100.1 + tid;
  __syncthreads();

  double2 &reg = *((double2 *)buf + tid);
  // double2  a   = {0, 0};

  __syncthreads();
  printf("[%d, %f, %f]\n", tid, reg.x, reg.y);
}

__global__ void
test_shfl()
{
  __shared__ double buf[64];

  int tid = threadIdx.x;

  const unsigned int row = tid / 4;
  const unsigned int col = tid & 3;

  buf[tid]      = 100.1 + tid;
  buf[tid + 32] = 100.1 + tid + 32;
  __syncthreads();

  double2 &reg = *((double2 *)(buf + (col + row / 4 * 4) * 8 + 2 * (row & 3)));

  // reg.y = __shfl_sync(0xffffffff, reg.x, 0);
  double a;

  a = __shfl_sync(0xffffffff, reg.x, 0);

  __syncthreads();
  printf("[%d, %f, %f, %f]\n", tid, reg.x, reg.y, a);
}

__global__ void
test_mma()
{
  __shared__ float A[16 * 16];
  __shared__ float B[16 * 16];
  __shared__ float C[16 * 16];


  int tid = threadIdx.x;

  const unsigned int row = tid / 4;
  const unsigned int col = tid & 3;

  if (tid < 16)
    for (int i = 0; i < 16; ++i)
      {
        A[i * 16 + tid] = tid + i * 1.5;
        B[i * 16 + tid] = tid + i * 2.5;
        C[i * 16 + tid] = 0;
      }
  __syncthreads();

  if (tid == 0)
    {
      for (int i = 0; i < 16; ++i)
        {
          for (int j = 0; j < 16; ++j)
            printf("%.2f, ", A[i * 16 + j]);
          printf("\n");
        }
      printf("\n");
      for (int i = 0; i < 16; ++i)
        {
          for (int j = 0; j < 16; ++j)
            printf("%.2f, ", B[i * 16 + j]);
          printf("\n");
        }
    }
  float a[4];
  float b[2];
  float c[4];

  int c_idx = row * 16 + 2 * col;

  c[0] = C[c_idx];
  c[1] = C[c_idx + 1];
  c[2] = C[c_idx + 8 * 16];
  c[3] = C[c_idx + 1 + 8 * 16];

  for (int cycle = 0; cycle < 2; ++cycle)
    {
      int a_idx = row * 16 + col + cycle * 8;
      int b_idx = col * 16 + row + cycle * 8 * 16;

      a[0] = A[a_idx];
      a[1] = A[a_idx + 8 * 16];
      a[2] = A[a_idx + 4];
      a[3] = A[a_idx + 4 + 8 * 16];

      b[0] = B[b_idx];
      b[1] = B[b_idx + 4 * 16];

      if (tid == 0)
        printf("A: %f, %f, %f, %f; B: %f, %f; C: %f, %f, %f, %f\n",
               a[0],
               a[1],
               a[2],
               a[3],
               b[0],
               b[1],
               c[0],
               c[1],
               c[2],
               c[3]);
      __syncthreads();

      uint32_t const *pA = reinterpret_cast<uint32_t const *>(&a);
      uint32_t const *pB = reinterpret_cast<uint32_t const *>(&b);

      asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                   "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                   : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                   : "r"(pA[0]),
                     "r"(pA[1]),
                     "r"(pA[2]),
                     "r"(pA[3]),
                     "r"(pB[0]),
                     "r"(pB[1]),
                     "f"(c[0]),
                     "f"(c[1]),
                     "f"(c[2]),
                     "f"(c[3]));
    }

  C[c_idx]              = c[0];
  C[c_idx + 1]          = c[1];
  C[c_idx + 8 * 16]     = c[2];
  C[c_idx + 1 + 8 * 16] = c[3];

  if (tid == 0)
    printf("A: %f, %f, %f, %f; B: %f, %f; C: %f, %f, %f, %f\n",
           a[0],
           a[1],
           a[2],
           a[3],
           b[0],
           b[1],
           c[0],
           c[1],
           c[2],
           c[3]);

  if (tid == 0)
    for (int i = 0; i < 16; ++i)
      {
        for (int j = 0; j < 16; ++j)
          printf("%.2f, ", C[i * 16 + j]);
        printf("\n");
      }
}

int
main()
{
  // global_func<<<1, 66>>>();
  // test_ldmatrix<<<1, 32>>>();
  // test_wideload<<<1, 32>>>();
  // test_shfl<<<1, 32>>>();
  test_mma<<<1, 32>>>();

  cudaDeviceSynchronize();

  std::cout << std::endl;
  return 0;
}
