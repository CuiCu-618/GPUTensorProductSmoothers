#include <mma.h>

#include <iomanip>
#include <iostream>
using namespace nvcuda;

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
test_wmmaload()
{
  __shared__ double buf[64];

  int tid       = threadIdx.x;
  int warpid    = tid / 32;
  buf[tid]      = 100.1 + tid;
  buf[tid + 32] = 100.1 + tid + 32;

  for (uint32_t i = tid, c = 0; i < 2048; i += 256, c++)
    printf("%d: %d\n", c, (warpid + c * 8 + 1) * 32 % 2048);

  __syncthreads();

  // double reg[2];
  // unsigned smem_ptr =
  //   static_cast<unsigned>(__cvta_generic_to_shared(buf));

  // asm volatile("wmma.load.c.sync.aligned.row.m8n8k4.shared.f64 {%0,%1},
  // [%2];"
  //              : "=d"(reg[0]), "=d"(reg[1])
  //              : "r"(smem_ptr));

  // __syncthreads();
  // printf("[%d, %f, %f, %f]\n", tid, reg[0], reg[1], buf[tid]);
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



__global__ void
test_mmatf32(float *result)
{
  // {
  //   __shared__ float shmem[128];

  //   if (threadIdx.x == 0)
  //     shmem[0] = 10001.925;
  //   __syncthreads();

  //   float fa = wmma::__float_to_tf32(shmem[0]);
  //   half  ha = shmem[0];

  //   if (threadIdx.x == 0)
  //     {
  //       printf("ld tf %.10f, %.10f, %.10f\n",
  //              shmem[0],
  //              wmma::__float_to_tf32(fa),
  //              __half2float(ha));
  //       printf("ld tf %.10f \n", (((shmem[0] - fa) * (1 << 11))));
  //     }
  // }

  float a[4];
  float b[2];
  float c[4];

  a[0] = 11.1;
  a[1] = 11.1;
  a[2] = 11.1;
  a[3] = 11.1;

  b[0] = 22.2;
  b[1] = 22.2;

  c[0] = 0;
  c[1] = 0;
  c[2] = 0;
  c[3] = 0;

  constexpr int scale = 1 << 0;

  float da[4];
  float db[2];

  for (int i = 0; i < 4; ++i)
    {
      da[i] = (a[i] - wmma::__float_to_tf32(a[i])) * scale;
    }
  for (int i = 0; i < 2; ++i)
    {
      db[i] = (b[i] - wmma::__float_to_tf32(b[i])) * scale;
    }

  uint32_t const *pA = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const *pB = reinterpret_cast<uint32_t const *>(&b);

  uint32_t const *dpA = reinterpret_cast<uint32_t const *>(&da);
  uint32_t const *dpB = reinterpret_cast<uint32_t const *>(&db);

  float buf[4];
  buf[0] = 0;
  buf[1] = 0;
  buf[2] = 0;
  buf[3] = 0;

  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
               : "r"(dpA[0]),
                 "r"(dpA[1]),
                 "r"(dpA[2]),
                 "r"(dpA[3]),
                 "r"(dpB[0]),
                 "r"(dpB[1]),
                 "f"(buf[0]),
                 "f"(buf[1]),
                 "f"(buf[2]),
                 "f"(buf[3]));

  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
               : "r"(dpA[0]),
                 "r"(dpA[1]),
                 "r"(dpA[2]),
                 "r"(dpA[3]),
                 "r"(pB[0]),
                 "r"(pB[1]),
                 "f"(buf[0]),
                 "f"(buf[1]),
                 "f"(buf[2]),
                 "f"(buf[3]));
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
               : "r"(pA[0]),
                 "r"(pA[1]),
                 "r"(pA[2]),
                 "r"(pA[3]),
                 "r"(dpB[0]),
                 "r"(dpB[1]),
                 "f"(buf[0]),
                 "f"(buf[1]),
                 "f"(buf[2]),
                 "f"(buf[3]));

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
  for (int i = 0; i < 4; ++i)
    {
      c[i] += buf[i] / scale;
    }
  __syncthreads();

  if (threadIdx.x == 0)
    {
      printf(" %.10f, %.10f, %.10f\n", buf[0], da[0], db[0]);
    }

  result[threadIdx.x] = c[0];
}

__global__ void
test_mmaf16(float *result)
{
  // __shared__ float shmem[128];
  // __shared__ half  shmemhf[128];

  // if (threadIdx.x == 0)
  //   for (int i = 0; i < 128; ++i)
  //     {
  //       shmem[i]   = 1.1;
  //       shmemhf[i] = i + 1.1;
  //     }
  // __syncthreads();

  // half A[8];

  // float *A_ptr = reinterpret_cast<float *>(&A);

  // auto smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&shmem[0]));
  // auto smem_ptrhf =
  //   static_cast<uint32_t>(__cvta_generic_to_shared(&shmemhf[threadIdx.x *
  //   8]));

  // asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
  //              "{%0, %1, %2, %3}, [%4]; "
  //              : "=f"(A_ptr[0]), "=f"(A_ptr[1]), "=f"(A_ptr[2]),
  //              "=f"(A_ptr[3]) : "r"(smem_ptrhf));
  // __syncthreads();
  // if (threadIdx.x == 0)
  //   {
  //     printf("ld %.10f, %.10f, %.10f\n",
  //            shmem[0],
  //            __half2float(A[0]),
  //            __half2float(A[1]));
  //   }

  half  a[8];
  half  b[4];
  float c[4];

  a[0] = 1.1;
  a[1] = 1.1;
  a[2] = 1.1;
  a[3] = 1.1;
  a[4] = 1.1;
  a[5] = 1.1;
  a[6] = 1.1;
  a[7] = 1.1;

  b[0] = 2.2;
  b[1] = 2.2;
  b[2] = 2.2;
  b[3] = 2.2;

  c[0] = 0;
  c[1] = 0;
  c[2] = 0;
  c[3] = 0;

  constexpr int scale = 1 << 11;

  half da[8];
  half db[4];

  for (int i = 0; i < 8; ++i)
    {
      a[i]     = 11.1;
      float ha = 11.1;
      da[i]    = __float2half((ha - __half2float(a[i])) * scale);
    }
  for (int i = 0; i < 4; ++i)
    {
      b[i]     = 22.2;
      float hb = 22.2;
      db[i]    = __float2half((hb - __half2float(b[i])) * scale);
    }

  uint32_t const *pA = reinterpret_cast<uint32_t const *>(&a);
  uint32_t const *pB = reinterpret_cast<uint32_t const *>(&b);

  uint32_t const *dpA = reinterpret_cast<uint32_t const *>(&da);
  uint32_t const *dpB = reinterpret_cast<uint32_t const *>(&db);

  float buf[4];
  buf[0] = 0;
  buf[1] = 0;
  buf[2] = 0;
  buf[3] = 0;

  // asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
  //              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
  //              : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
  //              : "r"(dpA[0]),
  //                "r"(dpA[1]),
  //                "r"(dpA[2]),
  //                "r"(dpA[3]),
  //                "r"(dpB[0]),
  //                "r"(dpB[1]),
  //                "f"(buf[0]),
  //                "f"(buf[1]),
  //                "f"(buf[2]),
  //                "f"(buf[3]));

  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
               : "r"(dpA[0]),
                 "r"(dpA[1]),
                 "r"(dpA[2]),
                 "r"(dpA[3]),
                 "r"(pB[0]),
                 "r"(pB[1]),
                 "f"(buf[0]),
                 "f"(buf[1]),
                 "f"(buf[2]),
                 "f"(buf[3]));
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(buf[0]), "=f"(buf[1]), "=f"(buf[2]), "=f"(buf[3])
               : "r"(pA[0]),
                 "r"(pA[1]),
                 "r"(pA[2]),
                 "r"(pA[3]),
                 "r"(dpB[0]),
                 "r"(dpB[1]),
                 "f"(buf[0]),
                 "f"(buf[1]),
                 "f"(buf[2]),
                 "f"(buf[3]));

  for (int i = 0; i < 4; ++i)
    {
      c[i] += buf[i] / scale;
    }

  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
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


  __syncthreads();

  if (threadIdx.x == 0)
    printf("%f\n", c[0]);

  result[threadIdx.x] = c[0];
}

int
main()
{
  // global_func<<<1, 66>>>();
  // test_wmmaload<<<1, 256>>>();
  // test_ldmatrix<<<1, 32>>>();
  // test_wideload<<<1, 32>>>();
  // test_shfl<<<1, 32>>>();
  // test_mma<<<1, 32>>>();
  float *res = (float *)malloc(128 * sizeof(float));

  float *result;
  cudaMalloc(&result, 128 * sizeof(float));

  // test_mmatf32<<<1, 32>>>(result);
  test_mmaf16<<<1, 32>>>(result);

  cudaDeviceSynchronize();

  cudaMemcpy(res, result, 128 * sizeof(float), cudaMemcpyDeviceToHost);

  for (int i = 0; i < 32; ++i)
    std::cout << std::fixed << std::setprecision(10) << res[i] << " ";
  std::cout << std::endl;
  return 0;
}
