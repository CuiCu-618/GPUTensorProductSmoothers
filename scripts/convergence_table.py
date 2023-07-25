import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from postprocess import read_data, write_convergence_table

files_2d_d = []

for i in range(3, 11):
    files_2d_d.append(f'/scratch/cucui/GPUTensorProductSmoothers/build_DG/TESTING/poisson_2D_DGQ{i}_ConflictFree_ConflictFree_GLOBAL_ConflictFree_ExactRes_multiple_double.log')

data_2d_d = read_data(files_2d_d)

# print(data_2d_d[0])

write_convergence_table("/scratch/cucui/GPUTensorProductSmoothers/build_DG/TESTING/table_2d_global", data_2d_d, 3, "global", "frac", 3)
write_convergence_table("/scratch/cucui/GPUTensorProductSmoothers/build_DG/TESTING/table_2d_cf", data_2d_d, 3, "cf", "frac", 3)
write_convergence_table("/scratch/cucui/GPUTensorProductSmoothers/build_DG/TESTING/table_2d_exactres", data_2d_d, 3, "exactres", "frac", 3)


files_3d_d = []

for i in range(3, 8):
    files_3d_d.append(f'/scratch/cucui/GPUTensorProductSmoothers/build_DG/TESTING/poisson_3D_DGQ{i}_ConflictFree_ConflictFree_GLOBAL_ConflictFree_ExactRes_multiple_double.log')

data_3d_d = read_data(files_3d_d)

write_convergence_table("/scratch/cucui/GPUTensorProductSmoothers/build_DG/TESTING/table_3d_global", data_3d_d, 3, "global", "frac", 3)
write_convergence_table("/scratch/cucui/GPUTensorProductSmoothers/build_DG/TESTING/table_3d_cf", data_3d_d, 3, "cf", "frac", 3)
write_convergence_table("/scratch/cucui/GPUTensorProductSmoothers/build_DG/TESTING/table_3d_exactres", data_3d_d, 3, "exactres", "frac", 3)

print(data_3d_d[0].shape)
print(data_3d_d[0][:, 4])