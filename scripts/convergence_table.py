import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from postprocess import read_data, write_convergence_table

# files_2d_d = []
#
# for i in range(2, 11):
#     files_2d_d.append(f'/scratch/cucui/GPUTensorProductSmoothers/build_biharm/TEST_ALL/biharm_2D_Q{i}_Basic_Basic_GLOBAL_Exact_Bila_KSVD_multiple_double_1s.log')
#
# data_2d_d = read_data(files_2d_d)
#
# # print(data_2d_d[0].shape)
#
# write_convergence_table("/scratch/cucui/GPUTensorProductSmoothers/build_biharm/TEST_ALL/table_2d_exact", data_2d_d, 2, "exact", "frac", 3)
# write_convergence_table("/scratch/cucui/GPUTensorProductSmoothers/build_biharm/TEST_ALL/table_2d_bila", data_2d_d, 2, "bila", "frac", 3)
# write_convergence_table("/scratch/cucui/GPUTensorProductSmoothers/build_biharm/TEST_ALL/table_2d_ksvd", data_2d_d, 2, "ksvd", "frac", 3)
#
#
#
# for i in range(2, 7):
#     files_3d_d.append(f'/scratch/cucui/GPUTensorProductSmoothers/build_biharm/TEST_ALL/biharm_3D_Q{i}_ConflictFree_ConflictFree_GLOBAL_Exact_multiple_double_1s.log')
#
# data_3d_d = read_data(files_3d_d)
#
# write_convergence_table("/scratch/cucui/GPUTensorProductSmoothers/build_biharm/TEST_ALL/table_3d_exact", data_3d_d, 2, "exact", "frac_3d", 1)

files_3d_d = []
for i in range(2, 7):
    files_3d_d.append(
        f"/scratch/cucui/GPUTensorProductSmoothers/build_biharm/damping/biharm_3D_Q{i}_ConflictFree_ConflictFree_GLOBAL_Bila_multiple_double_1s_damp0.7.log"
    )

data_3d_d = read_data(files_3d_d)

write_convergence_table(
    "/scratch/cucui/GPUTensorProductSmoothers/build_biharm/damping/table_3d_cf",
    data_3d_d,
    2,
    "exact",
    "frac_3d",
    1,
)
