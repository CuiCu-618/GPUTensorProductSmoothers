import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from postprocess import read_data, extract_component
import os


csfont = {'fontname':'Times New Roman', 'size': 18}
# plt.subplot(121)
# plt.subplots_adjust(wspace = 0.3)

fig = plt.gcf()
ax = plt.gca()
fig.set_figheight(5)
fig.set_figwidth(6)
fig.set_dpi(300)


files_d = []

for i in range(2, 11):
    files_d.append(f'/scratch/cucui/GPUTensorProductSmoothers/build_biharm/TEST_ALL/biharm_2D_Q{i}_Basic_Basic_GLOBAL_Exact_Bila_KSVD_multiple_double_1s.log')
data_d = read_data(files_d)

scale = 1e-6

data_0 = extract_component(data_d, "Gmres", 3, 0)
data_0_L = np.array([data_p[-1] for data_p in data_0]) / scale

data_1 = extract_component(data_d, "Gmres", 3, 1)
data_1_L = np.array([data_p[-1] for data_p in data_1]) / scale

data_2 = extract_component(data_d, "Gmres", 3, 2)
data_2_L = np.array([data_p[-1] for data_p in data_2]) / scale

fe = range(2, 11)

print(data_d)

plt.plot(fe, data_0_L, 'go-', label = "Exact")
plt.plot(fe, data_1_L, 'ro-', label = "Bila")
plt.plot(fe, data_2_L, 'bo-', label = "KSVD")



plt.title('2D',**csfont)
# plt.yscale('log')
plt.grid(linestyle='dashed')
plt.xlabel('Polynomial degree',**csfont)
plt.ylabel('s / Million DoF',**csfont)
plt.xticks(fe)
plt.tick_params(labelsize=14)
plt.ylim(bottom=0)

# plt.subplot(122)


# files_d = []

# for i in range(3, 8):
#     files_d.append(f'/scratch/cucui/GPUTensorProductSmoothers/build_DG/TESTING/poisson_3D_DGQ{i}_ConflictFree_ConflictFree_GLOBAL_ConflictFree_ExactRes_multiple_double.log')

# data_d = read_data(files_d)

# scale = 1e-9

# data_0 = extract_component(data_d, "Gmres", 3, 0)
# data_0_L = np.array([data_p[-1] for data_p in data_0]) / scale

# data_1 = extract_component(data_d, "Gmres", 3, 1)
# data_1_L = np.array([data_p[-1] for data_p in data_1]) / scale

# data_2 = extract_component(data_d, "Gmres", 3, 2)
# data_2_L = np.array([data_p[-1] for data_p in data_2]) / scale

# fe = range(3, 8)


# plt.plot(fe, data_0_L, 'go-', label = "Global")
# plt.plot(fe, data_1_L, 'ro-', label = "Conflict Free")
# plt.plot(fe, data_2_L, 'bo-', label = "Exact Residual")


# plt.title('3D',**csfont)
# # plt.yscale('log')
# plt.grid(linestyle='dashed')
# plt.xlabel('Polynomial degree',**csfont)
# plt.ylabel('s / Billion DoF',**csfont)
# plt.xticks(fe)
# plt.tick_params(labelsize=14)
# plt.ylim(bottom=0)

legend = plt.legend(loc='upper left',
            fancybox=True, frameon=True, shadow=False, ncol=1, prop={'family': 'Times New Roman', 'size': 16})
legend.get_frame().set_edgecolor('k') 

plt.subplots_adjust(bottom=0.2)
plt.subplots_adjust(left=0.2)

fig.savefig(f'/scratch/cucui/GPUTensorProductSmoothers/build_biharm/Figures/TimeSolution_2D.png', dpi=fig.dpi)
fig.savefig(f'/scratch/cucui/GPUTensorProductSmoothers/build_biharm/Figures/TimeSolution_2D.pdf', dpi=fig.dpi)