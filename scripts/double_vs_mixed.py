import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.ticker import FormatStrFormatter
from postprocess import read_data, extract_col
import os


files_d = []
files_f = []

for i in range(3, 8):
    files_d.append(f'/scratch/cucui/GPUTensorProductSmoothers/build_DG/TESTING/poisson_3D_DGQ{i}_ConflictFree_ConflictFree_GLOBAL_ConflictFree_ExactRes_multiple_double.log')
    files_f.append(f'/scratch/cucui/GPUTensorProductSmoothers/build_DG/TESTING_MIXED/poisson_3D_DGQ{i}_ConflictFree_ConflictFree_ConflictFree_ExactRes_multiple_mixed.log');

data_d = read_data(files_d)
data_f = read_data(files_f)

data_col = extract_col(data_d, 4, 3, 1)
data_Ax_d = np.array([data_p[-1] for data_p in data_col])

data_col = extract_col(data_f, 6, 2, 0)
data_Ax_f = np.array([data_p[-1] for data_p in data_col])

data_col = extract_col(data_d, 6, 3, 1)
data_S_d = np.array([data_p[-1] for data_p in data_col])

data_col = extract_col(data_f, 8, 2, 0)
data_S_f = np.array([data_p[-1] for data_p in data_col])

data_col = extract_col(data_d, 13, 3, 1)
data_gmres_d = np.array([data_p[-1] for data_p in data_col])

data_col = extract_col(data_f, 13, 2, 0)
data_gmres_f = np.array([data_p[-1] for data_p in data_col])

fe = np.arange(3, 8)

csfont = {'fontname':'Times New Roman', 'size': 18}


fig = plt.gcf()
ax = plt.gca()
fig.set_figheight(5)
fig.set_figwidth(6)
fig.set_dpi(300)

width = 0.25
width_b = 0.2

plt.bar(fe - width, data_Ax_f / data_Ax_d, width_b, label = "Ax")
plt.bar(fe, data_S_f / data_S_d, width_b, label = "Smoothing")
plt.bar(fe + width, data_gmres_d / data_gmres_f, width_b,label = "Gmres")

plt.title('Double precision versus Mixed precision',**csfont)
# plt.yscale('log')
plt.grid(linestyle='dashed', axis='y')
ax.set_axisbelow(True)
plt.xlabel('Polynomial degree',**csfont)
plt.ylabel('Relative Speedup',**csfont)
plt.xticks(fe)
plt.tick_params(labelsize=14)
plt.ylim(bottom=1)

ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

legend = plt.legend(loc='upper center', bbox_to_anchor=(0.45, -0.2),
            fancybox=True, frameon=True, shadow=False, ncol=3, prop={'family': 'Times New Roman', 'size': 16})
legend.get_frame().set_edgecolor('k') 

plt.subplots_adjust(bottom=0.3)
plt.subplots_adjust(left=0.2)

fig.savefig(f'/scratch/cucui/GPUTensorProductSmoothers/build_DG/Figures/double_mixed.png', dpi=fig.dpi)
fig.savefig(f'/scratch/cucui/GPUTensorProductSmoothers/build_DG/Figures/double_mixed.pdf', dpi=fig.dpi)