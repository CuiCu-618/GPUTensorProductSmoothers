#!/bin/bash                                                                                                                           

cd ..
make release
cd -

# DOF=16777216
# DOF=134217728
# for DOF in 4096 32768 262144 2097152 16777216 134217728
for DOF in 134217728
do
for p in 3 
do
    echo "/////////////////////"
    echo "Starting 3D degree=$p Dof=$DOF"
    echo "/////////////////////"
    python3 ../scripts/ct_parameter.py -DIM 3 -DEG $p -MAXSIZE $DOF -REDUCE 1e-8 -MAXIT 20 \
        -LA ConflictFree -SMV ConflictFree -SMI GLOBAL -VNUM double -SETS error_analysis -G none
    cd ..
    make benchmark_mg 
    cd -
    echo "/////////////////////"
    echo "Running 3D degree=$p Dof=$DOF"
    echo "/////////////////////"
    ../apps/benchmark_mg -device=2
    # ncu -k regex:laplace_kernel \
    #     --metrics gpc__cycles_elapsed.avg.per_second,gpu__time_duration.sum,dram__bytes.sum,dram__bytes.sum.per_second,dram__bytes_read.sum,dram__bytes_write.sum,sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    #     ../apps/benchmark_mg -device=2 > CC_CF_K1_E0_Q${p}_${DOF}
done
done


# ncu -k regex:laplace_kernel --metrics gpu__cycles_elapsed.avg.per_second,gpu__time_duration.sum,sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed  ../apps/benchmark_mg -device=2
