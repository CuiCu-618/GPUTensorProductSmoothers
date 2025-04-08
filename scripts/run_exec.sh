#!/bin/bash                                                                                                                           

cd ..
make release
cd -


for dofs in 500000
do
for p in 3 4 5 6 7
do
    echo "/////////////////////"
    echo "Starting 3D degree $p"
    echo "/////////////////////"
    python3 ../scripts/ct_parameter.py -DIM 3 -DEG $p -MAXSIZE $dofs \
            -LA ConflictFree -SMV ConflictFree -SMI ConflictFree -VNUM double -REP 1
    cd ..
    make benchmark_mg 
    cd -
    echo "/////////////////////"
    echo "Running 3D degree=$p Dof=$dofs"
    echo "/////////////////////"
    # ../apps/benchmark_mg -device=2
    # ncu -f -o Ax_DP_Q$p -k regex:laplace_kernel -c 2 --set full --import-source yes ../apps/benchmark_mg -device=2
    ncu -k regex:laplace_kernel -s 1 -c 1 \
      --metrics gpc__cycles_elapsed.avg.per_second,\
gpu__time_duration.sum,\
dram__bytes.sum,\
dram__bytes.sum.per_second,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_dmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
lts__t_sectors_lookup_hit.sum,\
lts__t_sectors_lookup_miss.sum,\
l1tex__t_sectors_lookup_hit.sum,\
l1tex__t_sectors_lookup_miss.sum,\
launch__registers_per_thread,\
launch__grid_size,\
launch__block_size,\
launch__shared_mem_per_block_dynamic,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__sass_data_bytes_mem_shared.sum,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__sass_thread_inst_executed_op_dadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_dmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_dfma_pred_on.sum,\
smsp__sass_inst_executed_op_local_ld.sum,\
smsp__sass_inst_executed_op_local_st.sum,\
sm__sass_inst_executed_op_shared_ld.sum,\
sm__sass_inst_executed_op_shared_st.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum.pct_of_peak_sustained_elapsed,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum.pct_of_peak_sustained_elapsed,\
l1tex__data_pipe_lsu_wavefronts.avg.pct_of_peak_sustained_elapsed \
        ../apps/benchmark_mg -device=2 > tmp_Q${p} # trans_rest_Q${p} 
done
done


# ncu -k regex:laplace_kernel --metrics gpu__cycles_elapsed.avg.per_second,gpu__time_duration.sum,sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed  ../apps/benchmark_mg -device=2
