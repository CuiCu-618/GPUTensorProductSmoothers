import sys

import numpy as np


def convert_to_Tbytes(value, unit):
    if unit == "Kbyte":
        return float(value) / 1024 / 1024 / 1024
    elif unit == "Mbyte":
        return float(value) / 1024 / 1024
    elif unit == "Gbyte":
        return float(value) / 1024
    else:
        return float(-1)


def convert_to_seconds(value, unit):
    if unit == "usecond":
        return float(value) / 1000000
    elif unit == "msecond":
        return float(value) / 1000
    else:
        return float(value)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py file_path")
        sys.exit(1)

    file_path = sys.argv[1]

    dram_bytes = []
    gpu_time_duration = []
    fma = []
    fp64 = []
    dmma = []
    hmma = []
    L2_hit = []
    L2_miss = []
    shared_ld = []
    shared_st = []
    L1_peak = []

    L2_flag = 0
    L1_flag = 0

    with open(file_path, "r") as file:
        for line in file:
            if "dram__bytes.sum" in line:
                parts = line.strip().split()
                value = parts[-1]
                unit = parts[-2]
                converted_value = convert_to_Tbytes(value, unit)
                if converted_value > 0:
                    dram_bytes.append(converted_value)
            elif "gpu__time_duration.sum" in line:
                parts = line.strip().split()
                value = parts[-1]
                unit = parts[-2]
                converted_value = convert_to_seconds(value, unit)
                gpu_time_duration.append(converted_value)
            elif "sm__pipe_fma_cycles_active" in line:
                parts = line.strip().split()
                value = parts[-1]
                fma.append(float(value) / 100)
            elif "sm__pipe_fp64_cycles_active" in line:
                parts = line.strip().split()
                value = parts[-1]
                fp64.append(float(value) / 100)
            elif "sm__pipe_tensor_op_dmma_cycles_active" in line:
                parts = line.strip().split()
                value = parts[-1]
                dmma.append(float(value) / 100)
            elif "sm__pipe_tensor_op_hmma_cycles_active" in line:
                parts = line.strip().split()
                value = parts[-1]
                hmma.append(float(value) / 100)
            elif "lts__t_sectors_lookup_hit.sum " in line:
                L2_flag = 1
                parts = line.strip().split()
                value = parts[-1]
                L2_hit.append(float(value))
            elif "lts__t_sectors_lookup_miss.sum " in line:
                parts = line.strip().split()
                value = parts[-1]
                L2_miss.append(float(value))
            elif "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum" in line:
                L1_flag = 1
                parts = line.strip().split()
                value = parts[-1]
                shared_ld.append(float(value))
            elif "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum" in line:
                parts = line.strip().split()
                value = parts[-1]
                shared_st.append(float(value))
            elif "l1tex__data_pipe_lsu_wavefronts.avg" in line:
                parts = line.strip().split()
                value = parts[-1]
                L1_peak.append(float(value) / 100)

    print("DRAM Bytes [TB]      :", dram_bytes)
    print("GPU Time Duration [s]:", gpu_time_duration)

    if L1_flag == 1:
        print("\nL1 peak [%]          :", L1_peak[0] * 100, L1_peak[1] * 100)
        print(
            "Shared peak [%]      :",
            (shared_ld[0] + shared_st[0]),
            (shared_ld[1] + shared_st[1]),
        )

    if L2_flag == 1:
        print(
            "L2 cache hit rate [%]:",
            L2_hit[0] / (L2_hit[0] + L2_miss[0]) * 100,
            L2_hit[1] / (L2_hit[1] + L2_miss[1]) * 100,
        )

    print("\nfma  :", fma)
    print("fp64 :", fp64)
    print("dmma :", dmma)
    print("hmma :", hmma)

    perf = []
    ai = []

    peak_fp64 = 8.81
    peak_fp32 = 17.62
    peak_dmma = 17.62
    peak_hmma = 281.92
    peak_tf32 = 140.96

    if "WMMA" in file_path:
        peak_tmp = peak_tf32
    else:
        peak_tmp = peak_hmma

    perf.append(
        fma[0] * peak_fp32
        + fp64[0] * peak_fp64
        + dmma[0] * peak_dmma
        + hmma[0] * peak_tmp
    )
    perf.append(
        fma[1] * peak_fp32
        + fp64[1] * peak_fp64
        + dmma[1] * peak_dmma
        + hmma[1] * peak_tmp
    )

    ai.append(perf[0] * gpu_time_duration[0] / dram_bytes[0])
    ai.append(perf[1] * gpu_time_duration[1] / dram_bytes[1])

    print("\nPerf [TFLOP/s]   :", perf)
    if L1_flag == 1:
        print("Sh Perf [TFLOP/s]:", perf[0] / L1_peak[0], perf[1] / L1_peak[1])

    print("\nAI [FLOP/byte]    :", ai)

    Bsh = 17.145
    if L1_flag == 1:
        sh_peak = (shared_ld[0] + shared_st[0]) / 100
        sh_peak1 = (shared_ld[1] + shared_st[1]) / 100
        print(
            "Sh AI [FLOP/byte] :",
            perf[0] / sh_peak / Bsh,
            perf[1] / sh_peak1 / Bsh,
            # perf[0] / L1_peak[0] / Bsh,
            # perf[1] / L1_peak[1] / Bsh,
        )
