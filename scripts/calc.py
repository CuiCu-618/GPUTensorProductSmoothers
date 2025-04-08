#!/usr/bin/env python3
import argparse
import glob
import os
import re

import pandas as pd


def extract_metric_value(lines, metric):
    """
    Search for lines containing the metric,
    extract the numeric value at the end (removing commas),
    and convert to float.
    Returns None if not found.
    """
    for line in lines:
        if metric in line:
            # Directly match the last field
            m = re.search(r"([\d,\.]+)$", line.strip())
            if m:
                value_str = m.group(1).replace(",", "")
                try:
                    return float(value_str)
                except ValueError:
                    return None
    return None


def extract_metric_value_with_unit(lines, metric):
    """
    For metric lines with units, assuming format:
       <metric_name> <unit> <value>
    Returns (unit, value); if not found returns (None, None)
    """
    for line in lines:
        if metric in line:
            fields = line.strip().split()
            if len(fields) >= 3:
                # Assume the second column is the unit, and the last column is the value
                unit = fields[1]
                try:
                    value = float(fields[-1].replace(",", ""))
                except ValueError:
                    value = None
                return unit, value
    return None, None


def process_file(filename, pattern):
    """
    Process a single file:
      - Extract fe from filename
      - Read file content, extract required metrics
      - Choose FLOPS calculation based on whether filename contains SP
      - Calculate byte/DoF, FLOP/DoF, FLOP/Byte
    Returns a dictionary with calculation results; returns None if metrics extraction fails.
    """
    # Extract fe from filename, e.g.: tmp_4 => fe = 4
    basename = os.path.basename(filename)
    m = re.match(rf"{pattern}(\d+)", basename)
    if not m:
        print(
            f"Filename {filename} does not match format '{pattern}${{fe}}', skipping."
        )
        return None
    fe = int(m.group(1))

    # Check if using single precision
    use_sp = "SP" in basename

    with open(filename, "r") as f:
        lines = f.readlines()

    # Extract dram__bytes.sum metric and its unit
    dram_unit, dram_val = extract_metric_value_with_unit(lines, "dram__bytes.sum")
    if dram_val is None:
        print(
            f"Failed to extract dram__bytes.sum metric from file {filename}, skipping."
        )
        return None

    # Convert units to bytes, supports Gbyte, Mbyte, Kbyte, byte
    unit_conversion = {"Gbyte": 1024**3, "Mbyte": 1024**2, "Kbyte": 1024, "byte": 1}
    factor = unit_conversion.get(dram_unit, 1)
    dram_bytes = dram_val * factor

    # Extract other metrics
    time_unit, time_duration = extract_metric_value_with_unit(
        lines, "gpu__time_duration.sum"
    )
    unit_conversion = {"usecond": 1e-6, "msecond": 1e-3, "second": 1}
    factor = unit_conversion.get(time_unit, 1)
    time_duration *= factor

    shared_ld = extract_metric_value(
        lines, "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum"
    )
    shared_st = extract_metric_value(
        lines, "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"
    )
    shared_bytes = (shared_ld + shared_st) * 17.125 / 100 * time_duration * 1024**4

    l2_hit = extract_metric_value(lines, "lts__t_sectors_lookup_hit.sum")
    l2_miss = extract_metric_value(lines, "lts__t_sectors_lookup_miss.sum")
    l2_hitrate = l2_hit / (l2_hit + l2_miss)

    grid_size_val = extract_metric_value(lines, "launch__grid_size")
    dadd_val = extract_metric_value(
        lines, "sm__sass_thread_inst_executed_op_dadd_pred_on.sum"
    )
    dfma_val = extract_metric_value(
        lines, "sm__sass_thread_inst_executed_op_dfma_pred_on.sum"
    )
    dmul_val = extract_metric_value(
        lines, "sm__sass_thread_inst_executed_op_dmul_pred_on.sum"
    )
    fadd_val = extract_metric_value(
        lines, "sm__sass_thread_inst_executed_op_fadd_pred_on.sum"
    )
    ffma_val = extract_metric_value(
        lines, "sm__sass_thread_inst_executed_op_ffma_pred_on.sum"
    )
    fmul_val = extract_metric_value(
        lines, "sm__sass_thread_inst_executed_op_fmul_pred_on.sum"
    )
    n_dofs = extract_metric_value(lines, "Number of degrees of freedom:")

    if None in (grid_size_val, dadd_val, dfma_val, dmul_val):
        print(f"Failed to extract some metrics from file {filename}, skipping.")
        return None

    # Calculate results
    dofs = (2 * fe + 2) ** 3  # Example: fe=4 => (2*4+2)^3 = 10^3 = 1000
    block = int(grid_size_val)
    # FLOP count: flops = dmul + dadd + 2 * dfma
    flops = dmul_val + dadd_val + dfma_val * 2
    flops_sp = fmul_val + fadd_val + ffma_val * 2

    # Select FLOPS calculation method based on filename
    if use_sp:
        active_flops = flops_sp
        flop_type = "SP"
    else:
        active_flops = flops
        flop_type = "DP"

    byte_dof = dram_bytes / block / dofs
    flop_dof = active_flops / block / dofs
    flop_byte = flop_dof / byte_dof
    byte_dof_shared = shared_bytes / block / dofs
    flop_byte_shared = flop_dof / byte_dof_shared

    tflops_per_second = active_flops / time_duration / 1e12
    gdofs_per_second = n_dofs / time_duration / 1e9

    if "smooth" in basename:
        gdofs_per_second = gdofs_per_second / 8

    base_result = {
        "file": filename,
        "fe": fe,
        "precision": flop_type,
        "dofs": dofs,
        "block": block,
        "flops": int(active_flops),
        "dram_bytes": int(dram_bytes),
        "shared_bytes": int(shared_bytes),
        "L2 hit": l2_hitrate,
        "byte/DoF": byte_dof,
        "byte_shared/DoF": byte_dof_shared,
        "FLOP/DoF": flop_dof,
        "FLOP/Byte": flop_byte,
        "FLOP/Byte_shared": flop_byte_shared,
        "TFLOPS/s": tflops_per_second,
        "GDoF/s": gdofs_per_second,
    }

    # Add counters based on precision type
    if use_sp:
        base_result.update(
            {
                "fadd": int(fadd_val),
                "ffma": int(ffma_val),
                "fmul": int(fmul_val),
            }
        )
    else:
        base_result.update(
            {
                "dadd": int(dadd_val),
                "dfma": int(dfma_val),
                "dmul": int(dmul_val),
            }
        )

    return base_result


def main():
    parser = argparse.ArgumentParser(description="Process GPU performance metrics")
    parser.add_argument(
        "--pattern",
        type=str,
        default="smooth_Q",
        help="Filename pattern, e.g.: smooth_Q or Ax_Q",
    )
    args = parser.parse_args()

    # Find all files matching the pattern
    files = glob.glob(f"{args.pattern}*")
    if not files:
        print(
            f"No files matching pattern '{args.pattern}${{fe}}' found in current directory."
        )
        return

    results = []
    for filename in files:
        print(f"Processing file: {filename}")
        res = process_file(filename, args.pattern)
        if res is not None:
            results.append(res)

    if not results:
        print("No files processed successfully.")
        return

    # Create DataFrame, sort by fe, and save to CSV file
    df = pd.DataFrame(results)
    df = df.sort_values("fe")  # Sort by fe in ascending order
    output_file = f"result_{args.pattern.rstrip('_')}.csv"
    df.to_csv(output_file, index=False)
    print("Calculation results:")
    print(df)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()
