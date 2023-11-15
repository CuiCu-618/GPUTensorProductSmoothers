import sys

def parse_timing_info(line):
    """Parse timing info from a line."""
    words = line.split()
    return int(words[-2])  # Assuming the timing info value is always at the second-to-last position

def compute_averages(file_path):
    timings = {'loading': 0, 'Vmult': 0, 'Storing': 0}
    counts = {'loading': 0, 'Vmult': 0, 'Storing': 0}

    max_loading = 0
    max_Vmult = 0
    max_Storing = 0

    with open(file_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        if 'loading' in line:
            data = parse_timing_info(line)
            max_loading = max(max_loading, data)
            timings['loading'] += data
            counts['loading'] += 1
        elif 'Vmult' in line:
            data = parse_timing_info(line)
            max_Vmult = max(max_Vmult, data)
            timings['Vmult'] += data
            counts['Vmult'] += 1
        elif 'Storing' in line:
            data = parse_timing_info(line)
            max_Storing = max(max_Storing, data)
            timings['Storing'] += data
            counts['Storing'] += 1

    # Compute averages
    avg_loading = timings['loading'] / counts['loading'] if counts['loading'] > 0 else 0
    avg_Vmult = timings['Vmult'] / counts['Vmult'] if counts['Vmult'] > 0 else 0
    avg_Storing = timings['Storing'] / counts['Storing'] if counts['Storing'] > 0 else 0
    total = avg_loading + avg_Vmult + avg_Storing
    max_total = max_loading + max_Vmult + max_Storing

    # Print results
    print(f"Average loading timing: {avg_loading:.2f} cycles")
    print(f"Average Vmult timing: {avg_Vmult:.2f} cycles")
    print(f"Average Storing timing: {avg_Storing:.2f} cycles")
    print(f"Average Total timing: {total:.2f} cycles")

    print(f"Max loading timing: {max_loading:.2f} cycles")
    print(f"Max Vmult timing: {max_Vmult:.2f} cycles")
    print(f"Max Storing timing: {max_Storing:.2f} cycles")
    print(f"Max Total timing: {max_total:.2f} cycles")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    compute_averages(file_path)
