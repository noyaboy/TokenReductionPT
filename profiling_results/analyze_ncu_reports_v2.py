#!/usr/bin/env python3
"""
Extract key metrics from NCU profiling reports (long format CSV)
"""
import subprocess
import csv
import sys
from collections import defaultdict

def get_ncu_csv(report_path):
    """Extract CSV data from NCU report"""
    cmd = ['ncu', '--import', report_path, '--csv']
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

def analyze_report(report_path, name):
    """Analyze a single NCU report"""
    print(f"\n{'='*100}")
    print(f"Configuration: {name}")
    print(f"{'='*100}\n")

    csv_data = get_ncu_csv(report_path)
    lines = csv_data.strip().split('\n')

    if len(lines) < 2:
        print("No data found")
        return

    # Parse CSV (long format: each row is a metric for a kernel)
    reader = csv.DictReader(lines)

    # Group by kernel
    kernels = defaultdict(lambda: {'metrics': {}, 'time': None})

    for row in reader:
        kernel_id = row['ID']
        kernel_name = row['Kernel Name']
        metric_name = row.get('Metric Name', '')
        metric_value = row.get('Metric Value', '')
        section_name = row.get('Section Name', '')

        # Store kernel info
        if kernel_name:
            kernels[kernel_id]['name'] = kernel_name
            kernels[kernel_id]['time'] = row.get('Kernel Time', '')

        # Store metric
        if metric_name and metric_value:
            full_metric_name = f"{section_name}/{metric_name}" if section_name else metric_name
            kernels[kernel_id]['metrics'][full_metric_name] = metric_value

    print(f"Total unique kernels profiled: {len(kernels)}\n")

    # Aggregate key metrics
    key_metrics_to_track = [
        ('SM [%]', 'GPU Speed Of Light/SM [%]'),
        ('Memory [%]', 'GPU Speed Of Light/Memory [%]'),
        ('Compute (SM) [%]', 'GPU Speed Of Light/Compute (SM) [%]'),
        ('L1/TEX Hit Rate', 'Memory Workload Analysis/L1/TEX Hit Rate'),
        ('L2 Hit Rate', 'Memory Workload Analysis/L2 Hit Rate'),
        ('Achieved Occupancy', 'Occupancy/Achieved Occupancy'),
    ]

    metrics_values = defaultdict(list)
    kernel_times = []

    for kid, kdata in kernels.items():
        kernel_name = kdata.get('name', 'unknown')

        # Track metrics
        for label, metric_key in key_metrics_to_track:
            if metric_key in kdata['metrics']:
                try:
                    # Remove commas and % signs
                    val_str = kdata['metrics'][metric_key].replace(',', '').replace('%', '')
                    val = float(val_str)
                    metrics_values[label].append(val)
                except (ValueError, AttributeError):
                    pass

    # Print average metrics
    print("Average Metrics Across All Kernels:")
    print("-" * 100)

    for label, _ in key_metrics_to_track:
        if label in metrics_values and metrics_values[label]:
            avg = sum(metrics_values[label]) / len(metrics_values[label])
            unit = '%' if any(x in label for x in ['[%]', 'Rate', 'Occupancy']) else ''
            print(f"  {label:<40} {avg:>10.2f} {unit}")
        else:
            print(f"  {label:<40} {'N/A':>10}")

    # Find top kernels by execution count
    kernel_counts = defaultdict(int)
    for kid, kdata in kernels.items():
        kernel_counts[kdata.get('name', 'unknown')] += 1

    sorted_kernels = sorted(kernel_counts.items(), key=lambda x: x[1], reverse=True)

    print("\nTop 10 Most Frequently Executed Kernels:")
    print("-" * 100)
    for i, (kernel, count) in enumerate(sorted_kernels[:10], 1):
        kernel_short = kernel[:80]
        print(f"  {i:2d}. [{count:3d}x] {kernel_short}")

def main():
    configs = [
        ('profiling_results/ncu_reports/ncu_224_bs1_baseline.ncu-rep', 'is=224 bs=1 BASELINE'),
        ('profiling_results/ncu_reports/ncu_224_bs1_with_tr.ncu-rep', 'is=224 bs=1 WITH TR'),
        ('profiling_results/ncu_reports/ncu_448_bs8_with_tr.ncu-rep', 'is=448 bs=8 WITH TR'),
        ('profiling_results/ncu_reports/ncu_448_bs4_with_tr.ncu-rep', 'is=448 bs=4 WITH TR (ANOMALY)'),
        ('profiling_results/ncu_reports/ncu_448_bs1_with_tr.ncu-rep', 'is=448 bs=1 WITH TR (BEST)'),
    ]

    for report_path, name in configs:
        try:
            analyze_report(report_path, name)
        except Exception as e:
            print(f"Error analyzing {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print("="*100)

if __name__ == '__main__':
    main()
