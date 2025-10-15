#!/usr/bin/env python3
"""
Analyze nsys profiling results to quantify TR overhead
"""
import sqlite3
import sys

def analyze_trace(sqlite_path, name):
    """Analyze a single trace file"""
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    
    # Get total GPU time
    cursor.execute("""
        SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL
    """)
    total_gpu_time = cursor.fetchone()[0] / 1e6  # Convert to ms
    
    # Get kernel count
    cursor.execute("""
        SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL
    """)
    kernel_count = cursor.fetchone()[0]
    
    # Get TR-specific kernels (topk, gather, reduce)
    cursor.execute("""
        SELECT 
            shortName,
            COUNT(*) as count,
            SUM(end - start) / 1e6 as total_time_ms,
            AVG(end - start) / 1e3 as avg_time_us
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE shortName LIKE '%gatherTopK%' 
           OR shortName LIKE '%bitonicSort%'
           OR shortName LIKE '%reduce_kernel%'
           OR shortName LIKE '%scatter_gather%'
        GROUP BY shortName
    """)
    tr_kernels = cursor.fetchall()
    
    # Get largest kernels by time
    cursor.execute("""
        SELECT 
            shortName,
            COUNT(*) as count,
            SUM(end - start) / 1e6 as total_time_ms,
            AVG(end - start) / 1e3 as avg_time_us,
            SUM(end - start) * 100.0 / (SELECT SUM(end - start) FROM CUPTI_ACTIVITY_KIND_KERNEL) as pct
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        GROUP BY shortName
        ORDER BY total_time_ms DESC
        LIMIT 10
    """)
    top_kernels = cursor.fetchall()
    
    conn.close()
    
    return {
        'name': name,
        'total_gpu_time_ms': total_gpu_time,
        'kernel_count': kernel_count,
        'tr_kernels': tr_kernels,
        'top_kernels': top_kernels
    }

def main():
    traces = [
        ('profiling_results/nsys_traces/quick_224_bs1_baseline.sqlite', 'is=224 bs=1 baseline'),
        ('profiling_results/nsys_traces/quick_224_bs1_with_tr.sqlite', 'is=224 bs=1 WITH TR'),
        ('profiling_results/nsys_traces/quick_448_bs8_baseline.sqlite', 'is=448 bs=8 baseline'),
        ('profiling_results/nsys_traces/quick_448_bs8_with_tr.sqlite', 'is=448 bs=8 WITH TR'),
    ]
    
    results = []
    for path, name in traces:
        try:
            result = analyze_trace(path, name)
            results.append(result)
        except Exception as e:
            print(f"Error analyzing {name}: {e}")
    
    # Print summary
    print("=" * 100)
    print("NSYS PROFILING RESULTS: TR OVERHEAD QUANTIFICATION")
    print("=" * 100)
    print()
    
    print("-" * 100)
    print("SUMMARY: Total GPU Time and Kernel Counts")
    print("-" * 100)
    print(f"{'Configuration':<25} {'Total GPU Time (ms)':<20} {'Kernel Count':<15} {'Avg per Kernel (μs)':<20}")
    print("-" * 100)
    
    for r in results:
        avg_per_kernel = (r['total_gpu_time_ms'] * 1000) / r['kernel_count']
        print(f"{r['name']:<25} {r['total_gpu_time_ms']:>18.2f}  {r['kernel_count']:>13,}  {avg_per_kernel:>18.2f}")
    
    print()
    print("-" * 100)
    print("COMPARISON: Baseline vs TR")
    print("-" * 100)
    
    # is=224 comparison
    if len(results) >= 2:
        base_224 = results[0]
        tr_224 = results[1]
        overhead_ms = tr_224['total_gpu_time_ms'] - base_224['total_gpu_time_ms']
        overhead_pct = (overhead_ms / base_224['total_gpu_time_ms']) * 100
        kernel_increase = tr_224['kernel_count'] - base_224['kernel_count']
        
        print(f"\nis=224 bs=1:")
        print(f"  Baseline GPU time: {base_224['total_gpu_time_ms']:.2f} ms")
        print(f"  With TR GPU time:  {tr_224['total_gpu_time_ms']:.2f} ms")
        print(f"  Overhead:          {overhead_ms:+.2f} ms ({overhead_pct:+.1f}%)")
        print(f"  Kernel count:      {base_224['kernel_count']:,} → {tr_224['kernel_count']:,} ({kernel_increase:+,})")
    
    # is=448 comparison
    if len(results) >= 4:
        base_448 = results[2]
        tr_448 = results[3]
        speedup_ms = base_448['total_gpu_time_ms'] - tr_448['total_gpu_time_ms']
        speedup_pct = (speedup_ms / base_448['total_gpu_time_ms']) * 100
        kernel_increase = tr_448['kernel_count'] - base_448['kernel_count']
        
        print(f"\nis=448 bs=8:")
        print(f"  Baseline GPU time: {base_448['total_gpu_time_ms']:.2f} ms")
        print(f"  With TR GPU time:  {tr_448['total_gpu_time_ms']:.2f} ms")
        print(f"  Speedup:           {speedup_ms:-.2f} ms ({speedup_pct:-.1f}%)")
        print(f"  Kernel count:      {base_448['kernel_count']:,} → {tr_448['kernel_count']:,} ({kernel_increase:+,})")
    
    # TR-specific kernels
    print()
    print("-" * 100)
    print("TR-SPECIFIC KERNELS (gatherTopK, bitonicSort, reduce_kernel, scatter_gather)")
    print("-" * 100)
    
    for r in results:
        if 'WITH TR' in r['name']:
            print(f"\n{r['name']}:")
            if r['tr_kernels']:
                print(f"  {'Kernel':<60} {'Count':<10} {'Total (ms)':<12} {'Avg (μs)':<12}")
                print(f"  {'-'*60} {'-'*10} {'-'*12} {'-'*12}")
                total_tr_time = 0
                for kernel, count, total_ms, avg_us in r['tr_kernels']:
                    kernel_short = kernel[:60]
                    print(f"  {kernel_short:<60} {count:<10,} {total_ms:<12.3f} {avg_us:<12.2f}")
                    total_tr_time += total_ms
                print(f"  {'':<60} {'':<10} {'-'*12}")
                print(f"  {'TOTAL TR OVERHEAD':<60} {'':<10} {total_tr_time:<12.3f} ({total_tr_time/r['total_gpu_time_ms']*100:.2f}%)")
            else:
                print("  No TR-specific kernels detected (pattern matching may need adjustment)")
    
    print()
    print("-" * 100)
    print("TOP 10 KERNELS BY TIME")
    print("-" * 100)
    
    for r in results:
        print(f"\n{r['name']}:")
        print(f"  {'Kernel':<60} {'Count':<10} {'Total (ms)':<12} {'% of Total':<12}")
        print(f"  {'-'*60} {'-'*10} {'-'*12} {'-'*12}")
        for row in r['top_kernels']:
            if len(row) >= 5:
                kernel, count, total_ms, avg_us, pct = row
                kernel_str = str(kernel) if kernel is not None else 'Unknown'
                kernel_short = kernel_str[:60]
                print(f"  {kernel_short:<60} {count:<10,} {total_ms:<12.2f} {pct:<12.1f}")
    
    print()
    print("=" * 100)

if __name__ == '__main__':
    main()
