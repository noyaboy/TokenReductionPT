#!/usr/bin/env python3
import sqlite3

# Get kernels from baseline
conn_base = sqlite3.connect('profiling_results/nsys_traces/quick_224_bs1_baseline.sqlite')
cursor_base = conn_base.cursor()
cursor_base.execute("SELECT DISTINCT demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL")
baseline_kernels = set(row[0] for row in cursor_base.fetchall() if row[0])
conn_base.close()

# Get kernels from TR
conn_tr = sqlite3.connect('profiling_results/nsys_traces/quick_224_bs1_with_tr.sqlite')
cursor_tr = conn_tr.cursor()
cursor_tr.execute("SELECT DISTINCT demangledName FROM CUPTI_ACTIVITY_KIND_KERNEL")
tr_kernels = set(row[0] for row in cursor_tr.fetchall() if row[0])

# Find TR-specific kernels
tr_only = tr_kernels - baseline_kernels

print("TR-SPECIFIC KERNELS (present in TR but not baseline):")
print("=" * 80)
for kernel in sorted(tr_only):
    # Get stats for this kernel
    cursor_tr.execute("""
        SELECT COUNT(*), SUM(end-start)/1e6, AVG(end-start)/1e3
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE demangledName = ?
    """, (kernel,))
    count, total_ms, avg_us = cursor_tr.fetchone()
    print(f"\n{kernel[:80]}")
    print(f"  Count: {count:,}, Total: {total_ms:.3f} ms, Avg: {avg_us:.2f} μs")

conn_tr.close()
