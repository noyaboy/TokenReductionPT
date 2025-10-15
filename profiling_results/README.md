# Profiling Investigation: Token Reduction Performance Paradox

## Problem Statement

Token Reduction (TR) shows contradictory performance characteristics depending on batch size and input resolution:

### Small Batch Case (bs=1, is=224):
- **Without TR**: 169.7 img/s, 5.89ms latency, 17.57 GFLOPs
- **With TR (keep_rate=0.1)**: 140.9 img/s, 7.09ms latency, 5.55 GFLOPs
- ❌ **Result**: 68% FLOPs reduction but 17% SLOWER

### Large Batch Case (bs=8, is=448):
- **Without TR**: 108.7 img/s, 73.62ms latency, 78.52 GFLOPs
- **With TR (keep_rate=0.1)**: 321.3 img/s, 24.9ms latency, 24.73 GFLOPs
- ✅ **Result**: 69% FLOPs reduction and 3x FASTER

## Investigation Goal

Understand why TR overhead dominates in small batch scenarios despite reducing computational workload.

## Profiling Plan

This directory contains scripts and results for systematic profiling analysis.

### Phase 1: Timeline Analysis (nsys)
Capture system-wide execution traces to identify:
- Kernel launch overhead
- CPU-GPU synchronization points
- Memory transfers
- Total kernel count

### Phase 2: Kernel Analysis (ncu)
Deep dive into specific kernels to measure:
- Compute vs memory bound ratios
- SM occupancy and warp utilization
- Achieved vs theoretical bandwidth
- Overhead of TR operations (topk, gather, indexing)

### Phase 3: Root Cause & Recommendations
- Identify the crossover point (batch size where TR becomes beneficial)
- Quantify overhead sources
- Propose optimization strategies

## Requirements

- CUDA 11.4+ compatible PyTorch installation
- NVIDIA Nsight Systems (nsys)
- NVIDIA Nsight Compute (ncu)
- GPU with at least 11GB VRAM

## Scripts

All profiling scripts are in this directory. Run them on a CUDA-enabled machine.
