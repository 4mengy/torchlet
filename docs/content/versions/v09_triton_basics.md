# v09_triton_basics

## What This Version Introduces

This planned version introduces Triton with a small GQA-adjacent kernel before replacing the full attention path.

The goal is to teach Triton execution structure without mixing in the full complexity of paged attention.

## Why Introduce It

Jumping directly from PyTorch paged GQA to a full Triton paged attention kernel would hide too many new ideas at once: program IDs, block sizes, masks, pointer arithmetic, compile-time constants, and launch grids.

## Core Principle

A Triton kernel is written as a program that operates on a tile of data. Many programs run in parallel, and each program computes offsets from its program IDs.

The teaching target is to make these ideas visible:

- Program IDs map work to tiles.
- Masks protect boundary loads and stores.
- Block sizes are explicit.
- Kernel launch shape becomes part of the design.

## Files To Compare

- The new Triton kernel file once implemented.
- The PyTorch reference function it is checked against.
- Any test or smoke script that compares outputs.

## Teaching Tradeoff

This version should stay intentionally small. Its job is confidence-building, not performance heroics.
