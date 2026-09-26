# MPI Matrix-Vector Multiplication

A parallel matrix-vector multiplication program written in C using **MPI
(Message Passing Interface)**. The matrix is split row-wise across
processes with `MPI_Scatter`, the vector is broadcast to every process
with `MPI_Bcast`, each process computes its own block of dot products in
parallel, and the partial results are collected back at the root process
with `MPI_Gather`. The program also times the parallel section with
`MPI_Wtime` and writes the full result vector to a file.

This project was built as part of a lab/seminar exercise on parallel and
distributed computing (MIET Jammu, CSE – AI & ML).

## Features

- Row-wise decomposition of an N × N matrix across `p` MPI processes
- Vector broadcast + partial dot-product computation + gather pattern
- Wall-clock timing of the parallel computation via `MPI_Wtime`
- Truncated console preview for large matrices/vectors (full result is
  always written to `result_matrix.txt`)
- Configurable matrix size via a command-line argument
- Heap-allocated buffers (no fixed-size stack arrays), which avoids the
  stack-overflow segfaults that fixed-size 2D arrays cause once N gets
  large
- `long long` accumulation, which avoids the silent 32-bit integer
  overflow that shows up as negative results for larger matrices

## Requirements

- An MPI implementation (tested with [MPICH](https://www.mpich.org/); Open
  MPI works the same way)
- A C compiler (via `mpicc`)

On Ubuntu/Debian:

```bash
sudo apt-get install mpich libmpich-dev
# or, for Open MPI instead:
# sudo apt-get install openmpi-bin libopenmpi-dev
```

## Build

```bash
mpicc src/mpi_matrix_vector.c -o mpi_exec -Wall
```

## Run

```bash
mpirun -np <num_processes> ./mpi_exec [N]
```

- `num_processes` — number of MPI ranks to launch
- `N` (optional) — size of the square matrix/vector (default: `20`)

**N must be evenly divisible by the number of processes** — the program
exits with a clear error message if it isn't, rather than crashing.

Examples:

```bash
mpirun -np 4 ./mpi_exec          # 20x20 matrix, 4 processes
mpirun -np 10 ./mpi_exec 100     # 100x100 matrix, 10 processes
mpirun -np 4 ./mpi_exec > output.txt   # redirect full output to a file
```

## Sample output

Running with `N = 20`:

```
Matrix:
1 2 3 4 5 6 7 8 9 10  ...
21 22 23 24 25 26 27 28 29 30  ...
41 42 43 44 45 46 47 48 49 50  ...
...
Vector: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
Matrix-vector multiplication complete.
Result: 2870 7070 11270 15470 19670 23870 28070 32270 36470 40670 44870 49070 53270 57470 61670 65870 70070 74270 78470 82670
Execution Time: 0.000013 seconds
Saving result to 'result_matrix.txt' ...
Matrix saved successfully!
```

The full console capture is in [`docs/sample_output.txt`](docs/sample_output.txt).
Timing numbers will vary by machine, process count, and matrix size —
larger matrices take noticeably longer and the speedup becomes more
visible as `N` grows.

## How it works

1. **Rank 0** allocates and fills the full `N x N` matrix (values `1`
   through `N*N`, row-major) and the length-`N` vector (values `1`
   through `N`).
2. `MPI_Bcast` sends a copy of the vector to every process.
3. `MPI_Scatter` splits the matrix into `N / p` contiguous rows per
   process.
4. Each process computes the dot product of its local rows with the
   vector.
5. `MPI_Gather` collects every process's partial results back into the
   full result vector on rank 0.
6. Rank 0 prints a preview, the elapsed time, and writes the complete
   result to `result_matrix.txt`.

## Project structure

```
mpi-matrix-vector/
├── src/
│   └── mpi_matrix_vector.c   # main MPI program
├── docs/
│   └── sample_output.txt     # example run captured to a file
├── README.md
├── LICENSE
└── .gitignore
```

## Notes / lessons learned

- **Segfaults on larger N**: caused by declaring the matrix as a
  fixed-size stack array (`int matrix[N][N]`). Fixed by switching to
  `malloc`-based heap allocation for all matrix/vector buffers.
- **Wrong (negative) results on larger N**: caused by 32-bit `int`
  overflow in the row sums. Fixed by using `long long` for matrix,
  vector, and accumulator values.
- **`N` not divisible by process count**: instead of letting
  `MPI_Scatter` silently misbehave, the program checks this up front
  and exits with a clear error message.

## License

Released under the [MIT License](LICENSE).

## Author

**Piyush Sambyal**
B.Tech CSE (AI & ML), Model Institute of Engineering and Technology, Jammu
