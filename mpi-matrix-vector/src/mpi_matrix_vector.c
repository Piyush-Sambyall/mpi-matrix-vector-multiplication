#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define DEFAULT_N 20
#define PREVIEW_LIMIT 10

static void print_matrix_preview(const long long *matrix, int n) {
    int preview = (n > PREVIEW_LIMIT) ? PREVIEW_LIMIT : n;
    for (int i = 0; i < preview; i++) {
        for (int j = 0; j < preview; j++) {
            printf("%lld ", matrix[(long long)i * n + j]);
        }
        if (n > preview) {
            printf(" ...");
        }
        printf("\n");
    }
    if (n > preview) {
        printf("...\n");
    }
}

/* Print a truncated preview of a length-n vector. */
static void print_vector_preview(const long long *vec, int n) {
    int preview = (n > PREVIEW_LIMIT * 2) ? PREVIEW_LIMIT * 2 : n;
    for (int i = 0; i < preview; i++) {
        printf("%lld ", vec[i]);
    }
    if (n > preview) {
        printf("...");
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    int rank, num_procs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int n = DEFAULT_N;
    if (argc > 1) {
        n = atoi(argv[1]);
        if (n <= 0) {
            if (rank == 0) {
                fprintf(stderr, "Error: N must be a positive integer.\n");
            }
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    if (n % num_procs != 0) {
        if (rank == 0) {
            fprintf(stderr,
                    "Error: matrix size N (%d) must be evenly divisible "
                    "by the number of processes (%d).\n",
                    n, num_procs);
        }
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    int rows_per_proc = n / num_procs;

    long long *matrix = NULL;         /* full N x N matrix, root only   */
    long long *result = NULL;         /* full result vector, root only  */
    long long *vector = malloc((size_t)n * sizeof(long long));
    long long *local_matrix = malloc((size_t)rows_per_proc * n * sizeof(long long));
    long long *local_result = malloc((size_t)rows_per_proc * sizeof(long long));

    if (!vector || !local_matrix || !local_result) {
        fprintf(stderr, "Rank %d: memory allocation failed.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (rank == 0) {
        matrix = malloc((size_t)n * n * sizeof(long long));
        result = malloc((size_t)n * sizeof(long long));
        if (!matrix || !result) {
            fprintf(stderr, "Rank 0: memory allocation failed.\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        /* Fill the matrix sequentially: 1, 2, 3, ..., n*n (row-major). */
        for (long long i = 0; i < (long long)n * n; i++) {
            matrix[i] = i + 1;
        }
        /* Fill the vector: 1, 2, 3, ..., n. */
        for (int i = 0; i < n; i++) {
            vector[i] = i + 1;
        }

        printf("Matrix:\n");
        print_matrix_preview(matrix, n);
    }

    double start_time = MPI_Wtime();

    /* Every process needs the full vector to compute its rows. */
    MPI_Bcast(vector, n, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    /* Distribute the matrix rows evenly across all processes. */
    MPI_Scatter(matrix, rows_per_proc * n, MPI_LONG_LONG,
                local_matrix, rows_per_proc * n, MPI_LONG_LONG,
                0, MPI_COMM_WORLD);

    /* Each process computes the dot product for its block of rows. */
    for (int i = 0; i < rows_per_proc; i++) {
        long long sum = 0;
        for (int j = 0; j < n; j++) {
            sum += local_matrix[(long long)i * n + j] * vector[j];
        }
        local_result[i] = sum;
    }

    /* Collect all partial results back at the root. */
    MPI_Gather(local_result, rows_per_proc, MPI_LONG_LONG,
               result, rows_per_proc, MPI_LONG_LONG,
               0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (rank == 0) {
        printf("Vector: ");
        print_vector_preview(vector, n);

        printf("Matrix-vector multiplication complete.\n");

        printf("Result: ");
        print_vector_preview(result, n);

        printf("Execution Time: %f seconds\n", end_time - start_time);

        FILE *fp = fopen("result_matrix.txt", "w");
        if (fp) {
            for (int i = 0; i < n; i++) {
                fprintf(fp, "%lld\n", result[i]);
            }
            fclose(fp);
            printf("Saving result to 'result_matrix.txt' ...\n");
            printf("Matrix saved successfully!\n");
        } else {
            fprintf(stderr, "Warning: could not open result_matrix.txt for writing.\n");
        }

        free(matrix);
        free(result);
    }

    free(vector);
    free(local_matrix);
    free(local_result);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
