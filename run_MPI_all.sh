mpicc -std=c99 hermitian_MPI_All.c -o hermitian_MPI_All -lm -Wall
mpirun -np 2 hermitian_MPI_All
