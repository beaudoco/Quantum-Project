mpicc -std=c99 hermitian_MPI.c -o hermitian_MPI -lm
mpirun -np 2 hermitian_MPI
