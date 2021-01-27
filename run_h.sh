mpicc -std=c99 hermitian_1.c -o hermitian_1 -lm
mpirun -np 2 hermitian_1
