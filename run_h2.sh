mpicc -std=c99 hermitian_2.c -o hermitian_2 -lm
mpirun -np 4 hermitian_2
