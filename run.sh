mpicc -std=c99 demo.c -o demo -lm
mpirun -np 2 demo
