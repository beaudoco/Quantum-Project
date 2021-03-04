
#include <stdio.h>   // for printf
#include <stdlib.h>  // for malloc
#include <complex.h> // for double complex
#include <string.h>  // for memcpy()
#include <unistd.h>  // for sleep()
#include <omp.h>     // for openMP
#include <math.h>    // for pow()
#include <stdbool.h> // for bool
// #include <time.h>    // for time testing

typedef struct Qureg
{

    int rank;

    int numQubits;
    long long int numAmpsTotal;
    long long int numAmpsPerRank;

    double complex *stateVector;
    double complex *bufferVector;
} Qureg;

Qureg createQureg(int numQubits)
{

    Qureg qureg;

    qureg.numQubits = numQubits;
    qureg.numAmpsTotal = 1LL << numQubits; // 2^numQubits
    qureg.numAmpsPerRank = qureg.numAmpsTotal;

    qureg.stateVector = calloc(qureg.numAmpsPerRank, sizeof *qureg.stateVector);
    qureg.bufferVector = calloc(qureg.numAmpsPerRank, sizeof *qureg.bufferVector);

    return qureg;
}

void initRandomQureg(Qureg qureg)
{

    double norm = 0;

    // generate random unnormalised complex amplitudes
    for (int i = 0; i < qureg.numAmpsPerRank; i++)
    {

        // all nodes generate every random number, to keep seed consistent
        double complex n = (rand() / (double)RAND_MAX) + 1i * (rand() / (double)RAND_MAX);

        // but only the nominated node adopts the value
        qureg.stateVector[i] = n;

        // conveniently, all nodes know/agree on the norm
        norm += pow(cabs(n), 2);
    }

    // normalise the amplitudes
    for (int i = 0; i < qureg.numAmpsPerRank; i++)
        qureg.stateVector[i] /= sqrt(norm);
}

void printQureg(Qureg qureg)
{

    for (int i = 0; i < qureg.numAmpsPerRank; i++)
    {
        double complex amp = qureg.stateVector[i];
        int ind = i;
        printf("qureg[%d] = %g + (%g)i\n", ind, creal(amp), cimag(amp));
    }
}

int main()
{
    //SET THREADS ALLOWED IN PROGRAM
    omp_set_num_threads(2);

    srand(12345);

    // create and print a random qureg
    int numQubits = 14;
    Qureg qureg = createQureg(numQubits);
    initRandomQureg(qureg);

    // MAKE THIS HERMITIAN USING SOMETHING LIKE
    Qureg *dens = calloc(1LL << numQubits, sizeof qureg);
    for (long long int i = 0; i < qureg.numAmpsTotal; i++)
    {

        dens[i] = createQureg(numQubits);
        initRandomQureg(dens[i]);
    }

    //SETUP TIMER FOR FILE
    // struct timespec begin, end;
    // clock_gettime(CLOCK_REALTIME, &begin);
    // int i, j;

// #pragma omp parallel for private(i)
    for (i = 0; i < qureg.numAmpsTotal; i++)
    {

#pragma omp parallel for private(j)
        for (j = 0; j < qureg.numAmpsTotal; j++)
        {
#pragma omp critical
            qureg.bufferVector[j] += qureg.stateVector[i] * dens[i].stateVector[j];
        }
    }

    //END CLOCK AND GET TIME
    // clock_gettime(CLOCK_REALTIME, &end);
    // long seconds = end.tv_sec - begin.tv_sec;
    // long nanoseconds = end.tv_nsec - begin.tv_nsec;
    // double elapsed = seconds + nanoseconds * 1e-9;

    // printf("time taken for CPU: %f\n", elapsed);

    size_t numBytes = qureg.numAmpsTotal * sizeof qureg.stateVector * sizeof *qureg.stateVector;
    memcpy(qureg.stateVector, qureg.bufferVector, numBytes);

    // sleep(1);
    // printQureg(qureg);
    // sleep(1);

    return 0;
}