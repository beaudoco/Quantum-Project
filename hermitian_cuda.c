
#include <stdio.h>   // for printf
#include <stdlib.h>  // for malloc
#include <complex.h> // for double complex
#include <string.h>  // for memcpy()
#include <unistd.h>  // for sleep()
#include <math.h>    // for pow()
// #include <mpi.h>     // for MPI
#include <stdbool.h> // for bool

extern void matrixMultiplication(double *quregReal, double *quregImg, double *qureg2Real, double *qureg2Img, long long arraySize);

typedef struct Qureg
{

    int rank;
    // int numRanks;

    int numQubits;
    long long int numAmpsTotal;
    long long int numAmpsPerRank;

    double complex *stateVector;
    double complex *bufferVector;
} Qureg;

Qureg createQureg(int numQubits)
{

    Qureg qureg;
    // MPI_Comm_rank(MPI_COMM_WORLD, &qureg.rank);
    // MPI_Comm_size(MPI_COMM_WORLD, &qureg.numRanks);

    qureg.numQubits = numQubits;
    qureg.numAmpsTotal = 1LL << numQubits; // 2^numQubits
    // qureg.numAmpsPerRank = qureg.numAmpsTotal / qureg.numRanks;
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

    // setup MPI, seed all nodes the same
    // MPI_Init(NULL, NULL);
    srand(12345);

    // create and print a random qureg
    int numQubits = 3;
    Qureg qureg = createQureg(numQubits);
    initRandomQureg(qureg);
    double *tmpDensMatReal;
    double *tmpDensMatImg;
    double *tmpStateVectReal;
    double *tmpStateVectImg;

    // MAKE THIS HERMITIAN USING SOMETHING LIKE
    Qureg *dens = calloc(1LL << numQubits, sizeof qureg);
    for (long long int i = 0; i < qureg.numAmpsTotal; i++)
    {

        dens[i] = createQureg(numQubits);
        initRandomQureg(dens[i]);
    }

    tmpDensMatReal = calloc(qureg.numAmpsPerRank * qureg.numAmpsPerRank, sizeof *qureg.stateVector);
    tmpDensMatImg = calloc(qureg.numAmpsPerRank * qureg.numAmpsPerRank, sizeof *qureg.stateVector);
    tmpStateVectReal = calloc(qureg.numAmpsPerRank, sizeof *qureg.stateVector);
    tmpStateVectImg = calloc(qureg.numAmpsPerRank, sizeof *qureg.stateVector);

    for (int i = 0; i < qureg.numAmpsTotal; i++)
    {

        for (int j = 0; j < qureg.numAmpsTotal; j++)
        {

            tmpDensMatReal[j + i * qureg.numAmpsTotal] = creal(dens[i].stateVector[j]);
            tmpDensMatImg[j + i * qureg.numAmpsTotal] = cimag(dens[i].stateVector[j]);
        }
        tmpStateVectReal[i] = creal(qureg.stateVector[i]);
        tmpStateVectImg[i] = cimag(qureg.stateVector[i]);
    }

    // HERE WE ARE GOING TO ATTEMPT TO MATRIX MULTIPLY
    matrixMultiplication(tmpStateVectReal, tmpStateVectImg, tmpDensMatReal, tmpDensMatImg, qureg.numAmpsTotal);

    for (int i = 0; i < qureg.numAmpsTotal; i++)
    {
        int ind = i;
        printf("qureg[%d] = %g + (%g)i \n", ind, tmpStateVectReal[i], tmpStateVectImg[i]);
    }

    // sleep(1);
    // printQureg(qureg);
    // sleep(1);

    for (int i = 0; i < qureg.numAmpsTotal; i++)
    {
        qureg.bufferVector[i] = 0;
        for (int j = 0; j < qureg.numAmpsTotal; j++)
        {
            qureg.bufferVector[i] += qureg.stateVector[j] * dens[j].stateVector[i];
        }
    }

    size_t numBytes = qureg.numAmpsTotal * sizeof qureg.stateVector * sizeof *qureg.stateVector;
    memcpy(qureg.stateVector, qureg.bufferVector, numBytes);

    sleep(1);
    printQureg(qureg);
    sleep(1);

    // MPI_Finalize();
    return 0;
}