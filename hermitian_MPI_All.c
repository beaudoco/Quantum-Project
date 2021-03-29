
#include <stdio.h>   // for printf
#include <stdlib.h>  // for malloc
#include <complex.h> // for double complex
#include <string.h>  // for memcpy()
#include <unistd.h>  // for sleep()
#include <math.h>    // for pow()
#include <mpi.h>     // for MPI
#include <stdbool.h> // for bool
#include <time.h>    // for time testing

typedef struct Qureg
{

    int rank;
    int numRanks;

    int numQubits;
    long long int numAmpsTotal;
    long long int numAmpsPerRank;

    double complex *stateVector;
    double complex *bufferVector;
    double complex *calcVector;
} Qureg;

Qureg createQureg(int numQubits)
{

    Qureg qureg;
    MPI_Comm_rank(MPI_COMM_WORLD, &qureg.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &qureg.numRanks);

    qureg.numQubits = numQubits;
    qureg.numAmpsTotal = 1LL << numQubits; // 2^numQubits
    qureg.numAmpsPerRank = qureg.numAmpsTotal / qureg.numRanks;

    qureg.stateVector = calloc(qureg.numAmpsPerRank, sizeof *qureg.stateVector);
    qureg.bufferVector = calloc(qureg.numAmpsPerRank, sizeof *qureg.bufferVector);
    qureg.calcVector = calloc(qureg.numAmpsTotal, sizeof *qureg.calcVector);

    return qureg;
}

void initRandomQureg(Qureg qureg)
{

    double norm = 0;

    // generate random unnormalised complex amplitudes
    for (int r = 0; r < qureg.numRanks; r++)
    {

        for (int i = 0; i < qureg.numAmpsPerRank; i++)
        {

            // all nodes generate every random number, to keep seed consistent
            double complex n = (rand() / (double)RAND_MAX) + 1i * (rand() / (double)RAND_MAX);

            // but only the nominated node adopts the value
            if (qureg.rank == r)
                qureg.stateVector[i] = n;

            // conveniently, all nodes know/agree on the norm
            norm += pow(cabs(n), 2);
        }
    }

    // normalise the amplitudes
    for (int i = 0; i < qureg.numAmpsPerRank; i++)
        qureg.stateVector[i] /= sqrt(norm);
}

void printQureg(Qureg qureg)
{

    // ensure all nodes are ready (sp we don't interrupt another node's previous print)
    MPI_Barrier(MPI_COMM_WORLD);

    for (int r = 0; r < qureg.numRanks; r++)
    {

        // only one node prints at a time
        if (qureg.rank == r)
        {
            printf("(node %d)\n", r);

            for (int i = 0; i < qureg.numAmpsPerRank; i++)
            {
                double complex amp = qureg.stateVector[i];
                int ind = i + r * qureg.numAmpsPerRank;
                printf("qureg[%d] = %g + (%g)i\n", ind, creal(amp), cimag(amp));
            }
        }

        // prevent other nodes printing prematurely by racing
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void exchangeStateVectors(Qureg qureg, Qureg *qureg2, int pairRank)
{

    // exchange state-vectors through multiple messages to avoid MPI limits
    long long int maxNumMessages = 1LL << 28;
    if (qureg.numAmpsPerRank < maxNumMessages)
        maxNumMessages = qureg.numAmpsPerRank;
    int numMessages = qureg.numAmpsPerRank / maxNumMessages;

    double complex *stateVector;
    stateVector = calloc(qureg.numAmpsPerRank * 2, sizeof stateVector);

    int TAG = 100;
    MPI_Status status;

    // send this node's stateVector to pairRank's bufferVector
    // receive pairRank's stateVector into this node's bufferVector
    for (int i = 0; i < numMessages; i++)

        MPI_Sendrecv(
            &qureg.stateVector[i * maxNumMessages], maxNumMessages, MPI_C_DOUBLE_COMPLEX, pairRank, TAG,
            &stateVector[i * maxNumMessages], maxNumMessages, MPI_C_DOUBLE_COMPLEX, pairRank, TAG,
            MPI_COMM_WORLD, &status);

    // NOW WE HAVE THE VALUES LETS CALC
    for (int i = 0; i < qureg.numAmpsPerRank; i++)
    {

        for (int j = 0; j < qureg.numAmpsPerRank; j++)
        {

            qureg.bufferVector[i] += stateVector[j] * qureg2[j + pairRank * qureg.numAmpsPerRank].stateVector[i];
        }
    }
    free(stateVector);
}

void matrixMultiplication(Qureg qureg, Qureg *qureg2)
{
    // exchange state-vectors through multiple messages to avoid MPI limits
    long long int maxNumMessages = 1LL << 28;
    if (qureg.numAmpsPerRank < maxNumMessages)
        maxNumMessages = qureg.numAmpsPerRank;
    int numMessages = qureg.numAmpsPerRank / maxNumMessages;

    // double complex *stateVector;
    // stateVector = calloc(qureg.numAmpsPerRank * 200, sizeof stateVector);

    // send this node's stateVector to pairRank's bufferVector
    // receive pairRank's stateVector into this node's bufferVector
    for (int i = 0; i < numMessages; i++)
        MPI_Allgather(&qureg.stateVector[i * maxNumMessages], maxNumMessages, MPI_C_DOUBLE_COMPLEX,
                      &qureg.calcVector[i * maxNumMessages], maxNumMessages, MPI_C_DOUBLE_COMPLEX, MPI_COMM_WORLD);

    for (int r = 0; r < qureg.numRanks; r++)
    {

        if (qureg.rank == r)
        {

            for (int i = 0; i < qureg.numAmpsPerRank; i++)
            {

                for (int j = 0; j < qureg.numAmpsPerRank; j++)
                {

                    qureg.bufferVector[i] += qureg.stateVector[j] * qureg2[j + r * qureg.numAmpsPerRank].stateVector[i];
                }
            }
        }
        else
        {

            // THIS IS FOR OFF NODE CALCULATIONS
            // exchangeStateVectors(qureg, qureg2, r);
            for (int i = 0; i < qureg.numAmpsPerRank; i++)
            {

                for (int j = 0; j < qureg.numAmpsPerRank; j++)
                {
                    // exchangeStateVectors(qureg, qureg2, r);
                    // printf("qureg[%d] = %g + (%g)i\n", j, creal(qureg.stateVector[j]), cimag(qureg.stateVector[j]));
                    // printf("qureg[%d] = %g + (%g)i\n", j, creal(qureg.calcVector[j + r * qureg.numAmpsPerRank]), cimag(qureg.calcVector[j + r * qureg.numAmpsPerRank]));
                    qureg.bufferVector[i] += qureg.calcVector[j + r * qureg.numAmpsPerRank] * qureg2[j + r * qureg.numAmpsPerRank].stateVector[i];
                }
            }
        }

        // free(stateVector);
    }

    for (int r = 0; r < qureg.numRanks; r++)
    {

        if (qureg.rank == r)
        {

            // OVERWRITE THE STATE VECTOR P WITH ITS BUFFER
            size_t numBytes = qureg.numAmpsTotal * sizeof *qureg.stateVector;
            memcpy(qureg.stateVector, qureg.bufferVector, numBytes);
        }
    }
}

int main()
{

    // setup MPI, seed all nodes the same
    MPI_Init(NULL, NULL);
    srand(12345);

    // create and print a random qureg
    int numQubits = 3;
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
    // sleep(1);
    // printQureg(qureg);
    // sleep(1);

    // HERE WE ARE GOING TO ATTEMPT TO MATRIX MULTIPLY
    matrixMultiplication(qureg, dens);

    //END CLOCK AND GET TIME
    // clock_gettime(CLOCK_REALTIME, &end);
    // long seconds = end.tv_sec - begin.tv_sec;
    // long nanoseconds = end.tv_nsec - begin.tv_nsec;
    // double elapsed = seconds + nanoseconds * 1e-9;

    // printf("time taken for GPU: %f\n", elapsed);

    sleep(1);
    printQureg(qureg);
    sleep(1);

    MPI_Finalize();
    return 0;
}