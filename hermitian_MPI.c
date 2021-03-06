
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

void exchangeStateVectors(Qureg qureg, int pairRank)
{

    // exchange state-vectors through multiple messages to avoid MPI limits
    long long int maxNumMessages = 1LL << 28;
    if (qureg.numAmpsPerRank < maxNumMessages)
        maxNumMessages = qureg.numAmpsPerRank;
    int numMessages = qureg.numAmpsPerRank / maxNumMessages;

    int TAG = 100;
    MPI_Status status;

    double complex *stateVector;
    stateVector = calloc(qureg.numAmpsPerRank, sizeof *stateVector);

    // send this node's stateVector to pairRank's bufferVector
    // receive pairRank's stateVector into this node's bufferVector
    for (int i = 0; i < numMessages; i++)
    {
        //FIGURE OUT WHAT TO SEND AND FOR CALC AND HOW TO HANDLE
        // MPI_Send(&qureg.stateVector[i * maxNumMessages], maxNumMessages, MPI_C_DOUBLE_COMPLEX, pairRank, TAG, MPI_COMM_WORLD);
        // MPI_Recv(&stateVector[i * maxNumMessages], maxNumMessages, MPI_C_DOUBLE_COMPLEX, pairRank, TAG, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(
            &qureg.stateVector[i * maxNumMessages], maxNumMessages, MPI_C_DOUBLE_COMPLEX, pairRank, TAG,
            &stateVector[i * maxNumMessages], maxNumMessages, MPI_C_DOUBLE_COMPLEX, pairRank, TAG,
            MPI_COMM_WORLD, &status);
    }

    // ensure all nodes are ready (sp we don't interrupt another node's previous print)
    // MPI_Barrier(MPI_COMM_WORLD);

    // for (int r = 0; r < qureg.numRanks; r++)
    // {

    //     // only one node prints at a time
    //     if (qureg.rank == r)
    //     {
    //         printf("(node %d)\n", r);

    //         for (int i = 0; i < qureg.numAmpsPerRank; i++)
    //         {
    //             double complex amp = stateVector[i];
    //             int ind = i + r * qureg.numAmpsPerRank;
    //             printf("qureg[%d] = %g + (%g)i\n", ind, creal(amp), cimag(amp));
    //         }
    //     }

    //     // prevent other nodes printing prematurely by racing
    //     MPI_Barrier(MPI_COMM_WORLD);
    // }
}

void matrixMultiplication(Qureg qureg, Qureg *qureg2)
{
    bool tmp = true;

    for (int r = 0; r < qureg.numRanks; r++)
    {
        if (qureg.rank == 0)
        {
            for (long long int i = 0; i < qureg.numAmpsPerRank; i++)
            {
                // double complex amp = qureg.stateVector[i];
                long long int ind = i + r * qureg.numAmpsPerRank;
                // printf("qureg[%lld]\n", ind);
                for (long long int j = 0; j < qureg.numAmpsTotal; j++)
                {
                    // double complex amp2 = qureg2[ind].stateVector[j];
                    long long int ind2 = j;

                    if ((i < ind && j >= r * qureg.numAmpsPerRank) || (i == ind && j < qureg.numAmpsPerRank))
                    {
                        // printf("qureg[%lld] = qureg2[%lld][%lld]\n", ind, ind2, ind);
                        qureg.bufferVector[ind] += qureg.stateVector[ind2] * qureg2[ind2].stateVector[ind];
                    }
                    else
                    {
                        long long int tmp = 1;
                        printf("qureg[%lld] = %g + (%g)i\n", ind, creal(qureg.stateVector[tmp]), cimag(qureg.stateVector[tmp]));
                        // exchangeStateVectors(qureg, r);
                    }
                }
            }
        }
    }

    // OVERWRITE THE STATE VECTOR P WITH ITS BUFFER
    size_t numBytes = qureg.numAmpsTotal * sizeof *qureg.stateVector;
    memcpy(qureg.stateVector, qureg.bufferVector, numBytes);
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
    printQureg(qureg);

    // THIS IS THE SECOND STATE VECTOR
    // MAKE THIS HERMITIAN USING SOMETHING LIKE
    // Qureg dens = createQureg(2 * numQubits);
    // OR
    Qureg *dens = calloc(1LL << numQubits, sizeof qureg);
    for (long long int i = 0; i < qureg.numAmpsTotal; i++)
    {
        // sleeping to force-flush stdout (else lines jumbled despite process synch)
        sleep(1);
        dens[i] = createQureg(numQubits);
        initRandomQureg(dens[i]);
        // printQureg(dens[i]);
        sleep(1);
    }

    // HERE WE ARE GOING TO ATTEMPT TO MATRIX MULTIPLY
    // sleeping to force-flush stdout (else lines jumbled despite process synch)
    sleep(1);
    matrixMultiplication(qureg, dens);
    sleep(1);
    printf("The updated matrix:\n");
    sleep(1);
    printQureg(qureg);

    sleep(1);

    MPI_Finalize();
    return 0;
}