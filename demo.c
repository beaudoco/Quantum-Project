
#include <stdio.h>      // for printf
#include <stdlib.h>     // for malloc
#include <complex.h>    // for double complex
#include <string.h>     // for memcpy()
#include <unistd.h>     // for sleep()
#include <math.h>       // for pow()
#include <mpi.h>        // for MPI


typedef struct Qureg {
    
    int rank;
    int numRanks;
    
    int numQubits;
    long long int numAmpsTotal;
    long long int numAmpsPerRank;
    
    double complex* stateVector;
    double complex* bufferVector;
} Qureg;


Qureg createQureg(int numQubits) {
    
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


void initRandomQureg(Qureg qureg) {
    
    double norm = 0;
    
    // generate random unnormalised complex amplitudes
    for (int r=0; r<qureg.numRanks; r++) {
        
        for (int i=0; i<qureg.numAmpsPerRank; i++) {
            
            // all nodes generate every random number, to keep seed consistent        
            double complex n = (rand() / (double) RAND_MAX) + 1i * (rand() / (double) RAND_MAX);
            
            // but only the nominated node adopts the value
            if (qureg.rank == r)
                qureg.stateVector[i] = n;
                
            // conveniently, all nodes know/agree on the norm
            norm += pow(cabs(n), 2);
        }
    }
    
    // normalise the amplitudes
    for (int i=0; i<qureg.numAmpsPerRank; i++)
        qureg.stateVector[i] /= sqrt(norm);
}


void printQureg(Qureg qureg) {
    
    // ensure all nodes are ready (sp we don't interrupt another node's previous print)
    MPI_Barrier(MPI_COMM_WORLD);
    
    for (int r=0; r<qureg.numRanks; r++) {
        
        // only one node prints at a time
        if (qureg.rank == r) {
            printf("(node %d)\n", r);
            
            for (int i=0; i<qureg.numAmpsPerRank; i++) {
                double complex amp = qureg.stateVector[i];
                int ind = i + r * qureg.numAmpsPerRank;
                printf("qureg[%d] = %g + (%g)i\n", ind, creal(amp), cimag(amp));
            }
        }
        
        // prevent other nodes printing prematurely by racing
        MPI_Barrier(MPI_COMM_WORLD);
    }
}


void exchangeStateVectors(Qureg qureg, int pairRank) {

    // exchange state-vectors through multiple messages to avoid MPI limits
    long long int maxNumMessages = 1LL<<28;
    if (qureg.numAmpsPerRank < maxNumMessages) 
        maxNumMessages = qureg.numAmpsPerRank;
    int numMessages = qureg.numAmpsPerRank/maxNumMessages;
    
    int TAG=100;
    MPI_Status status;
    
    // send this node's stateVector to pairRank's bufferVector
    // receive pairRank's stateVector into this node's bufferVector
    for (int i=0; i<numMessages; i++)
    
        MPI_Sendrecv(
            &qureg.stateVector[i*maxNumMessages], maxNumMessages, MPI_C_DOUBLE_COMPLEX, pairRank, TAG,
            &qureg.bufferVector[i*maxNumMessages], maxNumMessages, MPI_C_DOUBLE_COMPLEX, pairRank, TAG, 
            MPI_COMM_WORLD, &status);
}


void applyNotGate(Qureg qureg, int targetQubit) {
    
    // determine this node's pair node
    long long int globalInd = qureg.rank * qureg.numAmpsPerRank;        // ind of 01100 (e.g.)
    long long int pairGlobalInd = globalInd ^ (1LL << targetQubit);     // ind of 01110 (flipping target)
    int pairRank = pairGlobalInd / qureg.numAmpsPerRank;                // rank containing 01110
    
    // if each node contains all needed amplitudes, then no communication needed
    if (pairRank == qureg.rank) {
        
        long long int sizeHalfBlock = 1LL << targetQubit;  // number of contiguous bit sequences where targetQubit is 0
        long long int sizeBlock     = 2LL * sizeHalfBlock;
        
        // each iteration updates two amplitudes (by swapping)
        for (long long int i=0; i<qureg.numAmpsPerRank/2; i++) {
            
            // work out indices of to-be-swapped amplitudes
            long long int blockInd = i / sizeHalfBlock;
            long long int ampInd1 = blockInd * sizeBlock + i%sizeHalfBlock;
            long long int ampInd2 = ampInd1 + sizeHalfBlock;
            
            // swap amplitudes
            double complex amp1 = qureg.stateVector[ampInd1];
            double complex amp2 = qureg.stateVector[ampInd2];
            qureg.stateVector[ampInd1] = amp2;
            qureg.stateVector[ampInd2] = amp1;
        }
        
    // if communication needed, all amplitudes swap with pair node, respecting relative positions in-node
    } else {
        
        // load pairRank's state-vector into this node's buffer
        exchangeStateVectors(qureg, pairRank);
        
        // overwrite this node's state-vector with its buffer
        size_t numBytes = qureg.numAmpsPerRank * sizeof *qureg.stateVector;
        memcpy(qureg.stateVector, qureg.bufferVector, numBytes);
    }
}


int main() {
    
    // setup MPI, seed all nodes the same
    MPI_Init(NULL, NULL);
    srand(12345);
    
    // create and print a random qureg
    int numQubits = 3;
    Qureg qureg = createQureg(numQubits);
    initRandomQureg(qureg);
    printQureg(qureg);
    
    for (int i=0; i<numQubits; i++) {
        
        // sleeping to force-flush stdout (else lines jumbled despite process synch)
        sleep(1);
        
        if (qureg.rank == 0)
            printf("\napplying X to qubit %d produces:\n", i);
        
        // sleeping to force-flush stdout (else lines jumbled despite process synch)
        sleep(1);
            
        // NOT each qubit, one by one
        applyNotGate(qureg, i);
        printQureg(qureg);
    }
    
    MPI_Finalize();
    return 0;
}