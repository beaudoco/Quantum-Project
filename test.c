#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdbool.h>
#include <string.h>
#include <complex.h> // for double complex

#define ARR_SIZE 8

/*********************************************************
This section is used to declare the methods that shall be
used throughout the program.
*********************************************************/

void createArrays();
void matrixMultiplication();
void createCSV(int *matchArr, int mainArrLen, int compareArrLen);

/*********************************************************
This is the main function of the code, it is used to
accept the parameters that shall be used throughout the
program.

@parameter mainFile: This is used to select the file that
will be parsed against to see if it has any matches
@parameter compareFile: This is used to select the file
that to parse the main file with for matches
*********************************************************/

int main()
{
    createArrays();

    return 0;
}

/*********************************************************
This function is used to take a given file and process
the data into a usable array.

@parameter mainFile: The name of the file to process.
@parameter compareFile: used to name desired output file
@return: none
*********************************************************/

void createArrays()
{

    double complex stateVector[ARR_SIZE] = {
        0.416544 + (0.185924 * 1i),
        0.221074 + (0.173629 * 1i),
        0.056435 + (0.0645494 * 1i),
        0.0497726 + (0.287299 * 1i),
        0.284243 + (0.334581 * 1i),
        0.231751 + (0.383256 * 1i),
        0.210251 + (0.228927 * 1i),
        0.358434 + (0.0459774 * 1i)};

    double complex densityMatrix[ARR_SIZE][ARR_SIZE] = {
        {0.0631268 + (0.273539 * 1i),
         0.371035 + (0.0125303 * 1i),
         0.222853 + (0.276819 * 1i),
         0.386746 + (0.213651 * 1i),
         0.050326 + (0.0794385 * 1i),
         0.304065 + (0.0245374 * 1i),
         0.372029 + (0.269266 * 1i),
         0.344612 + (0.174035 * 1i)},
        {0.25837 + (0.125253 * 1i),
         0.152907 + (0.0125253 * 1i),
         0.13561 + (0.282548 * 1i),
         0.425889 + (0.174231 * 1i),
         0.338425 + (0.257385 * 1i),
         0.355461 + (0.0786767 * 1i),
         0.0983743 + (0.4285 * 1i),
         0.271178 + (0.0625225 * 1i)},
        {0.066071 + (0.143977 * 1i),
         0.00981416 + (0.189161 * 1i),
         0.386115 + (0.271509 * 1i),
         0.344739 + (0.366338 * 1i),
         0.388958 + (0.0431597 * 1i),
         0.291383 + (0.235694 * 1i),
         0.00677986 + (0.342565 * 1i),
         0.0624378 + (0.219716 * 1i)},
        {0.256895 + (0.258501 * 1i),
         0.0992593 + (0.218061 * 1i),
         0.387288 + (0.388609 * 1i),
         0.204927 + (0.0792415 * 1i),
         0.155658 + (0.282319 * 1i),
         0.264838 + (0.116505 * 1i),
         0.177231 + (0.251207 * 1i),
         0.411574 + (0.0945532 * 1i)},
        {0.250458 + (0.205802 * 1i),
         0.0981065 + (0.346668 * 1i),
         0.325638 + (0.247503 * 1i),
         0.286947 + (0.0573371 * 1i),
         0.272741 + (0.32528 * 1i),
         0.282524 + (0.197415 * 1i),
         0.0423849 + (0.278937 * 1i),
         0.135207 + (0.339587 * 1i)},
        {0.20248 + (0.315114 * 1i),
         0.269091 + (0.26966 * 1i),
         0.350586 + (0.155107 * 1i),
         0.000533435 + (0.0354795 * 1i),
         0.154462 + (0.0666583 * 1i),
         0.323208 + (0.271003 * 1i),
         0.367567 + (0.335463 * 1i),
         0.148724 + (0.300638 * 1i)},
        {0.313464 + (0.256217 * 1i),
         0.110086 + (0.293525 * 1i),
         0.370871 + (0.272734 * 1i),
         0.231522 + (0.0171637 * 1i),
         0.283468 + (0.258785 * 1i),
         0.155058 + (0.132634 * 1i),
         0.11072 + (0.340643 * 1i),
         0.202155 + (0.337446 * 1i)},
        {0.172135 + (0.379653 * 1i),
         0.0210012 + (0.0158552 * 1i),
         0.0600708 + (0.0570945 * 1i),
         0.140166 + (0.021666 * 1i),
         0.359398 + (0.252566 * 1i),
         0.118678 + (0.542957 * 1i),
         0.36567 + (0.265521 * 1i),
         0.00819212 + (0.304023 * 1i)}};

    //CHECK THE COMPLEXITY OF OUR INFORMATION
    matrixMultiplication(stateVector, densityMatrix);
}

/*********************************************************
This function is used to iterate over the main file's
data and compare it to the secondary file to check for
the highest matching score

@parameter mainBuff: The main file data to be processed.
@parameter mainArrLen: The overall size of the main file
@parameter compareBuff: The secondary file data to be
compared against the main
@parameter compareArrLen: The overall size of the
secondary file
@return: none
*********************************************************/

void matrixMultiplication(double complex stateVector[ARR_SIZE], double complex densityMatrix[ARR_SIZE][ARR_SIZE])
{
    double complex *tmpStateVector = calloc(ARR_SIZE*2, sizeof tmpStateVector);

    //ITERATE OVER THE MAIN FILE AS WE WILL BE WAVEFORM ANALYZING IT
    for (int i = 0; i < ARR_SIZE; i++)
    {

        for (int j = 0; j < ARR_SIZE; j++)
        {
            tmpStateVector[i] += stateVector[j] * densityMatrix[j][i];
        }
    }

    for (int i = 0; i < ARR_SIZE; i++)
    {
        printf("qureg[%d] = %g + (%g)i\n", i, creal(tmpStateVector[i]), cimag(tmpStateVector[i]));
    }

    //OUTPUT THE MATCH SCORE ARRAY
    // createCSV(matchArr, mainArrLen, compareArrLen);

    //FREE MEMORY
    // free(matchArr);
}

/*********************************************************
This function is used to output the processed match score.
This is for comparison purpose to ensure the program works

@parameter matchArr: The file data to be output.
@parameter mainArrLen: The overall row sizes
@parameter compareArrLen: The overall column sizes
@return: none
*********************************************************/
void createCSV(int *matchArr, int mainArrLen, int compareArrLen)
{
    //DECLARE VARS
    FILE *filep;
    int i = 0, j = 0;

    //OPEN FILE
    filep = fopen("test.txt", "w+");

    //WRITE RESULTS TO FILE
    for (i = 0; i < mainArrLen + 1; i++)
    {
        for (j = 0; j < compareArrLen + 1; j++)
        {
            fprintf(filep, "%d,", matchArr[j + (i * (compareArrLen + 1))]);
        }
        fprintf(filep, "\n");
    }

    //CLOSE FILE
    fclose(filep);

    //LET USER KNOW PROGRAM IS DONE
    printf("file created \n");
}
