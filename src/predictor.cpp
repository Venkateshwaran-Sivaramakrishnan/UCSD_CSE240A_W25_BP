//========================================================//
//  predictor.c                                           //
//  Source file for the Branch Predictor                  //
//                                                        //
//  Implement the various branch predictors below as      //
//  described in the README                               //
//========================================================//
#include <stdio.h>
#include <math.h>
#include "predictor.h"
#include <bitset>
//#include <execinfo.h>
//#include <signal.h>
//#include <exception>
//#include <iostream>
//#include <csignal>
//#include <cstdlib>
//#include <unistd.h>

using namespace std;

//
// TODO:Student Information
//
const char *studentName = "Venkateshwaran Sivaramakrishnan";
const char *studentID = "A69031111";
const char *email = "vsivaramakrishnan@ucsd.edu";

//------------------------------------//
//      Predictor Configuration       //
//------------------------------------//

// Handy Global for use in output routines
const char *bpName[4] = {"Static", "Gshare",
                         "Tournament", "Custom"};

// define number of bits required for indexing the BHT here.
int ghistoryBits = 17; // Number of bits used for Global History
int bpType;            // Branch Prediction Type
int verbose;

//------------------------------------//
//      Predictor Data Structures     //
//------------------------------------//

//
// TODO: Add your own Branch Predictor data structures here
//
// gshare
uint8_t *bht_gshare;
uint64_t ghistory;

// tournament
uint8_t *bht_local;
uint8_t *bht_global;
uint8_t *choice_t;
uint64_t *pht_local;
uint64_t tghistory;
int tghistoryBits = 15; // Number of bits used for Global History
int tlhistoryBits = 15; // Number of bits used for Local History
int phtIndexBits = 12;

// custom: tage
#define NUM_TAGE_TABLES 12
#define TAGE_MAX 7 			// 3-bit Tage Predictor
#define BIMODAL_SIZE 13
#define BHT_INIT 2 			// Because loops are initially biased to be Taken
#define BHT_MAX 3			// 2-bit Bimodal Predictor
#define GHIST_MAX_LEN 641
#define GHIST bitset<GHIST_MAX_LEN>
#define USE_ALT_ON_NA_INIT 8
#define USE_ALT_ON_NA_MAX 15
#define WEAK_TAKEN 4
#define WEAK_NOT_TAKEN 3
#define UBIT_MAX 3
#define MAX_INSTR_COUNT 18 // 256K Instruction: (1 << 18) yields 262144 instructions
#define PHR_SIZE 16
#define LOOP_TABLE_SIZE 10
#define LOOP_TAG_SIZE 14
#define LOOP_CTR_MAX 3 // For 2-bit counter
#define LOOP_AGE_MAX 8
#define LOOP_COUNT_MAX 14

typedef struct tage {
  uint16_t tag;
  uint16_t ctr;
  uint16_t u;
} tage_t;

typedef struct prediction {
  bool pred;
  uint32_t table;
  uint32_t index;
  bool alterPred;
  uint32_t alterTable;
  uint32_t alterIndex;
} prediction_t;

typedef struct shiftReg {
  uint16_t data;
  uint16_t actLength; // Actual Length
  uint16_t newLength; 
} cShiftReg_t;

typedef struct loopTable {
  bool prediction;
  bool used;
  uint32_t tag; 
  uint32_t ctr; // Confidence counter 
  uint32_t presentIter;
  uint32_t loopCount;
  uint32_t age;
} loopTable_t;

GHIST GHR;
uint8_t *BHT;
uint32_t PHR;
prediction_t predict;
uint32_t hist[NUM_TAGE_TABLES] = {640, 403, 254, 160, 101, 64, 40, 25, 16, 10, 6, 4}; // History Length from high to low
uint32_t tableSize[NUM_TAGE_TABLES] = {9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 10, 10}; // Number of 2^n entries from high to flow
uint32_t tagWidth[NUM_TAGE_TABLES] = {15, 14, 13, 12, 12, 11, 10, 9, 8, 8, 7, 7}; // Descending order helps in finding prediction with most history-bits first
uint32_t *tageTableSize;
uint32_t *tageTagSize;
uint32_t *tageHistory;
uint32_t bimodalEntries;
uint32_t *tageIndex;
uint32_t *tageTag;
uint32_t use_alt_on_na; // Use Alternate Prediction on New Allocation
tage_t **tageTables;
uint64_t count;
bool toggle;
cShiftReg_t *csrIndex;
cShiftReg_t **csrTag;
loopTable_t *loopTable;

// custom: piecewise linear
#define N 8        // Modulo values from 
#define M 118
#define H 26       // Global History Length
#define THETA 76   // (2.14*(h+1))+20.58 training threshold

bool pwGHR[H+1]; // Global History Register
// 0-Not Taken, 1-taken

int GA[H+1];   // Global Address Register (address % M). Address is shifted into the first position of the array.
// GHR and GA together give the path history. 
// Used for predicting the current branch.

int W[N][M][H+1];   // 3D-Matrix indexed by branch address, address of branch in path history, and position in the history
// Keeps tracks of correlation of every branch
// Acts as the weights that keeps track of the tendency of branch B to be taken
// 8-bit signed weights (+127 and -128)

//------------------------------------//
//        Predictor Functions         //
//------------------------------------//

// Initialize the predictor
//

// DEBUG: Stack Trace
//void signalHandler(int signal) {
//    if (signal == SIGSEGV) {
//        cerr << "Error: Segmentation fault (SIGSEGV) caught!\n";
//
//        void *array[10];
//        size_t size = backtrace(array, 10); // Capture stack trace
//       char **messages = backtrace_symbols(array, size); // Convert to symbols
//
//        cerr << "Stack trace:\n";
//        for (size_t i = 0; i < size; ++i) {
//            cerr << messages[i] << endl; // Print raw stack trace symbol
//
//            // Extract function address
//            string addr = messages[i];
//
//            // Use addr2line to get file name and line number
//            char command[256];
//            snprintf(command, sizeof(command), "addr2line -e predictor %p", array[i]);
//            system(command); // Run addr2line
//        }
//
//       free(messages); // Free memory allocated by backtrace_symbols()
//        exit(EXIT_FAILURE);
//    }
//}

// piecewise linear
int boundedIncrement(int data)   // Increment with saturation limits -128 and +127
{
  if (data < 127) return data+1; // 8-bit signed max value
  else return data;
}

int boundedDecrement(int data)   // Decrement with saturation limits -128 and +127
{
  if (data > -128) return data-1; // 8-bit signed min value
  else return data;
}

void init_piecewise_linear()
{
  pwGHR[0] = 1; // Initialize the first branch in GHR with TAKEN state

  for(uint32_t i = 1; i <= H; i++) pwGHR[i] = 0; // Initialize GHR to NOT TAKEN

  for(uint32_t i = 0; i <= H; i++) GA[i] = 0;  // Initialize Branch Address entries

  for(uint32_t i=0; i<N; i++) // Initialize the weight matrix
  {
    for(uint32_t j=0; j<M; j++)
    {
      for(uint32_t k=0; k<=H; k++)
      {  
        W[i][j][k] = 0;
      }
    }
  }
}

uint8_t piecewise_linear_predict(uint32_t pc)
{
  uint32_t pcIndex = pc % N; // PC modulo N gives compressed PC to index the branch in the weight table
  // Weight Matrix (W) size is constrained within the hardware budget. Hence the modulo operation to map the PC to the limited matrix dimensions

  int pcWeight = W[pcIndex][0][0];

  for(uint32_t i = 1; i <= H; i++)
  {
    if(pwGHR[i] == 1) pcWeight = pcWeight + W[pcIndex][GA[i]][i];
    else if (pwGHR[i] == 0) pcWeight = pcWeight - W[pcIndex][GA[i]][i];
  }

  if(pcWeight >= 0) return true; // Positive value favours branch TAKEN
  else return false; // Negative value favors branch NOT TAKEN
}

void train_piecewise_linear(uint32_t pc, uint8_t outcome)
{ 
  uint32_t pcIndex = pc % N;
  int pcWeight = W[pcIndex][0][0];
  uint8_t predDir = piecewise_linear_predict(pc);

  for(uint32_t i = 1; i <= H; i++)
  {
    if(pwGHR[i] == 1) pcWeight = pcWeight + W[pcIndex][GA[i]][i];
    else if (pwGHR[i] == 0) pcWeight = pcWeight - W[pcIndex][GA[i]][i];
  }

  if(pcWeight < 0) pcWeight = -pcWeight; // To get modulus value (positive) 

  if((outcome != predDir) || (pcWeight < THETA)) // On branch mispredict with weights under threshold, train the weight matrix
  {
    if(outcome) W[pcIndex][0][0] = W[pcIndex][0][0] + 1; // Branch TAKEN favors positive enforcement
    else W[pcIndex][0][0] = W[pcIndex][0][0] - 1; // Branch NOT TAKEN favors negative enforcement

    for(uint32_t i = 1; i <= H; i++)
    {
      if(outcome == pwGHR[i]) W[pcIndex][GA[i]][i] = boundedIncrement(W[pcIndex][GA[i]][i]); // Increment Weight Matrix if branch outcome follows Global History
      else W[pcIndex][GA[i]][i] = boundedDecrement(W[pcIndex][GA[i]][i]); // Decrement Weight Matrix if branch outcome diverges from Global History
    }
  }
  
  for(uint32_t i = 2; i <= H; i++) // Right shift GHR and GA by 1-bit. Preserve the LSB (0th bias bit). 
  {
    pwGHR[i-1] = pwGHR[i];
    GA[i-1] = GA[i];
  }

  pwGHR[H] = outcome; // Append the outcome at first position
  GA[H] = pc % M; // Store hashed (modulo) PC in Global Address Register
}

// tage functions
void initializeFoldReg(cShiftReg_t *csr, uint32_t actLength, uint32_t newLength)
{
  csr->data = 0;
  csr->actLength = actLength;
  csr->newLength = newLength; 
}

void foldReg(cShiftReg_t* csr)
{
  csr->data = (csr->data << 1) | GHR[0]; // Left-shift the CSR data and append the latest GHR bit to the LSB of CSR
  csr->data = (csr->data) ^ (GHR[csr->actLength] << (csr->actLength % csr->newLength));// Hash the GHR information at actual length with the CSR (one of many techniques from PPM)
  csr->data = (csr->data) ^ ((csr->data & (1 << csr->newLength)) >> csr->newLength); // Preserve any overflows from the previous operation by XORing with LSB of CSR
  //csr->data = (csr->data) ^ (csr->data >> csr->newLength);
  csr->data = (csr->data) & ((1 << csr->newLength) - 1);
}

uint32_t calTag(uint32_t pc, uint32_t tageNum, uint32_t tageTagSize)
{
  uint32_t tag = (pc ^ csrTag[0][tageNum].data ^ (csrTag[1][tageNum].data << 1));
  tag = (tag & ((1 << tageTagSize) - 1));
  return tag;
}

uint32_t calIndex(uint32_t pc, uint32_t tageNum, uint32_t tageTagSize)
{
  uint32_t index = pc ^ (pc >> tageTagSize) ^ csrIndex[tageNum].data ^ PHR;
  index = (index & ((1 << tageTagSize) - 1));
  return index;
}

void init_tage()
{
  GHR.reset(); // Set all bits of GHR to 0
  PHR = 0;
  bimodalEntries = 1 << BIMODAL_SIZE;
  predict.pred = -1;
  predict.alterPred = -1;
  predict.table = NUM_TAGE_TABLES;
  predict.alterTable = NUM_TAGE_TABLES;
  use_alt_on_na = USE_ALT_ON_NA_INIT;
  count = 0;
  toggle = 0;

  BHT = (uint8_t *)malloc(bimodalEntries * sizeof(uint8_t));
  tageTableSize = (uint32_t *)malloc(NUM_TAGE_TABLES * sizeof(uint32_t));
  tageTagSize = (uint32_t *)malloc(NUM_TAGE_TABLES * sizeof(uint32_t));
  tageHistory = (uint32_t *)malloc(NUM_TAGE_TABLES * sizeof(uint32_t));
  tageIndex = (uint32_t *)malloc(NUM_TAGE_TABLES * sizeof(uint32_t));
  tageTag = (uint32_t *)malloc(NUM_TAGE_TABLES * sizeof(uint32_t));
  csrIndex = (cShiftReg_t *)malloc(NUM_TAGE_TABLES * sizeof(cShiftReg_t));
  tageTables = new tage_t*[NUM_TAGE_TABLES];
  csrTag = new cShiftReg_t*[2];
  csrTag[0] = new cShiftReg_t[NUM_TAGE_TABLES]; 
  csrTag[1] = new cShiftReg_t[NUM_TAGE_TABLES];
  uint32_t tempSize;

  uint32_t loopTableSize = (1 << LOOP_TABLE_SIZE);
  loopTable = new loopTable_t[loopTableSize];

  for(uint32_t i = 0; i < loopTableSize; i++)
  {
    loopTable[i].prediction = false;
    loopTable[i].used = false;
    loopTable[i].tag = 0;
    loopTable[i].ctr = 0;
    loopTable[i].presentIter = 0;
    loopTable[i].loopCount = 0;
    loopTable[i].age = 0; 
  }

  for(int i = 0; i < NUM_TAGE_TABLES; i++)
  {
    tageTableSize[i] = tableSize[i];
    tageTagSize[i] = tagWidth[i];
    tageHistory[i] = hist[i];
    tageIndex[i] = 0;
    tageTag[i] = 0;
     
	
    tempSize = 1 << tageTableSize[i];
    tageTables[i] = new tage_t[tempSize];
    for(uint32_t j = 0; j < tempSize; j++)
    {
      tageTables[i][j].tag = 0;
      tageTables[i][j].ctr = 0;
      tageTables[i][j].u = 0;

    }

    // Create Circular Shift Registers for folding the Global History Register
    initializeFoldReg(&csrIndex[i], tageTagSize[i], tageHistory[i]);
    initializeFoldReg(&csrTag[0][i], tageTagSize[i], tageHistory[i]);
    initializeFoldReg(&csrTag[1][i], tageTagSize[i]-1, tageHistory[i]);

  }

  for(uint32_t i = 0; i < bimodalEntries; i++)
  {
    BHT[i] = BHT_INIT; 
  }

}

uint8_t tage_predict(uint32_t pc)
{
  //signal(SIGSEGV, signalHandler);
  // get lower ghistoryBits of pc
  uint32_t bimodalIndex = (pc & ((1 << BIMODAL_SIZE) - 1));
  uint32_t loopTableIndex = (pc & ((1 << LOOP_TABLE_SIZE) - 1));
  uint32_t loopTag = (pc >> LOOP_TABLE_SIZE) & ((1 << LOOP_TAG_SIZE) - 1);
  predict.pred = -1;
  predict.alterPred = -1;
  predict.table = NUM_TAGE_TABLES;
  predict.alterTable = NUM_TAGE_TABLES;
  
  // Loop table prediction
  if(loopTable[loopTableIndex].tag == loopTag)
  {
    if(loopTable[loopTableIndex].loopCount > loopTable[loopTableIndex].presentIter) loopTable[loopTableIndex].prediction = TAKEN; // Loop is taken if iterator is less than total iteration count
    else if(loopTable[loopTableIndex].loopCount == loopTable[loopTableIndex].presentIter) loopTable[loopTableIndex].prediction = NOTTAKEN; // Loop exit

    if(loopTable[loopTableIndex].ctr == LOOP_CTR_MAX)
    {
      loopTable[loopTableIndex].used = true;
      return loopTable[loopTableIndex].prediction;
    }
  }

  loopTable[loopTableIndex].used = false;

  for(int i = 0; i < NUM_TAGE_TABLES; i++)
  {
    tageIndex[i] = calIndex(pc, i, tageTableSize[i]);
    tageTag[i] = calTag(pc, i, tageTagSize[i]); 
  }
  
  // Check for Tag hits in the Tage Tables
  for(int i = 0; i < NUM_TAGE_TABLES; i++)
  {
    if(tageTables[i][tageIndex[i]].tag == tageTag[i])
    {
      predict.table = i;
      predict.index = tageIndex[i]; 
      break;
    }
  }
  
  // Check for Tag hits in the Alternate Tage Tables starting from predicted + 1 index
  for(int i = predict.table + 1; i < NUM_TAGE_TABLES; i++)
  {
    if(tageTables[i][tageIndex[i]].tag == tageTag[i])
    {
      predict.alterTable = i;
      predict.alterIndex = tageIndex[i]; 
      break;
    }
  }

  // Prediction Logic
  if(predict.table < NUM_TAGE_TABLES) // Indicates hit in Tage Table
  {
    if(predict.alterTable == NUM_TAGE_TABLES) // Indicates miss for Alternative Tage Table lookup
    {
      predict.alterPred = (BHT[bimodalIndex] > BHT_MAX/2); // Bimodal prediction becomes the Alternative prediction
    }
    else
    {
      if(tageTables[predict.alterTable][predict.alterIndex].ctr > TAGE_MAX/2) predict.alterPred = TAKEN;
      else predict.alterPred = NOTTAKEN;
    }
    if(tageTables[predict.table][predict.index].ctr != WEAK_TAKEN || tageTables[predict.table][predict.index].ctr != WEAK_NOT_TAKEN
       || tageTables[predict.table][predict.index].u != 0 || use_alt_on_na < USE_ALT_ON_NA_INIT)
    {
      predict.pred = tageTables[predict.table][predict.index].ctr > TAGE_MAX/2;
      return predict.pred;
    }
    else
    {
      return predict.alterPred;
    }
  }
  else // In case of no Hits in Tage tables(complete Miss)
  {
    predict.alterPred = (BHT[bimodalIndex] > BHT_MAX/2);
    return predict.alterPred;
  }

}

void train_tage(uint32_t pc, uint8_t outcome)
{
  // get lower ghistoryBits of pc
  uint32_t prediction = -1;
  uint32_t alterPrediction = -1;
  uint32_t bimodalIndex = (pc & ((1 << BIMODAL_SIZE) - 1));
  bool steal;

  uint32_t predDir = tage_predict(pc);
  
  // Loop updation logic
  uint32_t loopTableIndex = (pc & ((1 << LOOP_TABLE_SIZE) - 1));
  uint32_t loopTag = (pc >> LOOP_TABLE_SIZE) & ((1 << LOOP_TAG_SIZE) - 1);
  
  if((loopTable[loopTableIndex].tag != loopTag) && (loopTable[loopTableIndex].age > 0)) loopTable[loopTableIndex].age--; // In case of Tag Miss in Loop Table
  else // Indicates both Hit, and Miss-with age equals 0 
  {
    if(loopTable[loopTableIndex].age == 0) // In case of old/blank entry for both HIT and MISS cases
    { 
      loopTable[loopTableIndex].prediction = false;
      loopTable[loopTableIndex].tag = loopTag;
      loopTable[loopTableIndex].ctr = 0;
      loopTable[loopTableIndex].presentIter = 1;
      loopTable[loopTableIndex].loopCount = (1 << LOOP_COUNT_MAX) - 1;
      loopTable[loopTableIndex].age = (1 << LOOP_AGE_MAX) - 1;
    }
    else
    {
      if(loopTable[loopTableIndex].prediction == outcome) // Correct Prediction
      {
        if(loopTable[loopTableIndex].presentIter != loopTable[loopTableIndex].loopCount) loopTable[loopTableIndex].presentIter++; // Increment iteration count when iterator is less than the loop count
	else if(loopTable[loopTableIndex].presentIter != loopTable[loopTableIndex].loopCount)
	{
	  loopTable[loopTableIndex].presentIter = 0; // Reset because of predicted loop exit
	  if(loopTable[loopTableIndex].ctr < LOOP_CTR_MAX) loopTable[loopTableIndex].ctr++;
	}	
      }
      else // Incorrect Prediction
      {
	if(loopTable[loopTableIndex].age == (1 << LOOP_AGE_MAX) - 1)
	{
	  loopTable[loopTableIndex].loopCount = loopTable[loopTableIndex].presentIter;
	  loopTable[loopTableIndex].presentIter = 0;
	  loopTable[loopTableIndex].ctr = 1;
	}
	else
	{
	  loopTable[loopTableIndex].prediction = false;
      	  loopTable[loopTableIndex].tag = 0;
          loopTable[loopTableIndex].ctr = 0;
      	  loopTable[loopTableIndex].presentIter = 0;
      	  loopTable[loopTableIndex].loopCount = 0;
          loopTable[loopTableIndex].age = 0;
	}	
      }  
    }
    if(loopTable[loopTableIndex].used) return;
  } 

  if(predict.table < NUM_TAGE_TABLES) // If there is a tag Hit in Tage Tables
  {
    prediction = tageTables[predict.table][predict.index].ctr;
    if(outcome && (prediction < TAGE_MAX)) tageTables[predict.table][predict.index].ctr++; // Increment the counter if prediction is TAKEN
    else if (!outcome && (prediction > 0)) tageTables[predict.table][predict.index].ctr--; // Decrement the counter if prediction is NOT TAKEN

    if(predict.alterTable != NUM_TAGE_TABLES) alterPrediction = tageTables[predict.alterTable][predict.alterIndex].ctr; // If alternate prediction exists, read the prediction counter

    if(tageTables[predict.table][predict.index].u == 0 && alterPrediction != -1)
    {
      // Update the counters of alternate prediction
      if(outcome && (alterPrediction < TAGE_MAX)) tageTables[predict.alterTable][predict.alterIndex].ctr++; // Increment on TAKEN
      else if(!outcome && (alterPrediction > 0)) tageTables[predict.alterTable][predict.alterIndex].ctr--;  // Decrement on NOT TAKEN
    }
  }
  else
  {
   prediction = BHT[bimodalIndex];
   if(outcome && (prediction < BHT_MAX)) BHT[bimodalIndex]++; //Increment the Bimodal counter if prediction is TAKEN
   else if(!outcome && (prediction > 0)) BHT[bimodalIndex]--; // Decrement the Bimodal counter if prediction is NOT TAKEN
  }
  
  // If there is a prediction from Tage table, check for newly allocated entry in the tage table
  if(predict.table < NUM_TAGE_TABLES)
  {
    if(tageTables[predict.table][predict.index].ctr == WEAK_TAKEN || tageTables[predict.table][predict.index].ctr == WEAK_NOT_TAKEN
       || tageTables[predict.table][predict.index].u == 0)
    {
      if(predict.pred != predict.alterPred) // In case the predictions in prediction provider and alternate prediction are different
      {
	if(predict.alterPred == outcome) // If alternate prediction is correct
	{
	  if(use_alt_on_na < USE_ALT_ON_NA_MAX) use_alt_on_na++;
	}
	else if(use_alt_on_na > 0) use_alt_on_na--;
      } 
    }
  }

  // Updation policy on overall incorrect prediction
  if((outcome != predDir) && (predict.table > 0)) // Steal an entry if the prediction is wrong and it doesn't use the longest history length to predict
  {
    steal = false;
   for(uint32_t i = 0; i < predict.table; i++)
   {
     if(tageTables[i][tageIndex[i]].u == 0) steal = true; // On mis-predict, an entry which isn't useful can be allocated in bigger tage tables
   }
   if(!steal)
   { // Decrement the usefulness counter for misprediction
     for(int i = predict.table - 1; i >= 0; i--)
     {
       tageTables[i][tageIndex[i]].u--;
     }
   }
   else
   {
     for(int i = predict.table - 1; i>=0 ; i--)
     {
       if((tageTables[i][tageIndex[i]].u == 0) && !(rand()%10)) // One in 10 times probability that the usefulness counter is 0
       {
         if(outcome) tageTables[i][tageIndex[i]].ctr = WEAK_TAKEN; // If branch is TAKEN, set the saturation counter to WEAK_TAKEN indicating it is a new allocation
	 else tageTables[i][tageIndex[i]].ctr = WEAK_NOT_TAKEN; // If branch is NOT TAKEN, set the saturation counter to WEAK_NOT_TAKEN indicating it is a new allocation
	 tageTables[i][tageIndex[i]].tag = tageTag[i]; // Allocate the new tag
	 tageTables[i][tageIndex[i]].u = 0; // Reset the useful counter, indicating it is a new allocation and may not be a trust prediction
	 break;
       }
     }
   }
  }
  
  // Update usefulness counter of predictions which are derived from the Tage tables
  if(predict.table < NUM_TAGE_TABLES)
  {
    if(predict.alterPred != predDir)
    {
      if(outcome == predDir && tageTables[predict.table][predict.index].u < UBIT_MAX) tageTables[predict.table][predict.index].u++; // Correct prediction
      else if(outcome != predDir && tageTables[predict.table][predict.index].u > 0) tageTables[predict.table][predict.index].u--; // Incorrect Predicion
    }
  }

  GHR = (GHR << 1);
  PHR = (PHR << 1);
  if(outcome == TAKEN) 
  {
	  GHR.set(0,1);
	  PHR = PHR + 1;
  }
  //if(pc & 1) PHR = PHR + 1; // On unaligned PC
  PHR = (PHR & ((1 << PHR_SIZE) - 1)); // Apply mask to restrict PHR to it's size

  // CSR folding operation
  for(int i = 0; i < NUM_TAGE_TABLES; i++)
  {
    foldReg(&csrIndex[i]);
    foldReg(&csrTag[0][i]);
    foldReg(&csrTag[1][i]); 
  }

  count++;
  if(count == (1 << MAX_INSTR_COUNT))
  {
    count = 0; // Reset
    if(toggle == 0) 
    {
      for(int i = 0; i < NUM_TAGE_TABLES; i++)
      {
        for(int j = 0; j < (1 << tageTableSize[j]); j++)
        {
          tageTables[i][j].u = tageTables[i][j].u & 2; // Mask with 10 LSB
        }
      }
      toggle = 1;
    }
    else
    {
      for(int i = 0; i < NUM_TAGE_TABLES; i++)
      {
        for(int j = 0; j < (1 << tageTableSize[j]); j++)
        {
          tageTables[i][j].u = tageTables[i][j].u & 1; // Mask with 01 LSB
        }
      }
      toggle = 0;
    }
  }

}

void cleanup_tage()
{
  free(BHT);
  free(tageTableSize);
  free(tageTagSize);
  free(tageHistory);
  free(tageIndex);
  free(tageTag);
  free(csrIndex);
}
 
// gshare functions
void init_gshare()
{
  int bht_entries = 1 << ghistoryBits;
  bht_gshare = (uint8_t *)malloc(bht_entries * sizeof(uint8_t));
  int i = 0;
  for (i = 0; i < bht_entries; i++)
  {
    bht_gshare[i] = WN;
  }
  ghistory = 0;
}

uint8_t gshare_predict(uint32_t pc)
{
  // get lower ghistoryBits of pc
  uint32_t bht_entries = 1 << ghistoryBits;
  uint32_t pc_lower_bits = pc & (bht_entries - 1); // For a 32-bit Machine we don't care about lower 2-bits assuming an aligned memory
  uint32_t ghistory_lower_bits = ghistory & (bht_entries - 1);
  uint32_t index = pc_lower_bits ^ ghistory_lower_bits;
  switch (bht_gshare[index])
  {
  case WN:
    return NOTTAKEN;
  case SN:
    return NOTTAKEN;
  case WT:
    return TAKEN;
  case ST:
    return TAKEN;
  default:
    printf("Warning: Undefined state of entry in GSHARE BHT!\n");
    return NOTTAKEN;
  }
}

void train_gshare(uint32_t pc, uint8_t outcome)
{
  // get lower ghistoryBits of pc
  uint32_t bht_entries = 1 << ghistoryBits;
  uint32_t pc_lower_bits = pc & (bht_entries - 1);
  uint32_t ghistory_lower_bits = ghistory & (bht_entries - 1);
  uint32_t index = pc_lower_bits ^ ghistory_lower_bits;

  // Update state of entry in bht based on outcome
  switch (bht_gshare[index])
  {
  case WN:
    bht_gshare[index] = (outcome == TAKEN) ? WT : SN;
    break;
  case SN:
    bht_gshare[index] = (outcome == TAKEN) ? WN : SN;
    break;
  case WT:
    bht_gshare[index] = (outcome == TAKEN) ? ST : WN;
    break;
  case ST:
    bht_gshare[index] = (outcome == TAKEN) ? ST : WT;
    break;
  default:
    printf("Warning: Undefined state of entry in GSHARE BHT!\n");
    break;
  }

  // Update history register
  ghistory = ((ghistory << 1) | outcome);
}

void cleanup_gshare()
{
  free(bht_gshare);
}

// tournament functions
void init_tournament()
{
  int lbht_entries = 1 << tlhistoryBits;
  int gbht_entries = 1 << tghistoryBits;
  int pht_entries = 1 << phtIndexBits;
  bht_local = (uint8_t *)malloc(lbht_entries * sizeof(uint8_t));
  bht_global = (uint8_t *)malloc(gbht_entries * sizeof(uint8_t));
  choice_t = (uint8_t *)malloc(gbht_entries * sizeof(uint8_t));
  pht_local = (uint64_t *)malloc(pht_entries * sizeof(uint64_t)); // Assign more memory than required to play with entry size, but extra bits are masked during access
  int i = 0;
  for (i = 0; i < gbht_entries; i++)
  {
    bht_global[i] = WN;
    choice_t[i] = WT;
  }
  for (i = 0; i < lbht_entries; i++)
  {
    bht_local[i] = WN;
  }
  for (i = 0; i < pht_entries; i++)
  {
    pht_local[i] = 0;
  }
  tghistory = 0;
}

uint8_t tournament_predict(uint32_t pc)
{
  // get lower ghistoryBits of pc
  // discard last two bits of PC
  uint32_t pht_entries = 1 << (phtIndexBits);
  uint32_t pc_bits = (pc & (pht_entries - 1)); // For a 32-bit Machine we don't care about lower 2-bits assuming an aligned memory
  uint64_t bhtLocalIndex = pht_local[pc_bits] & ((1 << tlhistoryBits) - 1);
  uint64_t tghistoryIndex = tghistory & ((1 << tghistoryBits) - 1); 
  uint8_t localPrediction;
  uint8_t globalPrediction;
  uint8_t choice;
  
  switch (bht_local[bhtLocalIndex])
  {
  case WN:
    localPrediction = NOTTAKEN;
    break;
  case SN:
    localPrediction = NOTTAKEN;
    break;
  case WT:
    localPrediction = TAKEN;
    break;
  case ST:
    localPrediction = TAKEN;
    break;
  default:
    printf("Warning: Undefined state of entry in GSHARE BHT!\n");
    localPrediction = NOTTAKEN;
    break;
  }

  switch (bht_global[tghistoryIndex])
  {
  case WN:
    globalPrediction = NOTTAKEN;
    break;
  case SN:
    globalPrediction = NOTTAKEN;
    break;
  case WT:
    globalPrediction = TAKEN;
    break;
  case ST:
    globalPrediction = TAKEN;
    break;
  default:
    printf("Warning: Undefined state of entry in GSHARE BHT!\n");
    globalPrediction = NOTTAKEN;
    break;
  }

  switch (choice_t[tghistoryIndex])
  {
  case WN:
    choice = NOTTAKEN; // Choose Local Pattern
    break;
  case SN:
    choice = NOTTAKEN; // Choose Local Pattern
    break;
  case WT:
    choice = TAKEN; // Choose Global Pattern
    break;
  case ST:
    choice = TAKEN; // Choose Global Pattern
    break;
  default:
    printf("Warning: Undefined state of entry in GSHARE BHT!\n");
    choice = NOTTAKEN;
    break;
  }

  switch (choice)
  {
  case NOTTAKEN:
    return localPrediction;
  case TAKEN:
    return globalPrediction;
  default:
    printf("Warning: Undefined state of entry in CHOICE TABLE!\n");
    return localPrediction;
  }
  
}

void train_tournament(uint32_t pc, uint8_t outcome)
{
  // get lower ghistoryBits of pc
  // discard last two bits of PC
  uint32_t pht_entries = 1 << (phtIndexBits);
  uint32_t pc_bits = (pc & (pht_entries - 1)); // For a 32-bit Machine we don't care about lower 2-bits assuming an aligned memory
  uint64_t bhtLocalIndex = pht_local[pc_bits] & ((1 << tlhistoryBits) - 1);
  uint64_t tghistoryIndex = tghistory & ((1 << tghistoryBits) - 1); 
  uint8_t localPrediction;
  uint8_t globalPrediction;
  uint8_t prediction = tournament_predict(pc);
  uint8_t choice;
  
  switch (bht_local[bhtLocalIndex])
  {
  case WN:
    localPrediction = NOTTAKEN;
    break;
  case SN:
    localPrediction = NOTTAKEN;
    break;
  case WT:
    localPrediction = TAKEN;
    break;
  case ST:
    localPrediction = TAKEN;
    break;
  default:
    printf("Warning: Undefined state of entry in GSHARE BHT!\n");
    localPrediction = NOTTAKEN;
    break;
  }

  switch (bht_global[tghistoryIndex])
  {
  case WN:
    globalPrediction = NOTTAKEN;
    break;
  case SN:
    globalPrediction = NOTTAKEN;
    break;
  case WT:
    globalPrediction = TAKEN;
    break;
  case ST:
    globalPrediction = TAKEN;
    break;
  default:
    printf("Warning: Undefined state of entry in GSHARE BHT!\n");
    globalPrediction = NOTTAKEN;
    break;
  }
  
  if (localPrediction != globalPrediction)
  {
    if(prediction == outcome) // Correct Prediction
    {
      if(choice_t[tghistoryIndex] == WN) choice_t[tghistoryIndex] = SN;
      else if(choice_t[tghistoryIndex] == WT) choice_t[tghistoryIndex] = ST;
    }
    else // Incorrect Prediction with other entry as correct
    {
      if(choice_t[tghistoryIndex] == ST && globalPrediction != outcome) choice_t[tghistoryIndex] = WT;
      else if (choice_t[tghistoryIndex] == SN && localPrediction != outcome) choice_t[tghistoryIndex] = WN;
      else if (choice_t[tghistoryIndex] == WT && globalPrediction != outcome) choice_t[tghistoryIndex] = WN;
      else if (choice_t[tghistoryIndex] == WN && localPrediction != outcome) choice_t[tghistoryIndex] = WT;      
    }
  }

  // Update state of entry in bht based on outcome
  switch (bht_local[bhtLocalIndex])
  {
  case WN:
    bht_local[bhtLocalIndex] = (outcome == TAKEN) ? WT : SN;
    break;
  case SN:
    bht_local[bhtLocalIndex] = (outcome == TAKEN) ? WN : SN;
    break;
  case WT:
    bht_local[bhtLocalIndex] = (outcome == TAKEN) ? ST : WN;
    break;
  case ST:
    bht_local[bhtLocalIndex] = (outcome == TAKEN) ? ST : WT;
    break;
  default:
    printf("Warning: Undefined state of entry in GSHARE BHT!\n");
    break;
  }

  switch (bht_global[tghistoryIndex])
  {
  case WN:
    bht_global[tghistoryIndex] = (outcome == TAKEN) ? WT : SN;
    break;
  case SN:
    bht_global[tghistoryIndex] = (outcome == TAKEN) ? WN : SN;
    break;
  case WT:
    bht_global[tghistoryIndex] = (outcome == TAKEN) ? ST : WN;
    break;
  case ST:
    bht_global[tghistoryIndex] = (outcome == TAKEN) ? ST : WT;
    break;
  default:
    printf("Warning: Undefined state of entry in GSHARE BHT!\n");
    break;
  }

  // Update history register
  pht_local[pc_bits] = ((pht_local[pc_bits] << 1) | outcome) & ((1 << tlhistoryBits) - 1);
  tghistory = ((tghistory << 1) | outcome) & ((1 << tghistoryBits) - 1);
}

void cleanup_tournament()
{
  free(bht_local);
  free(bht_global);
  free(choice_t);
  free(pht_local);
}

void init_predictor()
{
  switch (bpType)
  {
  case STATIC:
    break;
  case GSHARE:
    init_gshare();
    break;
  case TOURNAMENT:
    init_tournament();
    break;
  case CUSTOM:
    init_piecewise_linear();
    break;
  default:
    break;
  }
}

// Make a prediction for conditional branch instruction at PC 'pc'
// Returning TAKEN indicates a prediction of taken; returning NOTTAKEN
// indicates a prediction of not taken
//
uint32_t make_prediction(uint32_t pc, uint32_t target, uint32_t direct)
{

  // Make a prediction based on the bpType
  switch (bpType)
  {
  case STATIC:
    return TAKEN;
  case GSHARE:
    return gshare_predict(pc);
  case TOURNAMENT:
    return tournament_predict(pc);
  case CUSTOM:
    return piecewise_linear_predict(pc);
  default:
    break;
  }

  // If there is not a compatable bpType then return NOTTAKEN
  return NOTTAKEN;
}

// Train the predictor the last executed branch at PC 'pc' and with
// outcome 'outcome' (true indicates that the branch was taken, false
// indicates that the branch was not taken)
//

void train_predictor(uint32_t pc, uint32_t target, uint32_t outcome, uint32_t condition, uint32_t call, uint32_t ret, uint32_t direct)
{
  if (condition)
  {
    switch (bpType)
    {
    case STATIC:
      return;
    case GSHARE:
      return train_gshare(pc, outcome);
    case TOURNAMENT:
      return train_tournament(pc, outcome);
    case CUSTOM:
      return train_piecewise_linear(pc, outcome);
    default:
      break;
    }
  }
}
