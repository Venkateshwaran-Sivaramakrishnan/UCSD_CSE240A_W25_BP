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
int tghistoryBits = 17; // Number of bits used for Global History
int tlhistoryBits = 10; // Number of bits used for Local History
int phtIndexBits = 10;

//------------------------------------//
//        Predictor Functions         //
//------------------------------------//

// Initialize the predictor
//

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
    choice_t[i] = WN;
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
  uint32_t pht_entries = 1 << (phtIndexBits + 2);
  uint32_t pc_bits = (pc & (pht_entries - 1)) >> 2; // For a 32-bit Machine we don't care about lower 2-bits assuming an aligned memory
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
  uint32_t pht_entries = 1 << (phtIndexBits + 2);
  uint32_t pc_bits = (pc & (pht_entries - 1)) >> 2; // For a 32-bit Machine we don't care about lower 2-bits assuming an aligned memory
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

  if ((globalPrediction == outcome) && (localPrediction != outcome) && (choice_t[tghistoryIndex] != ST))
  {
    choice_t[tghistoryIndex]++; 
  }

   if ((globalPrediction != outcome) && (localPrediction == outcome) && (choice_t[tghistoryIndex] != SN))
  {
    choice_t[tghistoryIndex]--; 
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
    return NOTTAKEN;
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
      return;
    default:
      break;
    }
  }
}
