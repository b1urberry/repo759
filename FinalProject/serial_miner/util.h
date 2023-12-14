#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include "sha256.h"

void hexStringToByteArray(const char* hexstr, unsigned char *output);

void hexStringToIntArray(const char *hexstr, uint32_t *outputloc);

void hexstr_to_intarray(const char* hexstr, uint32_t* outputloc);

unsigned char* hexstr_to_char(const char* hexstr);

uint32_t reverse32(uint32_t value);

void uint32_to_little_endian(uint32_t value, unsigned char *buffer);

void print_bytes(const unsigned char *data, size_t dataLen, int format); 

void print_bytes_reversed(const unsigned char *data, size_t dataLen, int format); 

// set the difficulty
void setDifficulty(uint32_t bits, uint32_t *difficulty);

void prepare_blockHeader(BYTE *blockHeader, char *version, char *prev_block_hash, char *merkle_root, char *time, char *bits);

// hash a block 
void hashBlock(uint32_t nonce, BYTE *blockHeader, uint32_t *result);

// mine a block
uint32_t mineBlock(uint32_t noncestart, char *version, 
    char *prev_block_hash, char *merkle_root, char *time, char *bits);


#endif // UTILS_H