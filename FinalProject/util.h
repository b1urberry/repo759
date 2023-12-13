#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>
#include "sha256.h"

// convert a hex string to a byte array
void hexStringToByteArray(char *hexString, BYTE *byteArray);

// convert a hex string to an int array
void hexStringToIntArray(char *hexString, uint32_t *intArray);

uint32_t reverse32(uint32_t value);

void uint32_to_little_endian(uint32_t value, unsigned char *buffer);

void print_bytes(const unsigned char *data, size_t dataLen, int format); 

void print_bytes_reversed(const unsigned char *data, size_t dataLen, int format); 

// set the difficulty
void setDifficulty(uint32_t bits, uint32_t *difficulty);

// hash a block 
void hashBlock(uint32_t nonce, char *version, char *prev_block_hash, 
    char *merkle_root, char *time, char *bits, uint32_t *result);

// mine a block
uint32_t mineBlock(uint32_t noncestart, char *version, 
    char *prev_block_hash, char *merkle_root, char *time, char *bits);


#endif // UTILS_H