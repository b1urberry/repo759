#include <stdatomic.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>


#include "sha256.h"
#include "util.h"

void hexStringToByteArray(char *hexString, BYTE *byteArray)
{
    int i;
    for(i = 0; i < 32; i++)
    {
        char hexPair[3];
        hexPair[0] = hexString[i * 2];
        hexPair[1] = hexString[i * 2 + 1];
        hexPair[2] = '\0';

        byteArray[i] = (BYTE)strtol(hexPair, NULL, 16);
    }
}

void hexStringToIntArray(char *hexString, uint32_t *intArray)
{
    int i;
    for(i = 0; i < 8; i++)
    {
        char hexPair[3];
        hexPair[0] = hexString[i * 2];
        hexPair[1] = hexString[i * 2 + 1];
        hexPair[2] = '\0';

        intArray[i] = (uint32_t)strtol(hexPair, NULL, 16);
    }
}

uint32_t reverse32(uint32_t value)
{
    return (((value & 0x000000FF) << 24) |
            ((value & 0x0000FF00) <<  8) |
            ((value & 0x00FF0000) >>  8) |
            ((value & 0xFF000000) >> 24));
}

void uint32_to_little_endian(uint32_t value, unsigned char *buffer) {
    buffer[0] = value & 0xFF;         // Extracts the least significant byte
    buffer[1] = (value >> 8) & 0xFF;
    buffer[2] = (value >> 16) & 0xFF;
    buffer[3] = (value >> 24) & 0xFF; // Extracts the most significant byte
}


void print_bytes(const unsigned char *data, size_t dataLen, int format) 
{
    for(size_t i = 0; i < dataLen; ++i) {
        printf("%02x", data[i]);
        if (format) {
            printf(((i + 1) % 16 == 0) ? "\n" : " ");
        }
    }
    printf("\n");
}

void print_bytes_reversed(const unsigned char *data, size_t dataLen, int format) 
{
    for(size_t i = dataLen; i > 0; --i) {
        printf("%02x", data[i - 1]);
        if (format) {
            printf(((i - 1) % 16 == 0) ? "\n" : " ");
        }
    }
    printf("\n");
}


void setDifficulty (uint32_t bits, uint32_t *difficulty) 
{
    for(int i = 0; i < 8; i++)
        difficulty[i] = 0;

    bits = reverse32(bits);

    char exponent = bits & 0xff;
    uint32_t significand = bits >> 8;

    for(int i = 0; i < 3; i++)
    {
        // Endianness is reversed in this step
        unsigned char thisvalue = (unsigned char)(significand >> (8 * i));

        int index = 32 - exponent + i;
        difficulty[index / 4] = difficulty[index / 4] |
            ((unsigned int)thisvalue << (8 * (3 - (index % 4))));
    }
}

void hashBlock(uint32_t nonce, char *version, char *prev_block_hash, char *merkle_root, char *time, char *bits, uint32_t *result)
{
    BYTE *blockHeader = malloc(80 * sizeof(BYTE));

    hexStringToByteArray(version, blockHeader);
    hexStringToByteArray(prev_block_hash, blockHeader + 4);
    hexStringToByteArray(merkle_root, blockHeader + 36);
    hexStringToByteArray(time, blockHeader + 68);
    hexStringToByteArray(bits, blockHeader + 72);
    uint32_to_little_endian(nonce, blockHeader + 76);

    print_bytes((unsigned char *)blockHeader, 80, 1);

    // hash the block header
    BYTE buf[SHA256_BLOCK_SIZE];
    SHA256_CTX ctx;

    sha256_init(&ctx);
    sha256_update(&ctx, blockHeader, 80);
    sha256_final(&ctx, buf);

    // hash the hash
    sha256_init(&ctx);
    sha256_update(&ctx, buf, SHA256_BLOCK_SIZE);
    sha256_final(&ctx, buf);

    memcpy(result, buf, 32);

    free(blockHeader);
}

uint32_t mineBlock(uint32_t noncestart, char *version, char *prev_block_hash, char *merkle_root, char *time, char *bits)
{
    uint32_t difficulty[8];
    uint32_t bitsInt[1];
    hexStringToIntArray(bits, bitsInt);
    setDifficulty(bitsInt[0], difficulty);

    char solved = 0;
    uint32_t hash[8];
    uint32_t nonce = noncestart - 1;

    while(!solved)
    {
        nonce++;
        hashBlock(nonce, version, prev_block_hash, merkle_root, time, bits, hash);

        solved = 1;
        for(int i = 0; i < 8; i++)
        {
            if(hash[i] > difficulty[i])
            {
                solved = 0;
                break;
            }
        }
    }
    
    return nonce;
}