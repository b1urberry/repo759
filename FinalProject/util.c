#include <stdatomic.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>


#include "sha256.h"
#include "util.h"

void hexStringToByteArray(const char *hexstr, unsigned char *output) {
    while (*hexstr && hexstr[1]) {
        sscanf(hexstr, "%2hhx", output++);
        hexstr += 2;
    }
}

unsigned char* hexstr_to_char(const char* hexstr)
{
    size_t len = strlen(hexstr);
    size_t final_len = len / 2;
    unsigned char* chars = (unsigned char*)malloc((final_len + 1));
    for(size_t i = 0, j = 0; j < final_len; i += 2, j++)
        chars[j] = (hexstr[i] % 32 + 9) % 25 * 16 + (hexstr[i+1] % 32 + 9) % 25;
    chars[final_len] = '\0';
    return chars;
}

void hexstr_to_intarray(const char* hexstr, uint32_t* outputloc)
{
    size_t len = strlen(hexstr);
    size_t intlen = (len + 7) / 8; // +7 ensures that we do a ceiling divide
    unsigned char* bytes = hexstr_to_char(hexstr);

    for(size_t i = 0; i < intlen; i++)
    {
        uint32_t a = (uint32_t)bytes[i * 4 + 3] << 24;
        *(outputloc + i) = ((uint32_t)bytes[i * 4])
            + ((uint32_t)bytes[i * 4 + 1] << 8)
            + ((uint32_t)bytes[i * 4 + 2] << 16)
            + ((uint32_t)bytes[i * 4 + 3] << 24);
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


void setDifficulty(uint32_t bits, uint32_t *difficulty) 
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

    // print_bytes((unsigned char *)blockHeader, 80, 1);

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

uint32_t mineBlock(uint32_t noncestart, char *version, char *prev_block_hash, char *merkle_root, char *time, char *nbits)
{
    // First convert bits to a uint32_t, then convert this to a difficulty
    uint32_t difficulty[8];
    uint32_t bits[1];
    hexstr_to_intarray(nbits, bits);
    setDifficulty(*bits, difficulty);

    char solved = 0;
    uint32_t hash[8];
    uint32_t nonce = noncestart-1;

    while(1)
    {
        nonce++;

        hashBlock(nonce, version, prev_block_hash, merkle_root, time, nbits, hash);

        // print out the hash with the current nonce, for testing
        // print_bytes_reversed((unsigned char *)hash, 32, 1);

        for(int i = 0; i < 8; i++)
        {
            if(hash[7-i] < difficulty[i])
            {
                solved = 1;
                return nonce;
            }
            else if(hash[7-i] > difficulty[i])
                break;
            // And if they're equal, we keep going!
        }
    }

}