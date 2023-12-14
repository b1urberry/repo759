#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "util.h"
#include "sha256.h"

int main() 
{
    // Genesis block info
    char *version = "01000000";
    char *prev_block_hash = "0000000000000000000000000000000000000000000000000000000000000000";
    char *merkle_root = "3BA3EDFD7A7B12B27AC72C3E67768F617FC81BC3888A51323A9FB8AA4B1E5E4A";
    char *time = "29AB5F49";
    char *bits = "FFFF001D";
    
    // testing mineBlock function with nonce = 2083236793, 10 nounces before the correct nonce
    // the following line verifies the genesis block in 10 hashes
    uint32_t nonce = mineBlock(2000000000, version, prev_block_hash, merkle_root, time, bits);
    printf("found nonce:");
    printf("%d\n", nonce);

    return 0;
}
