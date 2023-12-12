#include <stdio.h>
#include <memory.h>
#include <stdlib.h>
#include <string.h>
#include "sha256.h"

int main(int argc, char *argv[])
{
    if (argc != 2) {
        printf("Usage: %s <string>\n", argv[0]);
        return 1;
    }

    int len = strlen(argv[1]);

    BYTE *text = malloc(len * sizeof(BYTE));
    memcpy(text, argv[1], len);
    BYTE buf[SHA256_BLOCK_SIZE];
    
    SHA256_CTX ctx;

    sha256_init(&ctx);
    sha256_update(&ctx, text, len);
    sha256_final(&ctx, buf);

    // print out the hash
    for (int i = 0; i < SHA256_BLOCK_SIZE; i++)
        printf("%02x", buf[i]);
}
