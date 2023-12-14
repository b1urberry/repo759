#include <cstdio>
#include <cstdlib>
#include <stdbool.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdlib.h>
#include <memory.h>
#include <stddef.h>

// sha256 macros 
#define SHA256_BLOCK_SIZE 32            // SHA256 outputs a 32 byte digest
#define ROTLEFT(a,b) (((a) << (b)) | ((a) >> (32-(b))))
#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))
#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

// data types
typedef unsigned char BYTE;             // 8-bit byte
typedef unsigned int  WORD;             // 32-bit word, change to "long" for 16-bit machines

// convenience structure passed to all sha256 functions
typedef struct {
	BYTE data[64];
	WORD datalen;
	unsigned long long bitlen;
	WORD state[8];
} SHA256_CTX; 

__device__ WORD k[64] = {
	0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
	0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
	0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
	0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
	0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
	0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
	0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
	0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

// sha256 hashing functions
__device__ void sha256_transform(SHA256_CTX *ctx, const BYTE data[]);
__device__ void sha256_init(SHA256_CTX *ctx);
__device__ void sha256_update(SHA256_CTX *ctx, const BYTE data[], size_t len);
__device__ void sha256_final(SHA256_CTX *ctx, BYTE hash[]);

// global variables to store the correct nonce and hash
__managed__ uint32_t correct_nonce;
__managed__ uint32_t correct_hash[8];
__managed__ uint32_t difficulty[8];

// miner kernel call: each thread verifies one nonce
__global__ void mine_kernel(uint32_t nonceStart, BYTE *blockHeader);
// make kernel call
void launch_mine(int threads_per_block, uint32_t nonceStart, BYTE *blockHeader);

// utility functions
__device__ void uint32_to_little_endian(uint32_t value, unsigned char *buffer);
void hexStringToByteArray(const char* hexstr, unsigned char *output);
void hexStringToIntArray(const char *hexstr, uint32_t *outputloc);
void hexstr_to_intarray(const char* hexstr, uint32_t* outputloc);
unsigned char* hexstr_to_char(const char* hexstr);
uint32_t reverse32(uint32_t value);
void print_bytes(const unsigned char *data, size_t dataLen, int format); 
void print_bytes_reversed(const unsigned char *data, size_t dataLen, int format); 
void setDifficulty(uint32_t bits, uint32_t *difficulty);









int main(){
    // Genesis block info
    const char *version = "01000000";
    const char *prev_block_hash = "0000000000000000000000000000000000000000000000000000000000000000";
    const char *merkle_root = "3BA3EDFD7A7B12B27AC72C3E67768F617FC81BC3888A51323A9FB8AA4B1E5E4A";
    const char *time = "29AB5F49";
    const char *nbits = "FFFF001D";

    uint32_t h_correct_nonce;
    unsigned h_correct_hash[8];

	// prepare the block header (except for the nonce of each thread) for hashing
    BYTE *blockHeader = (BYTE *)malloc(80 * sizeof(BYTE));
    hexStringToByteArray(version, blockHeader);
    hexStringToByteArray(prev_block_hash, blockHeader + 4);
    hexStringToByteArray(merkle_root, blockHeader + 36);
    hexStringToByteArray(time, blockHeader + 68);
    hexStringToByteArray(nbits, blockHeader + 72);

	print_bytes(blockHeader, 80, 1);
    
    // First convert bits to a uint32_t, then convert this to a difficulty
    uint32_t bits[1];
    hexstr_to_intarray(nbits, bits);
    setDifficulty(*bits, difficulty);
	
    // launch_mine(512, 2083236393, blockHeader);

    // cudaMemcpyFromSymbol(&h_correct_nonce, correct_nonce, sizeof(uint32_t));
    // cudaMemcpyFromSymbol(h_correct_hash, correct_hash, sizeof(uint32_t) * 8);
    
    // printf("%d\n", h_correct_nonce);
    // print_bytes_reversed((unsigned char *)h_correct_hash, 32, 1);
}










/*********************** FUNCTION DEFINITIONS ***********************/
__device__ void uint32_to_little_endian(uint32_t value, unsigned char *buffer) {
    buffer[0] = value & 0xFF;         // Extracts the least significant byte
    buffer[1] = (value >> 8) & 0xFF;
    buffer[2] = (value >> 16) & 0xFF;
    buffer[3] = (value >> 24) & 0xFF; // Extracts the most significant byte
}

__global__ void mine_kernel(uint32_t nonce_blockStart, BYTE *blockHeader) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t nonce = nonce_blockStart + threadId;

    uint32_t hash[8];

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

    memcpy(hash, buf, 32);

    for(int i = 0; i < 8; i++)
    {
        if(hash[7-i] < difficulty[i])
        {
            // atomicExch(&correct_nonce, nonce);
			correct_nonce = nonce;
            // Copy the hash to correct_hash
            for (int i = 0; i < 8; i++) {
				correct_hash[i] = hash[i];
                // atomicExch(&correct_hash[i], hash[i]);
            }
            return;
        }
        else if(hash[7-i] > difficulty[i])
            break;
        // And if they're equal, we keep going!
    }
}

// Define the launch_mine function
void launch_mine(int threads_per_block, uint32_t nonceStart, BYTE *blockHeader) {

    // Calculate the number of blocks needed
    // int numBlocks = (MAX_NONCE - nonce + threads_per_block - 1) / threads_per_block;
    int numBlocks = 1;

    // Launch the mine_kernel
    mine_kernel<<<numBlocks, threads_per_block>>>(nonceStart, blockHeader);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
}

__device__ void sha256_transform(SHA256_CTX *ctx, const BYTE data[])
{
	WORD a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];

	for (i = 0, j = 0; i < 16; ++i, j += 4)
		m[i] = (data[j] << 24) | (data[j + 1] << 16) | (data[j + 2] << 8) | (data[j + 3]);
	for ( ; i < 64; ++i)
		m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];

	a = ctx->state[0];
	b = ctx->state[1];
	c = ctx->state[2];
	d = ctx->state[3];
	e = ctx->state[4];
	f = ctx->state[5];
	g = ctx->state[6];
	h = ctx->state[7];

	for (i = 0; i < 64; ++i) {
		t1 = h + EP1(e) + CH(e,f,g) + k[i] + m[i];
		t2 = EP0(a) + MAJ(a,b,c);
		h = g;
		g = f;
		f = e;
		e = d + t1;
		d = c;
		c = b;
		b = a;
		a = t1 + t2;
	}

	ctx->state[0] += a;
	ctx->state[1] += b;
	ctx->state[2] += c;
	ctx->state[3] += d;
	ctx->state[4] += e;
	ctx->state[5] += f;
	ctx->state[6] += g;
	ctx->state[7] += h;
}

__device__ void sha256_init(SHA256_CTX *ctx)
{
	ctx->datalen = 0;
	ctx->bitlen = 0;
	ctx->state[0] = 0x6a09e667;
	ctx->state[1] = 0xbb67ae85;
	ctx->state[2] = 0x3c6ef372;
	ctx->state[3] = 0xa54ff53a;
	ctx->state[4] = 0x510e527f;
	ctx->state[5] = 0x9b05688c;
	ctx->state[6] = 0x1f83d9ab;
	ctx->state[7] = 0x5be0cd19;
}

__device__ void sha256_update(SHA256_CTX *ctx, const BYTE data[], size_t len)
{
	WORD i;

	for (i = 0; i < len; ++i) {
		ctx->data[ctx->datalen] = data[i];
		ctx->datalen++;
		if (ctx->datalen == 64) {
			sha256_transform(ctx, ctx->data);
			ctx->bitlen += 512;
			ctx->datalen = 0;
		}
	}
}

__device__ void sha256_final(SHA256_CTX *ctx, BYTE hash[])
{
	WORD i;

	i = ctx->datalen;

	// Pad whatever data is left in the buffer.
	if (ctx->datalen < 56) {
		ctx->data[i++] = 0x80;
		while (i < 56)
			ctx->data[i++] = 0x00;
	}
	else {
		ctx->data[i++] = 0x80;
		while (i < 64)
			ctx->data[i++] = 0x00;
		sha256_transform(ctx, ctx->data);
		memset(ctx->data, 0, 56);
	}

	// Append to the padding the total message's length in bits and transform.
	ctx->bitlen += ctx->datalen * 8;
	ctx->data[63] = ctx->bitlen;
	ctx->data[62] = ctx->bitlen >> 8;
	ctx->data[61] = ctx->bitlen >> 16;
	ctx->data[60] = ctx->bitlen >> 24;
	ctx->data[59] = ctx->bitlen >> 32;
	ctx->data[58] = ctx->bitlen >> 40;
	ctx->data[57] = ctx->bitlen >> 48;
	ctx->data[56] = ctx->bitlen >> 56;
	sha256_transform(ctx, ctx->data);

	// Since this implementation uses little endian byte ordering and SHA uses big endian,
	// reverse all the bytes when copying the final state to the output hash.
	for (i = 0; i < 4; ++i) {
		hash[i]      = (ctx->state[0] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 4]  = (ctx->state[1] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 8]  = (ctx->state[2] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 12] = (ctx->state[3] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 16] = (ctx->state[4] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 20] = (ctx->state[5] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 24] = (ctx->state[6] >> (24 - i * 8)) & 0x000000ff;
		hash[i + 28] = (ctx->state[7] >> (24 - i * 8)) & 0x000000ff;
	}
}


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
        // uint32_t a = (uint32_t)bytes[i * 4 + 3] << 24;
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


void setDifficulty(uint32_t bits, uint32_t *difficulty) {
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