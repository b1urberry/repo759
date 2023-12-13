#include <cstdio>
#include <cstdlib>
#include <stdbool.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

extern "C" {
	#include "sha256.h"
	#include "util.h"
}

__global__ void kernel_sha256d();

int main(){
    // Genesis block info
    char *version = "01000000";
    char *prev_block_hash = "0000000000000000000000000000000000000000000000000000000000000000";
    char *merkle_root = "3BA3EDFD7A7B12B27AC72C3E67768F617FC81BC3888A51323A9FB8AA4B1E5E4A";
    char *time = "29AB5F49";
    char *bits = "FFFF001D";


}