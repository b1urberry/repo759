CC = gcc
NVCC = nvcc
CPP = g++

CFLAGS = -Wall -Wextra -g
NVCC_FLAGS = -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17
CPP_FLAGS = -Wall -O3 -std=c++17

all: serial_miner

# main:
# 	$(NVCC) main.cu kernel.cu sha256.cu util.cu -o main $(NVCC_FLAGS)

serial: 
	$(CC) serial_miner.c sha256.c util.c -o serial $(CFLAGS) 

# sha256_test: sha256_test.o sha256.o
# 	$(CC) $(CFLAGS) -o sha256_test sha256_test.o sha256.o

# serial: serial_miner.o sha256.o util.o
# 	$(CC) $(CFLAGS) -o serial_miner serial_miner.o sha256.o util.o 

# sha256_test.o: sha256_test.c 
# 	$(CC) $(CFLAGS) -c sha256_test.c

# sha256.o: sha256.c
# 	$(CC) $(CFLAGS) -c sha256.c

# serial_miner.o: serial_miner.c
# 	$(CC) $(CFLAGS) -c serial_miner.c

# util.o: util.c
# 	$(CC) $(CFLAGS) -c util.c

# main:
# 	$(NVCC) main.cu kernel.cu -o main $(NVCC_FLAGS)

clean:
	rm -rf *.o *.out *.err *.dSYM main serial serial 

