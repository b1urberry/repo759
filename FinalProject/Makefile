# Specify the compiler
CC = gcc

# Specify the compiler flags
CFLAGS = -Wall -Wextra -g

sha256_test: sha256_test.o sha256.o
	$(CC) $(CFLAGS) -o sha256_test sha256_test.o sha256.o

serial: serial_miner.o sha256.o util.o
	$(CC) $(CFLAGS) -o serial_miner serial_miner.o sha256.o util.o 

sha256_test.o: sha256_test.c 
	$(CC) $(CFLAGS) -c sha256_test.c

sha256.o: sha256.c
	$(CC) $(CFLAGS) -c sha256.c

serial_miner.o: serial_miner.c
	$(CC) $(CFLAGS) -c serial_miner.c

util.o: util.c
	$(CC) $(CFLAGS) -c util.c

clean:
	rm -f *.o sha256_test serial_miner 

