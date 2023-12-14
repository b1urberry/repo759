CC = gcc
NVCC = nvcc
CPP = g++

CFLAGS = -Wall -Wextra -g
NVCC_FLAGS = -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17
CPP_FLAGS = -Wall -O3 -std=c++17


main: 
	$(NVCC) main.cu -o main $(NVCC_FLAGS) 

clean:
	rm -f *.o *.out *.err main

