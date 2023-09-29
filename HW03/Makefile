# NVCC is the CUDA compiler
NVCC = nvcc

# Compilation flags
NVCCFLAGS = -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17

all: task1 task2 task3

task1: 
	$(NVCC) $(NVCCFLAGS) task1.cu -o task1

task2: 
	$(NVCC) $(NVCCFLAGS) task2.cu -o task2

task3: 
	$(NVCC) $(NVCCFLAGS) task3.cu vscale.cu -o task3

clean:
	rm -f task1 task2 task3 task1.out task1.err task2.out task2.err task3.out task3.err results.txt
