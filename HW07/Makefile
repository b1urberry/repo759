# Variables
NVCC = nvcc
CPP = g++
NVCC_FLAGS = -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17
CPP_FLAGS = -Wall -O3 -std=c++17

# Targets
all: task1_thrust task1_cub task2 task3

task1_thrust:
	$(NVCC) task1_thrust.cu -o task1_thrust $(NVCC_FLAGS)

task1_cub:
	$(NVCC) task1_cub.cu -o task1_cub $(NVCC_FLAGS)

task2:
	$(NVCC) task2.cu count.cu -o task2 $(NVCC_FLAGS)

task3:
	$(CPP) task3.cpp -o task3 $(CPP_FLAGS) -fopenmp

clean:
	rm -f task1_thrust task1_cub task2 task3 *.out *.err
