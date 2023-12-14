CXX = g++
MPICXX = mpicxx
CLANGXX = clang++
CXXFLAGS = -Wall -O3 -std=c++17
OMPFLAGS = -fopenmp
MPIFLAGS = 
CLANGFLAGS = -O3 -fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target=nvptx64-nvidia-cuda --offload-arch=sm_61 -march=native

all: task1 task2 task3 task4

task1: task1.cpp cluster.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) task1.cpp cluster.cpp -o task1

task2: task2.cpp montecarlo.cpp
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -fno-tree-vectorize -march=native -fopt-info-vec task2.cpp montecarlo.cpp -o task2

task3: task3.cpp
	$(MPICXX) $(MPIFLAGS) $(CXXFLAGS) task3.cpp -o task3

task4: task4.cpp convolve.cpp
	$(CLANGXX) $(CLANGFLAGS) convolve.cpp task4.cpp -o task4

clean:
	rm -f task1 task2 task3 task4 *.out *.err
	
