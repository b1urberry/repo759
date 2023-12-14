# Compiler and flags
CXX = g++ 
MPICXX = mpicxx
CXXFLAGS = -Wall -O3 -std=c++17 -fno-tree-vectorize
MPIFLAGS = -fopenmp -fno-tree-vectorize -march=native -fopt-info-vec

# Executable names
TASK1 = task1
TASK2_MPI = task2
TASK2_OMP = task2_pure_omp

# Build rules
all: $(TASK1) $(TASK2_MPI) $(TASK2_OMP)

$(TASK1): task1.cpp optimize.cpp
	$(CXX) task1.cpp optimize.cpp $(CXXFLAGS) -o $(TASK1)

$(TASK2_MPI): task2.cpp reduce.cpp
	$(MPICXX) task2.cpp reduce.cpp $(CXXFLAGS) $(MPIFLAGS) -o $(TASK2_MPI)

$(TASK2_OMP): task2_pure_omp.cpp reduce.cpp
	$(CXX) task2_pure_omp.cpp reduce.cpp $(CXXFLAGS) $(MPIFLAGS) -o $(TASK2_OMP)

clean:
	rm -f $(TASK1) $(TASK2_MPI) $(TASK2_OMP) *.err *.out *.txt

.PHONY: all clean
