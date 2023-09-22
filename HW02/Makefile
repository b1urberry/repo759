# Compiler settings
CXX = g++
CXXFLAGS = -Wall -O3 -std=c++17

# Targets
task1: scan.cpp task1.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

task2: convolution.cpp task2.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

task3: matmul.cpp task3.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

clean:
	rm -f task1 task2 task3
