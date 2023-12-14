# Compiler
CXX = g++

# Compiler Flags
CXXFLAGS = -Wall -O3 -std=c++17 -fopenmp

# Build Targets and their respective source files
TARGET1 = task1
SRC1 = task1.cpp matmul.cpp

TARGET2 = task2
SRC2 = task2.cpp convolution.cpp

TARGET3 = task3
SRC3 = task3.cpp msort.cpp

# All Targets (default)
all: $(TARGET1) $(TARGET2) $(TARGET3)

# Build rules for each target
$(TARGET1):
	$(CXX) $(CXXFLAGS) $(SRC1) -o $(TARGET1)

$(TARGET2):
	$(CXX) $(CXXFLAGS) $(SRC2) -o $(TARGET2)

$(TARGET3):
	$(CXX) $(CXXFLAGS) $(SRC3) -o $(TARGET3)

# Phony targets for cleaning (i.e., removing compiled files)
.PHONY: clean

clean:
	rm -f $(TARGET1) $(TARGET2) $(TARGET3) *.out *.err
