# Makefile for scenario_mpc C++ library
# This is an alternative to CMake for simple builds

CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -I/usr/include/eigen3 -Iinclude
LDFLAGS =

# Detect Eigen path on different systems
EIGEN_PATH := $(shell pkg-config --cflags eigen3 2>/dev/null || echo "-I/usr/include/eigen3")
CXXFLAGS += $(EIGEN_PATH)

# Source files
SRC_DIR = src
INCLUDE_DIR = include
BUILD_DIR = build
TEST_DIR = tests
EXAMPLE_DIR = examples

SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(SRCS:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

# Targets
.PHONY: all clean test example

all: $(BUILD_DIR)/libscenario_mpc.a test_core example_mpc

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Create static library
$(BUILD_DIR)/libscenario_mpc.a: $(OBJS)
	ar rcs $@ $^

# Build test executable
test_core: $(BUILD_DIR)/libscenario_mpc.a $(TEST_DIR)/test_core.cpp
	$(CXX) $(CXXFLAGS) $(TEST_DIR)/test_core.cpp -L$(BUILD_DIR) -lscenario_mpc -o $(BUILD_DIR)/$@

# Build example executable
example_mpc: $(BUILD_DIR)/libscenario_mpc.a $(EXAMPLE_DIR)/example_mpc.cpp
	$(CXX) $(CXXFLAGS) $(EXAMPLE_DIR)/example_mpc.cpp -L$(BUILD_DIR) -lscenario_mpc -o $(BUILD_DIR)/$@

# Run tests
test: test_core
	./$(BUILD_DIR)/test_core

# Run example
example: example_mpc
	./$(BUILD_DIR)/example_mpc

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR)

# Help
help:
	@echo "Available targets:"
	@echo "  all       - Build library, tests, and examples"
	@echo "  test      - Build and run tests"
	@echo "  example   - Build and run example"
	@echo "  clean     - Remove build artifacts"
	@echo ""
	@echo "Requirements:"
	@echo "  - g++ with C++17 support"
	@echo "  - Eigen3 library"
	@echo ""
	@echo "Install dependencies (Ubuntu):"
	@echo "  sudo apt-get install g++ libeigen3-dev"
