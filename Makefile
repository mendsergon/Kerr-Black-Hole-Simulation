# Kerr Black Hole Simulation — Linux Makefile

CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++17 -O3
LIBS = -lglfw -lGL -lGLU -lOpenCL -lm
TARGET = blackhole
SRCS = main.cpp blackhole.cpp
OBJS = $(SRCS:.cpp=.o)

# OpenMP for CPU fallback parallelization
CXXFLAGS += -fopenmp
LIBS += -fopenmp

all: $(TARGET)
	@echo "Build complete: ./$(TARGET)"

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $(TARGET) $(CXXFLAGS) $(LIBS)

%.o: %.cpp blackhole.h
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

run: all
	./$(TARGET)

debug: CXXFLAGS = -Wall -Wextra -std=c++17 -g -O0 -fopenmp
debug: clean $(TARGET)
	@echo "Debug build complete: ./$(TARGET)"

gpu-info:
	@echo "=== OpenCL Devices ==="
	@clinfo --list 2>/dev/null || echo "clinfo not installed — run: make deps"
	@echo ""
	@echo "=== OpenGL Info ==="
	@glxinfo 2>/dev/null | head -20 || echo "glxinfo not installed"

deps:
	@echo "Installing dependencies..."
	@if [ -f /etc/arch-release ]; then \
		sudo pacman -S --needed glfw mesa glu glm opencl-headers ocl-icd; \
		echo ""; \
		echo "For AMD GPU:    sudo pacman -S rocm-opencl-runtime"; \
		echo "For NVIDIA GPU: sudo pacman -S opencl-nvidia"; \
		echo "For Intel GPU:  sudo pacman -S intel-compute-runtime"; \
	elif [ -f /etc/debian_version ]; then \
		sudo apt install libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev libglm-dev opencl-headers ocl-icd-opencl-dev; \
		echo ""; \
		echo "For AMD GPU:    sudo apt install mesa-opencl-icd"; \
		echo "For NVIDIA GPU: sudo apt install nvidia-opencl-dev"; \
		echo "For Intel GPU:  sudo apt install intel-opencl-icd"; \
	elif [ -f /etc/fedora-release ]; then \
		sudo dnf install glfw-devel mesa-libGL-devel mesa-libGLU-devel glm-devel opencl-headers ocl-icd-devel; \
		echo ""; \
		echo "For AMD GPU:    sudo dnf install mesa-libOpenCL"; \
	else \
		echo "Unknown distro. Install manually: glfw, mesa, glu, glm, opencl-headers, ocl-icd"; \
	fi

.PHONY: all clean run debug gpu-info deps
