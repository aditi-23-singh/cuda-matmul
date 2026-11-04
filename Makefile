# RTX 3050 (Laptop) is Ampere -> compute capability 8.6
ARCH ?= sm_86
NVCC ?= nvcc

matmul: src/matmul.cu
	$(NVCC) -O3 -arch=$(ARCH) -lineinfo -o $@ $< -lcublas

bench: matmul
	python3 bench.py

clean:
	rm -rf matmul matmul.exe results

.PHONY: bench clean
