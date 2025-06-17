all: GPU3D
CC = mpicxx
NVCC = nvcc
LIBS = -lcudart -lm  -lnccl -lxml2 
FLAGS = -O3 -w
CINC = -I/usr/include/libxml2
NVINC = -I$(CUDA_PATH)/include

GPU3D : main.o kernel.o
	$(CC) -o $@ $^ $(LIBS) $(FLAGS)

%.o : %.cu
	$(NVCC) -o $@ -c $^ $(NVINC) $(FLAGS)

%.o : %.cpp
	$(CC) -o $@ -c $^ $(CINC) $(FLAGS)

clean:
	rm -f main.o kernel.o GPU3D
