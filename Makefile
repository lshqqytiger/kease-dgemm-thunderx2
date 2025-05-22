KERNEL_PATH := ./kernel
OUTPUT_PATH := ./out

KERNELS := $(shell find $(KERNEL_PATH) -name '*.c')
OUTPUTS_EXECUTABLE := $(KERNELS:$(KERNEL_PATH)/%.c=$(OUTPUT_PATH)/%.out)
OUTPUTS_SHARED := $(KERNELS:$(KERNEL_PATH)/%.c=$(OUTPUT_PATH)/%.so)

# compiler
CC     := armclang

# default c flags
CFLAGS := -O3 -mcpu=native
CFLAGS += -fopenmp
CFLAGS += -armpl -lm -lnuma
CFLAGS += -Wall -Werror

all: $(OUTPUTS_EXECUTABLE)

shared: $(OUTPUTS_SHARED)

# executables
$(OUTPUT_PATH)/cblas.out: dgemm_flops.c $(KERNEL_PATH)/cblas.c
	$(CC) -o $@ $^ $(CFLAGS) -DKERNEL=\"cblas\"

$(OUTPUT_PATH)/play.out: dgemm_flops.c $(KERNEL_PATH)/play.c
	$(CC) -o $@ $^ $(CFLAGS) -DKERNEL=\"play\"

$(OUTPUT_PATH)/%.out: dgemm_flops.c $(KERNEL_PATH)/%.c
	$(CC) -o $@ $^ $(CFLAGS) -DKERNEL=\"$@\" -DVERIFY

# shared objects
$(OUTPUT_PATH)/cblas.so: $(KERNEL_PATH)/cblas.c
	$(CC) -shared -o $@ $^ $(CFLAGS) -DKERNEL=\"cblas\"

$(OUTPUT_PATH)/play.so: $(KERNEL_PATH)/play.c
	$(CC) -shared -o $@ $^ $(CFLAGS) -DKERNEL=\"play\"

$(OUTPUT_PATH)/%.so: $(KERNEL_PATH)/%.c
	$(CC) -shared -o $@ $^ $(CFLAGS) -DKERNEL=\"$@\" -DVERIFY

clean:
	rm -f $(OUTPUT_PATH)/*.out
	rm -f $(OUTPUT_PATH)/*.so
