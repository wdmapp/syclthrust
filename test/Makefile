CPP = clang++
CPP_FLAGS = -fsycl -I../include

BUILD_DIR=build

SYCL_SOURCES = $(wildcard *.cxx)
EXES = $(addprefix $(BUILD_DIR)/,$(basename $(SYCL_SOURCES)))

.PHONY: all
all: $(EXES)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/% : %.cxx | $(BUILD_DIR)
	@echo "Compiling "$<
	$(CPP) $(CPP_FLAGS) -std=c++17 -o $@ $<


.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)/*
