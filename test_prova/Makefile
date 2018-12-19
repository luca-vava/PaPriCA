
#-------------------------------------------------------------------------------
#
# Makefile usage
#
# make <target> [sm=<XXX,...>] [cdp=<0|1>] [force32=<0|1>] [abi=<0|1>] [open64=<0|1>] [verbose=<0|1>] [keep=<0|1>] [quicktest=<0|1>] [quickertest=<0|1>]
#
#-------------------------------------------------------------------------------

include ../common.mk 
 
#-------------------------------------------------------------------------------
# Compiler and compilation platform
#-------------------------------------------------------------------------------

# Includes
INC += -I$(CUB_DIR) -I$(CUB_DIR)test 

# Suffix to append to each binary
SUFFIX = $(BIN_SUFFIX)_$(TEST_SUFFIX)

# Define test arch
DEFINES += -DTEST_ARCH=$(TEST_ARCH)

#-------------------------------------------------------------------------------
# Dependency Lists
#-------------------------------------------------------------------------------

PROVA1 = 		test_block_scan_raking \
	 				test_block_scan_raking_memoize \
	 				test_block_scan_warp_scans
	 					test_device_radix_sort_segmented	
ALL = 				link \
	 				test_iterator \
	 				test_allocator \
	 				test_warp_scan \
	 				test_warp_reduce \
	 				$(BLOCK_REDUCE) \
	 				$(BLOCK_SCAN) \
	 				$(BLOCK_RADIX_SORT) \
					test_block_prova \
	 				test_block_load_store \
	 				test_block_histogram \
				 	test_device_reduce \
			 		test_device_histogram \
			 		test_device_scan \
					test_device_prova \
	 				$(DEVICE_RADIX_SORT) \
					test_device_reduce_by_key\
					test_device_run_length_encode\
		 	
#-------------------------------------------------------------------------------
# make test_device_prova
#-------------------------------------------------------------------------------

test_device_scan: bin/test_device_prova_$(SUFFIX)

bin/test_device_scan_$(SUFFIX) : test_device_prova.cu $(DEPS)
	mkdir -p bin
	$(NVCC) $(DEFINES) $(SM_TARGETS) -o bin/test_device_prova_$(SUFFIX) test_device_prova.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3
		test_device_select_unique \
					test_device_select_if 

#-------------------------------------------------------------------------------
# make all
#-------------------------------------------------------------------------------

all : $(ALL)

#-------------------------------------------------------------------------------
# make run
#-------------------------------------------------------------------------------

run : 
	for i in $(ALL); do $(MEMCHECK) ./bin/$${i}_$(SUFFIX) --device=$(device) || exit 1; done

run_prova1 :
	for i in $(PROVA1); do $(MEMCHECK) ./bin/$${i}_$(SUFFIX) --device=$(device) || exit 1; done

#-------------------------------------------------------------------------------
# make link
#-------------------------------------------------------------------------------

link : bin/link_$(SUFFIX)

bin/link_$(SUFFIX) : link_a.cu link_b.cu link_main.cpp $(DEPS)
	mkdir -p bin
	$(NVCC) $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(DEFINES) $(SM_TARGETS) link_a.cu -c -o bin/link_a.obj
	$(NVCC) $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(DEFINES) $(SM_TARGETS) link_b.cu -c -o bin/link_b.obj
	$(NVCC) $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(DEFINES) $(SM_TARGETS) link_main.cpp bin/link_a.obj bin/link_b.obj -o bin/link_$(SUFFIX)
#-------------------------------------------------------------------------------
# make test_block_prova
#-------------------------------------------------------------------------------

test_prova: $(PROVA1)

