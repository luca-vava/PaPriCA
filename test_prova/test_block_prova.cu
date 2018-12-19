
/******************************************************************************
 * Test of BlockScan utilities
 ******************************************************************************/

// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <iostream>
#include <limits>
#include <typeinfo>

#include <cub/block/block_scan.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_allocator.cuh>

#include "test_util.h"


using namespace cub;

//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose       = false;
int                     g_repeat        = 0;
CachingDeviceAllocator  g_allocator(true);


/**
 * Primitive variant to test
 */
enum TestMode
{
    BASIC,
    AGGREGATE,
    PREFIX,
};


/**
 * Scan mode to test
 */
enum ScanMode
{
    EXCLUSIVE,
    INCLUSIVE
};


/**
 * \brief WrapperFunctor (for precluding test-specialized dispatch to *Sum variants)
 */
template<typename OpT>
struct WrapperFunctor
{
    OpT op;

    WrapperFunctor(OpT op) : op(op) {}

    template <typename T>
    __host__ __device__ __forceinline__ T operator()(const T &a, const T &b) const
    {
        return op(a, b);
    }
};

/**
 * Stateful prefix functor
 */
template <
    typename T,
    typename ScanOpT>
struct BlockPrefixCallbackOp
{
    int     linear_tid;
    T       prefix;
    ScanOpT  scan_op;

    __device__ __forceinline__
    BlockPrefixCallbackOp(int linear_tid, T prefix, ScanOpT scan_op) :
        linear_tid(linear_tid),
        prefix(prefix),
        scan_op(scan_op)
    {}

    __device__ __forceinline__
    T operator()(T block_aggregate)
    {
        // For testing purposes
        T retval = (linear_tid == 0) ? prefix  : T();
        prefix = scan_op(prefix, block_aggregate);
        return retval;
    }
};


//---------------------------------------------------------------------
// Exclusive scan
//---------------------------------------------------------------------

/// Exclusive scan (BASIC, 1)
template <typename BlockScanT, typename T, typename ScanOpT, typename PrefixCallbackOp, typename IsPrimitiveT>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[1], T &initial_value, ScanOpT &scan_op, T &block_aggregate, PrefixCallbackOp &prefix_op,
    Int2Type<EXCLUSIVE> scan_mode, Int2Type<BASIC> test_mode, IsPrimitiveT is_primitive)
{
    block_scan.ExclusiveScan(data[0], data[0], initial_value, scan_op);
}


//---------------------------------------------------------------------
// Exclusive sum
//---------------------------------------------------------------------

/// Exclusive sum (BASIC, 1)
template <typename BlockScanT, typename T, typename PrefixCallbackOp>
__device__ __forceinline__ void DeviceTest(
    BlockScanT &block_scan, T (&data)[1], T &initial_value, Sum &scan_op, T &block_aggregate, PrefixCallbackOp &prefix_op,
    Int2Type<EXCLUSIVE> scan_mode, Int2Type<BASIC> test_mode, Int2Type<true> is_primitive)
{
    block_scan.ExclusiveSum(data[0], data[0]);
}


//---------------------------------------------------------------------
// Test kernels
//---------------------------------------------------------------------

/**
 * BlockScan test kernel.
 */
template <
    int                 BLOCK_DIM_X,
    int                 BLOCK_DIM_Y,
    int                 BLOCK_DIM_Z,
    int                 ITEMS_PER_THREAD,
    ScanMode            SCAN_MODE,
    TestMode            TEST_MODE,
    BlockScanAlgorithm  ALGORITHM,
    typename            T,
    typename            ScanOpT>
__launch_bounds__ (BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z)
__global__ void BlockScanKernel(
    T                   *d_in,
    T                   *d_out,
    T                   *d_aggregate,
    ScanOpT              scan_op,
    T                   initial_value,
    clock_t             *d_elapsed)
{
    const int BLOCK_THREADS     = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
    const int TILE_SIZE         = BLOCK_THREADS * ITEMS_PER_THREAD;

    // Parameterize BlockScan type for our thread block
    typedef BlockScan<T, BLOCK_DIM_X, ALGORITHM, BLOCK_DIM_Y, BLOCK_DIM_Z> BlockScanT;

    // Allocate temp storage in shared memory
    __shared__ typename BlockScanT::TempStorage temp_storage;

    int linear_tid = RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z);

    // Per-thread tile data
    T data[ITEMS_PER_THREAD];
    LoadDirectBlocked(linear_tid, d_in, data);

    __threadfence_block();      // workaround to prevent clock hoisting
    clock_t start = clock();
    __threadfence_block();      // workaround to prevent clock hoisting

    // Test scan
    T                                   block_aggregate;
    BlockScanT                          block_scan(temp_storage);
    BlockPrefixCallbackOp<T, ScanOpT>   prefix_op(linear_tid, initial_value, scan_op);

    DeviceTest(block_scan, data, initial_value, scan_op, block_aggregate, prefix_op,
        Int2Type<SCAN_MODE>(), Int2Type<TEST_MODE>(), Int2Type<Traits<T>::PRIMITIVE>());

    // Stop cycle timer
    __threadfence_block();      // workaround to prevent clock hoisting
    clock_t stop = clock();
    __threadfence_block();      // workaround to prevent clock hoisting

    // Store output
    StoreDirectBlocked(linear_tid, d_out, data);

    // Store block_aggregate
    if (TEST_MODE != BASIC)
        d_aggregate[linear_tid] = block_aggregate;

    // Store prefix
    if (TEST_MODE == PREFIX)
    {
        if (linear_tid == 0)
            d_out[TILE_SIZE] = prefix_op.prefix;
    }

    // Store time
    if (linear_tid == 0)
        *d_elapsed = (start > stop) ? start - stop : stop - start;
}



//---------------------------------------------------------------------
// Host utility subroutines
//---------------------------------------------------------------------

/**
 * Initialize exclusive-scan problem (and solution)
 */
template <typename T, typename ScanOpT>
T Initialize(
    GenMode     gen_mode,
    T           *h_in,
    T           *h_reference,
    int         num_items,
    ScanOpT     scan_op,
    T           initial_value,
    Int2Type<EXCLUSIVE>)
{
    InitValue(gen_mode, h_in[0], 0);

    T block_aggregate   = h_in[0];
    h_reference[0]      = initial_value;
    T inclusive         = scan_op(initial_value, h_in[0]);

    for (int i = 1; i < num_items; ++i)
    {
        InitValue(gen_mode, h_in[i], i);
        h_reference[i] = inclusive;
        inclusive = scan_op(inclusive, h_in[i]);
        block_aggregate = scan_op(block_aggregate, h_in[i]);
    }

    return block_aggregate;
}


/**
 * Test thread block scan.
 */
template <
    int                 BLOCK_DIM_X,
    int                 BLOCK_DIM_Y,
    int                 BLOCK_DIM_Z,
    int                 ITEMS_PER_THREAD,
    ScanMode            SCAN_MODE,
    TestMode            TEST_MODE,
    BlockScanAlgorithm  ALGORITHM,
    typename            ScanOpT,
    typename            T>
void Test(
    GenMode             gen_mode,
    ScanOpT             scan_op,
    T                   initial_value)
{
    // Check size of smem storage for the target arch to make sure it will fit
    typedef BlockScan<T, BLOCK_DIM_X, ALGORITHM, BLOCK_DIM_Y, BLOCK_DIM_Z> BlockScanT;

    enum
    {
#if defined(SM100) || defined(SM110) || defined(SM130)
        sufficient_smem         = (sizeof(typename BlockScanT::TempStorage)     <= 16 * 1024),
        sufficient_threads      = ((BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z)    <= 512),
#else
        sufficient_smem         = (sizeof(typename BlockScanT::TempStorage)     <= 16 * 1024),
        sufficient_threads      = ((BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z)    <= 1024),
#endif

#if defined(_WIN32) || defined(_WIN64)
        // Accommodate ptxas crash bug (access violation) on Windows
        special_skip            = ((TEST_ARCH <= 130) && (Equals<T, TestBar>::VALUE) && (BLOCK_DIM_Z > 1)),
#else
        special_skip            = false,
#endif
        sufficient_resources    = (sufficient_smem && sufficient_threads && !special_skip),
    };

    Test<BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z, ITEMS_PER_THREAD, SCAN_MODE, TEST_MODE, ALGORITHM>(
        gen_mode, scan_op, initial_value, Int2Type<sufficient_resources>());
}



/**
 * Main
 */
int main(int argc, char** argv)
{
    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("repeat", g_repeat);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--device=<device-id>] "
            "[--repeat=<repetitions of entire test suite>]"
            "[--v] "
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());

#ifdef QUICK_TEST

    Test<128, 1, 1, 1, EXCLUSIVE, AGGREGATE, BLOCK_SCAN_WARP_SCANS>(UNIFORM, Sum(), int(0));

    // Compile/run quick tests
    Test<128, 1, 1, 4, EXCLUSIVE, AGGREGATE, BLOCK_SCAN_WARP_SCANS>(UNIFORM, Sum(), int(0));
    Test<128, 1, 1, 4, EXCLUSIVE, AGGREGATE, BLOCK_SCAN_RAKING>(UNIFORM, Sum(), int(0));
    Test<128, 1, 1, 4, EXCLUSIVE, AGGREGATE, BLOCK_SCAN_RAKING_MEMOIZE>(UNIFORM, Sum(), int(0));

    Test<128, 1, 1, 2, INCLUSIVE, PREFIX, BLOCK_SCAN_RAKING>(INTEGER_SEED, Sum(), TestFoo::MakeTestFoo(17, 21, 32, 85));
    Test<128, 1, 1, 1, EXCLUSIVE, AGGREGATE, BLOCK_SCAN_WARP_SCANS>(UNIFORM, Sum(), make_longlong4(17, 21, 32, 85));


#else

    // Compile/run thorough tests
    for (int i = 0; i <= g_repeat; ++i)
    {
        // Run tests for different thread block sizes
        Test<17>();
        Test<32>();
        Test<62>();
        Test<65>();
//            Test<96>();             // TODO: file bug for UNREACHABLE error for Test<96, 9, BASIC, BLOCK_SCAN_RAKING>(UNIFORM, Sum(), NullType(), make_ulonglong2(17, 21));
        Test<128>();
    }

#endif

    return 0;
}

