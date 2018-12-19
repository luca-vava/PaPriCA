// Ensure printing of CUDA runtime errors to console
#define CUB_STDERR

#include <stdio.h>
#include <typeinfo>

#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cub/util_allocator.cuh>
#include <cub/iterator/constant_input_iterator.cuh>
#include <cub/iterator/discard_output_iterator.cuh>
#include <cub/device/device_scan.cuh>

#include "test_util.h"

using namespace cub;


//---------------------------------------------------------------------
// Globals, constants and typedefs
//---------------------------------------------------------------------

bool                    g_verbose           = false;
int                     g_timing_iterations = 0;
int                     g_repeat            = 0;
double                  g_device_giga_bandwidth;
CachingDeviceAllocator  g_allocator(true);

// Dispatch types
enum Backend
{
    CUB,        // CUB method
    THRUST,     // Thrust method
    CDP,        // GPU-based (dynamic parallelism) dispatch to CUB method
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


//---------------------------------------------------------------------
// Dispatch to different CUB DeviceScan entrypoints
//---------------------------------------------------------------------

/**
 * Dispatch to exclusive scan entrypoint
 */
template <typename IsPrimitiveT, typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, typename InitialValueT, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>       dispatch_to,
    IsPrimitiveT        is_primitive,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    ScanOpT             scan_op,
    InitialValueT       initial_value,
    OffsetT             num_items,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, scan_op, initial_value, num_items, stream, debug_synchronous);
    }
    return error;
}


/**
 * Dispatch to exclusive sum entrypoint
 */
/**
template <typename InputIteratorT, typename OutputIteratorT, typename InitialValueT, typename OffsetT>
CUB_RUNTIME_FUNCTION __forceinline__
cudaError_t Dispatch(
    Int2Type<CUB>       dispatch_to,
    Int2Type<true>      is_primitive,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t&             temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    Sum                 scan_op,
    InitialValueT       initial_value,
    OffsetT             num_items,
    cudaStream_t        stream,
    bool                debug_synchronous)
{
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < timing_timing_iterations; ++i)
    {
        error = DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream, debug_synchronous);
    }
    return error;
}

*/
//---------------------------------------------------------------------
// CUDA Nested Parallelism Test Kernel
//---------------------------------------------------------------------

/**
 * Simple wrapper kernel to invoke DeviceScan
 */
template <typename IsPrimitiveT, typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, typename InitialValueT, typename OffsetT>
__global__ void CnpDispatchKernel(
    IsPrimitiveT        is_primitive,
    int                 timing_timing_iterations,
    size_t              *d_temp_storage_bytes,
    cudaError_t         *d_cdp_error,

    void*               d_temp_storage,
    size_t              temp_storage_bytes,
    InputIteratorT      d_in,
    OutputIteratorT     d_out,
    ScanOpT             scan_op,
    InitialValueT       initial_value,
    OffsetT             num_items,
    bool                debug_synchronous)
{
#ifndef CUB_CDP
    *d_cdp_error = cudaErrorNotSupported;
#else
    *d_cdp_error = Dispatch(
        Int2Type<CUB>(),
        is_primitive,
        timing_timing_iterations,
        d_temp_storage_bytes,
        d_cdp_error,
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        scan_op,
        initial_value,
        num_items,
        0,
        debug_synchronous);

    *d_temp_storage_bytes = temp_storage_bytes;
#endif
}

//---------------------------------------------------------------------
// Main
//---------------------------------------------------------------------

/**
 * Main
 */
int main(int argc, char** argv)
{
    int num_items = -1;

    // Initialize command line
    CommandLineArgs args(argc, argv);
    g_verbose = args.CheckCmdLineFlag("v");
    args.GetCmdLineArgument("n", num_items);
    args.GetCmdLineArgument("i", g_timing_iterations);
    args.GetCmdLineArgument("repeat", g_repeat);

    // Print usage
    if (args.CheckCmdLineFlag("help"))
    {
        printf("%s "
            "[--n=<input items> "
            "[--i=<timing iterations> "
            "[--device=<device-id>] "
            "[--repeat=<repetitions of entire test suite>]"
            "[--v] "
            "[--cdp]"
            "\n", argv[0]);
        exit(0);
    }

    // Initialize device
    CubDebugExit(args.DeviceInit());
    g_device_giga_bandwidth = args.device_giga_bandwidth;
    printf("\n");

// Compile/run thorough tests
    for (int i = 0; i <= g_repeat; ++i)
    {
        // Test different input+output data types
        TestSize<unsigned char>(num_items,      (int) 0, (int) 99);

        // Test same intput+output data types
	TestSize<unsigned int>(num_items,       (unsigned int) 0,       (unsigned int) 99);
        /**TestSize<unsigned char>(num_items,      (unsigned char) 0,      (unsigned char) 99);
        TestSize<char>(num_items,               (char) 0,               (char) 99);
        TestSize<unsigned short>(num_items,     (unsigned short) 0,     (unsigned short)99);
        
        TestSize<unsigned long long>(num_items, (unsigned long long) 0, (unsigned long long) 99);

        TestSize<uchar2>(num_items,     make_uchar2(0, 0),              make_uchar2(17, 21));
        TestSize<char2>(num_items,      make_char2(0, 0),               make_char2(17, 21));
        TestSize<ushort2>(num_items,    make_ushort2(0, 0),             make_ushort2(17, 21));
        TestSize<uint2>(num_items,      make_uint2(0, 0),               make_uint2(17, 21));
        TestSize<ulonglong2>(num_items, make_ulonglong2(0, 0),          make_ulonglong2(17, 21));
        TestSize<uchar4>(num_items,     make_uchar4(0, 0, 0, 0),        make_uchar4(17, 21, 32, 85));
        TestSize<char4>(num_items,      make_char4(0, 0, 0, 0),         make_char4(17, 21, 32, 85));

        TestSize<ushort4>(num_items,    make_ushort4(0, 0, 0, 0),       make_ushort4(17, 21, 32, 85));
        TestSize<uint4>(num_items,      make_uint4(0, 0, 0, 0),         make_uint4(17, 21, 32, 85));
        TestSize<ulonglong4>(num_items, make_ulonglong4(0, 0, 0, 0),    make_ulonglong4(17, 21, 32, 85));

        TestSize<TestFoo>(num_items,
            TestFoo::MakeTestFoo(0, 0, 0, 0),
            TestFoo::MakeTestFoo(1ll << 63, 1 << 31, short(1 << 15), char(1 << 7)));

        TestSize<TestBar>(num_items,
            TestBar(0, 0),
            TestBar(1ll << 63, 1 << 31));
  */  }

#endif

    return 0;
}



