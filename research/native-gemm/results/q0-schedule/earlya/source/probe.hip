// Standalone native gfx90a FP16 GEMM and its benchmark harness.
// No BLAS/CK is used by either the GPU computation or the CPU oracle.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "data.hpp"
#ifdef NATIVE_FP8_B
#include "data_fp8.hpp"
#ifdef NATIVE_SCALED_B
#include "data_scaled.hpp"
using InputMatrices = ScaledMatrices;
#else
using InputMatrices = Fp8Matrices;
#endif
constexpr bool kFp8B = true;
#else
using InputMatrices = TestMatrices;
constexpr bool kFp8B = false;
#endif

#define HIP(expression)                                                        \
    do {                                                                       \
        const auto status = (expression);                                      \
        if (status != hipSuccess) {                                            \
            throw std::runtime_error(std::string(#expression) + ": " +          \
                                     hipGetErrorString(status));               \
        }                                                                      \
    } while (0)

// F16 means a vector of sixteen FP32 accumulators, not an FP16 scalar.
using F16 = float __attribute__((ext_vector_type(16)));
#include "compute.hpp"

#ifdef NATIVE_SCALED_B
#include "scaled_kernel.hpp"
#else
// Kernel arguments and operations intentionally retain the measured ordering.
// Output tile and StageK are compile-time constants from compute.hpp.
__global__ __launch_bounds__(256) void native_gemm(
    const uint32_t* a,
    const uint32_t* b,
    float* out,
    uint64_t* stamps,
    int m,
    int n,
    int mp,
    int np,
    int tiles,
    int swizzle) {
    extern __shared__ __attribute__((aligned(16))) uint32_t smem[];

    const int grid_columns = np / kTile;
    const int tile_row = blockIdx.x / grid_columns;
    const int tile_column = blockIdx.x % grid_columns;
    const int wave = threadIdx.x / 64;
    const int lane = threadIdx.x % 64;
    const int wave_row = wave / 2;
    const int wave_column = wave % 2;
    const int wave_tile = kTile / 2;
    const int half_fragment_bytes = kTile * 8;

    // FP8 mode redistributes both producers by K8 slice; output ownership and
    // the LDS consumer layout are deliberately identical in both modes.
#ifndef NATIVE_FP8_B
    const int panel_bytes = kTile * kStageK;
    const uintptr_t producer_source =
        reinterpret_cast<uintptr_t>(wave < 2 ? a : b)
        + size_t((wave < 2 ? tile_row : tile_column) * 2 + wave % 2) * panel_bytes
        + lane * 16;

    // The emitter advances its pointer by 4096 between four-packet groups.
    // Its final increment must subtract those already-performed increments.
    const uint32_t source_increment = __builtin_amdgcn_readfirstlane(
        uint32_t((wave < 2 ? mp : np) * kStageK * 2
                 - (kQ * (kTile / 128) / 4 - 1) * 4096));
#endif
    const uint32_t lds_base = uint32_t(reinterpret_cast<uintptr_t>(smem));
    const int lds_lane = swizzle ? (lane ^ (lane >> 3)) : lane;
#ifndef NATIVE_FP8_B
    const uint32_t producer_destination =
        lds_base
        + (wave < 2 ? wave * half_fragment_bytes
                    : kTile * kStageK * 2 + (wave - 2) * half_fragment_bytes)
        + lds_lane * 16;
#endif
    const uint32_t operand_a_address =
        lds_base + wave_row * half_fragment_bytes + lds_lane * 16;
    const uint32_t operand_b_address =
        lds_base + kTile * kStageK * 2
        + wave_column * half_fragment_bytes + lds_lane * 16;

    F16 accumulators[kR * kR] = {};
    uint64_t core_begin, core_end, steady_begin, steady_end;
#ifdef NATIVE_FP8_B
    static_assert(kTile == 128 && kStageK == 32);
    const uintptr_t a_source = reinterpret_cast<uintptr_t>(a)
        + size_t(tile_row) * 8192 + wave * 2048 + lane * 16;
    const uintptr_t b_source = reinterpret_cast<uintptr_t>(b)
        + size_t(tile_column) * 4096 + wave * 1024 + lane * 16;
    const uint32_t a_destination = lds_base + wave * 2048 + lds_lane * 16;
    pipeline(
        uint32_t(a_source), uint32_t(a_source >> 32),
        __builtin_amdgcn_readfirstlane(uint32_t(mp * 64)), a_destination,
        operand_a_address, operand_b_address,
        uint32_t(b_source), uint32_t(b_source >> 32),
        __builtin_amdgcn_readfirstlane(uint32_t(np * 32)), a_destination + 8192,
        tiles, accumulators, core_begin, core_end, steady_begin, steady_end);
#else
    pipeline(
        uint32_t(producer_source), uint32_t(producer_source >> 32),
        source_increment, producer_destination, operand_a_address, operand_b_address,
        tiles, accumulators, core_begin, core_end, steady_begin, steady_end);
#endif

    if (lane == 0) {
        auto wave_stamps = stamps + (size_t(blockIdx.x) * 4 + wave) * 4;
        wave_stamps[0] = core_begin;
        wave_stamps[1] = core_end;
        wave_stamps[2] = steady_begin;
        wave_stamps[3] = steady_end;
    }

    // MFMA's sixteen accumulator values/lane map to a 32x32 microtile.
    // This is the inverse of the operand fragment convention in data.hpp.
#pragma unroll
    for (int micro_row = 0; micro_row < kR; ++micro_row) {
#pragma unroll
        for (int micro_column = 0; micro_column < kR; ++micro_column) {
#pragma unroll
            for (int element = 0; element < 16; ++element) {
                int row = tile_row * kTile + wave_row * wave_tile + micro_row * 32
                        + 8 * (element / 4) + 4 * (lane / 32) + element % 4;
                int column = tile_column * kTile + wave_column * wave_tile
                           + micro_column * 32 + lane % 32;
                if (row < m && column < n) {
                    out[size_t(row) * n + column] =
                        accumulators[micro_row * kR + micro_column][element];
                }
            }
        }
    }
}

#endif // NATIVE_SCALED_B

// Everything below is host-only setup, timing, verification and reporting.
// None of these helpers is part of the timed device K loop.
template<class T>
struct DeviceBuffer {
    T* pointer = nullptr;

    explicit DeviceBuffer(size_t count) {
        HIP(hipMalloc(&pointer, std::max(size_t(1), count) * sizeof(T)));
    }
    ~DeviceBuffer() {
        if (pointer) {
            (void)hipFree(pointer);
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct TimingEvents {
    hipEvent_t begin;
    hipEvent_t end;

    TimingEvents() {
        HIP(hipEventCreate(&begin));
        HIP(hipEventCreate(&end));
    }
    ~TimingEvents() {
        (void)hipEventDestroy(begin);
        (void)hipEventDestroy(end);
    }
};

struct Options {
    int rows = 1664;
    int columns = 1024;
    int reduction = 8192;
    int rounds = 9;
    int warmup_ms = 200;
    int swizzle = 0;
    int b_pattern = 0;
    int scale_pattern = 1;
    uint32_t seed_a = 1729;
    uint32_t seed_b = 2718;
    std::string stamp_prefix;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key == "--help") {
            std::printf(
                "--m N --n N --k N --rounds N --warmup-ms N "
                "--seed-a N --seed-b N --swizzle 0|1 --stamps-prefix PATH; "
                "compiled tile%d,StageK%d,FP16 packed panels,FP32 C; native only\n",
                kTile, kStageK);
            std::exit(0);
        }
        if (++index == argc) {
            throw std::runtime_error("missing value");
        }
        const std::string value = argv[index];
        if (key == "--m") {
            options.rows = std::stoi(value);
        } else if (key == "--n") {
            options.columns = std::stoi(value);
        } else if (key == "--k") {
            options.reduction = std::stoi(value);
        } else if (key == "--rounds") {
            options.rounds = std::stoi(value);
        } else if (key == "--warmup-ms") {
            options.warmup_ms = std::stoi(value);
        } else if (key == "--stamps-prefix") {
            options.stamp_prefix = value;
        } else if (key == "--scale-pattern") {
            options.scale_pattern = std::stoi(value);
        } else if (key == "--b-pattern") {
            options.b_pattern = std::stoi(value);
        } else if (key == "--swizzle") {
            options.swizzle = std::stoi(value);
        } else if (key == "--seed-a") {
            options.seed_a = uint32_t(std::stoul(value));
        } else if (key == "--seed-b") {
            options.seed_b = uint32_t(std::stoul(value));
        } else {
            throw std::runtime_error("unknown option " + key);
        }
    }
    if (options.rows < 1 || options.columns < 1
        || options.rows > 16384 || options.columns > 16384
        || options.reduction < 0 || options.reduction > 65536
        || options.rounds < 1 || options.warmup_ms < 0
        || (options.swizzle != 0 && options.swizzle != 1)
        || options.b_pattern < 0 || options.b_pattern > 1
        || (options.b_pattern && !kFp8B)) {
        throw std::runtime_error("dimensions/options out of range");
    }
    return options;
}

struct ReferenceOutputs {
    bool full;
    std::vector<size_t> indices;
    std::vector<float> values;
};

ReferenceOutputs make_reference(const InputMatrices& matrices, int stage_count) {
    const int m = matrices.rows;
    const int n = matrices.columns;
    const int k = matrices.reduction;
    ReferenceOutputs reference;
    reference.full = uint64_t(m) * n * std::max(k, 1) <= 30000000;

    if (reference.full) {
        reference.indices.resize(size_t(m) * n);
        for (size_t index = 0; index < reference.indices.size(); ++index) {
            reference.indices[index] = index;
        }
    } else {
        std::mt19937 random(314159);
        for (int sample = 0; sample < 2048; ++sample) {
            reference.indices.push_back(size_t(random()) % (size_t(m) * n));
        }
        // Exercise row/column microtile boundaries and all four matrix edges.
        for (int row = 0; row < m; row += 32) {
            for (int column : {0, n - 1}) {
                reference.indices.push_back(size_t(row) * n + column);
            }
        }
        for (int column = 0; column < n; column += 32) {
            for (int row : {0, m - 1}) {
                reference.indices.push_back(size_t(row) * n + column);
            }
        }
        for (int row : {0, m - 1}) {
            for (int column : {0, n - 1}) {
                reference.indices.push_back(size_t(row) * n + column);
            }
        }
        std::sort(reference.indices.begin(), reference.indices.end());
        reference.indices.erase(
            std::unique(reference.indices.begin(), reference.indices.end()),
            reference.indices.end());
    }

    reference.values.reserve(reference.indices.size());
    for (size_t index : reference.indices) {
        const int row = int(index / n);
        const int column = int(index % n);
        // The diagnostic deliberately repeats only the first StageK fragment.
        // Ordinary GEMM always references all original row-major operands.
        reference.values.push_back(
            kResident
                ? matrices.reference_dot(row, column, std::min(k, kStageK)) * stage_count
                : matrices.reference_dot(row, column));
    }
    return reference;
}

struct Verification {
    int mismatches = 0;
    int nonfinite = 0;
    int guards = 0;

    bool valid() const {
        return !mismatches && !nonfinite && !guards;
    }
};

Verification verify_output(
    const std::vector<float>& output,
    size_t element_count,
    const ReferenceOutputs& reference) {
    Verification result;
    for (size_t index = 0; index < element_count; ++index) {
        result.nonfinite += !std::isfinite(output[index + 16]);
    }
    for (size_t sample = 0; sample < reference.indices.size(); ++sample) {
        const size_t index = reference.indices[sample];
        if (output[index + 16] != reference.values[sample]) {
            if (result.mismatches < 4) {
                std::fprintf(
                    stderr, "MISMATCH %zu expected %.9g actual %.9g\n",
                    index, reference.values[sample], output[index + 16]);
            }
            ++result.mismatches;
        }
    }
    for (int index = 0; index < 16; ++index) {
        result.guards +=
            (std::bit_cast<uint32_t>(output[index]) != 0xffffffffu)
            + (std::bit_cast<uint32_t>(output[element_count + 16 + index]) != 0xffffffffu);
    }
    return result;
}

double median(std::vector<double>& values) {
    std::sort(values.begin(), values.end());
    return (values[(values.size() - 1) / 2] + values[values.size() / 2]) / 2;
}

struct ClockSummary {
    double grid_body_us;
    double median_core_mhz;
    double median_cta_share_of_cu_peak;
};

ClockSummary summarize_clocks(
    const std::vector<uint64_t>& timestamps,
    int blocks,
    int stage_count,
    int steady_clock_khz) {
    uint64_t grid_steady_begin = UINT64_MAX;
    uint64_t grid_steady_end = 0;
    std::vector<double> core_rates;
    std::vector<double> cta_shares;
    const double executed_flops_per_cta = 2.0 * kTile * kTile * stage_count * kStageK;

    for (int block = 0; block < blocks; ++block) {
        const auto stamps = timestamps.data() + size_t(block) * 16;
        uint64_t core_begin = UINT64_MAX;
        uint64_t core_end = 0;
        uint64_t steady_begin = UINT64_MAX;
        uint64_t steady_end = 0;
        for (int wave = 0; wave < 4; ++wave) {
            core_begin = std::min(core_begin, stamps[wave * 4]);
            core_end = std::max(core_end, stamps[wave * 4 + 1]);
            steady_begin = std::min(steady_begin, stamps[wave * 4 + 2]);
            steady_end = std::max(steady_end, stamps[wave * 4 + 3]);
        }
        // Core clocks are compared only within a CTA/CU. The common steady
        // clock, not core timestamps from different CUs, defines grid time.
        grid_steady_begin = std::min(grid_steady_begin, steady_begin);
        grid_steady_end = std::max(grid_steady_end, steady_end);
        const double body_us = double(steady_end - steady_begin) * 1000 / steady_clock_khz;
        core_rates.push_back(double(core_end - core_begin) / body_us);
        cta_shares.push_back(
            executed_flops_per_cta / (double(core_end - core_begin) * 1024));
    }
    return {
        double(grid_steady_end - grid_steady_begin) * 1000 / steady_clock_khz,
        median(core_rates),
        median(cta_shares),
    };
}

void write_clock_dump(
    const std::string& prefix, int round, const std::vector<uint64_t>& timestamps) {
    if (prefix.empty()) {
        return;
    }
    std::ofstream file(prefix + "-r" + std::to_string(round) + ".bin", std::ios::binary);
    file.write(
        reinterpret_cast<const char*>(timestamps.data()),
        std::streamsize(timestamps.size() * 8));
    if (!file) {
        throw std::runtime_error("timestamp output failed");
    }
}

int main(int argc, char** argv) try {
    const Options options = parse_options(argc, argv);
    const int m = options.rows;
    const int n = options.columns;
    const int k = options.reduction;

#ifdef NATIVE_SCALED_B
    if (options.swizzle || options.scale_pattern < 0 || options.scale_pattern > 3)
        throw std::runtime_error("scaled variant requires linear LDS and scale pattern0..3");
#endif
    HIP(hipSetDevice(0));
    int steady_clock_khz;
    HIP(hipDeviceGetAttribute(&steady_clock_khz, hipDeviceAttributeWallClockRate, 0));
    hipDeviceProp_t device;
    HIP(hipGetDeviceProperties(&device, 0));
    char pci_bus_id[64];
    HIP(hipDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), 0));

    const int padded_m = (m + kTile - 1) / kTile * kTile;
    const int padded_n = (n + kTile - 1) / kTile * kTile;
    const int stage_count = (k + kStageK - 1) / kStageK;
    const int blocks = padded_m / kTile * (padded_n / kTile);
    int resident_blocks;
    HIP(hipOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident_blocks, native_gemm, 256, kLdsBytes));

    std::fprintf(stderr, "kind=%s\n", kResident ? "resident-control" : "gemm");
    if (kResident) {
        std::fprintf(stderr,
                     "Resident control repeats first padded stage fragment, not ordinary full GEMM\n");
    }
    std::fprintf(
        stderr, "LDS lane permutation=%s (address-only, producer/consumer agree)\n",
        options.swizzle ? "lane XOR lane>>3" : "linear");

    InputMatrices matrices(m, n, k, options.seed_a, options.seed_b);
#ifdef NATIVE_FP8_B
    matrices.set_finite_pattern(options.b_pattern != 0);
#ifdef NATIVE_SCALED_B
    matrices.set_scales(options.scale_pattern, kScaled);
#endif
#endif
    const auto packing_begin = std::chrono::steady_clock::now();
    auto packed_a = matrices.pack_operand(false, kTile, kStageK);
    auto packed_b = matrices.pack_operand(true, kTile, kStageK);
    const double packing_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - packing_begin).count();

    DeviceBuffer<uint32_t> device_a(packed_a.size());
    DeviceBuffer<uint32_t> device_b(packed_b.size());
#ifdef NATIVE_SCALED_B
#ifdef NATIVE_SCALE_BATCH
    const auto upload_scales = matrices.pack_scale_pairs();
#else
    const auto& upload_scales = matrices.scales;
#endif
    DeviceBuffer<float> device_scales(upload_scales.size());
    if (!upload_scales.empty())
        HIP(hipMemcpy(device_scales.pointer, upload_scales.data(), upload_scales.size() * 4,
                      hipMemcpyHostToDevice));
    std::fprintf(stderr, "SCALES K128/N128 FP32 pattern%d enabled%d logical%zu uploaded%zu\n",
                 options.scale_pattern, int(kScaled), matrices.scales.size(), upload_scales.size());
#endif
    DeviceBuffer<float> device_output(size_t(m) * n + 32);
    DeviceBuffer<uint64_t> device_stamps(size_t(blocks) * 16);
    if (!packed_a.empty()) {
        HIP(hipMemcpy(device_a.pointer, packed_a.data(), packed_a.size() * 4,
                      hipMemcpyHostToDevice));
    }
    if (!packed_b.empty()) {
        HIP(hipMemcpy(device_b.pointer, packed_b.data(), packed_b.size() * 4,
                      hipMemcpyHostToDevice));
    }
    std::fprintf(
        stderr,
        "HIP0 BDF=%s O3 gfx90a, CUs=%d tile=%d stageK=%d singleLDS=%d "
        "residentCTA/CU=%d grid=%dx%d(%d) block256 M/N/K=%d/%d/%d "
        "padded=%d/%d/%d; %s packed/FP32 C,packing %.3f ms outside timing,"
        "no BLAS/CK,model/context=none,external NUMA,no clock override\n",
        pci_bus_id, device.multiProcessorCount, kTile, kStageK, kLdsBytes,
        resident_blocks, padded_m / kTile, padded_n / kTile, blocks,
        m, n, k, padded_m, padded_n, stage_count * kStageK,
        kFp8B ? "FP16 A/E4M3FN B" : "FP16", packing_ms);

    const ReferenceOutputs reference = make_reference(matrices, stage_count);
    std::fprintf(
        stderr, "REFERENCE %s %zu outputs from original row-major operands; "
        "all C scanned finite/guards every round\n",
        reference.full ? "FULL" : "SAMPLED", reference.indices.size());

    auto launch = [&] {
        native_gemm<<<blocks, 256, kLdsBytes>>>(
            device_a.pointer, device_b.pointer, device_output.pointer + 16,
            device_stamps.pointer, m, n, padded_m, padded_n, stage_count, options.swizzle
#ifdef NATIVE_SCALED_B
            , device_scales.pointer
#endif
        );
        HIP(hipGetLastError());
    };

    // Preserve the measured protocol: initial dispatch, selected-kernel
    // warmup, then per-round output poison. CPU checks are outside events.
    launch();
    HIP(hipDeviceSynchronize());
    const auto warmup_end = std::chrono::steady_clock::now()
                          + std::chrono::milliseconds(options.warmup_ms);
    while (std::chrono::steady_clock::now() < warmup_end) {
        launch();
        HIP(hipDeviceSynchronize());
    }

    TimingEvents events;
    std::vector<float> output(size_t(m) * n + 32);
    std::vector<uint64_t> timestamps(size_t(blocks) * 16);
    std::puts(
        "kind,round,tile,stage_k,swizzle,m,n,k,mp,np,kp,ctas,resident_ctas_per_cu,"
        "lds_bytes,event_us,grid_body_us,useful_TFLOPs,executed_TFLOPs,"
        "median_core_MHz,median_cta_share_of_cu_peak,global_clock_adjusted_efficiency,"
        "reference_outputs,full_reference,mismatches,nonfinite,guards,valid,pack_ms");

    for (int round = 0; round < options.rounds; ++round) {
        HIP(hipMemset(device_output.pointer, 0xff, output.size() * 4));
        HIP(hipEventRecord(events.begin));
        launch();
        HIP(hipEventRecord(events.end));
        HIP(hipEventSynchronize(events.end));
        float event_ms;
        HIP(hipEventElapsedTime(&event_ms, events.begin, events.end));

        HIP(hipMemcpy(output.data(), device_output.pointer, output.size() * 4,
                      hipMemcpyDeviceToHost));
        HIP(hipMemcpy(timestamps.data(), device_stamps.pointer, timestamps.size() * 8,
                      hipMemcpyDeviceToHost));
        const Verification verification = verify_output(output, size_t(m) * n, reference);
        const ClockSummary clocks = summarize_clocks(
            timestamps, blocks, stage_count, steady_clock_khz);
        write_clock_dump(options.stamp_prefix, round, timestamps);

        const double useful_tflops = 2.0 * m * n * k / (event_ms * 1e9);
        const double executed_tflops =
            2.0 * padded_m * padded_n * stage_count * kStageK / (event_ms * 1e9);
        const double global_efficiency = useful_tflops
            / (device.multiProcessorCount * 1024.0 * clocks.median_core_mhz / 1e6);

        std::printf(
            "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,"
            "%.6f,%.6f,%.6f,%.6f,%.3f,%.9f,%.9f,%zu,%d,%d,%d,%d,%d,%.3f\n",
            kResident ? "resident-control" : "gemm", round, kTile, kStageK,
            options.swizzle, m, n, k, padded_m, padded_n, stage_count * kStageK,
            blocks, resident_blocks, kLdsBytes, event_ms * 1000, clocks.grid_body_us,
            useful_tflops, executed_tflops, clocks.median_core_mhz,
            clocks.median_cta_share_of_cu_peak, global_efficiency,
            reference.indices.size(), reference.full, verification.mismatches,
            verification.nonfinite, verification.guards, verification.valid(), packing_ms);
        if (!verification.valid()) {
            throw std::runtime_error("GEMM verification failed");
        }
    }
    std::fprintf(stderr, "COMPLETE all reference outputs/finite C/guards exact\n");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "ERROR: %s\n", error.what());
    return 1;
}
