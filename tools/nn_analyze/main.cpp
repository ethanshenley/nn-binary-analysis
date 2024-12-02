#include "nn_binary_analysis/analyzer.hpp"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>
#include <iostream>

using namespace llvm;
using namespace nn_binary_analysis;

// Command line options
static cl::opt<std::string> InputFilename(cl::Positional,
    cl::desc("<input binary>"),
    cl::Required);

static cl::opt<std::string> TargetTriple("triple",
    cl::desc("Target triple (default: host triple)"),
    cl::init(sys::getDefaultTargetTriple()));

static cl::opt<bool> Verbose("v",
    cl::desc("Enable verbose output"),
    cl::init(false));

int main(int argc, char *argv[]) {
    // Parse command line options
    cl::ParseCommandLineOptions(argc, argv, "Neural Network Binary Analyzer\n");

    // Load input file
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
        MemoryBuffer::getFile(InputFilename);
    if (std::error_code EC = BufferOrErr.getError()) {
        errs() << "Error loading file '" << InputFilename << "': " 
               << EC.message() << "\n";
        return 1;
    }
    std::unique_ptr<MemoryBuffer> Buffer = std::move(BufferOrErr.get());

    // Initialize analyzer
    AnalysisConfig config;
    config.enable_pattern_detection = true;
    config.enable_resource_monitoring = true;
    config.confidence_threshold = 0.7f;

    try {
        // Create binary analyzer
        BinaryAnalyzer analyzer(TargetTriple, config);

        // Analyze binary
        const uint8_t* data = reinterpret_cast<const uint8_t*>(
            Buffer->getBufferStart());
        size_t size = Buffer->getBufferSize();

        AnalysisResult result = analyzer.analyze(data, size);

        // Report results
        std::cout << "\nAnalysis Results:\n";
        std::cout << "================\n";
        std::cout << "Neural Network Detection Confidence: " 
                  << result.overall_confidence.overall() * 100 << "%\n\n";

        if (result.has_neural_network()) {
            std::cout << "Detected Operations:\n";
            for (const auto& op : result.detected_ops) {
                std::cout << "- Operation at 0x" << std::hex << op.start_address
                         << " - 0x" << op.end_address << std::dec << "\n";
                std::cout << "  Type: " << getTensorOpTypeName(op.type) << "\n";
                std::cout << "  Confidence: " 
                         << op.confidence.overall() * 100 << "%\n";
                
                if (Verbose) {
                    std::cout << "  Dimensions: ";
                    for (auto dim : op.dimensions) {
                        std::cout << dim << " ";
                    }
                    std::cout << "\n";
                    
                    if (op.type == TensorOpType::Convolution2D) {
                        std::cout << "  Kernel Size: " << op.metadata.kernel_size << "\n";
                        std::cout << "  Stride: " << op.metadata.stride << "\n";
                        std::cout << "  Padding: " << op.metadata.padding << "\n";
                    }
                }
                std::cout << "\n";
            }

            if (Verbose) {
                std::cout << "Memory Access Patterns:\n";
                for (const auto& pattern : result.access_patterns) {
                    std::cout << "- Base Address: 0x" << std::hex 
                             << pattern.base_address << std::dec << "\n";
                    std::cout << "  Primary Stride: " << pattern.stride_primary << "\n";
                    std::cout << "  Secondary Stride: " << pattern.stride_secondary << "\n";
                    std::cout << "  Consistency: " 
                             << pattern.consistency_ratio * 100 << "%\n\n";
                }

                std::cout << "Resource Utilization:\n";
                std::cout << "- SIMD Usage: " 
                         << result.resource_usage.simd_utilization * 100 << "%\n";
                std::cout << "- Memory Bandwidth: " 
                         << result.resource_usage.memory_bandwidth * 100 << "%\n";
                std::cout << "- Cache Efficiency: " 
                         << result.resource_usage.cache_efficiency * 100 << "%\n";
            }
        } else {
            std::cout << "No neural network operations detected with sufficient confidence.\n";
        }

    } catch (const std::exception& e) {
        errs() << "Error during analysis: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

// Helper function to convert TensorOpType to string
std::string getTensorOpTypeName(TensorOpType type) {
    switch (type) {
        case TensorOpType::MatrixMultiplication:
            return "Matrix Multiplication";
        case TensorOpType::Convolution2D:
            return "2D Convolution";
        case TensorOpType::ElementWise:
            return "Element-wise Operation";
        case TensorOpType::Pooling:
            return "Pooling";
        case TensorOpType::Normalization:
            return "Normalization";
        default:
            return "Unknown";
    }
}