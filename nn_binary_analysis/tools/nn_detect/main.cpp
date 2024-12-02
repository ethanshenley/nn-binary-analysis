#include "include/nn_binary_analysis/analyzer.hpp"
#include "llvm/Support/Host.h"
#include "llvm/ADT/Triple.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <cstring>


using namespace nn_binary_analysis;

void printResults(const AnalysisResult& result) {
    std::cout << "\n=== Neural Network Analysis Results ===\n\n";
    
    // Framework Detection
    std::cout << "Framework Detection:\n";
    std::cout << "--------------------\n";
    std::cout << "Detected Framework: ";
    switch (result.framework.type) {
        case MLFramework::LibTorch:
            std::cout << "LibTorch";
            break;
        case MLFramework::PyTorch:
            std::cout << "PyTorch";
            break;
        case MLFramework::TensorFlow:
            std::cout << "TensorFlow";
            break;
        default:
            std::cout << "Unknown";
    }
    std::cout << "\nVersion: " << (result.framework.version.empty() ? "Unknown" : result.framework.version);
    std::cout << "\nConfidence: " << std::fixed << std::setprecision(2) 
              << (result.framework.confidence * 100) << "%\n\n";

    // Architecture Detection
    std::cout << "Architecture Detection:\n";
    std::cout << "----------------------\n";
    std::cout << "Model Type: ";
    switch (result.architecture.type) {
        case ModelArchitecture::Transformer:
            std::cout << "Transformer";
            break;
        case ModelArchitecture::CNN:
            std::cout << "CNN";
            break;
        case ModelArchitecture::RNN:
            std::cout << "RNN";
            break;
        default:
            std::cout << "Unknown";
    }
    if (!result.architecture.specific_type.empty()) {
        std::cout << " (" << result.architecture.specific_type << ")";
    }
    std::cout << "\nDimensions: ";
    for (auto dim : result.architecture.layer_sizes) {
        std::cout << dim << " ";
    }
    std::cout << "\nConfidence: " << std::fixed << std::setprecision(2) 
              << (result.architecture.confidence * 100) << "%\n\n";

    // Resource Usage
    std::cout << "Resource Utilization:\n";
    std::cout << "--------------------\n";
    std::cout << "SIMD Usage: " << (result.resource_usage.simd_utilization * 100) << "%\n";
    std::cout << "Memory Bandwidth: " << (result.resource_usage.memory_bandwidth * 100) << "%\n";
    std::cout << "Cache Efficiency: " << (result.resource_usage.cache_efficiency * 100) << "%\n\n";

    // Overall Results
    std::cout << "Overall Analysis:\n";
    std::cout << "----------------\n";
    std::cout << "Neural Network Detected: " << (result.has_neural_network() ? "Yes" : "No") << "\n";
    std::cout << "Confidence Score: " << (result.overall_confidence.overall() * 100) << "%\n";
    
    if (result.is_transformer()) {
        std::cout << "\nTransformer Model Details:\n";
        std::cout << "------------------------\n";
        std::cout << "Architecture appears to be a transformer-based model.\n";
        std::cout << "Likely to be a BERT variant based on architecture patterns.\n";
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <binary_file>\n";
        return 1;
    }

    try {
        // Read binary file
        std::ifstream file(argv[1], std::ios::binary);
        if (!file) {
            std::cerr << "Error: Could not open file: " << argv[1] << "\n";
            return 1;
        }

        // Get file size and read contents
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<uint8_t> buffer(size);
        file.read(reinterpret_cast<char*>(buffer.data()), size);

        // Configure analysis
        AnalysisConfig config;
        config.enable_pattern_detection = true;
        config.enable_framework_detection = true;
        config.enable_resource_monitoring = true;

        // Initialize analyzer with host triple
        BinaryAnalyzer analyzer(llvm::sys::getDefaultTargetTriple(), config);
        
        std::cout << "Analyzing binary: " << argv[1] << "\n";
        std::cout << "File size: " << size << " bytes\n";

        // Perform analysis
        auto result = analyzer.analyze(buffer.data(), buffer.size());

        // Print results
        printResults(result);

    } catch (const std::exception& e) {
        std::cerr << "Error during analysis: " << e.what() << "\n";
        return 1;
    }

    return 0;
}