#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <memory>

namespace nn_binary_analysis {

// ML Framework identification
enum class MLFramework {
    Unknown,
    PyTorch,
    TensorFlow,
    ONNX,
    LibTorch
};

// Model architecture types
enum class ModelArchitecture {
    Unknown,
    Transformer,    // BERT, GPT, etc.
    CNN,           // Convolutional networks
    RNN,           // Recurrent networks
    Linear         // Simple feedforward
};

// Forward declarations
class BinaryParser;
class PatternDetector;
class ResourceMonitor;

// Confidence metrics for analysis results
struct ConfidenceMetrics {
    float pattern_confidence{0.0f};    // Confidence in pattern detection
    float structure_confidence{0.0f};  // Confidence in structural analysis
    float operation_confidence{0.0f};  // Confidence in operation classification
    
    float overall() const {
        return (pattern_confidence + structure_confidence + operation_confidence) / 3.0f;
    }
};

// Framework detection result
struct FrameworkDetails {
    MLFramework type{MLFramework::Unknown};
    std::string version;
    float confidence{0.0f};
};

// Model architecture details
struct ArchitectureDetails {
    ModelArchitecture type{ModelArchitecture::Unknown};
    std::vector<size_t> layer_sizes;
    std::string specific_type;  // e.g., "BERT-base", "ResNet50"
    float confidence{0.0f};
};

// Represents a decoded instruction with analysis metadata
struct DecodedInstruction {
    uint64_t address{0};              // Instruction address
    std::vector<uint8_t> bytes;       // Raw instruction bytes
    std::string mnemonic;             // Instruction mnemonic
    bool potential_tensor_op{false};   // Initial classification
    ConfidenceMetrics confidence;      // Analysis confidence
};

// Memory access pattern information
struct MemoryAccessPattern {
    uint64_t base_address{0};         // Base address of access
    uint32_t stride_primary{0};       // Primary stride length
    uint32_t stride_secondary{0};     // Secondary stride (for nested patterns)
    uint32_t access_count{0};         // Number of accesses in pattern
    float consistency_ratio{0.0f};    // Pattern consistency metric
    
    // New fields for transformer pattern detection
    bool is_attention_pattern{false};
    bool is_feedforward_pattern{false};
    bool is_layer_norm_pattern{false};
};

// Types of tensor operations we can detect
enum class TensorOpType {
    Unknown,
    MatrixMultiplication,
    Convolution2D,
    ElementWise,
    Pooling,
    Normalization,
    Attention,          // Transformer attention operations
    LayerNorm,          // Layer normalization
    Embedding,          // Embedding lookup
    SoftMax            // Softmax operation
};

// Detected tensor operation
struct TensorOperation {
    TensorOpType type{TensorOpType::Unknown};
    uint64_t start_address{0};        // Start address in binary
    uint64_t end_address{0};          // End address in binary
    std::vector<uint32_t> dimensions; // Tensor dimensions if detected
    ConfidenceMetrics confidence;     // Detection confidence
    
    struct {
        uint32_t kernel_size{0};      // For convolution operations
        uint32_t stride{0};           // For strided operations
        uint32_t padding{0};          // For padded operations
        uint32_t heads{0};            // For attention operations
        uint32_t hidden_size{0};      // For transformer operations
    } metadata;
};

// Resource utilization metrics
struct ResourceMetrics {
    float simd_utilization{0.0f};     // SIMD instruction usage
    float memory_bandwidth{0.0f};     // Memory bandwidth utilization
    float cache_efficiency{0.0f};     // Cache hit rate
    
    float overall_efficiency() const {
        return (simd_utilization + memory_bandwidth + cache_efficiency) / 3.0f;
    }
};

// Enhanced analysis result including framework and architecture detection
struct AnalysisResult {
    std::vector<TensorOperation> detected_ops;
    std::vector<MemoryAccessPattern> access_patterns;
    ResourceMetrics resource_usage;
    ConfidenceMetrics overall_confidence;
    FrameworkDetails framework;
    ArchitectureDetails architecture;
    
    bool has_neural_network() const {
        return !detected_ops.empty() && overall_confidence.overall() > 0.7f;
    }
    
    bool is_transformer() const {
        return architecture.type == ModelArchitecture::Transformer && 
               framework.confidence > 0.7f;
    }
};

// Configuration for analysis
struct AnalysisConfig {
    bool enable_pattern_detection{true};
    bool enable_resource_monitoring{true};
    bool enable_framework_detection{true};
    float confidence_threshold{0.7f};
    uint32_t sample_rate{1000};       // Sample every N instructions
    
    struct {
        uint32_t pattern_buffer_size{1024};
        uint32_t min_pattern_length{3};
        uint32_t max_pattern_length{16};
    } pattern_detection;
    
    struct {
        uint32_t window_size{1000};
        float simd_threshold{0.3f};
        float bandwidth_threshold{0.5f};
    } resource_monitoring;
    
    struct {
        bool detect_transformers{true};
        bool detect_pytorch{true};
        uint32_t min_layer_size{32};
    } model_detection;
};

struct DecodedInstruction {
    uint64_t address{0};              // Instruction address
    std::vector<uint8_t> bytes;       // Raw instruction bytes
    std::string mnemonic;             // Instruction mnemonic
    bool potential_tensor_op{false};   // Initial classification
    ConfidenceMetrics confidence;      // Analysis confidence
    std::unique_ptr<llvm::MCInst> inst; // Stored LLVM instruction

    // Add conversion operator
    operator const llvm::MCInst&() const { return *inst; }
};

} // namespace nn_binary_analysis