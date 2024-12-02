#include "transformer_detector.hpp"
#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace nn_binary_analysis {

namespace {
    // Constants for transformer detection
    constexpr size_t MIN_ATTENTION_WINDOW = 32;
    constexpr size_t MAX_ATTENTION_WINDOW = 512;
    constexpr float MIN_PATTERN_CONFIDENCE = 0.7f;
    
    // Common transformer architecture sizes
    const std::vector<uint32_t> COMMON_HIDDEN_SIZES = {768, 512, 1024, 256};
    const std::vector<uint32_t> COMMON_HEAD_COUNTS = {8, 12, 16};
    
    bool isPowerOfTwo(uint32_t n) {
        return n && !(n & (n - 1));
    }
    
    bool isCommonTransformerSize(uint32_t size) {
        return std::find(COMMON_HIDDEN_SIZES.begin(), 
                        COMMON_HIDDEN_SIZES.end(), 
                        size) != COMMON_HIDDEN_SIZES.end();
    }
}

TransformerDetector::TransformerDetector(const AnalysisConfig& config)
    : config_(config) {
    // Deque doesn't need reserve, it will grow as needed
    instruction_window_.clear();
}

void TransformerDetector::processInstruction(const DecodedInstruction& inst) {
    instruction_window_.push_back(inst);
    if (instruction_window_.size() > config_.pattern_detection.pattern_buffer_size) {
        instruction_window_.pop_front();
    }
    
    // Process window if we have enough instructions
    if (instruction_window_.size() >= MIN_ATTENTION_WINDOW) {
        std::vector<DecodedInstruction> window_vec(instruction_window_.begin(),
                                                 instruction_window_.end());
                                                 
        // Check for transformer patterns
        if (detectAttentionPattern(window_vec)) {
            TransformerLayerPattern layer;
            layer.has_layer_norm1 = detectLayerNormPattern(window_vec);
            layer.has_feedforward = detectFeedForwardPattern(window_vec);
            
            // Only add if we have enough confidence
            layer.confidence = computePatternConfidence(layer);
            if (layer.confidence >= MIN_PATTERN_CONFIDENCE) {
                detected_layers_.push_back(layer);
                updateOverallConfidence();
            }
        }
    }
}

bool TransformerDetector::detectTransformerPatterns(
    const std::vector<MemoryAccessPattern>& patterns) {
    
    size_t attention_patterns = 0;
    size_t layer_norm_patterns = 0;
    
    for (const auto& pattern : patterns) {
        if (isAttentionMemoryPattern(pattern)) {
            attention_patterns++;
            updateDimensionEstimates(pattern);
        }
        else if (isLayerNormMemoryPattern(pattern)) {
            layer_norm_patterns++;
        }
    }
    
    // We expect certain ratios of patterns in a transformer
    float attention_ratio = static_cast<float>(attention_patterns) / patterns.size();
    float layer_norm_ratio = static_cast<float>(layer_norm_patterns) / patterns.size();
    
    return attention_ratio > 0.2f && layer_norm_ratio > 0.1f;
}

bool TransformerDetector::detectAttentionPattern(
    const std::vector<DecodedInstruction>& window) {
    
    // Look for characteristic attention patterns:
    // 1. Three similar matrix multiplications (Q, K, V projections)
    // 2. Matrix multiplication for attention scores
    // 3. Softmax operation
    // 4. Final projection
    
    size_t matrix_mul_count = 0;
    bool has_softmax = false;
    
    for (const auto& inst : window) {
        if (inst.mnemonic.find("vmul") != std::string::npos ||
            inst.mnemonic.find("vfma") != std::string::npos) {
            matrix_mul_count++;
        }
        else if (inst.mnemonic.find("vmax") != std::string::npos &&
                 inst.mnemonic.find("vadd") != std::string::npos) {
            // Simplified softmax detection
            has_softmax = true;
        }
    }
    
    // Expect at least 4 matrix multiplications (3 projections + attention)
    return matrix_mul_count >= 4 && has_softmax;
}

bool TransformerDetector::detectLayerNormPattern(
    const std::vector<DecodedInstruction>& window) {
    
    // Layer normalization typically involves:
    // 1. Mean calculation
    // 2. Variance calculation
    // 3. Normalization
    // 4. Scale and shift
    
    bool has_mean_calc = false;
    bool has_variance_calc = false;
    bool has_division = false;
    
    for (const auto& inst : window) {
        if (inst.mnemonic.find("vadd") != std::string::npos) {
            has_mean_calc = true;
        }
        else if (inst.mnemonic.find("vmul") != std::string::npos &&
                 inst.mnemonic.find("vsub") != std::string::npos) {
            has_variance_calc = true;
        }
        else if (inst.mnemonic.find("vdiv") != std::string::npos ||
                 inst.mnemonic.find("vsqrt") != std::string::npos) {
            has_division = true;
        }
    }
    
    return has_mean_calc && has_variance_calc && has_division;
}

bool TransformerDetector::detectFeedForwardPattern(
    const std::vector<DecodedInstruction>& window) {
    
    // Feed-forward network typically shows:
    // 1. Large matrix multiplication
    // 2. Activation function (GELU/ReLU)
    // 3. Another large matrix multiplication
    
    size_t matrix_mul_count = 0;
    bool has_activation = false;
    
    for (const auto& inst : window) {
        if (inst.mnemonic.find("vmul") != std::string::npos) {
            matrix_mul_count++;
        }
        else if (inst.mnemonic.find("vmax") != std::string::npos ||  // ReLU
                 inst.mnemonic.find("vtanh") != std::string::npos) {  // GELU
            has_activation = true;
        }
    }
    
    return matrix_mul_count >= 2 && has_activation;
}

bool TransformerDetector::isAttentionMemoryPattern(
    const MemoryAccessPattern& pattern) const {
    
    // Attention patterns typically show:
    // 1. Regular strides matching hidden size
    // 2. Secondary strides matching head size
    // 3. High consistency ratio
    
    if (pattern.consistency_ratio < 0.8f) {
        return false;
    }
    
    // Check if strides match common transformer dimensions
    bool primary_match = false;
    for (uint32_t size : COMMON_HIDDEN_SIZES) {
        if (pattern.stride_primary == size * sizeof(float)) {
            primary_match = true;
            break;
        }
    }
    
    bool secondary_match = false;
    for (uint32_t heads : COMMON_HEAD_COUNTS) {
        uint32_t head_size = pattern.stride_primary / (heads * sizeof(float));
        if (isPowerOfTwo(head_size)) {
            secondary_match = true;
            break;
        }
    }
    
    return primary_match && secondary_match;
}

bool TransformerDetector::isLayerNormMemoryPattern(
    const MemoryAccessPattern& pattern) const {
    
    // Layer norm shows:
    // 1. Consecutive access pattern
    // 2. High consistency
    // 3. Stride matching hidden size
    
    return pattern.consistency_ratio > 0.9f &&
           pattern.stride_secondary == 0 &&
           isCommonTransformerSize(pattern.stride_primary / sizeof(float));
}

void TransformerDetector::updateDimensionEstimates(
    const MemoryAccessPattern& pattern) {
    
    // Extract potential dimensions from stride patterns
    std::vector<uint32_t> strides = {
        pattern.stride_primary,
        pattern.stride_secondary
    };
    
    analyzeTensorDimensions(strides);
}

void TransformerDetector::analyzeTensorDimensions(
    const std::vector<uint32_t>& strides) {
    
    // Look for common transformer dimensions in strides
    for (uint32_t stride : strides) {
        uint32_t potential_size = stride / sizeof(float);
        
        if (isCommonTransformerSize(potential_size)) {
            // Update detected layer dimensions
            if (!detected_layers_.empty()) {
                auto& current_layer = detected_layers_.back();
                if (std::find(current_layer.dimensions.begin(),
                            current_layer.dimensions.end(),
                            potential_size) == current_layer.dimensions.end()) {
                    current_layer.dimensions.push_back(potential_size);
                }
            }
        }
    }
}

float TransformerDetector::computePatternConfidence(
    const TransformerLayerPattern& pattern) const {
    
    float confidence = 0.0f;
    
    // Base confidence from attention pattern
    confidence += pattern.attention.confidence * 0.4f;
    
    // Layer normalization adds confidence
    if (pattern.has_layer_norm1) confidence += 0.2f;
    if (pattern.has_layer_norm2) confidence += 0.2f;
    
    // Feed-forward network
    if (pattern.has_feedforward) confidence += 0.2f;
    
    // Dimension matching adds confidence
    if (!pattern.dimensions.empty()) {
        float dim_confidence = 0.0f;
        for (uint32_t dim : pattern.dimensions) {
            if (isCommonTransformerSize(dim)) {
                dim_confidence += 0.1f;
            }
        }
        confidence += std::min(0.2f, dim_confidence);
    }
    
    return std::min(1.0f, confidence);
}

void TransformerDetector::updateOverallConfidence() {
    if (detected_layers_.empty()) {
        overall_confidence_ = 0.0f;
        return;
    }
    
    float total_confidence = 0.0f;
    for (const auto& layer : detected_layers_) {
        total_confidence += layer.confidence;
    }
    
    overall_confidence_ = total_confidence / detected_layers_.size();
}

std::vector<TransformerDetector::TransformerLayerPattern>
TransformerDetector::getDetectedLayers() const {
    return detected_layers_;
}

float TransformerDetector::getOverallConfidence() const {
    return overall_confidence_;
}

ArchitectureDetails TransformerDetector::getArchitectureDetails() const {
    ArchitectureDetails details;
    details.type = ModelArchitecture::Transformer;
    details.confidence = overall_confidence_;
    
    if (!detected_layers_.empty()) {
        // Get most common hidden size
        std::unordered_map<uint32_t, size_t> size_counts;
        for (const auto& layer : detected_layers_) {
            for (uint32_t dim : layer.dimensions) {
                size_counts[dim]++;
            }
        }
        
        uint32_t most_common_size = 0;
        size_t max_count = 0;
        for (const auto& pair : size_counts) {
            if (pair.second > max_count) {
                most_common_size = pair.first;
                max_count = pair.second;
            }
        }
        
        if (most_common_size > 0) {
            details.layer_sizes = {most_common_size};
            
            // Try to identify specific architecture
            if (most_common_size == 768) {
                details.specific_type = "BERT-base";
            }
            else if (most_common_size == 1024) {
                details.specific_type = "BERT-large";
            }
        }
    }
    
    return details;
}

} // namespace nn_binary_analysis