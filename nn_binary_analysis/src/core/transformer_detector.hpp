#pragma once

#include "types.hpp"
#include <memory>
#include <vector>
#include <deque>

namespace nn_binary_analysis {

class TransformerDetector {
public:
    struct AttentionPattern {
        bool has_query_projection{false};
        bool has_key_projection{false};
        bool has_value_projection{false};
        bool has_attention_multiply{false};
        bool has_softmax{false};
        uint32_t hidden_size{0};
        uint32_t num_heads{0};
        float confidence{0.0f};
    };

    struct TransformerLayerPattern {
        AttentionPattern attention;
        bool has_layer_norm1{false};
        bool has_layer_norm2{false};
        bool has_feedforward{false};
        std::vector<uint32_t> dimensions;
        float confidence{0.0f};
    };

    explicit TransformerDetector(const AnalysisConfig& config);

    // Main detection interface
    void processInstruction(const DecodedInstruction& inst);
    bool detectTransformerPatterns(const std::vector<MemoryAccessPattern>& patterns);
    
    // Results access
    std::vector<TransformerLayerPattern> getDetectedLayers() const;
    float getOverallConfidence() const;
    ArchitectureDetails getArchitectureDetails() const;

private:
    const AnalysisConfig& config_;
    std::deque<DecodedInstruction> instruction_window_;
    std::vector<TransformerLayerPattern> detected_layers_;
    float overall_confidence_{0.0f};

    // Pattern detection helpers
    bool detectAttentionPattern(const std::vector<DecodedInstruction>& window);
    bool detectLayerNormPattern(const std::vector<DecodedInstruction>& window);
    bool detectFeedForwardPattern(const std::vector<DecodedInstruction>& window);
    
    // Memory pattern analysis
    bool isAttentionMemoryPattern(const MemoryAccessPattern& pattern) const;
    bool isLayerNormMemoryPattern(const MemoryAccessPattern& pattern) const;
    
    // Dimension analysis
    void updateDimensionEstimates(const MemoryAccessPattern& pattern);
    void analyzeTensorDimensions(const std::vector<uint32_t>& strides);
    
    // Confidence computation
    float computePatternConfidence(const TransformerLayerPattern& pattern) const;
    float computeArchitectureConfidence() const;
    void updateOverallConfidence();
};

} // namespace nn_binary_analysis