#pragma once

#include "types.hpp"
#include <vector>
#include <string>
#include <unordered_map>

namespace nn_binary_analysis {

class FrameworkDetector {
public:
    explicit FrameworkDetector(const AnalysisConfig& config);
    
    // Process instruction sequence for framework detection
    void processInstruction(const DecodedInstruction& inst);
    
    // Get current framework detection results
    FrameworkDetails getFrameworkDetails() const;
    
    // Check for specific frameworks
    bool isLikelyPyTorch() const;
    bool isLikelyLibTorch() const;
    
private:
    struct FrameworkSignature {
        std::vector<std::string> instruction_patterns;
        std::vector<std::string> symbol_patterns;
        std::vector<std::string> string_patterns;
    };
    
    struct DetectionState {
        size_t matched_patterns{0};
        float confidence{0.0f};
        std::string version;
    };

    const AnalysisConfig& config_;
    std::deque<DecodedInstruction> instruction_window_;
    std::unordered_map<MLFramework, DetectionState> detection_states_;
    
    // Framework-specific signature patterns
    std::unordered_map<MLFramework, FrameworkSignature> signatures_;
    
    // Detection methods
    void initializeSignatures();
    void updateDetectionState(const DecodedInstruction& inst);
    bool matchesPattern(const std::string& text, const std::vector<std::string>& patterns);
    float computeConfidence(const DetectionState& state) const;
    void detectVersion(MLFramework framework, const std::string& text);
};

} // namespace nn_binary_analysis