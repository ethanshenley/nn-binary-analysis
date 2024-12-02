#include "framework_detector.hpp"
#include <algorithm>
#include <regex>

namespace nn_binary_analysis {

namespace {
    // Common patterns for different frameworks
    const std::unordered_map<MLFramework, std::vector<std::string>> FRAMEWORK_SYMBOLS = {
        {MLFramework::LibTorch, {
            "torch", "c10", "caffe2", "aten",
            "_ZN3c108TensorId", // Tensor class symbols
            "_ZN5torch3jit"     // TorchScript symbols
        }},
        {MLFramework::PyTorch, {
            "pytorch", "torch.nn",
            "_ZN3c108autograd",
            "python3.torch"
        }},
        {MLFramework::TensorFlow, {
            "tensorflow", "tf.", "tf2",
            "_ZN10tensorflow",
            "tfl1_detect"
        }}
    };

    const std::unordered_map<MLFramework, std::vector<std::string>> FRAMEWORK_STRINGS = {
        {MLFramework::LibTorch, {
            "LibTorch", "torch::jit",
            "torch.compilation",
            ".pt", ".pth"
        }},
        {MLFramework::PyTorch, {
            "torch.nn.Module",
            "torch.utils",
            "pytorch_model"
        }},
        {MLFramework::TensorFlow, {
            "tf.keras",
            "saved_model.pb",
            "tensorflow.python"
        }}
    };
}

FrameworkDetector::FrameworkDetector(const AnalysisConfig& config)
    : config_(config) {
    initializeSignatures();
}

void FrameworkDetector::initializeSignatures() {
    // Initialize LibTorch signatures
    signatures_[MLFramework::LibTorch] = {
        // Instruction patterns
        {
            "movabs.*torch",
            "call.*aten",
            "call.*c10"
        },
        // Symbol patterns
        FRAMEWORK_SYMBOLS.at(MLFramework::LibTorch),
        // String patterns
        FRAMEWORK_STRINGS.at(MLFramework::LibTorch)
    };

    // Initialize other frameworks similarly
    signatures_[MLFramework::PyTorch] = {
        {
            "movabs.*pytorch",
            "call.*torch"
        },
        FRAMEWORK_SYMBOLS.at(MLFramework::PyTorch),
        FRAMEWORK_STRINGS.at(MLFramework::PyTorch)
    };
}

void FrameworkDetector::processInstruction(const DecodedInstruction& inst) {
    instruction_window_.push_back(inst);
    if (instruction_window_.size() > config_.pattern_detection.pattern_buffer_size) {
        instruction_window_.pop_front();
    }

    updateDetectionState(inst);
}

void FrameworkDetector::updateDetectionState(const DecodedInstruction& inst) {
    // Check instruction against all framework patterns
    for (const auto& [framework, signature] : signatures_) {
        auto& state = detection_states_[framework];

        // Check instruction patterns
        if (matchesPattern(inst.mnemonic, signature.instruction_patterns)) {
            state.matched_patterns++;
        }

        // Check for framework-specific strings in instruction bytes
        std::string bytes_as_string(inst.bytes.begin(), inst.bytes.end());
        if (matchesPattern(bytes_as_string, signature.string_patterns)) {
            state.matched_patterns++;
            detectVersion(framework, bytes_as_string);
        }

        // Update confidence
        state.confidence = computeConfidence(state);
    }
}

bool FrameworkDetector::matchesPattern(const std::string& text, 
                                     const std::vector<std::string>& patterns) {
    for (const auto& pattern : patterns) {
        std::regex rx(pattern, std::regex::icase);
        if (std::regex_search(text, rx)) {
            return true;
        }
    }
    return false;
}

float FrameworkDetector::computeConfidence(const DetectionState& state) const {
    // Basic confidence based on number of matched patterns
    float base_confidence = std::min(1.0f, 
        static_cast<float>(state.matched_patterns) / 10.0f);
    
    // Boost confidence if version information is found
    if (!state.version.empty()) {
        base_confidence *= 1.2f;
    }

    return std::min(1.0f, base_confidence);
}

void FrameworkDetector::detectVersion(MLFramework framework, const std::string& text) {
    // Version detection patterns
    std::unordered_map<MLFramework, std::regex> version_patterns = {
        {MLFramework::LibTorch, std::regex("torch\\s+version\\s+([0-9.]+)")},
        {MLFramework::PyTorch, std::regex("pytorch\\s+([0-9.]+)")}
    };

    auto it = version_patterns.find(framework);
    if (it != version_patterns.end()) {
        std::smatch match;
        if (std::regex_search(text, match, it->second)) {
            detection_states_[framework].version = match[1];
        }
    }
}

FrameworkDetails FrameworkDetector::getFrameworkDetails() const {
    FrameworkDetails result;
    float max_confidence = 0.0f;

    for (const auto& [framework, state] : detection_states_) {
        if (state.confidence > max_confidence) {
            max_confidence = state.confidence;
            result.type = framework;
            result.version = state.version;
            result.confidence = state.confidence;
        }
    }

    return result;
}

bool FrameworkDetector::isLikelyPyTorch() const {
    auto it = detection_states_.find(MLFramework::PyTorch);
    return it != detection_states_.end() && it->second.confidence > 0.7f;
}

bool FrameworkDetector::isLikelyLibTorch() const {
    auto it = detection_states_.find(MLFramework::LibTorch);
    return it != detection_states_.end() && it->second.confidence > 0.7f;
}

} // namespace nn_binary_analysis