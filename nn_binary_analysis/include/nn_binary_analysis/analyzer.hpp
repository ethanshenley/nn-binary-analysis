#pragma once

#include "types.hpp"
#include <string>
#include <memory>
#include <stdexcept>

namespace nn_binary_analysis {

class BinaryAnalyzer {
public:
    explicit BinaryAnalyzer(const std::string& triple, const AnalysisConfig& config);
    ~BinaryAnalyzer();

    // Prevent copying
    BinaryAnalyzer(const BinaryAnalyzer&) = delete;
    BinaryAnalyzer& operator=(const BinaryAnalyzer&) = delete;

    // Core analysis function
    AnalysisResult analyze(const uint8_t* data, size_t size);

    // Status and configuration
    bool isInitialized() const;
    const AnalysisConfig& getConfig() const;
    void updateConfig(const AnalysisConfig& config);

private:
    class Implementation;
    std::unique_ptr<Implementation> impl_;
};

// Exception classes for error handling
class AnalysisError : public std::runtime_error {
public:
    explicit AnalysisError(const std::string& msg) : std::runtime_error(msg) {}
};

class InitializationError : public AnalysisError {
public:
    explicit InitializationError(const std::string& msg) 
        : AnalysisError("Initialization failed: " + msg) {}
};

} // namespace nn_binary_analysis