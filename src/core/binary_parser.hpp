#pragma once

#include "types.hpp"
#include "transformer_detector.hpp"
#include "framework_detector.hpp"
#include <memory>
#include <string>
#include <vector>

namespace llvm {
class MCInst;
class MCInstrInfo;
class MCRegisterInfo;
class MCSubtargetInfo;
class MCContext;
class MCDisassembler;
class Target;
class MCAsmInfo;
class MCInstPrinter;
}

namespace nn_binary_analysis {

class BinaryParser {
public:
    explicit BinaryParser(const std::string& triple, const AnalysisConfig& config);
    ~BinaryParser();

    // Prevent copying due to LLVM resource management
    BinaryParser(const BinaryParser&) = delete;
    BinaryParser& operator=(const BinaryParser&) = delete;

    // Core parsing functionality
    bool parseSection(const uint8_t* data, size_t size, uint64_t base_addr);
    bool parseInstruction(const uint8_t* data, size_t max_size, uint64_t addr, DecodedInstruction& result);

    // Enhanced analysis results
    AnalysisResult getAnalysisResult() const;
    const std::vector<DecodedInstruction>& getCurrentSection() const;
    
    // Configuration and status
    bool isInitialized() const { return initialized_; }
    const std::string& getTriple() const { return triple_; }

private:
    // LLVM components (implementation details hidden in cpp)
    class LLVMComponents;
    std::unique_ptr<LLVMComponents> llvm_;

    // Analysis components
    std::unique_ptr<TransformerDetector> transformer_detector_;
    std::unique_ptr<FrameworkDetector> framework_detector_;
    AnalysisConfig config_;

    // Parser state
    std::string triple_;
    bool initialized_;
    std::vector<DecodedInstruction> current_section_;
    std::vector<MemoryAccessPattern> access_patterns_;
    ResourceMetrics resource_metrics_;

    // Internal helpers
    void initialize();
    bool detectPotentialTensorOp(const llvm::MCInst& inst, DecodedInstruction& decoded);
    void computeConfidenceMetrics(const llvm::MCInst& inst, DecodedInstruction& decoded);
    
    // Pattern analysis helpers
    bool analyzeMemoryAccess(const llvm::MCInst& inst);
    float analyzeMemoryAccessConfidence(const llvm::MCInst& inst);
    float analyzeInstructionSequence(const llvm::MCInst& inst);

    // Instruction classification helpers
    bool isVectorInstruction(const llvm::MCInst& inst) const;
    bool isMemoryInstruction(const llvm::MCInst& inst) const;
    bool hasRegularStridePattern(const llvm::MCInst& inst) const;
    
    // New analysis methods
    void updateResourceMetrics(const DecodedInstruction& inst);
    void processMemoryPattern(const MemoryAccessPattern& pattern);
    void finalizeAnalysis();
};

} // namespace nn_binary_analysis