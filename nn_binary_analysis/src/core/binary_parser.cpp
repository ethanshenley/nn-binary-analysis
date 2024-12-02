#include "llvm/InitializePasses.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/MC/MCTargetOptionsCommandFlags.h"
#include "llvm/Target/TargetMachine.h"
#include "binary_parser.hpp"
#include "memory_analyzer.hpp"
#include "transformer_detector.hpp"
#include "framework_detector.hpp"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include <sstream>

using namespace llvm;

namespace nn_binary_analysis {

class BinaryParser::LLVMComponents {
public:
    std::unique_ptr<const Target> target;
    std::unique_ptr<MCInstrInfo> instr_info;
    std::unique_ptr<MCRegisterInfo> reg_info;
    std::unique_ptr<MCAsmInfo> asm_info;
    std::unique_ptr<MCContext> context;
    std::unique_ptr<MCDisassembler> disassembler;
    std::unique_ptr<MCInstPrinter> inst_printer;
    std::unique_ptr<MCSubtargetInfo> subtarget_info;
    std::unique_ptr<MemoryAnalyzer> memory_analyzer;
    std::unique_ptr<PatternAnalyzer> pattern_analyzer;
    std::unique_ptr<SourceMgr> src_mgr;  // Add this line
};

BinaryParser::BinaryParser(const std::string& triple, const AnalysisConfig& config)
    : llvm_(std::make_unique<LLVMComponents>())
    , transformer_detector_(std::make_unique<TransformerDetector>(config))
    , framework_detector_(std::make_unique<FrameworkDetector>(config))
    , config_(config)
    , triple_(triple)
    , initialized_(false) {
    initialize();
}

BinaryParser::~BinaryParser() = default;

void BinaryParser::initialize() {
    // Initialize LLVM targets
    InitializeAllTargetInfos();
    InitializeAllTargetMCs();
    InitializeAllAsmParsers();
    InitializeAllDisassemblers();

    std::string error;
    auto target = TargetRegistry::lookupTarget(triple_, error);
    
    if (!target) {
        initialized_ = false;
        return;
    }

    llvm_->target.reset(target);

    // Initialize LLVM MC components
    llvm_->reg_info.reset(target->createMCRegInfo(triple_));
    if (!llvm_->reg_info) return;

    llvm_->asm_info.reset(target->createMCAsmInfo(*llvm_->reg_info, triple_));
    if (!llvm_->asm_info) return;

    // Create SourceMgr as member variable
    llvm_->src_mgr = std::make_unique<SourceMgr>();
    
    // Create MCContext with proper references
    auto *MRI = llvm_->reg_info.get();
    //llvm_->context.reset(new MCContext(*llvm_->asm_info, MRI, llvm_->src_mgr.get(), nullptr));
    //llvm_->context.reset(new MCContext(*llvm_->asm_info, *llvm_->reg_info, llvm_->src_mgr.get()));
    if (!llvm_->context) return;

    std::string CPU = "generic";
    std::string Features = "";
    
    llvm_->subtarget_info.reset(target->createMCSubtargetInfo(triple_, CPU, Features));
    if (!llvm_->subtarget_info) return;

    llvm_->disassembler.reset(target->createMCDisassembler(*llvm_->subtarget_info, *llvm_->context));
    if (!llvm_->disassembler) return;

    llvm_->instr_info.reset(target->createMCInstrInfo());
    if (!llvm_->instr_info) return;

    unsigned AsmPrinterVariant = llvm_->asm_info->getAssemblerDialect();
    llvm_->inst_printer.reset(target->createMCInstPrinter(
        Triple(triple_), AsmPrinterVariant, *llvm_->asm_info,
        *llvm_->instr_info, *llvm_->reg_info));
    if (!llvm_->inst_printer) return;

    // Initialize analyzers
    llvm_->memory_analyzer = std::make_unique<MemoryAnalyzer>();
    llvm_->pattern_analyzer = std::make_unique<PatternAnalyzer>();

    initialized_ = true;
}

bool BinaryParser::parseSection(const uint8_t* data, size_t size, uint64_t base_addr) {
    if (!initialized_) return false;

    // Register memory region
    if (!llvm_->memory_analyzer->registerMemoryRegion(base_addr, base_addr + size, 0x4, 4)) {
        return false;
    }

    current_section_.clear();
    current_section_.reserve(size / 4); // Estimate instruction count
    access_patterns_.clear();

    const uint8_t* ptr = data;
    uint64_t addr = base_addr;
    size_t remaining = size;

    while (remaining > 0) {
        DecodedInstruction decoded;
        if (!parseInstruction(ptr, remaining, addr, decoded)) {
            ptr++; addr++; remaining--;
            continue;
        }

        // Process through all detectors
        transformer_detector_->processInstruction(decoded);
        framework_detector_->processInstruction(decoded);
        
        // Memory and pattern analysis
        if (isMemoryInstruction(decoded)) {
            auto pattern = llvm_->pattern_analyzer->getCurrentStrideInfo();
            if (pattern.is_tensor_pattern) {
                MemoryAccessPattern access_pattern;
                access_pattern.base_address = addr;
                access_pattern.stride_primary = pattern.primary_stride;
                access_pattern.stride_secondary = pattern.secondary_stride;
                access_pattern.consistency_ratio = pattern.confidence;
                processMemoryPattern(access_pattern);
            }
        }

        // Update resource metrics
        updateResourceMetrics(decoded);

        current_section_.push_back(std::move(decoded));
        ptr += decoded.bytes.size();
        addr += decoded.bytes.size();
        remaining -= decoded.bytes.size();
    }

    finalizeAnalysis();
    return true;
}

bool BinaryParser::parseInstruction(const uint8_t* data, size_t max_size, 
                                  uint64_t addr, DecodedInstruction& result) {
    if (!initialized_) return false;

    MCInst inst;
    uint64_t inst_size;
    
    MCDisassembler::DecodeStatus status = 
        llvm_->disassembler->getInstruction(inst, inst_size, 
                                          ArrayRef<uint8_t>(data, max_size),
                                          addr, nulls());

    if (status != MCDisassembler::Success) {
        return false;
    }

    // Fill basic instruction information
    result.address = addr;
    result.bytes = std::vector<uint8_t>(data, data + inst_size);
    
    // Get instruction string
    std::string inst_string;
    raw_string_ostream os(inst_string);
    llvm_->inst_printer->printInst(&inst, os, "", *llvm_->subtarget_info);
    os.flush();  // Flush the stream
    result.mnemonic = os.str(); // Get the string from the stream

    // Analyze for tensor operations
    detectPotentialTensorOp(inst, result);
    computeConfidenceMetrics(inst, result);

    return true;
}

bool BinaryParser::detectPotentialTensorOp(const MCInst& inst, DecodedInstruction& decoded) {
    // Check for vector instructions
    if (isVectorInstruction(inst)) {
        decoded.potential_tensor_op = true;
        return true;
    }

    // Check for memory instructions with regular patterns
    if (isMemoryInstruction(inst) && hasRegularStridePattern(inst)) {
        decoded.potential_tensor_op = true;
        return true;
    }

    decoded.potential_tensor_op = false;
    return false;
}

void BinaryParser::computeConfidenceMetrics(const MCInst& inst, DecodedInstruction& decoded) {
    float pattern_conf = 0.0f;
    float struct_conf = 0.0f;
    float op_conf = 0.0f;

    // Pattern confidence from instruction characteristics
    if (isVectorInstruction(inst)) {
        pattern_conf += 0.4f;
    }
    if (isMemoryInstruction(inst)) {
        pattern_conf += analyzeMemoryAccessConfidence(inst);
    }

    // Structural confidence from instruction sequence
    struct_conf = analyzeInstructionSequence(inst);

    // Operation confidence
    if (hasRegularStridePattern(inst)) {
        op_conf += 0.3f;
    }

    decoded.confidence = ConfidenceMetrics{
        std::min(1.0f, pattern_conf),
        std::min(1.0f, struct_conf),
        std::min(1.0f, op_conf)
    };
}

bool BinaryParser::isVectorInstruction(const MCInst& inst) const {
    const MCInstrDesc& desc = llvm_->instr_info->get(inst.getOpcode());
    
    // Check for vector/SIMD instructions using available flags
    bool isVector = false;
    
    // Check instruction format and flags
    uint64_t Flags = desc.TSFlags;
    if (Flags & 0x1) {  // Check first bit for vector ops - you might need to adjust this based on LLVM version
        isVector = true;
    }
    
    // Check opcode range for known vector instruction sets
    unsigned Opcode = inst.getOpcode();
    if (Opcode >= 0x500 && Opcode <= 0x600) {  // Example range, adjust based on your target
        isVector = true;
    }
    
    return isVector;
}

bool BinaryParser::isMemoryInstruction(const MCInst& inst) const {
    const MCInstrDesc& desc = llvm_->instr_info->get(inst.getOpcode());
    return desc.mayLoad() || desc.mayStore();
}

bool BinaryParser::hasRegularStridePattern(const MCInst& inst) const {
    if (!isMemoryInstruction(inst)) return false;

    // Get memory operand information
    const MCInstrDesc& desc = llvm_->instr_info->get(inst.getOpcode());
    
    // Extract base address from instruction
    uint64_t base_addr = 0;
    for (unsigned i = 0; i < inst.getNumOperands(); ++i) {
        const MCOperand& op = inst.getOperand(i);
        if (op.isImm()) {
            base_addr = static_cast<uint64_t>(op.getImm());
            break;
        }
    }

    // Analyze the addressing pattern
    auto addr_info = llvm_->memory_analyzer->translateAddress(base_addr);
    if (!addr_info.is_valid) {
        return false;
    }

    // Record access and check pattern
    return llvm_->pattern_analyzer->getCurrentStrideInfo().is_tensor_pattern;
}

float BinaryParser::analyzeMemoryAccessConfidence(const MCInst& inst) {
    if (!isMemoryInstruction(inst)) return 0.0f;

    auto stride_info = llvm_->pattern_analyzer->getCurrentStrideInfo();
    return stride_info.confidence;
}

float BinaryParser::analyzeInstructionSequence(const MCInst& inst) {
    static thread_local std::deque<const MCInst*> instruction_window;
    static constexpr size_t WINDOW_SIZE = 8;
    
    float sequence_confidence = 0.0f;
    
    // Update instruction window
    instruction_window.push_back(&inst);
    if (instruction_window.size() > WINDOW_SIZE) {
        instruction_window.pop_front();
    }
    
    // Analyze instruction sequence patterns
    if (instruction_window.size() >= 3) {
        // Vector sequence
        bool has_vector_sequence = false;
        for (size_t i = 0; i < instruction_window.size() - 1; ++i) {
            if (isVectorInstruction(*instruction_window[i]) && 
                isVectorInstruction(*instruction_window[i + 1])) {
                has_vector_sequence = true;
                break;
            }
        }
        if (has_vector_sequence) sequence_confidence += 0.3f;
        
        // Memory sequence
        bool has_memory_sequence = false;
        for (size_t i = 0; i < instruction_window.size() - 2; ++i) {
            if (isMemoryInstruction(*instruction_window[i]) && 
                isMemoryInstruction(*instruction_window[i + 1]) &&
                isMemoryInstruction(*instruction_window[i + 2])) {
                has_memory_sequence = true;
                break;
            }
        }
        if (has_memory_sequence) sequence_confidence += 0.2f;
    }
    
    return std::min(1.0f, sequence_confidence);
}

void BinaryParser::updateResourceMetrics(const DecodedInstruction& inst) {
    if (isVectorInstruction(inst)) {
        resource_metrics_.simd_utilization += 0.1f;
        resource_metrics_.simd_utilization = std::min(1.0f, resource_metrics_.simd_utilization);
    }
    
    if (isMemoryInstruction(inst)) {
        resource_metrics_.memory_bandwidth += 0.1f;
        resource_metrics_.memory_bandwidth = std::min(1.0f, resource_metrics_.memory_bandwidth);
    }
}

void BinaryParser::processMemoryPattern(const MemoryAccessPattern& pattern) {
    access_patterns_.push_back(pattern);
    
    // Update cache efficiency based on stride patterns
    if (pattern.consistency_ratio > 0.8f && pattern.stride_primary <= 64) {
        resource_metrics_.cache_efficiency += 0.1f;
        resource_metrics_.cache_efficiency = std::min(1.0f, resource_metrics_.cache_efficiency);
    }
}

void BinaryParser::finalizeAnalysis() {
    if (config_.enable_pattern_detection) {
        transformer_detector_->detectTransformerPatterns(access_patterns_);
    }
}

AnalysisResult BinaryParser::getAnalysisResult() const {
    AnalysisResult result;
    
    result.framework = framework_detector_->getFrameworkDetails();
    result.architecture = transformer_detector_->getArchitectureDetails();
    result.resource_usage = resource_metrics_;
    result.access_patterns = access_patterns_;
    
    result.overall_confidence = ConfidenceMetrics{
        transformer_detector_->getOverallConfidence(),
        result.framework.confidence,
        result.resource_usage.overall_efficiency()
    };
    
    return result;
}

const std::vector<DecodedInstruction>& BinaryParser::getCurrentSection() const {
    return current_section_;
}

} // namespace nn_binary_analysis