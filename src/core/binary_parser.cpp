#include "binary_parser.hpp"
#include "memory_analyzer.hpp"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDisassembler/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include <sstream>

using namespace llvm;

namespace nn_binary_analysis {

// PIMPL class for LLVM components
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
    LLVMInitializeAllTargetInfos();
    LLVMInitializeAllTargetMCs();
    LLVMInitializeAllDisassemblers();

    std::string error;
    llvm_->target.reset(TargetRegistry::lookupTarget(triple_, error));
    
    if (!llvm_->target) {
        initialized_ = false;
        return;
    }

    // Initialize LLVM MC components
    llvm_->reg_info.reset(llvm_->target->createMCRegInfo(triple_));
    if (!llvm_->reg_info) return;

    llvm_->asm_info.reset(llvm_->target->createMCAsmInfo(*llvm_->reg_info, triple_));
    if (!llvm_->asm_info) return;

    llvm_->context.reset(new MCContext(llvm_->asm_info.get(), llvm_->reg_info.get(), nullptr));
    if (!llvm_->context) return;

    llvm_->subtarget_info.reset(llvm_->target->createMCSubtargetInfo(triple_, "", ""));
    if (!llvm_->subtarget_info) return;

    llvm_->disassembler.reset(llvm_->target->createMCDisassembler(*llvm_->subtarget_info, *llvm_->context));
    if (!llvm_->disassembler) return;

    llvm_->instr_info.reset(llvm_->target->createMCInstrInfo());
    if (!llvm_->instr_info) return;

    llvm_->inst_printer.reset(llvm_->target->createMCInstPrinter(
        Triple(triple_), 0, *llvm_->asm_info, *llvm_->instr_info, *llvm_->reg_info));
    if (!llvm_->inst_printer) return;

    // Initialize our analyzers
    llvm_->memory_analyzer = std::make_unique<MemoryAnalyzer>();
    llvm_->pattern_analyzer = std::make_unique<PatternAnalyzer>();

    initialized_ = true;
}

bool BinaryParser::parseSection(const uint8_t* data, size_t size, uint64_t base_addr) {
    if (!initialized_) return false;

    // Register the section with the memory analyzer
    if (!llvm_->memory_analyzer->registerMemoryRegion(
            base_addr, base_addr + size, 0x4, 4)) { // Read permission, 4-byte alignment
        return false;
    }

    current_section_.clear();
    current_section_.reserve(size / 4); // Estimate instruction count

    const uint8_t* ptr = data;
    uint64_t addr = base_addr;
    size_t remaining = size;

    while (remaining > 0) {
        DecodedInstruction decoded;
        if (!parseInstruction(ptr, remaining, addr, decoded)) {
            // Skip problematic byte and continue
            ptr++; addr++; remaining--;
            continue;
        }

        current_section_.push_back(std::move(decoded));
        ptr += decoded.bytes.size();
        addr += decoded.bytes.size();
        remaining -= decoded.bytes.size();
    }

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
    result.mnemonic = inst_string;

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

    // Structural confidence from instruction sequence analysis
    struct_conf = analyzeInstructionSequence(inst);

    // Operation confidence from specific operation patterns
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
    return (desc.TSFlags & MCID::UsesVREGS) != 0;
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
    llvm_->pattern_analyzer->recordAccess(addr_info);
    auto stride_info = llvm_->pattern_analyzer->getCurrentStrideInfo();

    return stride_info.is_tensor_pattern;
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
        // Pattern 1: Vector sequence
        bool has_vector_sequence = false;
        for (size_t i = 0; i < instruction_window.size() - 1; ++i) {
            if (isVectorInstruction(*instruction_window[i]) && 
                isVectorInstruction(*instruction_window[i + 1])) {
                has_vector_sequence = true;
                break;
            }
        }
        if (has_vector_sequence) sequence_confidence += 0.3f;
        
        // Pattern 2: Memory sequence
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

const std::vector<DecodedInstruction>& BinaryParser::getCurrentSection() const {
    return current_section_;
}

} // namespace nn_binary_analysis