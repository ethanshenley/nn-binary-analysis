# .gitignore

```
__pycache__

.env
```

# .vscode/c_cpp_properties.json

```json
{
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**",
                "/usr/include/llvm-15",
                "/usr/include/llvm-15/llvm",
                "${workspaceFolder}/include",
                "${workspaceFolder}/src"
            ],
            "defines": [],
            "compilerPath": "/usr/bin/clang++-15",
            "cStandard": "c17",
            "cppStandard": "c++17",
            "intelliSenseMode": "linux-clang-x64",
            "compileCommands": "${workspaceFolder}/build/compile_commands.json"
        }
    ],
    "version": 4
}
```

# .vscode/settings.json

```json
{
    "files.associations": {
        "array": "cpp",
        "atomic": "cpp",
        "bit": "cpp",
        "*.tcc": "cpp",
        "bitset": "cpp",
        "cctype": "cpp",
        "clocale": "cpp",
        "cmath": "cpp",
        "compare": "cpp",
        "concepts": "cpp",
        "cstdarg": "cpp",
        "cstddef": "cpp",
        "cstdint": "cpp",
        "cstdio": "cpp",
        "cstdlib": "cpp",
        "ctime": "cpp",
        "cwchar": "cpp",
        "cwctype": "cpp",
        "deque": "cpp",
        "string": "cpp",
        "unordered_map": "cpp",
        "vector": "cpp",
        "exception": "cpp",
        "algorithm": "cpp",
        "functional": "cpp",
        "iterator": "cpp",
        "memory": "cpp",
        "memory_resource": "cpp",
        "numeric": "cpp",
        "optional": "cpp",
        "random": "cpp",
        "regex": "cpp",
        "string_view": "cpp",
        "system_error": "cpp",
        "tuple": "cpp",
        "type_traits": "cpp",
        "utility": "cpp",
        "initializer_list": "cpp",
        "iosfwd": "cpp",
        "iostream": "cpp",
        "istream": "cpp",
        "limits": "cpp",
        "new": "cpp",
        "numbers": "cpp",
        "ostream": "cpp",
        "sstream": "cpp",
        "stdexcept": "cpp",
        "streambuf": "cpp",
        "typeinfo": "cpp",
        "cstring": "cpp",
        "fstream": "cpp",
        "iomanip": "cpp"
    }
}
```

# CMakeLists.txt

```txt
cmake_minimum_required(VERSION 3.15)
project(nn_binary_analysis VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find LLVM package
find_package(LLVM 15.0 REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Add LLVM flags
include_directories(SYSTEM ${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# Get required LLVM libraries
llvm_map_components_to_libnames(llvm_libs
    Support
    Core
    IRReader
    MCParser
    MC
    MCDisassembler
    Target
    AllTargetsDescs
    AllTargetsDisassemblers
    AllTargetsInfos
)

# Add subdirectory containing our actual project
add_subdirectory(nn_binary_analysis)
```

# compile_commands.json

```json
[
{
  "directory": "/home/ethan/Documents/GitHub/nn-binary-analysis/build/nn_binary_analysis",
  "command": "/usr/bin/c++ -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -I/home/ethan/Documents/GitHub/nn-binary-analysis/nn_binary_analysis/include -I/home/ethan/Documents/GitHub/nn-binary-analysis/nn_binary_analysis/src -isystem /usr/lib/llvm-15/include -std=gnu++17 -o CMakeFiles/nn_binary_analysis.dir/src/core/binary_parser.cpp.o -c /home/ethan/Documents/GitHub/nn-binary-analysis/nn_binary_analysis/src/core/binary_parser.cpp",
  "file": "/home/ethan/Documents/GitHub/nn-binary-analysis/nn_binary_analysis/src/core/binary_parser.cpp",
  "output": "nn_binary_analysis/CMakeFiles/nn_binary_analysis.dir/src/core/binary_parser.cpp.o"
},
{
  "directory": "/home/ethan/Documents/GitHub/nn-binary-analysis/build/nn_binary_analysis",
  "command": "/usr/bin/c++ -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -I/home/ethan/Documents/GitHub/nn-binary-analysis/nn_binary_analysis/include -I/home/ethan/Documents/GitHub/nn-binary-analysis/nn_binary_analysis/src -isystem /usr/lib/llvm-15/include -std=gnu++17 -o CMakeFiles/nn_binary_analysis.dir/src/core/memory_analyzer.cpp.o -c /home/ethan/Documents/GitHub/nn-binary-analysis/nn_binary_analysis/src/core/memory_analyzer.cpp",
  "file": "/home/ethan/Documents/GitHub/nn-binary-analysis/nn_binary_analysis/src/core/memory_analyzer.cpp",
  "output": "nn_binary_analysis/CMakeFiles/nn_binary_analysis.dir/src/core/memory_analyzer.cpp.o"
},
{
  "directory": "/home/ethan/Documents/GitHub/nn-binary-analysis/build/nn_binary_analysis",
  "command": "/usr/bin/c++ -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -I/home/ethan/Documents/GitHub/nn-binary-analysis/nn_binary_analysis/include -I/home/ethan/Documents/GitHub/nn-binary-analysis/nn_binary_analysis/src -isystem /usr/lib/llvm-15/include -std=gnu++17 -o CMakeFiles/nn_binary_analysis.dir/src/core/transformer_detector.cpp.o -c /home/ethan/Documents/GitHub/nn-binary-analysis/nn_binary_analysis/src/core/transformer_detector.cpp",
  "file": "/home/ethan/Documents/GitHub/nn-binary-analysis/nn_binary_analysis/src/core/transformer_detector.cpp",
  "output": "nn_binary_analysis/CMakeFiles/nn_binary_analysis.dir/src/core/transformer_detector.cpp.o"
},
{
  "directory": "/home/ethan/Documents/GitHub/nn-binary-analysis/build/nn_binary_analysis",
  "command": "/usr/bin/c++ -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -I/home/ethan/Documents/GitHub/nn-binary-analysis/nn_binary_analysis/include -I/home/ethan/Documents/GitHub/nn-binary-analysis/nn_binary_analysis/src -isystem /usr/lib/llvm-15/include -std=gnu++17 -o CMakeFiles/nn_binary_analysis.dir/src/core/framework_detector.cpp.o -c /home/ethan/Documents/GitHub/nn-binary-analysis/nn_binary_analysis/src/core/framework_detector.cpp",
  "file": "/home/ethan/Documents/GitHub/nn-binary-analysis/nn_binary_analysis/src/core/framework_detector.cpp",
  "output": "nn_binary_analysis/CMakeFiles/nn_binary_analysis.dir/src/core/framework_detector.cpp.o"
},
{
  "directory": "/home/ethan/Documents/GitHub/nn-binary-analysis/build/nn_binary_analysis",
  "command": "/usr/bin/c++ -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS -I/home/ethan/Documents/GitHub/nn-binary-analysis/nn_binary_analysis/include -isystem /usr/lib/llvm-15/include -std=gnu++17 -o CMakeFiles/nn_detect.dir/tools/nn_detect/main.cpp.o -c /home/ethan/Documents/GitHub/nn-binary-analysis/nn_binary_analysis/tools/nn_detect/main.cpp",
  "file": "/home/ethan/Documents/GitHub/nn-binary-analysis/nn_binary_analysis/tools/nn_detect/main.cpp",
  "output": "nn_binary_analysis/CMakeFiles/nn_detect.dir/tools/nn_detect/main.cpp.o"
}
]
```

# include/nn_binary_analysis/analyzer.hpp

```hpp
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
```

# nn_binary_analysis/CMakeLists.txt

```txt
# Library sources
set(LIB_SOURCES
    src/core/binary_parser.cpp
    src/core/memory_analyzer.cpp
    src/core/transformer_detector.cpp
    src/core/framework_detector.cpp
)

# Create library
add_library(nn_binary_analysis ${LIB_SOURCES})

target_include_directories(nn_binary_analysis
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${LLVM_INCLUDE_DIRS}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(nn_binary_analysis
    PUBLIC 
        ${llvm_libs}
)

# Demo executable
add_executable(nn_detect tools/nn_detect/main.cpp)
target_link_libraries(nn_detect PRIVATE nn_binary_analysis)
```

# nn_binary_analysis/include/nn_binary_analysis/analyzer.hpp

```hpp
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
```

# nn_binary_analysis/src/core/binary_parser.cpp

```cpp
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
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
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
```

# nn_binary_analysis/src/core/binary_parser.hpp

```hpp
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
```

# nn_binary_analysis/src/core/framework_detector.cpp

```cpp
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
```

# nn_binary_analysis/src/core/framework_detector.hpp

```hpp
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
```

# nn_binary_analysis/src/core/memory_analyzer.cpp

```cpp
#include "memory_analyzer.hpp"
#include <algorithm>
#include <cassert>

namespace nn_binary_analysis {

// MemoryAnalyzer Implementation
MemoryAnalyzer::MemoryAnalyzer(uint32_t default_alignment)
    : default_alignment_(default_alignment) {
    // Reserve space for typical number of memory regions
    memory_regions_.reserve(16);
}

bool MemoryAnalyzer::registerMemoryRegion(uint64_t start, uint64_t end,
                                        uint32_t perms, uint32_t alignment) {
    if (start >= end) return false;
    
    // Check for overlapping regions
    for (const auto& region : memory_regions_) {
        if ((start >= region.start_address && start < region.end_address) ||
            (end > region.start_address && end <= region.end_address)) {
            return false;
        }
    }

    // Ensure alignment is power of 2
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        alignment = default_alignment_;
    }

    memory_regions_.push_back({start, end, perms, alignment});
    
    // Keep regions sorted by start address
    std::sort(memory_regions_.begin(), memory_regions_.end(),
              [](const MemoryRegion& a, const MemoryRegion& b) {
                  return a.start_address < b.start_address;
              });
    
    return true;
}

MemoryAnalyzer::AddressTranslation 
MemoryAnalyzer::translateAddress(uint64_t virtual_addr) const {
    AddressTranslation result{virtual_addr, 0, false, default_alignment_};
    
    // Binary search for the containing region
    auto it = std::lower_bound(
        memory_regions_.begin(), memory_regions_.end(),
        virtual_addr,
        [](const MemoryRegion& region, uint64_t addr) {
            return region.end_address <= addr;
        });
    
    if (it != memory_regions_.end() && 
        virtual_addr >= it->start_address && 
        virtual_addr < it->end_address) {
        result.physical_addr = virtual_addr - it->start_address;
        result.is_valid = true;
        result.alignment = it->alignment;
    }
    
    return result;
}

bool MemoryAnalyzer::isAlignmentValid(uint64_t addr, uint32_t required_alignment) const {
    return (addr % required_alignment) == 0;
}

// PatternAnalyzer Implementation
class PatternAnalyzer::CircularBuffer {
public:
    explicit CircularBuffer(size_t capacity)
        : buffer_(capacity)
        , head_(0)
        , size_(0)
        , capacity_(capacity) {}

    void push_back(const MemoryAnalyzer::AddressTranslation& addr) {
        buffer_[head_] = addr;
        head_ = (head_ + 1) % capacity_;
        if (size_ < capacity_) size_++;
    }

    size_t size() const { return size_; }
    
    const MemoryAnalyzer::AddressTranslation& operator[](size_t idx) const {
        assert(idx < size_);
        return buffer_[(head_ + capacity_ - size_ + idx) % capacity_];
    }

private:
    std::vector<MemoryAnalyzer::AddressTranslation> buffer_;
    size_t head_;
    size_t size_;
    size_t capacity_;
};

PatternAnalyzer::PatternAnalyzer(size_t window_size)
    : access_window_(std::make_unique<CircularBuffer>(window_size))
    , minimum_pattern_length_(3) {
    current_stride_info_ = {0, 0, 0.0f, false};
}

void PatternAnalyzer::recordAccess(const MemoryAnalyzer::AddressTranslation& addr_info) {
    if (!addr_info.is_valid) return;
    
    access_window_->push_back(addr_info);
    updatePatternAnalysis();
}

void PatternAnalyzer::updatePatternAnalysis() {
    if (access_window_->size() < minimum_pattern_length_) return;

    std::unordered_map<uint32_t, uint32_t> stride_counts;
    uint32_t max_count = 0;
    uint32_t primary_stride = 0;

    // Analyze primary stride pattern
    for (size_t i = 1; i < access_window_->size(); ++i) {
        uint64_t curr_addr = (*access_window_)[i].physical_addr;
        uint64_t prev_addr = (*access_window_)[i-1].physical_addr;
        
        if (curr_addr > prev_addr) {
            uint32_t stride = static_cast<uint32_t>(curr_addr - prev_addr);
            uint32_t count = ++stride_counts[stride];
            
            if (count > max_count) {
                max_count = count;
                primary_stride = stride;
            }
        }
    }

    // Analyze secondary stride pattern
    uint32_t secondary_stride = 0;
    if (max_count >= minimum_pattern_length_) {
        std::unordered_map<uint32_t, uint32_t> secondary_counts;
        
        for (size_t i = 2; i < access_window_->size(); ++i) {
            uint64_t curr_addr = (*access_window_)[i].physical_addr;
            uint64_t prev_addr = (*access_window_)[i-2].physical_addr;
            
            if (curr_addr > prev_addr) {
                uint32_t stride = static_cast<uint32_t>(curr_addr - prev_addr);
                if (stride > primary_stride) {
                    uint32_t count = ++secondary_counts[stride];
                    if (count >= minimum_pattern_length_ / 2) {
                        secondary_stride = stride;
                        break;
                    }
                }
            }
        }
    }

    // Update stride information
    float confidence = static_cast<float>(max_count) / (access_window_->size() - 1);
    bool is_tensor = isTensorPattern(primary_stride, secondary_stride, confidence);
    
    current_stride_info_ = {
        primary_stride,
        secondary_stride,
        confidence,
        is_tensor
    };
}

bool PatternAnalyzer::isTensorPattern(uint32_t primary, uint32_t secondary, 
                                    float confidence) const {
    if (confidence < 0.7f) return false;

    // Check for power-of-2 strides (common in tensor operations)
    bool primary_pow2 = (primary & (primary - 1)) == 0;
    bool secondary_pow2 = secondary == 0 || (secondary & (secondary - 1)) == 0;

    // Check for typical tensor operation patterns
    bool typical_primary = primary == 4 || primary == 8 || 
                         primary == 16 || primary == 32;
    bool typical_secondary = secondary == 0 || secondary >= primary;

    return (primary_pow2 && secondary_pow2) || 
           (typical_primary && typical_secondary);
}

StrideInfo PatternAnalyzer::getCurrentStrideInfo() const {
    return current_stride_info_;
}

} // namespace nn_binary_analysis
```

# nn_binary_analysis/src/core/memory_analyzer.hpp

```hpp
#pragma once

#include "types.hpp"
#include <deque>
#include <unordered_map>

namespace nn_binary_analysis {

// Forward declarations and common structures
struct StrideInfo {
    uint32_t primary_stride{0};
    uint32_t secondary_stride{0};
    float confidence{0.0f};
    bool is_tensor_pattern{false};
};

class MemoryAnalyzer {
public:
    struct MemoryRegion {
        uint64_t start_address;
        uint64_t end_address;
        uint32_t permissions;
        uint32_t alignment;
    };

    struct AddressTranslation {
        uint64_t virtual_addr;
        uint64_t physical_addr;
        bool is_valid;
        uint32_t alignment;
    };

    explicit MemoryAnalyzer(uint32_t default_alignment = 4096);
    
    bool registerMemoryRegion(uint64_t start, uint64_t end, 
                            uint32_t perms, uint32_t alignment);
    AddressTranslation translateAddress(uint64_t virtual_addr) const;
    bool isAlignmentValid(uint64_t addr, uint32_t required_alignment) const;

private:
    std::vector<MemoryRegion> memory_regions_;
    uint32_t default_alignment_;
};

class PatternAnalyzer {
public:
    explicit PatternAnalyzer(size_t window_size = 1024);
    
    void recordAccess(const MemoryAnalyzer::AddressTranslation& addr_info);
    StrideInfo getCurrentStrideInfo() const;

private:
    class CircularBuffer;
    std::unique_ptr<CircularBuffer> access_window_;
    StrideInfo current_stride_info_;
    size_t minimum_pattern_length_;

    void updatePatternAnalysis();
    bool isTensorPattern(uint32_t primary, uint32_t secondary, float confidence) const;
};

} // namespace nn_binary_analysis
```

# nn_binary_analysis/src/core/transformer_detector.cpp

```cpp
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
```

# nn_binary_analysis/src/core/transformer_detector.hpp

```hpp
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
```

# nn_binary_analysis/src/core/types.hpp

```hpp
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

} // namespace nn_binary_analysis
```

# nn_binary_analysis/tools/nn_detect/main.cpp

```cpp
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
```

# README.md

```md
# Neural Network Detection In Binaries

This tool is based off of the paper "Architectural Reconnaissance of Neural Networks: From Binary to Actionable Intelligence" by Ethan Henley, Gianni Crivello, and Cpt. Robery Massey


```

# src/analysis/stride_analyzer.cpp

```cpp

```

# src/analysis/stride_analyzer.hpp

```hpp

```

# src/analysis/tensor_classification.cpp

```cpp

```

# src/analysis/tensor_classification.hpp

```hpp

```

# src/analysis/uncertainty.cpp

```cpp

```

# src/analysis/uncertainty.hpp

```hpp

```

# src/core/binary_analyzer.cpp

```cpp
#include "nn_binary_analysis/analyzer.hpp"
#include "core/binary_parser.hpp"
#include "core/memory_analyzer.hpp"
#include <unordered_map>
#include <deque>

namespace nn_binary_analysis {

class BinaryAnalyzer::Implementation {
public:
    explicit Implementation(const std::string& triple, const AnalysisConfig& config)
        : config_(config)
        , parser_(triple)
        , initialized_(parser_.isInitialized()) {}

    bool isInitialized() const { return initialized_; }

    AnalysisResult analyze(const uint8_t* data, size_t size) {
        if (!initialized_) {
            throw InitializationError("Analyzer not properly initialized");
        }

        AnalysisResult result;
        result.resource_usage = ResourceMetrics{};
        
        // Parse binary sections
        if (!parser_.parseSection(data, size, 0x0)) {
            throw AnalysisError("Failed to parse binary section");
        }

        // Analyze decoded instructions
        analyzeInstructions(parser_.getCurrentSection(), result);

        // Compute overall confidence
        computeOverallConfidence(result);

        return result;
    }

    const AnalysisConfig& getConfig() const { return config_; }
    void updateConfig(const AnalysisConfig& config) { config_ = config; }

private:
    AnalysisConfig config_;
    BinaryParser parser_;
    bool initialized_;

    // Helper struct for operation detection
    struct PotentialOperation {
        uint64_t start_address;
        uint64_t end_address;
        TensorOpType type;
        std::vector<uint32_t> dimensions;
        float confidence;
        size_t instruction_count;
    };

    void analyzeInstructions(const std::vector<DecodedInstruction>& instructions,
                           AnalysisResult& result) {
        std::deque<DecodedInstruction> window;
        std::vector<PotentialOperation> potential_ops;
        
        // Sliding window analysis for operation detection
        for (const auto& inst : instructions) {
            window.push_back(inst);
            if (window.size() > config_.pattern_detection.pattern_buffer_size) {
                window.pop_front();
            }

            if (inst.potential_tensor_op) {
                analyzeOperationWindow(window, potential_ops);
            }

            // Update resource metrics
            updateResourceMetrics(inst, result.resource_usage);
        }

        // Convert potential operations to final tensor operations
        for (const auto& pot_op : potential_ops) {
            if (pot_op.confidence >= config_.confidence_threshold) {
                TensorOperation op;
                op.type = pot_op.type;
                op.start_address = pot_op.start_address;
                op.end_address = pot_op.end_address;
                op.dimensions = pot_op.dimensions;
                
                // Set confidence metrics
                op.confidence.pattern_confidence = pot_op.confidence;
                op.confidence.structure_confidence = computeStructureConfidence(pot_op);
                op.confidence.operation_confidence = computeOperationConfidence(pot_op);

                result.detected_ops.push_back(op);
            }
        }
    }

    void analyzeOperationWindow(const std::deque<DecodedInstruction>& window,
                              std::vector<PotentialOperation>& potential_ops) {
        if (window.size() < config_.pattern_detection.min_pattern_length) {
            return;
        }

        // Try to detect operation type from instruction pattern
        auto op_type = detectOperationType(window);
        if (op_type == TensorOpType::Unknown) {
            return;
        }

        // Create or update potential operation
        PotentialOperation pot_op;
        pot_op.start_address = window.front().address;
        pot_op.end_address = window.back().address;
        pot_op.type = op_type;
        pot_op.instruction_count = window.size();
        
        // Estimate dimensions from memory access patterns
        estimateOperationDimensions(window, pot_op.dimensions);
        
        // Compute confidence based on instruction pattern strength
        pot_op.confidence = computeWindowConfidence(window, op_type);

        // Add to potential operations if confidence is above minimum threshold
        if (pot_op.confidence >= config_.confidence_threshold * 0.8f) {
            potential_ops.push_back(pot_op);
        }
    }

    TensorOpType detectOperationType(const std::deque<DecodedInstruction>& window) {
        // Count different instruction patterns
        size_t vector_ops = 0;
        size_t memory_ops = 0;
        size_t arithmetic_ops = 0;

        for (const auto& inst : window) {
            // Simple pattern matching for now - could be enhanced with machine learning
            if (inst.mnemonic.find("vmul") != std::string::npos ||
                inst.mnemonic.find("vfma") != std::string::npos) {
                vector_ops++;
            }
            else if (inst.mnemonic.find("load") != std::string::npos ||
                     inst.mnemonic.find("store") != std::string::npos) {
                memory_ops++;
            }
            else if (inst.mnemonic.find("add") != std::string::npos ||
                     inst.mnemonic.find("mul") != std::string::npos) {
                arithmetic_ops++;
            }
        }

        // Classify based on operation ratios
        float vector_ratio = static_cast<float>(vector_ops) / window.size();
        float memory_ratio = static_cast<float>(memory_ops) / window.size();

        if (vector_ratio > 0.4f && memory_ratio > 0.3f) {
            return TensorOpType::MatrixMultiplication;
        }
        else if (vector_ratio > 0.3f && memory_ratio > 0.4f) {
            return TensorOpType::Convolution2D;
        }
        else if (arithmetic_ops > memory_ops) {
            return TensorOpType::ElementWise;
        }

        return TensorOpType::Unknown;
    }

    void estimateOperationDimensions(const std::deque<DecodedInstruction>& window,
                                   std::vector<uint32_t>& dimensions) {
        // This is a simplified implementation - could be enhanced with more sophisticated analysis
        dimensions.clear();
        
        // Look for dimension hints in memory access patterns
        uint32_t stride_count = 0;
        uint32_t last_stride = 0;

        for (const auto& inst : window) {
            if (inst.confidence.pattern_confidence > 0.8f) {
                // Assume high confidence patterns indicate dimension boundaries
                stride_count++;
            }
        }

        // Make educated guesses about dimensions based on access patterns
        if (stride_count > 0) {
            dimensions.push_back(stride_count);
            if (stride_count > 16) {  // Arbitrary threshold
                dimensions.push_back(stride_count / 16);  // Estimate 2D shape
            }
        }
    }

    float computeWindowConfidence(const std::deque<DecodedInstruction>& window,
                                TensorOpType op_type) {
        float total_confidence = 0.0f;
        
        // Combine individual instruction confidences
        for (const auto& inst : window) {
            total_confidence += inst.confidence.overall();
        }
        
        return total_confidence / window.size();
    }

    float computeStructureConfidence(const PotentialOperation& op) {
        // Compute confidence based on typical operation characteristics
        float conf = 0.0f;
        
        // Size-based confidence
        if (op.instruction_count >= config_.pattern_detection.min_pattern_length) {
            conf += 0.3f;
        }
        
        // Dimension-based confidence
        if (!op.dimensions.empty()) {
            conf += 0.4f;
            // Higher confidence for power-of-2 dimensions
            for (uint32_t dim : op.dimensions) {
                if ((dim & (dim - 1)) == 0) {
                    conf += 0.1f;
                }
            }
        }
        
        return std::min(1.0f, conf);
    }

    float computeOperationConfidence(const PotentialOperation& op) {
        // Operation-specific confidence computation
        float conf = 0.0f;
        
        switch (op.type) {
            case TensorOpType::MatrixMultiplication:
                // Check for characteristic instruction ratios
                conf = 0.8f;
                break;
                
            case TensorOpType::Convolution2D:
                // Check for nested loop patterns
                conf = 0.7f;
                break;
                
            case TensorOpType::ElementWise:
                // Simpler patterns, but need high consistency
                conf = 0.6f;
                break;
                
            default:
                conf = 0.3f;
        }
        
        return conf * op.confidence;  // Scale by pattern confidence
    }

    void updateResourceMetrics(const DecodedInstruction& inst,
                             ResourceMetrics& metrics) {
        // Update SIMD utilization
        if (inst.mnemonic.find('v') == 0) {  // Vector instruction
            metrics.simd_utilization += 0.1f;
        }
        
        // Update memory bandwidth
        if (inst.mnemonic.find("load") != std::string::npos ||
            inst.mnemonic.find("store") != std::string::npos) {
            metrics.memory_bandwidth += 0.1f;
        }
        
        // Normalize metrics
        metrics.simd_utilization = std::min(1.0f, metrics.simd_utilization);
        metrics.memory_bandwidth = std::min(1.0f, metrics.memory_bandwidth);
        
        // Cache efficiency is currently a placeholder
        metrics.cache_efficiency = 0.5f;
    }

    void computeOverallConfidence(AnalysisResult& result) {
        if (result.detected_ops.empty()) {
            result.overall_confidence = ConfidenceMetrics{};
            return;
        }

        float pattern_conf = 0.0f;
        float struct_conf = 0.0f;
        float op_conf = 0.0f;

        for (const auto& op : result.detected_ops) {
            pattern_conf += op.confidence.pattern_confidence;
            struct_conf += op.confidence.structure_confidence;
            op_conf += op.confidence.operation_confidence;
        }

        size_t n = result.detected_ops.size();
        result.overall_confidence = ConfidenceMetrics{
            pattern_conf / n,
            struct_conf / n,
            op_conf / n
        };
    }
};

// Public API implementations
BinaryAnalyzer::BinaryAnalyzer(const std::string& triple, const AnalysisConfig& config)
    : impl_(std::make_unique<Implementation>(triple, config)) {}

BinaryAnalyzer::~BinaryAnalyzer() = default;

AnalysisResult BinaryAnalyzer::analyze(const uint8_t* data, size_t size) {
    return impl_->analyze(data, size);
}

bool BinaryAnalyzer::isInitialized() const {
    return impl_->isInitialized();
}

const AnalysisConfig& BinaryAnalyzer::getConfig() const {
    return impl_->getConfig();
}

void BinaryAnalyzer::updateConfig(const AnalysisConfig& config) {
    impl_->updateConfig(config);
}

} // namespace nn_binary_analysis
```

# src/core/binary_parser.cpp

```cpp
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
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
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
```

# src/core/binary_parser.hpp

```hpp
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
```

# src/core/framework_detector.cpp

```cpp
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
```

# src/core/framework_detector.hpp

```hpp
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
```

# src/core/memory_analyzer.cpp

```cpp
#include "memory_analyzer.hpp"
#include <algorithm>
#include <cassert>

namespace nn_binary_analysis {

// MemoryAnalyzer Implementation
MemoryAnalyzer::MemoryAnalyzer(uint32_t default_alignment)
    : default_alignment_(default_alignment) {
    // Reserve space for typical number of memory regions
    memory_regions_.reserve(16);
}

bool MemoryAnalyzer::registerMemoryRegion(uint64_t start, uint64_t end,
                                        uint32_t perms, uint32_t alignment) {
    if (start >= end) return false;
    
    // Check for overlapping regions
    for (const auto& region : memory_regions_) {
        if ((start >= region.start_address && start < region.end_address) ||
            (end > region.start_address && end <= region.end_address)) {
            return false;
        }
    }

    // Ensure alignment is power of 2
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        alignment = default_alignment_;
    }

    memory_regions_.push_back({start, end, perms, alignment});
    
    // Keep regions sorted by start address
    std::sort(memory_regions_.begin(), memory_regions_.end(),
              [](const MemoryRegion& a, const MemoryRegion& b) {
                  return a.start_address < b.start_address;
              });
    
    return true;
}

MemoryAnalyzer::AddressTranslation 
MemoryAnalyzer::translateAddress(uint64_t virtual_addr) const {
    AddressTranslation result{virtual_addr, 0, false, default_alignment_};
    
    // Binary search for the containing region
    auto it = std::lower_bound(
        memory_regions_.begin(), memory_regions_.end(),
        virtual_addr,
        [](const MemoryRegion& region, uint64_t addr) {
            return region.end_address <= addr;
        });
    
    if (it != memory_regions_.end() && 
        virtual_addr >= it->start_address && 
        virtual_addr < it->end_address) {
        result.physical_addr = virtual_addr - it->start_address;
        result.is_valid = true;
        result.alignment = it->alignment;
    }
    
    return result;
}

bool MemoryAnalyzer::isAlignmentValid(uint64_t addr, uint32_t required_alignment) const {
    return (addr % required_alignment) == 0;
}

// PatternAnalyzer Implementation
class PatternAnalyzer::CircularBuffer {
public:
    explicit CircularBuffer(size_t capacity)
        : buffer_(capacity)
        , head_(0)
        , size_(0)
        , capacity_(capacity) {}

    void push_back(const MemoryAnalyzer::AddressTranslation& addr) {
        buffer_[head_] = addr;
        head_ = (head_ + 1) % capacity_;
        if (size_ < capacity_) size_++;
    }

    size_t size() const { return size_; }
    
    const MemoryAnalyzer::AddressTranslation& operator[](size_t idx) const {
        assert(idx < size_);
        return buffer_[(head_ + capacity_ - size_ + idx) % capacity_];
    }

private:
    std::vector<MemoryAnalyzer::AddressTranslation> buffer_;
    size_t head_;
    size_t size_;
    size_t capacity_;
};

PatternAnalyzer::PatternAnalyzer(size_t window_size)
    : access_window_(std::make_unique<CircularBuffer>(window_size))
    , minimum_pattern_length_(3) {
    current_stride_info_ = {0, 0, 0.0f, false};
}

void PatternAnalyzer::recordAccess(const MemoryAnalyzer::AddressTranslation& addr_info) {
    if (!addr_info.is_valid) return;
    
    access_window_->push_back(addr_info);
    updatePatternAnalysis();
}

void PatternAnalyzer::updatePatternAnalysis() {
    if (access_window_->size() < minimum_pattern_length_) return;

    std::unordered_map<uint32_t, uint32_t> stride_counts;
    uint32_t max_count = 0;
    uint32_t primary_stride = 0;

    // Analyze primary stride pattern
    for (size_t i = 1; i < access_window_->size(); ++i) {
        uint64_t curr_addr = (*access_window_)[i].physical_addr;
        uint64_t prev_addr = (*access_window_)[i-1].physical_addr;
        
        if (curr_addr > prev_addr) {
            uint32_t stride = static_cast<uint32_t>(curr_addr - prev_addr);
            uint32_t count = ++stride_counts[stride];
            
            if (count > max_count) {
                max_count = count;
                primary_stride = stride;
            }
        }
    }

    // Analyze secondary stride pattern
    uint32_t secondary_stride = 0;
    if (max_count >= minimum_pattern_length_) {
        std::unordered_map<uint32_t, uint32_t> secondary_counts;
        
        for (size_t i = 2; i < access_window_->size(); ++i) {
            uint64_t curr_addr = (*access_window_)[i].physical_addr;
            uint64_t prev_addr = (*access_window_)[i-2].physical_addr;
            
            if (curr_addr > prev_addr) {
                uint32_t stride = static_cast<uint32_t>(curr_addr - prev_addr);
                if (stride > primary_stride) {
                    uint32_t count = ++secondary_counts[stride];
                    if (count >= minimum_pattern_length_ / 2) {
                        secondary_stride = stride;
                        break;
                    }
                }
            }
        }
    }

    // Update stride information
    float confidence = static_cast<float>(max_count) / (access_window_->size() - 1);
    bool is_tensor = isTensorPattern(primary_stride, secondary_stride, confidence);
    
    current_stride_info_ = {
        primary_stride,
        secondary_stride,
        confidence,
        is_tensor
    };
}

bool PatternAnalyzer::isTensorPattern(uint32_t primary, uint32_t secondary, 
                                    float confidence) const {
    if (confidence < 0.7f) return false;

    // Check for power-of-2 strides (common in tensor operations)
    bool primary_pow2 = (primary & (primary - 1)) == 0;
    bool secondary_pow2 = secondary == 0 || (secondary & (secondary - 1)) == 0;

    // Check for typical tensor operation patterns
    bool typical_primary = primary == 4 || primary == 8 || 
                         primary == 16 || primary == 32;
    bool typical_secondary = secondary == 0 || secondary >= primary;

    return (primary_pow2 && secondary_pow2) || 
           (typical_primary && typical_secondary);
}

StrideInfo PatternAnalyzer::getCurrentStrideInfo() const {
    return current_stride_info_;
}

} // namespace nn_binary_analysis
```

# src/core/memory_analyzer.hpp

```hpp
#pragma once

#include "types.hpp"
#include <deque>
#include <unordered_map>

namespace nn_binary_analysis {

// Forward declarations and common structures
struct StrideInfo {
    uint32_t primary_stride{0};
    uint32_t secondary_stride{0};
    float confidence{0.0f};
    bool is_tensor_pattern{false};
};

class MemoryAnalyzer {
public:
    struct MemoryRegion {
        uint64_t start_address;
        uint64_t end_address;
        uint32_t permissions;
        uint32_t alignment;
    };

    struct AddressTranslation {
        uint64_t virtual_addr;
        uint64_t physical_addr;
        bool is_valid;
        uint32_t alignment;
    };

    explicit MemoryAnalyzer(uint32_t default_alignment = 4096);
    
    bool registerMemoryRegion(uint64_t start, uint64_t end, 
                            uint32_t perms, uint32_t alignment);
    AddressTranslation translateAddress(uint64_t virtual_addr) const;
    bool isAlignmentValid(uint64_t addr, uint32_t required_alignment) const;

private:
    std::vector<MemoryRegion> memory_regions_;
    uint32_t default_alignment_;
};

class PatternAnalyzer {
public:
    explicit PatternAnalyzer(size_t window_size = 1024);
    
    void recordAccess(const MemoryAnalyzer::AddressTranslation& addr_info);
    StrideInfo getCurrentStrideInfo() const;

private:
    class CircularBuffer;
    std::unique_ptr<CircularBuffer> access_window_;
    StrideInfo current_stride_info_;
    size_t minimum_pattern_length_;

    void updatePatternAnalysis();
    bool isTensorPattern(uint32_t primary, uint32_t secondary, float confidence) const;
};

} // namespace nn_binary_analysis
```

# src/core/transformer_detector.cpp

```cpp
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
```

# src/core/transformer_detector.hpp

```hpp
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
```

# src/core/types.hpp

```hpp
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

} // namespace nn_binary_analysis
```

# src/rewrite/injection.cpp

```cpp

```

# src/rewrite/injection.hpp

```hpp

```

# src/rewrite/instrument.cpp

```cpp

```

# src/rewrite/instrument.hpp

```hpp

```

# src/utils/circular_buffer.hpp

```hpp

```

# src/utils/logging.hpp

```hpp

```

# src/utils/metrics.hpp

```hpp

```

# tools/nn_analyze/CMakeLists.txt

```txt

```

# tools/nn_analyze/main.cpp

```cpp
#include "nn_binary_analysis/analyzer.hpp"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include <memory>
#include <iostream>

using namespace llvm;
using namespace nn_binary_analysis;

// Command line options
static cl::opt<std::string> InputFilename(cl::Positional,
    cl::desc("<input binary>"),
    cl::Required);

static cl::opt<std::string> TargetTriple("triple",
    cl::desc("Target triple (default: host triple)"),
    cl::init(sys::getDefaultTargetTriple()));

static cl::opt<bool> Verbose("v",
    cl::desc("Enable verbose output"),
    cl::init(false));

int main(int argc, char *argv[]) {
    // Parse command line options
    cl::ParseCommandLineOptions(argc, argv, "Neural Network Binary Analyzer\n");

    // Load input file
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
        MemoryBuffer::getFile(InputFilename);
    if (std::error_code EC = BufferOrErr.getError()) {
        errs() << "Error loading file '" << InputFilename << "': " 
               << EC.message() << "\n";
        return 1;
    }
    std::unique_ptr<MemoryBuffer> Buffer = std::move(BufferOrErr.get());

    // Initialize analyzer
    AnalysisConfig config;
    config.enable_pattern_detection = true;
    config.enable_resource_monitoring = true;
    config.confidence_threshold = 0.7f;

    try {
        // Create binary analyzer
        BinaryAnalyzer analyzer(TargetTriple, config);

        // Analyze binary
        const uint8_t* data = reinterpret_cast<const uint8_t*>(
            Buffer->getBufferStart());
        size_t size = Buffer->getBufferSize();

        AnalysisResult result = analyzer.analyze(data, size);

        // Report results
        std::cout << "\nAnalysis Results:\n";
        std::cout << "================\n";
        std::cout << "Neural Network Detection Confidence: " 
                  << result.overall_confidence.overall() * 100 << "%\n\n";

        if (result.has_neural_network()) {
            std::cout << "Detected Operations:\n";
            for (const auto& op : result.detected_ops) {
                std::cout << "- Operation at 0x" << std::hex << op.start_address
                         << " - 0x" << op.end_address << std::dec << "\n";
                std::cout << "  Type: " << getTensorOpTypeName(op.type) << "\n";
                std::cout << "  Confidence: " 
                         << op.confidence.overall() * 100 << "%\n";
                
                if (Verbose) {
                    std::cout << "  Dimensions: ";
                    for (auto dim : op.dimensions) {
                        std::cout << dim << " ";
                    }
                    std::cout << "\n";
                    
                    if (op.type == TensorOpType::Convolution2D) {
                        std::cout << "  Kernel Size: " << op.metadata.kernel_size << "\n";
                        std::cout << "  Stride: " << op.metadata.stride << "\n";
                        std::cout << "  Padding: " << op.metadata.padding << "\n";
                    }
                }
                std::cout << "\n";
            }

            if (Verbose) {
                std::cout << "Memory Access Patterns:\n";
                for (const auto& pattern : result.access_patterns) {
                    std::cout << "- Base Address: 0x" << std::hex 
                             << pattern.base_address << std::dec << "\n";
                    std::cout << "  Primary Stride: " << pattern.stride_primary << "\n";
                    std::cout << "  Secondary Stride: " << pattern.stride_secondary << "\n";
                    std::cout << "  Consistency: " 
                             << pattern.consistency_ratio * 100 << "%\n\n";
                }

                std::cout << "Resource Utilization:\n";
                std::cout << "- SIMD Usage: " 
                         << result.resource_usage.simd_utilization * 100 << "%\n";
                std::cout << "- Memory Bandwidth: " 
                         << result.resource_usage.memory_bandwidth * 100 << "%\n";
                std::cout << "- Cache Efficiency: " 
                         << result.resource_usage.cache_efficiency * 100 << "%\n";
            }
        } else {
            std::cout << "No neural network operations detected with sufficient confidence.\n";
        }

    } catch (const std::exception& e) {
        errs() << "Error during analysis: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

// Helper function to convert TensorOpType to string
std::string getTensorOpTypeName(TensorOpType type) {
    switch (type) {
        case TensorOpType::MatrixMultiplication:
            return "Matrix Multiplication";
        case TensorOpType::Convolution2D:
            return "2D Convolution";
        case TensorOpType::ElementWise:
            return "Element-wise Operation";
        case TensorOpType::Pooling:
            return "Pooling";
        case TensorOpType::Normalization:
            return "Normalization";
        default:
            return "Unknown";
    }
}
```

# tools/nn_detect/CMakeLists.txt

```txt
cmake_minimum_required(VERSION 3.15)
project(nn_binary_analysis VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find LLVM
find_package(LLVM 15.0 REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

# Library sources
set(LIB_SOURCES
    src/core/binary_parser.cpp
    src/core/memory_analyzer.cpp
    src/core/transformer_detector.cpp
    src/core/framework_detector.cpp
)

# Create library
add_library(nn_binary_analysis ${LIB_SOURCES})

target_include_directories(nn_binary_analysis
    PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link against LLVM libraries
llvm_map_components_to_libnames(llvm_libs support core irreader mcparser mc)
target_link_libraries(nn_binary_analysis PRIVATE ${llvm_libs})

# Demo executable
add_executable(nn_detect tools/nn_detect/main.cpp)
target_link_libraries(nn_detect PRIVATE nn_binary_analysis)
```

# tools/nn_detect/main.cpp

```cpp
#include "nn_binary_analysis/analyzer.hpp"
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
```

