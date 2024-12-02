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