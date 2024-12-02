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