// logging.hpp — Debug and info logging utilities
// =============================================================================
//
// Lightweight logging framework for the accelerator driver.  Supports:
//   - Multiple log levels (TRACE, DEBUG, INFO, WARN, ERROR)
//   - Compile-time level filtering via ACCEL_LOG_LEVEL
//   - Optional file output
//   - Coloured terminal output
//   - printf-style and stream-style macros
//
// Usage:
//   LOG_INFO("Layer %s: %u tiles", layer.c_str(), num_tiles);
//   LOG_DEBUG("DMA transfer: %u bytes", bytes);
//   LOG_ERROR("Timeout on tile %u", tile_idx);
//
// =============================================================================
#pragma once

#include <string>
#include <cstdio>
#include <cstdarg>
#include <cstdint>
#include <chrono>
#include <ostream>

namespace accel {
namespace utils {

// =============================================================================
// Log levels
// =============================================================================
enum class LogLevel : uint8_t {
    Trace = 0,
    Debug = 1,
    Info  = 2,
    Warn  = 3,
    Error = 4,
    None  = 5    // Disable all logging
};

const char* logLevelName(LogLevel level) noexcept;
const char* logLevelColor(LogLevel level) noexcept;

// =============================================================================
// Logger — Global singleton
// =============================================================================
class Logger {
public:
    /// Get the global logger instance
    static Logger& instance();

    // -------------------------------------------------------------------------
    // Configuration
    // -------------------------------------------------------------------------

    /// Set minimum log level (messages below this are suppressed)
    void setLevel(LogLevel level) { min_level_ = level; }
    LogLevel level() const { return min_level_; }

    /// Enable/disable coloured output (ANSI escape codes)
    void setColor(bool enable) { color_ = enable; }

    /// Enable/disable timestamps
    void setTimestamps(bool enable) { timestamps_ = enable; }

    /// Set output file (in addition to stderr).  Pass nullptr to disable.
    void setOutputFile(const char* path);

    /// Close output file
    void closeOutputFile();

    // -------------------------------------------------------------------------
    // Logging
    // -------------------------------------------------------------------------

    /// Log a message at the given level (printf-style)
    void log(LogLevel level, const char* file, int line,
             const char* fmt, ...) __attribute__((format(printf, 5, 6)));

    /// Log a message at the given level (va_list)
    void logv(LogLevel level, const char* file, int line,
              const char* fmt, va_list args);

    /// Check if a given level would be logged
    bool isEnabled(LogLevel level) const { return level >= min_level_; }

    // -------------------------------------------------------------------------
    // Statistics
    // -------------------------------------------------------------------------
    uint32_t messageCount(LogLevel level) const;
    uint32_t totalMessages() const;
    void resetStats();

private:
    Logger();
    ~Logger();

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

    LogLevel min_level_;
    bool     color_;
    bool     timestamps_;
    FILE*    output_file_;

    // Per-level message counts
    uint32_t counts_[6] = {};

    // Start time for relative timestamps
    std::chrono::steady_clock::time_point start_time_;

    // Format and write a single log line
    void writeFormatted(LogLevel level, const char* file, int line,
                        const char* msg);
};

} // namespace utils
} // namespace accel

// =============================================================================
// Macros — Compile-time level filtering
// =============================================================================
#ifndef ACCEL_LOG_LEVEL
#define ACCEL_LOG_LEVEL 1   // Default: Debug and above
#endif

#define ACCEL_LOG(level, ...) \
    do { \
        if (static_cast<uint8_t>(level) >= ACCEL_LOG_LEVEL) { \
            ::accel::utils::Logger::instance().log( \
                level, __FILE__, __LINE__, __VA_ARGS__); \
        } \
    } while (0)

#define LOG_TRACE(...) ACCEL_LOG(::accel::utils::LogLevel::Trace, __VA_ARGS__)
#define LOG_DEBUG(...) ACCEL_LOG(::accel::utils::LogLevel::Debug, __VA_ARGS__)
#define LOG_INFO(...)  ACCEL_LOG(::accel::utils::LogLevel::Info,  __VA_ARGS__)
#define LOG_WARN(...)  ACCEL_LOG(::accel::utils::LogLevel::Warn,  __VA_ARGS__)
#define LOG_ERROR(...) ACCEL_LOG(::accel::utils::LogLevel::Error, __VA_ARGS__)
