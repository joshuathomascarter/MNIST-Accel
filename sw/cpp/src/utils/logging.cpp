// logging.cpp — Lightweight logging implementation
// =============================================================================
#include "utils/logging.hpp"

#include <cstdio>
#include <cstring>
#include <cstdarg>

#if defined(__APPLE__) || defined(__linux__)
#include <unistd.h>
#endif

namespace accel {
namespace utils {

// =============================================================================
// Level name / colour tables
// =============================================================================

const char* logLevelName(LogLevel level) noexcept {
    switch (level) {
        case LogLevel::Trace: return "TRACE";
        case LogLevel::Debug: return "DEBUG";
        case LogLevel::Info:  return "INFO ";
        case LogLevel::Warn:  return "WARN ";
        case LogLevel::Error: return "ERROR";
        case LogLevel::None:  return "NONE ";
    }
    return "?????";
}

const char* logLevelColor(LogLevel level) noexcept {
    switch (level) {
        case LogLevel::Trace: return "\033[90m";      // Gray
        case LogLevel::Debug: return "\033[36m";      // Cyan
        case LogLevel::Info:  return "\033[32m";      // Green
        case LogLevel::Warn:  return "\033[33m";      // Yellow
        case LogLevel::Error: return "\033[31;1m";    // Bold Red
        case LogLevel::None:  return "\033[0m";
    }
    return "\033[0m";
}

static const char* RESET_COLOR = "\033[0m";

// =============================================================================
// Logger — Singleton implementation
// =============================================================================

Logger& Logger::instance() {
    static Logger singleton;
    return singleton;
}

Logger::Logger()
    : min_level_(LogLevel::Info)
    , color_(true)
    , timestamps_(true)
    , output_file_(nullptr)
    , start_time_(std::chrono::steady_clock::now())
{
    // Detect if stderr is a terminal (for colour support)
#if defined(__APPLE__) || defined(__linux__)
    color_ = ::isatty(fileno(stderr)) != 0;
#else
    color_ = false;
#endif
}

Logger::~Logger() {
    closeOutputFile();
}

void Logger::setOutputFile(const char* path) {
    closeOutputFile();
    if (path) {
        output_file_ = std::fopen(path, "a");
    }
}

void Logger::closeOutputFile() {
    if (output_file_) {
        std::fclose(output_file_);
        output_file_ = nullptr;
    }
}

// =============================================================================
// Core logging
// =============================================================================

void Logger::log(LogLevel level, const char* file, int line,
                 const char* fmt, ...) {
    if (level < min_level_) return;

    va_list args;
    va_start(args, fmt);
    logv(level, file, line, fmt, args);
    va_end(args);
}

void Logger::logv(LogLevel level, const char* file, int line,
                  const char* fmt, va_list args) {
    if (level < min_level_) return;

    // Format the user message
    char msg_buf[2048];
    std::vsnprintf(msg_buf, sizeof(msg_buf), fmt, args);

    // Track statistics
    auto idx = static_cast<uint8_t>(level);
    if (idx < 6) counts_[idx]++;

    // Write formatted output
    writeFormatted(level, file, line, msg_buf);
}

void Logger::writeFormatted(LogLevel level, const char* file, int line,
                             const char* msg) {
    // Build timestamp
    char ts_buf[32] = "";
    if (timestamps_) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - start_time_).count();
        std::snprintf(ts_buf, sizeof(ts_buf), "[%6lld.%03lldms] ",
                      elapsed / 1000, elapsed % 1000);
    }

    // Extract just the filename from the path
    const char* basename = file;
    const char* slash = std::strrchr(file, '/');
    if (slash) basename = slash + 1;

    // Write to stderr
    if (color_) {
        std::fprintf(stderr, "%s%s%s[%s] %s:%d: %s%s\n",
                     ts_buf,
                     logLevelColor(level),
                     logLevelName(level),
                     logLevelName(level),
                     basename, line, msg,
                     RESET_COLOR);
    } else {
        std::fprintf(stderr, "%s[%s] %s:%d: %s\n",
                     ts_buf,
                     logLevelName(level),
                     basename, line, msg);
    }

    // Write to file (no colour)
    if (output_file_) {
        std::fprintf(output_file_, "%s[%s] %s:%d: %s\n",
                     ts_buf,
                     logLevelName(level),
                     basename, line, msg);
        std::fflush(output_file_);
    }
}

// =============================================================================
// Statistics
// =============================================================================

uint32_t Logger::messageCount(LogLevel level) const {
    auto idx = static_cast<uint8_t>(level);
    return (idx < 6) ? counts_[idx] : 0;
}

uint32_t Logger::totalMessages() const {
    uint32_t total = 0;
    for (auto c : counts_) total += c;
    return total;
}

void Logger::resetStats() {
    for (auto& c : counts_) c = 0;
}

} // namespace utils
} // namespace accel
