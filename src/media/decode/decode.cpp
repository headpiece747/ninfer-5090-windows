#include "media/decode/decode.h"
#include <stdexcept>
namespace ninfer::media::decode {
    Image decode_image(std::span<const std::uint8_t> bytes, const Policy& policy) {
        throw std::runtime_error("Compiled without FFMPEG support.");
    }
    Video decode_video(std::span<const std::uint8_t> bytes, const Policy& policy, double fps, int min_f, int max_f) {
        throw std::runtime_error("Compiled without FFMPEG support.");
    }
}
