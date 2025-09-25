//
// Created by progamers on 9/25/25.
//
#pragma once

#include <memory>

#include "cuda_wrappers/fwd.h"
#include "cuda_wrappers/stream.h"

namespace raw::cuda_wrappers {
class channel_format_description {
private:
	cudaChannelFormatDesc desc;

public:
	channel_format_description(cudaChannelFormatKind kind, int width, int height = 0, int depth = 0,
							   int alpha = 0) {
		desc = cudaCreateChannelDesc(width, height, depth, alpha, kind);
	}
	cudaChannelFormatDesc &get() {
		return desc;
	}

	uint32_t get_size() const {
		return (desc.x + desc.y + desc.z + desc.w) / 8;
	}
};
} // namespace raw::cuda_wrappers