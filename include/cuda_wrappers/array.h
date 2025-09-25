//
// Created by progamers on 9/25/25.
//

#pragma once

#include <cuda_runtime.h>

#include <complex>
#include <memory>

#include "cuda_wrappers/channel_format_description.h"
#include "cuda_wrappers/fwd.h"
#include "cuda_wrappers/resource_description.h"

namespace raw::cuda_wrappers {
class array {
private:
	cudaArray_t					 array_		 = nullptr;
	uint32_t					 width		 = 0;
	uint32_t					 height		 = 0;
	uint32_t					 single_size = 0;
	std::shared_ptr<cuda_stream> stream;

public:
	array(std::shared_ptr<cuda_stream> stream, channel_format_description format_description,
		  int width, int height)
		: width(width), height(height), stream(std::move(stream)) {
		CUDA_SAFE_CALL(cudaMallocArray(&array_, &format_description.get(), width, height));
	}

	~array() {
		CUDA_SAFE_CALL(cudaFreeArray(array_));
	}

	array(const array &)			= delete;
	array &operator=(const array &) = delete;
	array(array &&)					= default;
	array &operator=(array &&)		= default;

	cudaArray_t &get() {
		return array_;
	}

	uint32_t get_width() const {
		return width;
	}
	uint32_t get_height() const {
		return height;
	}

	void copy_from_host(void *src) {
		CUDA_SAFE_CALL(cudaMemcpy2DToArrayAsync(array_, 0, 0, src, width * single_size,
												width * single_size, height, cudaMemcpyHostToDevice,
												stream->stream()));
	}
	void copy_to_host(void *dst) {
		CUDA_SAFE_CALL(cudaMemcpy2DFromArrayAsync(dst, width * single_size, array_, 0, 0,
												  width * single_size, height,
												  cudaMemcpyDeviceToHost, stream->stream()));
	}
};
} // namespace raw::cuda_wrappers