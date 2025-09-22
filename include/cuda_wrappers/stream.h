//
// Created by progamers on 7/20/25.
//

#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <memory>

#include "fwd.h"
#include "error.h"
#include "exception.h"

namespace raw::cuda_wrappers {
class cuda_stream {
private:
	cudaStream_t _stream = nullptr;
	bool created;

private:
	void destroy_noexcept() noexcept {
        try {
            destroy();
        } catch (const cuda_exception &e) {
            std::cerr << std::format("[CRITICAL] Destroying CUDA stream failed. \n{}", e.what());
        }
    }

public:
	cuda_stream() : created(std::make_shared<bool>(false)) {
        CUDA_SAFE_CALL(cudaStreamCreate(&_stream));
        created = true;
    }
	cuda_stream(const cuda_stream& rhs) = delete;
	cuda_stream& operator=(const cuda_stream& rhs) = delete;
	cuda_stream(cuda_stream&& rhs) noexcept : _stream(rhs._stream), created(rhs.created) {
        rhs._stream = nullptr;
        rhs.created = false;
    }
	cuda_stream& operator=(cuda_stream&& rhs) noexcept {
        destroy_noexcept();
        _stream		= rhs._stream;
        created		= rhs.created;
        rhs._stream = nullptr;
        rhs.created = false;
        return *this;
    }
	void sync() {
        CUDA_SAFE_CALL(cudaStreamSynchronize(_stream));
    }
	void destroy() {
        if (created)
            CUDA_SAFE_CALL(cudaStreamDestroy(_stream));
        created = false;
    }
	void create() {
        destroy();
        if (!created)
            CUDA_SAFE_CALL(cudaStreamCreate(&_stream));
        created = true;
    }
	cudaStream_t& stream() {
        return _stream;
    }
	~cuda_stream() {
        destroy_noexcept();
    }
};
} // namespace raw::device_types::cuda