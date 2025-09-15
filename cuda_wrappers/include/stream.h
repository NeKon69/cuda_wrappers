//
// Created by progamers on 7/20/25.
//

#pragma once
#include <cuda_runtime.h>

#include "fwd.h"
namespace raw::cuda_wrappers {
class cuda_stream {
private:
	cudaStream_t _stream = nullptr;
	bool created;

private:
	void destroy_noexcept() noexcept;

public:
	cuda_stream();
	cuda_stream(const cuda_stream& rhs)			   = delete;
	cuda_stream& operator=(const cuda_stream& rhs) = delete;
	cuda_stream(cuda_stream&& rhs) noexcept;
	cuda_stream&				  operator=(cuda_stream&& rhs) noexcept;
	void						  sync();
	void						  destroy();
	void						  create();
	cudaStream_t&				  stream();
	~cuda_stream();
};
} // namespace raw::device_types::cuda

