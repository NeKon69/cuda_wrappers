#pragma once

#include <cuda_runtime.h>

#include <memory>

#include "cuda_wrappers/error.h"
#include "cuda_wrappers/stream.h"
#include "fwd.h"

namespace raw::cuda_wrappers {
class event {
private:
	cudaEvent_t					 event_ = nullptr;
	std::shared_ptr<cuda_stream> stream_;

public:
	event(std::shared_ptr<cuda_stream> stream) {
		CUDA_SAFE_CALL(cudaEventCreate(&event_));
		stream_ = stream;
	}

	event() {
		CUDA_SAFE_CALL(cudaEventCreate(&event_));
	}

	void record() {
		CUDA_SAFE_CALL(cudaEventRecord(event_, stream_->stream()));
	}

	bool ready() const {
		// Lazy to implement error handling here, surely nothing bad happens here
		return cudaEventQuery(event_) == cudaSuccess;
	}

	void sync() const {
		CUDA_SAFE_CALL(cudaEventSynchronize(event_));
	}

	float get_diff(event& _event) {
		float time;
		CUDA_SAFE_CALL(cudaEventElapsedTime(&time, event_, _event.event_));
		return time;
	}

	void record(std::shared_ptr<cuda_stream> stream) {
		stream_ = stream;
		record();
	}

	cudaEvent_t get() const {
		return event_;
	}

	~event() {
		if (cudaEventDestroy(event_) != cudaSuccess) {
			std::cerr
				<< "Goddang it, looks like we are screwed, but no matter, let's continue execution\n";
		}
	}

	event(event&&)			  = default;
	event& operator=(event&&) = default;

	event(const event&)			   = delete;
	event& operator=(const event&) = delete;
};

} // namespace raw::cuda_wrappers

