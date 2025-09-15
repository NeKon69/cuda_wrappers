//
// Created by progamers on 8/5/25.
//

#pragma once
#include "cuda_wrappers/resource.h"

namespace raw::cuda_wrappers::from_gl {
class image : resource {
private:
	cudaArray_t array;

public:
	using resource::resource;

	// If used default constructor and didn't set the data manually you are screwed (let it be UB)
	image() = default;

	explicit image(uint32_t texture_id);

	void set_data(uint32_t texture_id);

	cudaArray_t get();
};
} // namespace raw::cuda::from_gl

