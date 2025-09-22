//
// Created by progamers on 8/5/25.
//

#pragma once
#include "cuda_wrappers/resource.h"
#include "error.h"

namespace raw::cuda_wrappers::from_gl {
class image : resource {
private:
	cudaArray_t array;

public:
	using resource::resource;

	// If used default constructor and didn't set the data manually you are screwed (let it be UB)
	image() = default;

	explicit image(uint32_t texture_id) : resource(cudaGraphicsGLRegisterImage, nullptr, texture_id, GL_TEXTURE_2D,
			   cudaGraphicsRegisterFlagsSurfaceLoadStore) {
        CUDA_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&array, *get_resource(), 0, 0));
    }

	void set_data(uint32_t texture_id) {
        create(cudaGraphicsGLRegisterImage, texture_id, GL_TEXTURE_2D,
		   cudaGraphicsRegisterFlagsSurfaceLoadStore);
	    CUDA_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&array, *get_resource(), 0, 0));
    }

	cudaArray_t get() {
        map();
        return array;
    }
};
} // namespace raw::cuda::from_gl