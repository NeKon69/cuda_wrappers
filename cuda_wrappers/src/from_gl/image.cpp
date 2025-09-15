//
// Created by progamers on 8/5/25.
//
#include "from_gl/image.h"

#include "error.h"

namespace raw::cuda_wrappers::from_gl {
image::image(uint32_t texture_id)
	: resource(cudaGraphicsGLRegisterImage, nullptr, texture_id, GL_TEXTURE_2D,
			   cudaGraphicsRegisterFlagsSurfaceLoadStore) {
	CUDA_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&array, *get_resource(), 0, 0));
}

void image::set_data(uint32_t texture_id) {
	create(cudaGraphicsGLRegisterImage, texture_id, GL_TEXTURE_2D,
		   cudaGraphicsRegisterFlagsSurfaceLoadStore);
	CUDA_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&array, *get_resource(), 0, 0));
}

cudaArray_t image::get() {
	map();
	return array;
}
} // namespace raw::cuda_wrappers::from_gl
