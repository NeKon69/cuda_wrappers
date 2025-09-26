//
// Created by progamers on 8/5/25.
//

#pragma once
#include <cuda_runtime.h>

#include <cstring>

#include "fwd.h"

namespace raw::cuda_wrappers {
namespace resource_types {
struct array {
	static cudaResourceType res_type;
};
cudaResourceType array::res_type = cudaResourceTypeArray;
} // namespace resource_types

template<typename T>
class resource_description {
private:
	cudaResourceDesc description = {};

public:
	resource_description() {
		std::memset(&description, 0, sizeof(description));
		description.resType = T::res_type;
	}
	std::enable_if_t<std::is_same_v<T, resource_types::array>, void> set_array(
		const cudaArray_t& array) {
		description.res.array.array = array;
	}
	cudaResourceDesc& get() {
		return description;
	}
};
} // namespace raw::cuda_wrappers
