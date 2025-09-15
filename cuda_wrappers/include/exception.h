//
// Created by progamers on 9/6/25.
//

#pragma once
#include <stdexcept>

namespace raw::cuda_wrappers {
class cuda_exception : public std::runtime_error {
public:
	using std::runtime_error::runtime_error;
};
} // namespace raw::device_types::cuda
