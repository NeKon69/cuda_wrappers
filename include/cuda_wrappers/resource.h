//
// Created by progamers on 8/5/25.
//

#pragma once
#include <cuda_egl_interop.h>
#include <cuda_gl_interop.h>

#include <memory>
#include <iostream>

#include "error.h"
#include "fwd.h"
#include "stream.h"
#include "exception.h"

namespace raw::cuda_wrappers {
/**
 * @class resource
 * @brief Base class for CUDA data from opengl, takes in the constructor function to register the
 * resource, unmaps the stored resource in the destructor and unregisters it
 */
class resource {
private:
	cudaGraphicsResource_t		m_resource = nullptr;
	bool						mapped		= false;
	std::shared_ptr<cuda_stream> stream;

private:
	void unmap_noexcept() noexcept {
        if (mapped && m_resource) {
            try {
                CUDA_SAFE_CALL(
                    cudaGraphicsUnmapResources(1, &m_resource, stream ? stream->stream() : nullptr));
                mapped = false;
            } catch (const cuda_exception& e) {
                std::cerr << std::format("[CRITICAL] Failed to unmap graphics resource. \n{}",
                                         e.what());
            }
        }
    }
	void cleanup() noexcept {
        unmap_noexcept();
        if (m_resource) {
            try {
                CUDA_SAFE_CALL(cudaGraphicsUnregisterResource(m_resource));
            } catch (const cuda_exception& e) {
                std::cerr << std::format("[CRITICAL] Failed to unregister resource. \n{}", e.what());
            }
        }
    }

public:
	class mapped_resource {
	private:
		std::shared_ptr<cuda_stream> stream;
		cudaGraphicsResource_t		 resource;

	public:
		mapped_resource(std::shared_ptr<cuda_stream> stream, cudaGraphicsResource_t resource) : stream(stream), resource(resource) {
            CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &resource, stream->stream()));
        }
		~mapped_resource() {
            CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &resource, stream->stream()));
        }
		cudaGraphicsResource_t &operator*() {
            return resource;
        }
	};
	resource() = default;

	template<typename F, typename... Args>
		requires std::invocable<F, cudaGraphicsResource_t *, Args...>
	explicit resource(const F &&func, std::shared_ptr<cuda_stream> stream, Args &&...args)
		: stream(stream) {
		create(func, std::forward<Args &&>(args)...);
	}

	template<typename F, typename... Args>
	void create(const F &&func, Args &&...args) {
		cleanup();
		CUDA_SAFE_CALL(func(&m_resource, std::forward<Args &&>(args)...));
	}

	mapped_resource get_resource() {
        return mapped_resource {stream, m_resource};
    }

	void unmap() {
        if (mapped && m_resource) {
            CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &m_resource, stream->stream()));
            mapped = false;
        }
    }

	void map() {
        if (!mapped && m_resource) {
            CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &m_resource, stream->stream()));
            mapped = true;
        }
    }

	void set_stream(std::shared_ptr<cuda_stream> stream_) {
        stream = std::move(stream_);
    }

	virtual ~resource() {
        cleanup();
    }

	resource &operator=(const resource &) = delete;
	resource(const resource &)			  = delete;

	resource &operator=(resource &&rhs) noexcept {
        if (this == &rhs) {
            return *this;
        }
        cleanup();
        m_resource = rhs.m_resource;
        mapped	   = rhs.mapped;
        stream	   = std::move(rhs.stream);

        rhs.m_resource = nullptr;
        rhs.mapped	   = false;
        return *this;
    }
	resource(resource &&rhs) noexcept
		: m_resource(rhs.m_resource), mapped(rhs.mapped), stream(std::move(rhs.stream)) {
        rhs.m_resource = nullptr;
        rhs.mapped	   = false;
    }

protected:
};

} // namespace raw::cuda_wrappers