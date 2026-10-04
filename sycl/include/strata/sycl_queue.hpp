// include/strata/sycl_queue.hpp - the SYCL port's one addition to the engine's API surface.
//
// Every launcher takes `void* stream`, a cudaStream_t where null means the default stream. dpct migrates the
// cast to `(dpct::queue_ptr) stream` and dereferences it, so a null stream is a null sycl::queue* and a crash.
// q_of() is that cast with CUDA's null-stream meaning restored: the default in-order queue.
#pragma once
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

namespace strata {
inline sycl::queue* q_of(const void* stream) {
    return stream ? (sycl::queue*) stream : &dpct::get_in_order_queue();
}

// A copy with neither end in device memory (pinned USM host <-> pageable host) must not go through the
// queue. On an Arc that runs on the copy engine and hung it (dmesg "Engine reset: engine_class=bcs";
// repeated, the card was declared wedged). `unknown` is pageable host: get_pointer_type has no device
// allocation for it. A null pointer is not host-only; the caller rejects it.
inline bool host_only_copy(const void* dst, const void* src) {
    if (!dst || !src) return false;
    const sycl::context ctx = dpct::get_in_order_queue().get_context();
    const auto host = [&](const void* p) {
        const sycl::usm::alloc t = sycl::get_pointer_type(p, ctx);
        return t == sycl::usm::alloc::host || t == sycl::usm::alloc::unknown;
    };
    return host(dst) && host(src);
}
}  // namespace strata
