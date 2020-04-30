#ifndef THRUST_SYCL_H
#define THRUST_SYCL_H

#include <CL/sycl.hpp>

namespace thrust {
namespace sycl {

/*! Get the global singleton queue object used for all thrust::* operations.
 *
 * TODO: allow lib clients to customize this, via a cuda-like API
 * e.g. sylcSetDevice(int), syclGetDevice(int), for multi-gpu systems.
 * The idea is to support one device per MPI process type use case, and
 * not worry about more complex cases.
 */
static inline cl::sycl::queue& get_queue() {
  static cl::sycl::queue q{};
  return q;
}

/*! Copy data between device pointers or between host and device pointers. Note
 * that SYCL pointers returned by the USM allocation functions contain extra
 * information so the runtime can determine where they are located.
 *
 * This differs from the signature of CUDA thrust::copy. However because of
 * how gtensor uses the copy routine, this works within gtensor.
 *
 * TODO: make this more general, support iterators to be more compatible.
 */
template <typename T>
inline void copy(const T* src_start, const T* src_end, T* dest) {
  cl::sycl::queue& q = get_queue();
  q.memcpy(dest, src_start, src_end-src_start);
  q.wait();
}

} // namespace sycl
} // namespace thrust

#endif // THRUST_SYCL_H

// vim: ts=2 sw=2
