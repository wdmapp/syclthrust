#ifndef THRUST_SYCL_H
#define THRUST_SYCL_H

#include <CL/sycl.hpp>

namespace thrust {
namespace sycl {

// TODO: allow lib clients to customize this, via a cuda-like API
// e.g. sylcSetDevice(int), syclGetDevice(int), for multi-gpu systems.
// The idea is to support one device per MPI process type use case, and
// not worry about more complex cases.
static inline cl::sycl::queue& get_queue() {
  static cl::sycl::queue q{};
  return q;
}

// TODO: implement thrust::copy, used by gtensor
  /*
template <typename T>
inline copy(const T* src_start, const T* src_end, T* dest) {
  :
}

  thrust::copy(from.data(), from.data() + from.size(), to.data());

}
*/

} // namespace sycl
} // namespace thrust

#endif // THRUST_SYCL_H

// vim: ts=2 sw=2
