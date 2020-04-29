#include <CL/sycl.hpp>

namespace thrust {
namespace sycl {

static inline cl::sycl::queue& get_queue() {
  static cl::sycl::queue q{};
  return q;
}
  /*
template <typename T>
inline copy(const T* src_start, const T* src_end, T* dest) {
  :
}

  thrust::copy(from.data(), from.data() + from.size(), to.data());

}
*/
}
}

// vim: ts=2 sw=2
