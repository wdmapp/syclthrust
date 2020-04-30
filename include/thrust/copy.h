#ifndef THRUST_COPY_H
#define THRUST_COPY_H

#include "thrust/sycl.h"

namespace thrust {

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
  cl::sycl::queue& q = thrust::sycl::get_queue();
  q.memcpy(dest, src_start, (src_end-src_start) * sizeof(T));
  q.wait();
}

} // namespace thrust

#endif // THRUST_COPY_H
