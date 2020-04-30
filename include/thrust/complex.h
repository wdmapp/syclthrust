#ifndef THRUST_COMPLEX_H
#define THRUST_COMPLEX_H

#include <complex>

#include "thrust/sycl.h"

namespace thrust {

/*! Alias std::complex<T> as thrust::complex<T>
 */
template <typename T>
using complex = typename std::complex<T>;

} // namespace thrust

#endif // THRUST_COMPLEX_H
