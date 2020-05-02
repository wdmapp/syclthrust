#ifndef THRUST_DEVICE_PTR_H
#define THRUST_DEVICE_PTR_H

//#include <complex>

//#include "thrust/sycl.h"

namespace thrust {

/*! Device reference
 */
  /*
template <typename T>
class device_reference {
  public:
    using value_type = std::remove_const_t<T>;
    using pointer = std::remove_const_t<T>
    device_reference(const device_reference &) = default;
    device_reference(const device_pointer &ptr);

    device_pointer operater&() const {
      return
    };
}
*/

// define no-op device_pointer/raw ponter casts
template<typename Pointer>
inline Pointer raw_pointer_cast(Pointer p) {
  return p;
}

template<typename Pointer>
inline Pointer device_pointer_cast(Pointer p) {
  return p;
}

} // namespace thrust

#endif // THRUST_DEVICE_PTR_H

// vim: ts=2 sw=2
