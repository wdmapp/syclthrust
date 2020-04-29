#include <vector>

#include <CL/sycl.hpp>
#include <CL/sycl/usm.hpp>

#include <thrust/sycl.h>

namespace thrust {

template <typename T>
class device_vector {
  using device_alloc_type = cl::sycl::usm_allocator<T, cl::sycl::usm::alloc::device>;
  using device_vector_type = std::vector<T, device_alloc_type>;
  using size_type = typename std::size_t;

  private:
    cl::sycl::queue& m_queue;
    device_vector_type m_vec;

  public:
    device_vector(size_type count) : m_queue(thrust::sycl::get_queue()),
                                     m_vec(count, device_alloc_type(m_queue)) {}
    device_vector() : device_vector(0) {}
    T& operator[](size_type i);
    const T& operator[](size_type i) const;
    void resize(size_type new_size);
    size_type size();
    T* data();
};


template <typename T>
inline void device_vector<T>::resize(device_vector::size_type new_size) {
  m_vec.resize(new_size);
}


template <typename T>
inline T& device_vector<T>::operator[](device_vector::size_type i) {
  return m_vec[i];
}


template <typename T>
inline const T& device_vector<T>::operator[](device_vector::size_type i) const {
  return m_vec[i];
}



template <typename T>
inline typename device_vector<T>::size_type device_vector<T>::size() {
  return m_vec.size();
}

template <typename T>
inline T* device_vector<T>::data() {
  return m_vec.data();
}

} // end namespace thrust

// vim: ts=2 sw=2
