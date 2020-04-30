#ifndef THRUST_HOST_VECTOR_H
#define THRUST_HOST_VECTOR_H

#include <vector>

#include <CL/sycl.hpp>
#include <CL/sycl/usm.hpp>

#include <thrust/sycl.h>

namespace thrust {

template <typename T>
class host_vector {
  using host_alloc_type = cl::sycl::usm_allocator<T, cl::sycl::usm::alloc::host>;
  using host_vector_type = std::vector<T, host_alloc_type>;
  using size_type = typename std::size_t;

  private:
    cl::sycl::queue& m_queue;
    host_vector_type m_vec;

  public:
    host_vector(size_type count) : m_queue(thrust::sycl::get_queue()),
                                   m_vec(count, host_alloc_type(m_queue)) {}
    host_vector() : host_vector(0) {}
    T& operator[](size_type i);
    const T& operator[](size_type i) const;
    void resize(size_type new_size);
    size_type size();
    T* data();
    const T const* data() const;
};


template <typename T>
inline void host_vector<T>::resize(host_vector::size_type new_size) {
  m_vec.resize(new_size);
}


template <typename T>
inline T& host_vector<T>::operator[](host_vector::size_type i) {
  return m_vec[i];
}


template <typename T>
inline const T& host_vector<T>::operator[](host_vector::size_type i) const {
  return m_vec[i];
}



template <typename T>
inline typename host_vector<T>::size_type host_vector<T>::size() {
  return m_vec.size();
}

template <typename T>
inline T* host_vector<T>::data() {
  return m_vec.data();
}

template <typename T>
inline const T const* host_vector<T>::data() const {
  return m_vec.data();
}

} // end namespace thrust

#endif // THRUST_HOST_VECTOR_H

// vim: ts=2 sw=2
