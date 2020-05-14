#ifndef THRUST_HOST_VECTOR_H
#define THRUST_HOST_VECTOR_H

#include <vector>

#include <CL/sycl.hpp>
#include <CL/sycl/usm.hpp>

#include <thrust/sycl.h>

namespace thrust {

/*! A std::vector like container with storage in host memory. Does not
 * implementation all features of std::vector or cuda thrust::host_vector.
 * In particular, iterators are not yet supported.
 *
 * For the SYCL backend, a default singleton cl::sycl::queue object is used
 * for all operations.
 */
template <typename T>
class host_vector {
  public:
    using value_type = T;

    using host_alloc_type = cl::sycl::usm_allocator<value_type,
                                                cl::sycl::usm::alloc::host>;
    using host_vector_type = std::vector<value_type, host_alloc_type>;

    using pointer = typename host_vector_type::pointer;
    using const_pointer = typename host_vector_type::const_pointer;
    using reference = typename host_vector_type::reference;
    using const_reference = typename host_vector_type::const_reference;
    using size_type = typename host_vector_type::size_type;

    host_vector(size_type count) : m_queue(thrust::sycl::get_queue()),
                                     m_vec(count, host_alloc_type(m_queue)) {}
    host_vector() : host_vector(0) {}

    // copy and move constructors
    host_vector(const host_vector &dv)
      : m_queue(thrust::sycl::get_queue()),
        m_vec(dv.m_vec.size(), host_alloc_type(m_queue)) {
      assert(m_vec.size() == dv.m_vec.size());
      m_queue.memcpy(m_vec.data(), dv.data(), m_vec.size()*sizeof(T));
      m_queue.wait();
    }
    host_vector(host_vector &&dv) = delete;

    // element access
    reference operator[](size_type i);
    const_reference operator[](size_type i) const;

    // assignment
    host_vector& operator=(const host_vector &dv) {
      resize(dv.size());
      m_queue.memcpy(data(), dv.data(), size()*sizeof(T));
      m_queue.wait();

      return *this;
    }

    // Note: trying to move the vec results in a complex error related to
    // the sycl usm_allocator
    host_vector& operator=(host_vector &&dv) = delete;
    /*
    host_vector& operator=(host_vector &&dv) {
        m_vec = std::move(dv.m_vec);
        m_queue = std::move(dv.m_queue);

        return *this;
    }
    */

    // functions
    void resize(size_type new_size);
    size_type size() const;
    pointer data();
    const_pointer data() const;

  private:
    cl::sycl::queue& m_queue;
    host_vector_type m_vec;
};



template <typename T>
inline void host_vector<T>::resize(host_vector::size_type new_size) {
  m_vec.resize(new_size);
}


template <typename T>
inline typename host_vector<T>::reference
host_vector<T>::operator[](host_vector::size_type i) {
  return m_vec[i];
}


template <typename T>
inline typename host_vector<T>::const_reference
host_vector<T>::operator[](host_vector::size_type i) const {
  return m_vec[i];
}


template <typename T>
inline typename host_vector<T>::size_type
host_vector<T>::size() const {
  return m_vec.size();
}

template <typename T>
inline typename host_vector<T>::pointer
host_vector<T>::data() {
  return m_vec.data();
}


template <typename T>
inline typename host_vector<T>::const_pointer
host_vector<T>::data() const {
  return m_vec.data();
}


} // end namespace thrust

#endif // THRUST_HOST_VECTOR_H

// vim: ts=2 sw=2
