#ifndef THRUST_DEVICE_VECTOR_H
#define THRUST_DEVICE_VECTOR_H

#include <vector>

#include <CL/sycl.hpp>
#include <CL/sycl/usm.hpp>

#include <thrust/sycl.h>

namespace thrust {

/*! A std::vector like container with storage in device memory. Does not
 * implementation all features of std::vector or cuda thrust::device_vector.
 * In particular, iterators are not yet supported.
 *
 * For the SYCL backend, a default singleton cl::sycl::queue object is used
 * for all operations.
 */
template <typename T>
class device_vector {
  public:
    using value_type = T;

    using device_alloc_type = cl::sycl::usm_allocator<value_type,
                                                cl::sycl::usm::alloc::device>;
    using device_vector_type = std::vector<value_type, device_alloc_type>;

    using pointer = typename device_vector_type::pointer;
    using const_pointer = typename device_vector_type::const_pointer;
    using reference = typename device_vector_type::reference;
    using const_reference = typename device_vector_type::const_reference;
    using size_type = typename device_vector_type::size_type;

    device_vector(size_type count) : m_queue(thrust::sycl::get_queue()),
                                     m_vec(count, device_alloc_type(m_queue)) {}
    device_vector() : device_vector(0) {}

    // copy and move constructors
    device_vector(const device_vector &dv)
      : m_queue(thrust::sycl::get_queue()),
        m_vec(dv.m_vec.size(), device_alloc_type(m_queue)) {
      assert(m_vec.size() == dv.m_vec.size());
      m_queue.memcpy(m_vec.data(), dv.data(), m_vec.size()*sizeof(T));
      m_queue.wait();
    }
    device_vector(device_vector &&dv) = delete;

    // operators
    reference operator[](size_type i);
    const_reference operator[](size_type i) const;

    device_vector& operator=(const device_vector &dv) {
      resize(dv.size());
      m_queue.memcpy(data(), dv.data(), size()*sizeof(T));
      m_queue.wait();

      return *this;
    }

    device_vector& operator=(device_vector &&dv) = delete;
    /*
    device_vector& operator=(device_vector &&dv) {
        m_vec = std::move(dv.m_vec);

        // TODO: deinit vector
        //v.m_vec = v.device_vector_type(0, v.device_alloc_type(v.m_queue));

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
    device_vector_type m_vec;
};



template <typename T>
inline void device_vector<T>::resize(device_vector::size_type new_size) {
  m_vec.resize(new_size);
}


template <typename T>
inline typename device_vector<T>::reference
device_vector<T>::operator[](device_vector::size_type i) {
  return m_vec[i];
}


template <typename T>
inline typename device_vector<T>::const_reference
device_vector<T>::operator[](device_vector::size_type i) const {
  return m_vec[i];
}


template <typename T>
inline typename device_vector<T>::size_type
device_vector<T>::size() const {
  return m_vec.size();
}

template <typename T>
inline typename device_vector<T>::pointer
device_vector<T>::data() {
  return m_vec.data();
}


template <typename T>
inline typename device_vector<T>::const_pointer
device_vector<T>::data() const {
  return m_vec.data();
}


} // end namespace thrust

#endif // THRUST_DEVICE_VECTOR_H

// vim: ts=2 sw=2
