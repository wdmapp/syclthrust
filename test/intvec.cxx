#include <iostream>
#include <chrono>
#include <thread>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/complex.h>

using namespace std::chrono_literals;
using namespace cl::sycl;

int main(int argc, char **argv) {
    const int N = 32;
    int i;

    auto q = thrust::sycl::get_queue();
    auto dev = q.get_device();
    std::string type;
    if (dev.is_cpu()) {
        type = "CPU  ";
    } else if (dev.is_gpu()) {
        type = "GPU  ";
    } else if (dev.is_host()) {
        type = "HOST ";
    } else {
        type = "OTHER";
    }
    std::cout << "[" << type << "] "
              << dev.get_info<info::device::name>()
              << " {" << dev.get_info<info::device::vendor>() << "}"
              << std::endl;

    // make the int vector complex, just to exercise more things used by
    // gtensor... #hackz
    using value_type = thrust::complex<double>;

    thrust::device_vector<value_type> d_a{N};
    thrust::device_vector<value_type> d_b{N};
    thrust::device_vector<value_type> d_c{N};
    thrust::host_vector<value_type> h_a(N);
    thrust::host_vector<value_type> h_b(N);
    thrust::host_vector<value_type> h_c(N);

    value_type *d_a_data = d_a.data();
    value_type *d_b_data = d_b.data();
    value_type *d_c_data = d_c.data();

    auto e0 = q.submit([&](handler & cgh) {
        cgh.parallel_for<class DeviceVectorInit>(range<1>(N), [=](id<1> idx) {
            int i = idx[0];
            d_a_data[i] = static_cast<value_type>(i);
        });
    });
    e0.wait();

    // memcpy and wait
    d_b = d_a;

    auto e1 = q.submit([&](handler & cgh) {
        cgh.parallel_for<class DeviceVectorAdd>(range<1>(N), [=](id<1> idx) {
            int i = idx[0];
            d_c_data[i] = d_a_data[i].real() * d_b_data[i].real();
        });
    });
    e1.wait();

    thrust::copy(d_a.data(), d_a.data() + d_a.size(), h_a.data());
    thrust::copy(d_b.data(), d_b.data() + d_b.size(), h_b.data());
    // test compat with direct SYCL
    q.memcpy(h_c.data(), d_c_data, N*sizeof(value_type));
    q.wait();

    for (i=0; i<N; i++) {
        std::cout << "[" << i << "] " << h_a[i] << " * " << h_b[i]
                  << " = " << h_c[i] << std::endl;
    }

    return 0;
}

// vim: ts=2 sw=2
