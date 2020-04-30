#include <iostream>
#include <chrono>
#include <thread>

#include <thrust/device_vector.h>
//#include <thrust/sycl.h>

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

    thrust::device_vector<int> d_a{N};
    thrust::device_vector<int> d_b{N};
    thrust::device_vector<int> d_c{N};
    std::vector<int> h_a(N);
    std::vector<int> h_b(N);
    std::vector<int> h_c(N);

    int *d_a_data = d_a.data();
    int *d_b_data = d_b.data();
    int *d_c_data = d_c.data();

    auto e0 = q.submit([&](handler & cgh) {
        cgh.parallel_for<class DeviceVectorInit>(range<1>(N), [=](id<1> idx) {
            int i = idx[0];
            d_a_data[i] = i;
        });
    });
    e0.wait();

    // memcpy and wait
    d_b = d_a;

    auto e1 = q.submit([&](handler & cgh) {
        cgh.parallel_for<class DeviceVectorAdd>(range<1>(N), [=](id<1> idx) {
            int i = idx[0];
            d_c_data[i] = d_a_data[i] * d_b_data[i];
        });
    });
    e1.wait();

    q.memcpy(h_a.data(), d_a_data, N*sizeof(int));
    q.memcpy(h_b.data(), d_b_data, N*sizeof(int));
    q.memcpy(h_c.data(), d_c_data, N*sizeof(int));
    q.wait();

    for (i=0; i<N; i++) {
        std::cout << "[" << i << "] " << h_a[i] << " * " << h_b[i]
                  << " = " << h_c[i] << std::endl;
    }

    return 0;
}
