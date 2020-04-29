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

    thrust::device_vector<int> d_b{N};
    std::vector<int> h_b(N);

    int *d_b_data = d_b.data();

    auto e0 = q.submit([&](handler & cgh) {
        cgh.parallel_for<class DeviceVectorInit>(range<1>(N), [=](id<1> idx) {
            d_b_data[idx] = idx*idx;
        });
    });
    e0.wait();

    // TODO: fixes race on gpu, cpu and host work as expected
    q.memcpy(h_b.data(), d_b_data, N*sizeof(int));
    q.wait();

    for (i=0; i<N; i++) {
        std::cout << i << ": " << h_b[i] << std::endl;
    }

    return 0;
}
