# syclthrust

Partial [thrust](https://github.com/thrust/thrust) implementation using SYCL
USM extension. Will include at least `device_vector`, `host_vector`, and
`copy`. Used to prototype SYCL/DPC++ support for Intel GPUs backing
[gtensor](https://github.com/wdmapp/gtensor).

## Dependencies

Requires a SYCL implementation that supports the USM extension. As of April
2020, only Intel DPC++ (either installed via oneAPI beta06, or from source
at [intel/llvm github](https://github.com/intel/llvm).
