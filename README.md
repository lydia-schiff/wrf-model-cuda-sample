wrf-model-cuda-sample
===

A sample of my work for the high performance computing group at the [Space-Science and Engineering Center (SSEC)](http://www.ssec.wisc.edu/), at UW-Madison.

The [Weather Research and Forcasting model](http://www.wrf-model.org) is a numerical weather prediction model used for forcasting and running climate models for research.

It's Runge-Kutta numerical integration on huge parallel arrays, which is super expensive computationally, but ideal for GPGPU. (i.e. [embarrassingly parallel](http://en.wikipedia.org/wiki/Embarrassingly_parallel))

The original code is in Fortran-90, but my group was trying to use CUDA-C to do the hard parts on Nvidia GPU-clusters. I worked on translating a significant portion of the code to C, and rewrote most of the computationally difficult portions to run as massively-parallel algorithms. It's a game of micro-seconds, because it all adds up.


This is a sample of the work.
Fortran, C, and CUDA versions of the *advance_mu_t* module, which is a small dynamics module.

On a desktop with 3 GTX-680 GPUs:

Version of module  | Time
------------- | -------------
Original Fortran  | 152.0 ms
CUDA-C version | 0.051 ms


#####3235x speedup! ooh-e!

The idea is to use small desktop GPU-clusters to run simulations in a few minutes, that would have otherwise taken hours without a mad-expensive CPU-cluster.
