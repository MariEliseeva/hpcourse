#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.h>
#include "cl.hpp"

#include <vector>
#include <fstream>
#include <iostream>
#include <iterator>
#include <iomanip>

int main() {
    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> devices;
    std::vector<cl::Kernel> kernels;

    try {
        // create platform
        cl::Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        // create context
        cl::Context context(devices);

        // create command queue
        cl::CommandQueue queue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

        // load opencl source
        std::ifstream cl_file("convolution.cl");
        std::string cl_string(std::istreambuf_iterator<char>(cl_file), (std::istreambuf_iterator<char>()));
        cl::Program::Sources source(1, std::make_pair(cl_string.c_str(), cl_string.length() + 1));

        // create program
        cl::Program program(context, source);

        // compile opencl source
        try {
            program.build(devices);
        } catch (cl::Error const & e) {
            std::string log_str = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]);
            std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
            std::cout << log_str;
            return 0;
        }

        std::ifstream fin("input.txt");
        std::ofstream fout("output.txt");

        // create a message to send to kernel
        size_t const block_size = 256;
        size_t N, M;
        fin >> N >> M;

        std::vector<int> a(N * N);
        std::vector<int> b(M * M);
        std::vector<int> c(N * N);
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; j++) {
                fin >> a[i * N + j];
                c[i * N + j] = 0;
            }
        }
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < M; j++) {
                fin >> b[i * N + j];
            }
        }


        // allocate device buffer to hold message
        cl::Buffer dev_a (context, CL_MEM_READ_ONLY, sizeof(double) * N * N);
        cl::Buffer dev_b(context, CL_MEM_WRITE_ONLY, sizeof(double) * M * M);
        cl::Buffer dev_c  (context, CL_MEM_READ_ONLY, sizeof(double) * N * N);

        // copy from cpu to gpu
        queue.enqueueWriteBuffer(dev_a, CL_TRUE, 0, sizeof(double) * N * N, &a[0]);
        queue.enqueueWriteBuffer(dev_b, CL_TRUE, 0, sizeof(double) * M * M, &b[0]);

        // load named kernel from opencl source
        cl::Kernel kernel_gmem(program, "convolution");
        kernel_gmem.setArg(0, dev_a);
        kernel_gmem.setArg(1, dev_b);
        kernel_gmem.setArg(2, dev_c);
        kernel_gmem.setArg(3, static_cast<int>(N));
        kernel_gmem.setArg(4, static_cast<int>(M));
        queue.enqueueNDRangeKernel(kernel_gmem, cl::NullRange, cl::NDRange(N * N), cl::NDRange(block_size));

        queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, sizeof(double) * N * N, &c[0]);
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; j++) {
                fout << c[i * N + j] << " ";
            }
            fout << std::endl;
        }
    } catch (cl::Error e) {
        std::cout << std::endl << e.what() << " : " << e.err() << std::endl;
    }
    return 0;
}