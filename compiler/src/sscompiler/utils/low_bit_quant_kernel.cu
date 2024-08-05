// File: packbits_kernel.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void packbits_kernel(const uint8_t* __restrict__ input, uint8_t* __restrict__ output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * (cols / 8)) {
        int row = idx / (cols / 8);
        int col_start = (idx % (cols / 8)) * 8;

        uint8_t packed_byte = 0;
        for (int i = 0; i < 8; ++i) {
            packed_byte |= (input[row * cols + col_start + i] & 1) << (7 - i);
        }
        output[idx] = packed_byte;
    }
}

__global__ void unpackbits_kernel(uint8_t* input, uint8_t* output, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < rows * cols) {
        int row = idx / cols;
        int col = idx % cols;
        int bit = (input[row * (cols / 8) + col / 8] >> (7 - (col % 8))) & 1;
        output[row * cols + col] = bit;
    }
}

torch::Tensor packbits_cuda(torch::Tensor input) {
    int rows = input.size(0);
    int cols = input.size(1);
    auto output = torch::zeros({rows, cols / 8}, torch::dtype(torch::kUInt8).device(input.device()));

    int threads = 1024;
    int blocks = (rows * (cols / 8) + threads - 1) / threads;

    packbits_kernel<<<blocks, threads>>>(input.data_ptr<uint8_t>(), output.data_ptr<uint8_t>(), rows, cols);

    return output;
}

torch::Tensor unpackbits_cuda(torch::Tensor input, int rows, int cols) {
    auto output = torch::zeros({rows, cols}, torch::dtype(torch::kUInt8).device(input.device()));
    int threads = 1024;
    int blocks = (rows * cols + threads - 1) / threads;
    unpackbits_kernel<<<blocks, threads>>>(input.data_ptr<uint8_t>(), output.data_ptr<uint8_t>(), rows, cols);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("packbits_cuda", &packbits_cuda, "Pack bits (CUDA)");
    m.def("unpackbits_cuda", &unpackbits_cuda, "Unpack bits (CUDA)");
}
