
#include "cuda_runtime.h"
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <chrono>
#include <cufft.h>
#define M_PI 3.14159265358979f
#define CHECK_CUDA_ERROR(err) do { \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)
// Проверка ошибок cuFFT
#define CHECK_CUFFT_ERROR(call)                                             \
    do {                                                                    \
        cufftResult err = call;                                             \
        if (err != CUFFT_SUCCESS) {                                         \
            std::cerr << "cuFFT error at " << __FILE__ << ":" << __LINE__   \
                      << " code=" << err << "\n";                           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

std::vector<float> read_input_data(int N, int M, const std::string& filename) {
    size_t num_elements = static_cast<size_t>(M) * N;

    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Ошибка: не удалось открыть файл " + filename);
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    size_t expected_size = num_elements * sizeof(double);
    if (file_size != expected_size) {
        file.close();
        throw std::runtime_error("Ошибка: размер файла " + std::to_string(file_size) +
            " не соответствует ожидаемому " + std::to_string(expected_size));
    }

    std::vector<double> buffer(num_elements);
    file.read(reinterpret_cast<char*>(buffer.data()), expected_size);
    if (!file) {
        file.close();
        throw std::runtime_error("Ошибка: не удалось прочитать данные из файла");
    }

    file.close();

    std::vector<float> h_input(num_elements);
    std::transform(buffer.begin(), buffer.end(), h_input.begin(),
        [](double d) { return static_cast<float>(d); });

    return h_input;
}
// Функция вычисления ДПФ
void dft(const std::vector<float>& real_in, const std::vector<float>& imag_in,
    std::vector<float>& real_out, std::vector<float>& imag_out) {
    int N = real_in.size();
    real_out.resize(N);
    imag_out.resize(N);
    for (int k = 0; k < N; ++k) {
        real_out[k] = 0.0;
        imag_out[k] = 0.0;
        for (int n = 0; n < N; ++n) {
            float angle = -2.0 * M_PI * k * n / N;
            float cos_val = cos(angle);
            float sin_val = sin(angle);
            real_out[k] += real_in[n] * cos_val - imag_in[n] * sin_val;
            imag_out[k] += real_in[n] * sin_val + imag_in[n] * cos_val;
        }
    }
}

// Функция вычисления обратного ДПФ
void idft(const std::vector<float>& real_in, const std::vector<float>& imag_in,
    std::vector<float>& real_out, std::vector<float>& imag_out) {
    int N = real_in.size();
    real_out.resize(N);
    imag_out.resize(N);
    for (int n = 0; n < N; ++n) {
        real_out[n] = 0.0;
        imag_out[n] = 0.0;
        for (int k = 0; k < N; ++k) {
            float angle = 2.0 * M_PI * k * n / N;
            float cos_val = cos(angle);
            float sin_val = sin(angle);
            real_out[n] += real_in[k] * cos_val - imag_in[k] * sin_val;
            imag_out[n] += real_in[k] * sin_val + imag_in[k] * cos_val;
        }
        real_out[n] /= N;
        imag_out[n] /= N;
    }
}


// Запись сигнала в файл
void write_signal(const std::string& filename, const std::vector<float>& real_parts, const std::vector<float>& imag_parts) {
    std::ofstream file(filename);
    for (size_t i = 0; i < real_parts.size(); ++i) {
        file << real_parts[i] << " " << imag_parts[i] << std::endl;
    }
    file.close();
}

// Вычисление модуля ДПФ
std::vector<float> magnitude(const std::vector<float>& real_parts, const std::vector<float>& imag_parts) {
    std::vector<float> mag;
    for (size_t i = 0; i < real_parts.size(); ++i) {
        mag.push_back(std::sqrt(real_parts[i] * real_parts[i] + imag_parts[i] * imag_parts[i]));
    }
    return mag;
}
// CUDA ядро: каждый поток вычисляет X_k
__global__ void dft_kernel(const float* d_real_in, float* d_imag_in,
    float* d_real_out, float* d_imag_out, float N = 4301) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if ( k < N) {
        d_real_out[k] = 0.0;
        d_imag_out[k] = 0.0;
        for (int n = 0; n < N; ++n) {
            float angle = -2.0 * M_PI * k * n / N;
            float cos_val = cos(angle);
            float sin_val = sin(angle);
            d_real_out[k] += d_real_in[n] * cos_val - d_imag_in[n] * sin_val;
            d_imag_out[k] += d_real_in[n] * sin_val + d_imag_in[n] * cos_val;
        }
    }
}
__global__ void idft_kernel(const float* d_real_in, float* d_imag_in,
    float* d_real_out, float* d_imag_out, float N = 4301) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < N) {
        d_real_out[k] = 0.0;
        d_imag_out[k] = 0.0;
        for (int n = 0; n < N; ++n) {
            float angle = 2.0 * M_PI * k * n / N;
            float cos_val = cos(angle);
            float sin_val = sin(angle);
            d_real_out[k] += d_real_in[n] * cos_val - d_imag_in[n] * sin_val;
            d_imag_out[k] += d_real_in[k] * sin_val + d_imag_in[n] * cos_val;
        }
        d_real_out[k] /= N;
        d_imag_out[k] /= N;
    }
}
void dft_cuda(const std::vector<float>& real_in, const std::vector<float>& imag_in,
    std::vector<float>& real_out, std::vector<float>& imag_out) {
    int N = real_in.size();
    std::vector<float> h_real_out(N);
    std::vector<float> h_imag_out(N);

    float* d_real_in, float* d_imag_in;
    float* d_real_out, float* d_imag_out;
    CHECK_CUDA_ERROR(cudaMalloc(&d_real_in, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_imag_in, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_real_out, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_imag_out, N * sizeof(float)));

    std::cout << "GPU Execution Times:\n";
    int block_size = 256;
    dim3 blockDim(block_size);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x);

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    CHECK_CUDA_ERROR(cudaMemcpy(d_real_in, real_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_imag_in, imag_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    dft_kernel << <gridDim, blockDim >> > (d_real_in, d_imag_in, d_real_out, d_imag_out, N);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaMemcpy(h_real_out.data(), d_real_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_imag_out.data(), d_imag_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float time_ms;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_ms, start, stop));
    std::cout << "  Block=" << block_size << ": " << time_ms << " ms \n";
    real_out = h_real_out; imag_out = h_imag_out;

    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_real_in));
    CHECK_CUDA_ERROR(cudaFree(d_imag_in));
    CHECK_CUDA_ERROR(cudaFree(d_real_out));
    CHECK_CUDA_ERROR(cudaFree(d_imag_out));
}

void dft_cufft(const std::vector<float>& real_in,
    const std::vector<float>& imag_in,
    std::vector<float>& real_out,
    std::vector<float>& imag_out) {
    int N = static_cast<int>(real_in.size());

    cufftComplex* d_data = nullptr;

    CHECK_CUDA_ERROR(cudaMalloc(&d_data, sizeof(cufftComplex) * N));

    std::vector<cufftComplex> h_data(N);
    for (int i = 0; i < N; ++i) {
        h_data[i].x = real_in[i];
        h_data[i].y = imag_in[i];
    }

    CHECK_CUDA_ERROR(cudaMemcpy(d_data, h_data.data(),
        sizeof(cufftComplex) * N,
        cudaMemcpyHostToDevice));

    cufftHandle plan;
    CHECK_CUFFT_ERROR(cufftPlan1d(&plan, N, CUFFT_C2C, /*batch=*/1));

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    CHECK_CUFFT_ERROR(cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD));

    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float time_ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_ms, start, stop));
    std::cout << "cuFFT Execution Time: " << time_ms << " ms\n";


    CHECK_CUDA_ERROR(cudaMemcpy(h_data.data(), d_data,
        sizeof(cufftComplex) * N,
        cudaMemcpyDeviceToHost));

    real_out.resize(N);
    imag_out.resize(N);
    for (int i = 0; i < N; ++i) {
        real_out[i] = h_data[i].x;
        imag_out[i] = h_data[i].y;
    }


    CHECK_CUFFT_ERROR(cufftDestroy(plan));
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
}
__global__ void dft_reduction_kernel(const float* __restrict__ d_real_in,
    const float* __restrict__ d_imag_in,
    float* __restrict__ d_real_out,
    float* __restrict__ d_imag_out,
    int N) {
    extern __shared__ float sdata[];
    float* s_real = sdata;
    float* s_imag = sdata + blockDim.x;

    int k = blockIdx.x;           
    int tid = threadIdx.x;
    int stride = blockDim.x;

    float partial_real = 0.0f;
    float partial_imag = 0.0f;

    for (int n = tid; n < N; n += stride) {
        float angle = -2.0f * M_PI * k * n / N;
        float c = cosf(angle);
        float s = sinf(angle);
        partial_real += d_real_in[n] * c - d_imag_in[n] * s;
        partial_imag += d_real_in[n] * s + d_imag_in[n] * c;
    }
    s_real[tid] = partial_real;
    s_imag[tid] = partial_imag;
    __syncthreads();

    // reduction
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            s_real[tid] += s_real[tid + offset];
            s_imag[tid] += s_imag[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_real_out[k] = s_real[0];
        d_imag_out[k] = s_imag[0];
    }
}
__global__ void dft_warp_shuffle_kernel(const float* __restrict__ d_real_in,
    const float* __restrict__ d_imag_in,
    float* __restrict__ d_real_out,
    float* __restrict__ d_imag_out,
    int N) {
    int k = blockIdx.x;             
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    float real_sum = 0.0f;
    float imag_sum = 0.0f;

    for (int n = tid; n < N; n += blockSize) {
        float angle = -2.0f * M_PI * k * n / N;
        float c = cosf(angle);
        float s = sinf(angle);
        real_sum += d_real_in[n] * c - d_imag_in[n] * s;
        imag_sum += d_real_in[n] * s + d_imag_in[n] * c;
    }

    // warp-shuffle reduction for real
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        real_sum += __shfl_down_sync(0xFFFFFFFF, real_sum, offset);
        imag_sum += __shfl_down_sync(0xFFFFFFFF, imag_sum, offset);
    }

    // thread lane 0 of each warp writes its partial to shared mem
    __shared__ float warp_real[32]; 
    __shared__ float warp_imag[32];
    int lane = tid & (warpSize - 1);
    int warpId = tid >> 5;
    if (lane == 0) {
        warp_real[warpId] = real_sum;
        warp_imag[warpId] = imag_sum;
    }
    __syncthreads();

    if (warpId == 0) {
        float block_real = (tid < (blockSize + warpSize - 1) / warpSize) ? warp_real[lane] : 0.0f;
        float block_imag = (tid < (blockSize + warpSize - 1) / warpSize) ? warp_imag[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            block_real += __shfl_down_sync(0xFFFFFFFF, block_real, offset);
            block_imag += __shfl_down_sync(0xFFFFFFFF, block_imag, offset);
        }
        if (lane == 0) {
            d_real_out[k] = block_real;
            d_imag_out[k] = block_imag;
        }
    }
}

void dft_reduction_cuda(const std::vector<float>& real_in, const std::vector<float>& imag_in,
    std::vector<float>& real_out, std::vector<float>& imag_out) {
    int N = real_in.size();
    std::vector<float> h_real_out(N);
    std::vector<float> h_imag_out(N);

    float* d_real_in, float* d_imag_in;
    float* d_real_out, float* d_imag_out;
    CHECK_CUDA_ERROR(cudaMalloc(&d_real_in, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_imag_in, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_real_out, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_imag_out, N * sizeof(float)));

    std::cout << "GPU Execution Times:\n";
    int block_size = 256;
    dim3 blockDim(block_size);
    dim3 gridDim(N/*(N + blockDim.x - 1) / blockDim.x*/);

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));
    CHECK_CUDA_ERROR(cudaEventRecord(start));

    CHECK_CUDA_ERROR(cudaMemcpy(d_real_in, real_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_imag_in, imag_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    size_t shared_bytes = 2 * N * sizeof(float);
    dft_warp_shuffle_kernel << <gridDim, blockDim >> > (d_real_in, d_imag_in, d_real_out, d_imag_out, N);
    //dft_reduction_kernel << <gridDim, blockDim, shared_bytes >> > (d_real_in, d_imag_in, d_real_out, d_imag_out, N);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaMemcpy(h_real_out.data(), d_real_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_imag_out.data(), d_imag_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float time_ms;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_ms, start, stop));
    std::cout << "  Block=" << block_size << ": " << time_ms << " ms \n";
    real_out = h_real_out; imag_out = h_imag_out;

    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_real_in));
    CHECK_CUDA_ERROR(cudaFree(d_imag_in));
    CHECK_CUDA_ERROR(cudaFree(d_real_out));
    CHECK_CUDA_ERROR(cudaFree(d_imag_out));
}
int main() {
    const int N = 4301;
    const int M = 600;
    std::vector<float> seismogramma = read_input_data(N, M, "one_SP.bin");
    std::vector<float> h_input;
    int start_index = N * 200; // Начальный индекс
    h_input.assign(seismogramma.begin() + start_index, seismogramma.begin() + start_index + N);
    
    // Векторы для сигнала
    std::vector<float> real_signal = h_input;
    std::vector <float> imag_signal(real_signal.size(), 0.0f);

    // Вычисляем ДПФ
    std::vector<float> real_dft, imag_dft;
    auto start_time = std::chrono::high_resolution_clock::now();
    dft(real_signal, imag_signal, real_dft, imag_dft);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Time CPU: " << duration.count() << " ms" << std::endl;
    
    // Вычисляем обратное ДПФ для проверки
    std::vector<float> real_reconstructed, imag_reconstructed;
    idft(real_dft, imag_dft, real_reconstructed, imag_reconstructed);

    // Проверка корректности
    float max_error = 0.0;
    for (size_t i = 0; i < real_signal.size(); ++i) {
        float error = std::abs(real_signal[i] - real_reconstructed[i]);
        if (error > max_error) max_error = error;
    }
    std::cout << "max_error: " << max_error << std::endl;

    // Записываем исходный сигнал в файл
    write_signal("original_signal.txt", real_signal, imag_signal);

    // Вычисляем и записываем модуль ДПФ в файл
    auto mag = magnitude(real_dft, imag_dft);
    std::ofstream mag_file("magnitude.txt");
    for (float m : mag) {
        mag_file << m << std::endl;
    }
    mag_file.close();
    ///////////////////////////CUDA1
    dft_cuda(real_signal, imag_signal, real_dft, imag_dft);
    idft(real_dft, imag_dft, real_reconstructed, imag_reconstructed);
    // Проверка корректности
    max_error = 0.0;
    for (size_t i = 0; i < real_signal.size(); ++i) {
        float error = std::abs(real_signal[i] - real_reconstructed[i]);
        if (error > max_error) max_error = error;
    }
    std::cout << "max_errorCUDA: " << max_error << std::endl;
    dft_cufft(real_signal, imag_signal, real_dft, imag_dft);
    idft(real_dft, imag_dft, real_reconstructed, imag_reconstructed);

    // Проверка корректности
    max_error = 0.0;
    for (size_t i = 0; i < real_signal.size(); ++i) {
        float error = std::abs(real_signal[i] - real_reconstructed[i]);
        if (error > max_error) max_error = error;
    }
    std::cout << "max_errorCUFFT: " << max_error << std::endl;

    dft_reduction_cuda(real_signal, imag_signal, real_dft, imag_dft);
    idft(real_dft, imag_dft, real_reconstructed, imag_reconstructed);
    // Проверка корректности
    max_error = 0.0;
    for (size_t i = 0; i < real_signal.size(); ++i) {
        float error = std::abs(real_signal[i] - real_reconstructed[i]);
        if (error > max_error) max_error = error;
    }
    std::cout << "max_errorCUDAreduction: " << max_error << std::endl;
    return 0;
}