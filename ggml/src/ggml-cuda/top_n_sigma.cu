#include "top_n_sigma.cuh"


template <int BLOCK_SIZE>
__global__ void top_n_sigma_kernel(const float * x, float * y, const int ncols, const float n) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float * x_row = x + row * ncols;
    float * y_row = y + row * ncols;

    // Pass 1: max, sum, valid_count
    float local_max = -INFINITY;
    float local_sum = 0.0f;
    float local_count = 0.0f;

    for (int i = tid; i < ncols; i += BLOCK_SIZE) {
        float val = x_row[i];
        if (val != -INFINITY) {
            local_max = fmaxf(local_max, val);
            local_sum += val;
            local_count += 1.0f;
        }
    }

    // llama.cpp's block_reduce requires external shared memory for warp sync.
    // We use separate arrays to prevent race conditions during consecutive reductions.
    __shared__ float s_reduce_max[32];
    __shared__ float s_reduce_sum[32];
    __shared__ float s_reduce_count[32];

    float total_max   = block_reduce<block_reduce_method::MAX>(local_max, s_reduce_max);
    float total_sum   = block_reduce<block_reduce_method::SUM>(local_sum, s_reduce_sum);
    float total_count = block_reduce<block_reduce_method::SUM>(local_count, s_reduce_count);

    // block_reduce only guarantees the correct final value is returned to thread 0.
    // We must broadcast these results to all threads for the next pass.
    __shared__ float s_broadcast[3];
    if (tid == 0) {
        s_broadcast[0] = total_max;
        s_broadcast[1] = (total_count > 0.0f) ? (total_sum / total_count) : 0.0f;
        s_broadcast[2] = total_count;
    }
    __syncthreads();

    total_max   = s_broadcast[0];
    float mean  = s_broadcast[1];
    total_count = s_broadcast[2];

    // Pass 2: calculate variance
    float local_acc = 0.0f;
    for (int i = tid; i < ncols; i += BLOCK_SIZE) {
        float val = x_row[i];
        if (val != -INFINITY) {
            float diff = val - mean;
            local_acc += diff * diff;
        }
    }

    float total_acc = block_reduce<block_reduce_method::SUM>(local_acc, s_reduce_sum);

    // Broadcast standard deviation
    __shared__ float s_std_dev;
    if (tid == 0) {
        s_std_dev = (total_count > 0.0f) ? sqrtf(total_acc / total_count) : 0.0f;
    }
    __syncthreads();

    float std_dev = s_std_dev;
    float threshold = total_max - (n * std_dev);

    // Pass 3: apply mask
    for (int i = tid; i < ncols; i += BLOCK_SIZE) {
        float val = x_row[i];
        y_row[i] = (val < threshold) ? -INFINITY : val;
    }
}

void ggml_cuda_op_top_n_sigma(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;

    const int64_t ne00 = src0->ne[0];

    cudaStream_t stream = ctx.stream();

    float n_param;
    memcpy(&n_param, dst->op_params, sizeof(float));

    // 1 grid block, 256 threads per block
    top_n_sigma_kernel<256><<<1, 256, 0, stream>>>(src0_d, dst_d, ne00, n_param);
}