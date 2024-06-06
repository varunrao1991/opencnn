#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void batchnormalization(  
    __global const float* bias,
    __global const float* gamma,
    __global const float* input,
    __global float* output,
    const int input_width,
    const int input_height,
    const int channels
) {
    const int ix = get_global_id(0);
    const int iy = get_global_id(1);
    const int iz = get_global_id(2);

    const int input_idx = ix * input_height * channels + iy * channels;

    if (iz + 15 < channels) {
        float16 valueIn = vload16(iz, input + input_idx);
        valueIn *= vload16(iz, gamma);
        valueIn += vload16(iz, bias);

        int output_index = ix * channels * input_height + channels * iy;
        vstore16(valueIn, iz, output + output_index);
    } else {
        for (int offset = iz; offset < channels; ++offset) {
            int output_idx = input_idx + offset;
            float value = input[output_idx];
            value *= gamma[offset];
            value += bias[offset];
            output[output_idx] = value;
        }
    }
}
