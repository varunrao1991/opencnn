__kernel void leakyrelu(
    __global float* src,
    __global float* dst,
    __const int p,  // width
    __const int q,  // height
    __const int m,  // channels
	__const float leakyValue
) {
    const int ixy = get_global_id(0); // goes up to pq
    const int iz = get_global_id(1) * 16; // goes up to (m + 15) / 16

    int base_index = ixy * m + iz;

    if (iz + 15 < m) {
        float16 src_val = vload16(0, src + base_index);
        float16 result = max(leakyValue * src_val, src_val);
        vstore16(result, 0, dst + base_index);
    } else {
        for (int offset = 0; offset < 16 && (iz + offset) < m; ++offset) {
            float src_val = src[base_index + offset];
            dst[base_index + offset] =  max(leakyValue * src_val, src_val);
        }
    }
}