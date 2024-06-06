__kernel void add(__global float *src1, __global float *src2,
                  __global float *dst,
                  const int p, // width
                  const int q, // height
                  const int m  // channels
) {
  const int ixy = get_global_id(0);     // goes up to pq
  const int iz = get_global_id(1) * 16; // goes up to (m + 15) / 16

  int base_index = ixy * m + iz;

  if (iz + 15 < m) {
    float16 src1_val = vload16(0, src1 + base_index);
    float16 src2_val = vload16(0, src2 + base_index);
    float16 result = src1_val + src2_val;
    vstore16(result, 0, dst + base_index);
  } else {
    for (int offset = 0; offset < 16 && (iz + offset) < m; ++offset) {
      dst[base_index + offset] =
          src1[base_index + offset] + src2[base_index + offset];
    }
  }
}