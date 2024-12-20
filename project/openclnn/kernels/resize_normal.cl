__kernel void resize_normal(
    __global float* src,
    __global float* dst,
    __const int channels,
    __const int in_width,
    __const int in_height,
    __const int out_width,
    __const int out_height) {
    int gid_c = get_global_id(0);
    int gid_y = get_global_id(1);
    int gid_x = get_global_id(2);


    float scaleX = (float)in_width / out_width;
    float scaleY = (float)in_height / out_height;

    int inputX = (int)round(gid_x * scaleX);
    int inputY = (int)round(gid_y * scaleY);

    int inputIndex = gid_c + channels * (inputX * in_height + inputY);
    int outputIndex = gid_c + channels * (gid_x * out_height + gid_y);

    dst[outputIndex] = src[inputIndex];
}
