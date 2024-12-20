__kernel void softmax2(__global const float* inputMatrix,
                      __global float* inputVector,
                      __global float* outputMatrix,
                      const int channels)
{
    const int xy = get_global_id(0);  // pq
    const int channel = get_global_id(1);  // r

    const int index = xy * channels + channel;
    outputMatrix[index] = exp(inputMatrix[index]) / inputVector[xy];
}
