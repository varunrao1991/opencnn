__kernel void softmax1(__global const float* inputMatrix,
                      __global float* outputVector,
                      const int channels)
{
    const int index = get_global_id(0);  // rows * cols
	float sum = 0.0f;

	for (int channel = 0; channel < channels; channel++)
	{
		sum += exp(inputMatrix[index * channels + channel]);
	}
	outputVector[index] = sum;
}
