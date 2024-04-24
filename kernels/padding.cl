__kernel void padding(__global const float* input, 
						__global float* output, 
						int input_width, 
						int input_height, 
						int channels,
						int py1,
						int py2,
						int px1,
						int px2, 
						float pad_value)
{
    int j = get_global_id(0); // Width index
    int i = get_global_id(1); // Height index
    int k = get_global_id(2); // Channel index

	const output_height = input_height + py1 + py2;

    if (i >= (py1 + input_height) || j >= (px1 + input_width) || i < py1 || j < px1) {
        output[i * channels + j * output_height * channels + k] = pad_value; // Fill the padded region with the specified value
    }
    else {
        output[i * channels + j * output_height * channels + k] = input[(i-py1) * channels + (j-px1) * input_height * channels + k];
    }
}
