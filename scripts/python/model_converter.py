import numpy as np
import os
from tensorflow.keras.layers import Activation, Concatenate, Add, ZeroPadding2D, LeakyReLU, BatchNormalization, Conv2D, Input, MaxPooling2D, Dropout, concatenate, UpSampling2D, ReLU, InputLayer, Conv2DTranspose, DepthwiseConv2D

model_dir = "./models/"
model_name = "yolo"

epsilon = np.finfo(float).eps
lines_in_graph = []

model_data_dir = os.path.join(model_dir, model_name)
if not os.path.exists(model_data_dir):
    os.makedirs(model_data_dir)
    
for i, layer in enumerate(model.layers):
    print(i, ":", layer.__class__.__name__, layer.name)
    items_to_write = [layer.__class__.__name__, layer.name]
    if isinstance(layer,InputLayer):
        inpu_shape = layer.input.shape
        items_to_write.append("")
        items_to_write.append(str(inpu_shape[2]))
        items_to_write.append(str(inpu_shape[1]))
        items_to_write.append(str(inpu_shape[3]))
    elif isinstance(layer.input, list):
        input_layer_names = []
        for i in range(len(layer.input)):
            inpu_layer_name = layer.input[i].name.split('/')[0]
            input_layer_names.append(inpu_layer_name)
        items_to_write.append("&".join(input_layer_names))
        if isinstance(layer, Add):
            for i in range(len(layer.input)):
                inpu_shape = layer.input[i].shape
                items_to_write.append(str(inpu_shape[2]))
                items_to_write.append(str(inpu_shape[1]))
                items_to_write.append(str(inpu_shape[3]))
        elif isinstance(layer,Concatenate): 
            for i in range(len(layer.input)):
                inpu_shape = layer.input[i].shape
                items_to_write.append(str(inpu_shape[2]))
                items_to_write.append(str(inpu_shape[1]))
                items_to_write.append(str(inpu_shape[3]))
            '''
            elif isinstance(layer,HeadWrapper):
                for i in range(len(layer.input)):
                    inpu_shape = layer.input[i].shape
                    items_to_write.append(str(inpu_shape[2]))
                    items_to_write.append(str(inpu_shape[1]))
                    items_to_write.append(str(inpu_shape[3])) '''
        else:
            print("---------------------------------")
            print("Error: Multi input layer not known")
            print("---------------------------------")            
    else:
        inpu_layer_name = layer.input.name.split('/')[0]
        items_to_write.append(inpu_layer_name)
        if isinstance(layer,Conv2D):
            weights = layer.weights[0]
            path_to_save = os.path.join(model_data_dir, layer.name + '_weights.npy')
            np.save(path_to_save, weights)             
            if layer.use_bias:
                bias = layer.weights[1]
                path_to_save = os.path.join(model_data_dir, layer.name + '_bias.npy')
                np.save(path_to_save, bias)
            layer_activation = layer.activation
            if layer_activation:
                items_to_write.append(layer_activation.__name__)
            else:
                print("Warning: Activation layer present : ", layer_activation.__name__)                
                print("---------------------------------")
            weights_shape = layer.weights[0].shape
            items_to_write.append(str(weights_shape[0]))
            items_to_write.append(str(weights_shape[1]))
            items_to_write.append(str(weights_shape[2]))
            items_to_write.append(str(weights_shape[3]))
            inpu_shape = layer.input.shape
            items_to_write.append(str(inpu_shape[2]))
            items_to_write.append(str(inpu_shape[1]))
            items_to_write.append(str(inpu_shape[3]))
            strides = layer.strides
            if strides[0] != strides[1]:
                print("---------------------------------")
                print("Error: Strides must be equal")
                print("---------------------------------")
            dilation_rates = layer.dilation_rate
            if dilation_rates[0] != dilation_rates[1] and dilation_rates[1] != 1:
                print("---------------------------------")
                print("Error: dilation_rates must be equal")
                print("---------------------------------")                
            items_to_write.append(str(strides[0]))
            items_to_write.append(str(layer.use_bias))
            items_to_write.append(str(layer.padding))            
        elif isinstance(layer,Conv2DTranspose):
            weights = layer.weights[0]
            path_to_save = os.path.join(model_data_dir, layer.name + '_weights.npy')
            np.save(path_to_save, weights)             
            if layer.use_bias:
                bias = layer.weights[1]
                path_to_save = os.path.join(model_data_dir, layer.name + '_bias.npy')
                np.save(path_to_save, bias)
            weights_shape = layer.weights[0].shape
            items_to_write.append(str(weights_shape[0]))
            items_to_write.append(str(weights_shape[1]))
            items_to_write.append(str(weights_shape[2]))
            items_to_write.append(str(weights_shape[3]))
            inpu_shape = layer.input.shape
            items_to_write.append(str(inpu_shape[2]))
            items_to_write.append(str(inpu_shape[1]))
            items_to_write.append(str(inpu_shape[3]))
            strides = layer.strides
            if strides[0] != strides[1]:
                print("---------------------------------")
                print("Error: Strides must be equal")
                print("---------------------------------")
            items_to_write.append(str(strides[0]))
            items_to_write.append(str(layer.use_bias))
        elif isinstance(layer,DepthwiseConv2D):   
            weights = layer.weights[0]
            path_to_save = os.path.join(model_data_dir, layer.name + '_weights.npy')
            np.save(path_to_save, weights)             
            if layer.use_bias:
                bias = layer.weights[1]
                path_to_save = os.path.join(model_data_dir, layer.name + '_bias.npy')
                np.save(path_to_save, bias)
            weights_shape = layer.weights[0].shape
            items_to_write.append(str(weights_shape[0]))
            items_to_write.append(str(weights_shape[1]))
            items_to_write.append(str(weights_shape[2]))
            inpu_shape = layer.input.shape
            items_to_write.append(str(inpu_shape[2]))
            items_to_write.append(str(inpu_shape[1]))
            items_to_write.append(str(inpu_shape[3]))
            strides = layer.strides
            if strides[0] != strides[1]:
                print("---------------------------------")
                print("Error: Strides must be equal")
                print("---------------------------------")
            items_to_write.append(str(strides[0]))
            items_to_write.append(str(layer.use_bias))
        elif isinstance(layer,BatchNormalization):
            weights = layer.get_weights()
            gamma = weights[0]
            beta = weights[1]
            moving_mean = weights[2]
            moving_variance = weights[3]
            valueToSqrt = np.sqrt(moving_variance + epsilon)
            a = gamma / np.sqrt(valueToSqrt)
            b = - moving_mean * a + beta
            path_to_save = os.path.join(model_data_dir, layer.name + '_mean.npy')
            np.save(path_to_save, b)
            path_to_save = os.path.join(model_data_dir, layer.name + '_variance.npy')
            np.save(path_to_save, a) 

            inpu_shape = layer.input.shape
            items_to_write.append(str(inpu_shape[2]))
            items_to_write.append(str(inpu_shape[1]))
            items_to_write.append(str(inpu_shape[3]))
        elif isinstance(layer,MaxPooling2D):
            inpu_shape = layer.input.shape
            items_to_write.append(str(inpu_shape[2]))
            items_to_write.append(str(inpu_shape[1]))
            items_to_write.append(str(inpu_shape[3]))
        elif isinstance(layer,ZeroPadding2D):
            if(layer.padding[0][1] != layer.padding[1][1] and layer.padding[0][0] == 0 and layer.padding[1][0] != 0):
                print(layer.padding)
                print("---------------------------------")
                print("Error: This padding not supported") 
                print("---------------------------------")
            inpu_shape = layer.input.shape
            items_to_write.append(str(inpu_shape[2]))
            items_to_write.append(str(inpu_shape[1]))
            items_to_write.append(str(inpu_shape[3]))
            items_to_write.append(str(layer.padding[0][0]))
            items_to_write.append(str(layer.padding[0][1]))
            items_to_write.append(str(layer.padding[1][0]))
            items_to_write.append(str(layer.padding[1][1]))
        elif isinstance(layer, ReLU):
            maxValue = layer.max_value
            inpu_shape = layer.input.shape
            items_to_write.append(str(inpu_shape[2]))
            items_to_write.append(str(inpu_shape[1]))
            items_to_write.append(str(inpu_shape[3]))
            items_to_write.append(str(maxValue))
        elif isinstance(layer, LeakyReLU):
            alpha = layer.alpha.item()
            inpu_shape = layer.input.shape
            items_to_write.append(str(inpu_shape[2]))
            items_to_write.append(str(inpu_shape[1]))
            items_to_write.append(str(inpu_shape[3]))
            items_to_write.append(str(alpha))
        elif isinstance(layer, Activation):
            activation_type = layer.get_config()['activation']
            maxValue = 0
            if activation_type == 'relu':
                items_to_write.append("relu")
                maxValue = 0
                if hasattr(layer,'max_value'):
                    maxValue = layer.max_value
                inpu_shape = layer.input.shape
                items_to_write.append(str(inpu_shape[2]))
                items_to_write.append(str(inpu_shape[1]))
                items_to_write.append(str(inpu_shape[3]))
                items_to_write.append(str(maxValue))
            elif activation_type == 'sigmoid':
                items_to_write.append("sigmoid")
                inpu_shape = layer.input.shape
                items_to_write.append(str(inpu_shape[2]))
                items_to_write.append(str(inpu_shape[1]))
                items_to_write.append(str(inpu_shape[3]))
            elif activation_type == 'softmax':
                items_to_write.append("softmax")
                inpu_shape = layer.input.shape
                output_shape = layer.output.shape
                print(output_shape)
                items_to_write.append(str(inpu_shape[1]))
                items_to_write.append(str(inpu_shape[2]))
            else:
                print("---------------------------------")
                print("Error: Only Relu activation with max value supported, provided is ", activation_type)
                print("---------------------------------") 
        else:
            print(layer.input.shape)
            print("Error: Layer is not identified ", layer.__class__.__name__, "-> ", layer.name)
    lines_in_graph.append(",".join(items_to_write))
with open(os.path.join(model_dir, f'{model_name}.txt'), 'w') as f:
    # Write each line of the list to the file
    for line in lines_in_graph:            
        f.write(line + '\n')
    print("File is saved to " + f'{model_name}.txt')    