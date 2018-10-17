from tensorflow.python import pywrap_tensorflow
import numpy as np
checkpoint_path='models/VGGnet_fast_rcnn_iter_470000.ckpt'#your ckpt path
reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map=reader.get_variable_to_shape_map()

vgg19={}

for key in var_to_shape_map:
    print ("tensor_name",key)
    sStr_2=key[:-2]
    print sStr_2
    if not vgg19.has_key(sStr_2):
        vgg19[sStr_2]=[reader.get_tensor(key)]
    else:
        vgg19[sStr_2].append(reader.get_tensor(key))

np.save('name.npy',vgg19)
