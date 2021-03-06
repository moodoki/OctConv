import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras import constraints
from tensorflow.keras import activations
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.keras.backend import repeat_elements
import tensorflow.keras.backend as K

def _upsample_nn(x, factor=2, data_format='channels_last'):
    end_ax = len(x.shape)
    if data_format == 'channels_last':
        end_ax -= 1
        start_ax = 1
    else:
        start_ax = 2

    output = x
    for ax in range(start_ax, end_ax):
        output = repeat_elements(output, factor, axis=ax)
    return output

def _upsample_nn_shape(input_shape, factor, data_format='channels_last'):
    assert data_format in ['channels_last', 'channels_first']
    output_shape = [input_shape[0]] + list(map(lambda x: x*factor, input_shape[1:]))
    if data_format == 'channels_last':
        output_shape[-1] = input_shape[-1]
    else:
        output_shape[1] = input_shape[1]
    return output_shape

def UpSampleNN(factor=2, data_format='channels_last'):
    return layers.Lambda(lambda x: _upsample_nn(x, factor, data_format),
                         lambda x: _upsample_nn_shape(x, factor, data_format),
                        )

class OctConv(layers.Layer):
    def __init__(self, rank,
                 filters,
                 kernel_size,
                 alpha=0.5,
                 strides=1,
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 name=None,
                 **kwargs
                ):
        super(OctConv, self).__init__(
            trainable=trainable,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        self.rank = rank
        self.alpha = alpha
        self.filters = filters
        self.lr_filters = int(self.alpha * self.filters)
        self.hr_filters = self.filters - self.lr_filters
        self.kernel_size = conv_utils.normalize_tuple(
            kernel_size, rank, 'kernel_size')
        self.dilation_rate = conv_utils.normalize_tuple(
            dilation_rate, rank, 'dilation_rate')
        self.data_format = data_format
        self.strides = conv_utils.normalize_tuple(
            strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.bias_constraint = constraints.get(bias_constraint)
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.filter_names = ['hh', 'hl', 'lh', 'll']

    def _compute_lr_shape(self, hr_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        lr_shape = [hr_shape[0]] + list(map(lambda x: x//2, hr_shape[1:]))
        lr_shape[channel_axis] = hr_shape[channel_axis]
        return tf.TensorShape(lr_shape)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        if isinstance(input_shape, list):
            hr_shape = tf.TensorShape(input_shape[0])
            lr_shape = tf.TensorShape(input_shape[1])
            in_shapes = [hr_shape, lr_shape, lr_shape, lr_shape]
        else:
            hr_shape = tf.TensorShape(input_shape)
            lr_shape = self._compute_lr_shape(hr_shape)
            in_shapes = [hr_shape, lr_shape, None, None]


        input_dim = int(hr_shape[channel_axis])
        kernel_shapes = [self.kernel_size + (input_dim, self.hr_filters),
                         self.kernel_size + (input_dim, self.lr_filters),
                         self.kernel_size + (input_dim, self.hr_filters),
                         self.kernel_size + (input_dim, self.lr_filters),
                        ]
        #4 Kernels and (optional) biases weights
        add_kernel_weights = lambda name, shape : self.add_weight(name='kernel_' + name,
                                                           shape=shape,
                                                           initializer=self.kernel_initializer,
                                                           regularizer=self.kernel_regularizer,
                                                           constraint=self.kernel_constraint,
                                                           trainable=True,
                                                           dtype=self.dtype)
        add_bias_weights = lambda name, filters: self.add_weight(name='bias_' + name,
                                                           shape=(filters,),
                                                           initializer=self.bias_initializer,
                                                           regularizer=self.bias_regularizer,
                                                           constraint=self.bias_constraint,
                                                           trainable=True,
                                                           dtype=self.dtype)
        if self.padding == 'causal':
            op_padding = 'valid'
        else:
            op_padding = self.padding

        create_conv_op = lambda in_shape, filter_shape: nn_ops.Convolution(
            in_shape,
            filter_shape=filter_shape,
            dilation_rate=self.dilation_rate,
            strides=self.strides,
            padding=op_padding.upper(),
            data_format=conv_utils.convert_data_format(self.data_format,
                                                       self.rank + 2))
        self.kernels = {}
        self.biases = {}
        self._conv_ops = {}
        for k, f, ii in zip(self.filter_names, kernel_shapes, in_shapes):
            if ii is not None:
                self.kernels[k] = add_kernel_weights(k, f) if f[-1] > 0 else None
                self._conv_ops[k] = create_conv_op(ii, self.kernels[k].get_shape()) if f[-1] > 0 else None
                if self.use_bias:
                    self.biases[k] = add_bias_weights(k, f[-1]) if f[-1] > 0  else None
                else:
                    self.biases[k] = None
            else:
                self.kernels[k] = None
                self._conv_ops[k] = None
                self.biases[k] = None

        self._upsample_op = UpSampleNN(factor=2,
                                       data_format=self.data_format)
        self._downsample_op = lambda x: tf.nn.pool(
            x,
            window_shape=[2]*self.rank,
            pooling_type='AVG',
            padding='VALID',
            strides=[2]*self.rank,
            data_format=conv_utils.convert_data_format(self.data_format,
                                                       self.rank+2))
        #self.built = True
        super().build(input_shape)

    def _conv_add_bias(self, inputs, fname):
        """Returns None if filter or xx isn't defined,
        convolve x with specified filter and add bias if bias is used"""
        if inputs == None or self._conv_ops[fname] == None:
            return None

        outputs = self._conv_ops[fname](inputs, self.kernels[fname])

        if fname[-1] == 'h':
            filters = self.hr_filters
        else:
            filters = self.lr_filters

        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.biases[fname], (1, filters, 1))
                    outputs += bias
                if self.rank == 2:
                    outputs = nn.bias_add(outputs, self.biases[fname], data_format='NCHW')
                if self.rank == 3:
                    # As of Mar 2017, direct addition is significantly slower than
                    # bias_add when computing gradients. To use bias_add, we collapse Z
                    # and Y into a single dimension to obtain a 4D input tensor.
                    outputs_shape = outputs.shape.as_list()
                    if outputs_shape[0] is None:
                        outputs_shape[0] = -1
                    outputs_4d = array_ops.reshape(outputs,
                                                   [outputs_shape[0], outputs_shape[1],
                                                    outputs_shape[2] * outputs_shape[3],
                                                    outputs_shape[4]])
                    outputs_4d = nn.bias_add(outputs_4d, self.biases[fname], data_format='NCHW')
                    outputs = array_ops.reshape(outputs_4d, outputs_shape)
            else:
                outputs = nn.bias_add(outputs, self.biases[fname], data_format='NHWC')
        return outputs

    def call(self, xx):
        x_h, x_l = xx if isinstance(xx, list) else (xx, None)

        y_h = self._conv_add_bias(x_h, 'hh')
        y_l = self._downsample_op(x_h)
        y_l = self._conv_add_bias(y_l, 'hl')

        y_ll = self._conv_add_bias(x_l, 'll')
        y_l = y_l + y_ll if y_ll is not None else y_l

        y_lh = self._conv_add_bias(x_l, 'lh')
        y_h = y_h + self._upsample_op(y_lh) if y_lh is not None else y_h

        if self.activation:
            y_h = self.activation(y_h)
            y_l = self.activation(y_l)

        return [y_h, y_l]

    def _compute_causal_padding(self):
        """Calculates padding for 'causal' option for 1-d conv layers."""
        left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
        if self.data_format == 'channels_last':
            causal_padding = [[0, 0], [left_pad, 0], [0, 0]]
        else:
            causal_padding = [[0, 0], [0, 0], [left_pad, 0]]
        return causal_padding

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'alpha': self.alpha,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        }

        base_config = super(OctConv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return list([compute_output_shape_single_res(ii) for ii in input_shape])

        lr_shape = self._compute_lr_shape(input_shape)

        return [compute_output_shape_single_res(input_shape),
                compute_output_shape_single_res(lr_shape)]

    def compute_output_shape_single_res(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
                return tensor_shape.TensorShape([input_shape[0]] + new_space +
                                                [self.filters])
        else:
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
        return tensor_shape.TensorShape([input_shape[0], self.filters] +
                                                new_space)

class OctConv3D(OctConv):
    """docstring for OctConv3D"""
    def __init__(self,
                 filters,
                 kernel_size,
                 alpha=0.5,
                 strides=1,
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs
                ):
        super(OctConv3D, self).__init__(
            rank=3,
            filters=filters,
            kernel_size=kernel_size,
            alpha=alpha,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)


class OctConv2D(OctConv):
    """docstring for OctConv2D"""
    def __init__(self,
                 filters,
                 kernel_size,
                 alpha=0.5,
                 strides=1,
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs
                ):
        super(OctConv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            alpha=alpha,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)

class OctConv1D(OctConv):
    """docstring for OctConv2D"""
    def __init__(self,
                 filters,
                 kernel_size,
                 alpha=0.5,
                 strides=1,
                 padding='valid',
                 data_format='channels_last',
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs
                ):
        super(OctConv1D, self).__init__(
            rank=1,
            filters=filters,
            kernel_size=kernel_size,
            alpha=alpha,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)

    def call(self, inputs):
        if self.padding == 'causal':
            inputs = tf.pad(inputs, self._compute_causal_padding())
        return super(OctConv1D, self).call(inputs)

