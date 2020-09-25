"""
Implementation of UTime as described in:

Mathias Perslev, Michael Hejselbak Jensen, Sune Darkner, Poul Jørgen Jennum
and Christian Igel. U-Time: A Fully Convolutional Network for Time Series
Segmentation Applied to Sleep Staging. Advances in Neural Information
Processing Systems (NeurIPS 2019)
"""
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

from ..logging import ScreenLogger
from ..utils.conv_arithmetics import compute_receptive_fields


def get_padding_size(x, k, s, d):
    r""" Returns padding size which can make W_in == W_out.

    .. math::
        p = \lfloor \frac{1}{2} \{ (x-1)s - x + k + (k-1)(d-1) \} \rfloor
    
    Args:
        x (int): length of input/output sequence
        k (int): kernel size
        s (int): stride size
        d (int): dilation size
    """
    p = int(((x-1)*s - x + k + (k-1)*(d-1)) / 2)
    return p


class SingleConvBlock(nn.Module):
    """

    This block is consist from (Conv-BN) * 1.
    
    Todo:
         Move to the top of this file.
    """
    def __init__(
        self,
        in_ch,
        out_ch,
        n_periods,
        # conv layers
        afunc=F.elu,
        dilation=2,
        kernel_size=5,
        padding='same',
        stride=1,
    ):
        """
        Args:
            in_ch/out_ch (int):
                input/output channels
            n_periods (int):
                length of input sequence
            afunc (function):
                Activation function for convolution layers (Default: F.elu)
            dilation (int):
                (Default: 2)
            kernel_size (int):
                Kernel size for convolution layers. (Default: 5)
            padding (str):
                padding algorithm: `same` or `valid`.
            stride (int):
                (Default: 1)
        
        """
        super().__init__()
        if padding == 'same':
            pad = get_padding_size(
                n_periods, kernel_size, stride, dilation)
        else:
            pad = 0
        
        self.conv1 = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, pad),
            dilation=(1, dilation),
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.afunc = afunc
        
    def forward(self, x):
        x = self.afunc(self.bn1(self.conv1(x)))
        return x

class DoubleConvBlock(nn.Module):
    """

    This block is consist from (Conv-BN) * 2.
    
    Todo:
         Move to the top of this file.
    """
    def __init__(
        self,
        in_ch,
        out_ch,
        n_periods,
        mid_ch=None,
        afunc=F.elu,
        dilation=2,
        kernel_size=5,
        padding='same',
        stride=1,
    ):
        """
        Args:
            in_ch/out_ch (int):
                input/output channels
            n_periods (int):
                length of input sequence
            mid_ch (int or None):
                channels for out_ch of 1st conv and in_ch of 2nd conv.
                If mid_ch is None, mid_ch = out_ch.
            afunc (function):
                activation function for convolution layers (Default: F.elu)
            dilation (int):
                (Default: 2)
            kernel_size (int):
                kernel size for convolution layers. (Default: 5)
            padding (str):
                padding algorithm: `same` or `valid`.
            stride (int):
                (Default: 1)
        
        """
        super().__init__()
        if padding == 'same':
            pad = get_padding_size(
                n_periods, kernel_size, stride, dilation)
        else:
            pad = 0

        if mid_ch == None:
            mid_ch = out_ch
            
        self.conv1 = nn.Conv2d(
            in_ch,
            mid_ch,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, pad),
            dilation=(1, dilation),
        )
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(
            mid_ch,
            out_ch,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, pad),
            dilation=(1, dilation),
        )
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.afunc = afunc
        
    def forward(self, x):
        x = self.afunc(self.bn1(self.conv1(x)))
        x = self.afunc(self.bn2(self.conv2(x)))
        return x


class DownBlock(nn.Module):
    """This is a single block for U-Time encoder.
    The block consists from Conv2D - BN - Conv2d - BN - MaxPool.

    Attributes:
        None: -

    Note:
        ``padding`` is allways set to 'same'.

    """
    def __init__(self, in_ch, out_ch, n_periods, pool_size):
        """
        Args:
            in_ch/out_ch (int):
                input/output channels.
            n_periods (int):
                length of input sequence
            pool_size (int):
                kernel size of a pooling operation. When pool_size <= 0, pooling opperation
                will be skipped.
        """
        super().__init__()
        self.double_conv = DoubleConvBlock(in_ch, out_ch, n_periods)
        self.pool = nn.MaxPool2d(kernel_size=(1, pool_size))

        
    def forward(self, x):
        x = self.double_conv(x)
        x = self.pool(x)
        return x

    
class UTimeEncoder(nn.ModuleList):
    """
    Attributes:
        depth (int):
            Number of conv blocks in encoding layer (number of 2x2 max pools)
            Note: each block doubles the filter count while halving the spatial
            dimensions of the features.
        dilation (int):
            TODO
        activation (string):
            Activation function for convolution layers
        kernel_size (int):
            Kernel size for convolution layers
        pools ([int]):
            list of pooling size. (Default: None, required)

    """

    def __init__(
        self,
        in_ch,
        n_periods,
        depth=4,
        pools=None,
    ):
        super().__init__()
        self.depth = depth
        self.in_ch = in_ch
        filters = [] # list of output channels.
        
        # -- main blocks --
        l = []
        n_periods = n_periods
        for i in range(depth):            
            l.append(
                DownBlock(in_ch, in_ch * 2, n_periods, pools[i]),
            )
            filters.append(in_ch)
            in_ch = int(in_ch * 2)
            n_periods //= pools[i]
        self.conv_blocks = nn.ModuleList(l)
        
        # -- bottom --
        self.bottom = DoubleConvBlock(in_ch, in_ch, n_periods, dilation=1)
        filters.append(in_ch)
        
        # -- Store meta data --
        self.filters = tuple(filters)
        
        
    def forward(self, x):
        x_ = x

        # -- main block --
        residual_connections = []
        for i in range(self.depth):
            x_ = self.conv_blocks[i](x_)
            residual_connections.append(x_)
        # -- bottom --
        encoded = self.bottom(x_)

        return encoded, residual_connections

 
class UpBlock(nn.Module):
    """This is an implementation of a single decoder block for U-Time.
    The block consists from ConvTransposed2d - (Conv2d - BN) - Merge - (Conv2d - BN)
    - (Conv2d - BN).
    
    Attributes:
        None: -
    
    Note:
        ``padding`` is allways set to 'same'.

    """
    def __init__(
        self,
        in_ch,
        out_ch,
        n_periods,
        mode='bilinear',
    ):
        """
        Args:
            in_ch/out_ch (int):
                input/output channels.
            n_periods (int):
                length of input sequence
            mode (str):
                the upsampling algorithm: ``bilinear``(``nn.Upsample``), or 
                ``deconv``(``nn.ConvTransposed2d``, working). (Default: bilinear)
        
        """
        super().__init__()
        self.afunc = afunc
        
        # -- Upsamplomg Layer --
        if mode == 'bilinear':
            # FIXME: Why this convoltion is needed right after up()?
            # In the implementation by milesial, the conv is not used.
            # https://github.com/milesial/Pytorch-UNet/blob/67bf11b4db4c5f2891bd7e8e7f58bcde8ee2d2db/unet/unet_parts.py#L63        
            self.up = nn.Upsample(
                scale_factor=(1, 2),
                mode='bilinear',
                align_corners=True)
            self.conv0 = SingleConvBlock(in_ch, in_ch, n_periods, dilation=1)
        else:
            raise ValueError()
        
        # --  Double Conv Layer --
        self.double_conv = DoubleConvBlock(in_ch * 2, out_ch, n_periods)
        
                
    def forward(self, x1, x2):
        """ 
        Args:
            x1 (Tensor): a tensor from main stream. shape = (N, C, H, W)
            x2 (Tensor): a tensor from downsampling layer.
        """
        # -- upsampling --
        x1 = self.up(x1)
        x1 = self.conv0(x1)
        
        # -- Concat --
        diff_h = x2.size()[2] - x1.size()[2]
        diff_w = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_h // 2, diff_h - diff_h // 2,
                        diff_w // 2, diff_w - diff_w // 2])
        x = torch.cat([x1, x2], dim=1)
        
        # -- conv --
        x = self.double_conv(x)
        return x
    



    
class UTime(nn.Module):
    """
    Input must take channel-first format (BCHW).
    This model use 2D convolutional filter with kernel size = (1 x f).

    OBS: Uses 2D operations internally with a 'dummy' axis, so that a batch
         of shape [bs, d, c] is processed as [bs, d, 1, c]. These operations
         are (on our systems, at least) currently significantly faster than
         their 1D counterparts in tf.keras.

    See also original U-net paper at http://arxiv.org/abs/1505.04597

    Attributes:
       n_periods (int): length of sequence (?)


    """

    def __init__(
        self,
        n_classes,
        batch_shape,
        depth=4,
        dilation=2,
        activation="elu",
        dense_classifier_activation="tanh",
        kernel_size=5,
        transition_window=1,
        padding="same",
        complexity_factor=2,
        l2_reg=None,
        pools=(10, 8, 6, 4),
        data_per_prediction=None,
        logger=None,
        build=True,
        **kwargs,
    ):
        """

        Args:
            n_classes (int):
                The number of classes to model, gives the number of filters in the
                final 1x1 conv layer.
            batch_shape (list): Giving the shape of one one batch of data,
                potentially omitting the zero-th axis (the batchsize dim)
            depth (int):
                Number of conv blocks in encoding layer (number of 2x2 max pools)
                Note: each block doubles the filter count while halving the spatial
                dimensions of the features.
            dilation (int):
                TODO
            activation (string):
                Activation function for convolution layers
            dense_classifier_activation (string):
                TODO
            kernel_size (int):
                Kernel size for convolution layers
            transition_window (int):
                TODO
            padding (string):
                Padding type ('same' or 'valid')
            complexity_factor (int/float):
                Use int(N * sqrt(complexity_factor)) number of filters in each
                convolution layer instead of default N.
            l2_reg (float in [0, 1])
               L2 regularization on conv weights
            pools (int or list of ints):
               TODO
            data_per_prediction (int):
               TODO
            logger (logging.Logger | ScreenLogger):
               UTime.Logger object, logging to files or screen.
            build (bool):
               TODO
        """
        super(UTime, self).__init__()

        # Set logger or standard print wrapper
        self.logger = logger or ScreenLogger()

        # Set various attributes
        assert len(batch_shape) == 4
        self.n_channels = batch_shape[1]
        self.input_dims = batch_shape[2]
        self.n_periods = batch_shape[3]
        self.n_classes = int(n_classes)
        self.dilation = int(dilation)
        self.cf = complexity_factor
        self.init_filters = int(8 * self.cf)
        self.kernel_size = int(kernel_size)
        self.transition_window = transition_window
        self.activation = activation
        self.l2_reg = l2_reg
        self.depth = depth
        self.n_crops = 0
        self.pools = (
            [pools] * self.depth if not isinstance(pools, (list, tuple)) else pools
        )
        if len(self.pools) != self.depth:
            raise ValueError(
                "Argument 'pools' must be a single integer or a "
                "list of values of length equal to 'depth'."
            )
        self.padding = padding.lower()
        if self.padding != "same":
            raise ValueError("Currently, must use 'same' padding.")

        # FIXME: activation func?
        self.dense_classifier_activation = dense_classifier_activation
        self.data_per_prediction = data_per_prediction or self.input_dims
        if not isinstance(self.data_per_prediction, (int, np.integer)):
            raise TypeError("data_per_prediction must be an integer value")
        if self.input_dims % self.data_per_prediction:
            raise ValueError(
                "'input_dims' ({}) must be evenly divisible by "
                "'data_per_prediction' ({})".format(
                    self.input_dims, self.data_per_prediction
                )
            )

        # FIXME: remove?
        # if build:
        #     # Compute receptive field
        #     ind = [x.__class__.__name__ for x in self.layers].index("UpSampling2D")
        #     self.receptive_field = compute_receptive_fields(self.layers[:ind])[-1][-1]

        #     # Log the model definition
        #     self.log()
        # else:
        #     self.receptive_field = [None]
        self.receptive_field = [None]

    def create_encoder(
        in_,
        depth,
        pools,
        filters,
        kernel_size,
        activation,
        dilation,
        padding,
        kernel_reg=None,
        name="encoder",
        name_prefix="",
    ):
        """Returns a module list of an encoder.

        Define weight matrics for encoder.

        Note:
            This method should be called in __init__().
        """
        name = "{}{}".format(name_prefix, name)
        residual_connections = []
        for i in range(depth):
            l_name = name + "_L%i" % i
            # conv = Conv2D(
            #     filters,
            #     (kernel_size, 1),
            #     activation=activation,
            #     padding=padding,
            #     kernel_regularizer=kernel_reg,
            #     dilation_rate=dilation,
            #     name=l_name + "_conv1",
            # )(in_)
            conv = nn.Conv2d(
                filters,
                (1, kernel_size),
                padding=padding,
                dilation=dilation,
            )
            bn = BatchNormalization(name=l_name + "_BN1")(conv)
            conv = Conv2D(
                filters,
                (kernel_size, 1),
                activation=activation,
                padding=padding,
                kernel_regularizer=kernel_reg,
                dilation_rate=dilation,
                name=l_name + "_conv2",
            )(bn)
            bn = BatchNormalization(name=l_name + "_BN2")(conv)
            in_ = MaxPooling2D(pool_size=(pools[i], 1), name=l_name + "_pool")(bn)

            # add bn layer to list for residual conn.
            residual_connections.append(bn)
            filters = int(filters * 2)

        # Bottom
        name = "{}bottom".format(name_prefix)
        conv = Conv2D(
            filters,
            (kernel_size, 1),
            activation=activation,
            padding=padding,
            kernel_regularizer=kernel_reg,
            dilation_rate=1,
            name=name + "_conv1",
        )(in_)
        bn = BatchNormalization(name=name + "_BN1")(conv)
        conv = Conv2D(
            filters,
            (kernel_size, 1),
            activation=activation,
            padding=padding,
            kernel_regularizer=kernel_reg,
            dilation_rate=1,
            name=name + "_conv2",
        )(bn)
        encoded = BatchNormalization(name=name + "_BN2")(conv)

        return encoded, residual_connections, filters

    def create_upsample(
        self,
        in_,
        res_conns,
        depth,
        pools,
        filters,
        kernel_size,
        activation,
        dilation,  # NOT USED
        padding,
        kernel_reg=None,
        name="upsample",
        name_prefix="",
    ):
        name = "{}{}".format(name_prefix, name)
        residual_connections = res_conns[::-1]
        for i in range(depth):
            filters = int(filters / 2)
            l_name = name + "_L%i" % i

            # Up-sampling block
            fs = pools[::-1][i]
            up = UpSampling2D(size=(fs, 1), name=l_name + "_up")(in_)
            conv = Conv2D(
                filters,
                (fs, 1),
                activation=activation,
                padding=padding,
                kernel_regularizer=kernel_reg,
                name=l_name + "_conv1",
            )(up)
            bn = BatchNormalization(name=l_name + "_BN1")(conv)

            # Crop and concatenate
            cropped_res = self.crop_nodes_to_match(residual_connections[i], bn)
            # cropped_res = residual_connections[i]
            merge = Concatenate(axis=-1, name=l_name + "_concat")([cropped_res, bn])
            conv = Conv2D(
                filters,
                (kernel_size, 1),
                activation=activation,
                padding=padding,
                kernel_regularizer=kernel_reg,
                name=l_name + "_conv2",
            )(merge)
            bn = BatchNormalization(name=l_name + "_BN2")(conv)
            conv = Conv2D(
                filters,
                (kernel_size, 1),
                activation=activation,
                padding=padding,
                kernel_regularizer=kernel_reg,
                name=l_name + "_conv3",
            )(bn)
            in_ = BatchNormalization(name=l_name + "_BN3")(conv)
        return in_

    def create_dense_modeling(
        self, in_, in_reshaped, filters, dense_classifier_activation, name_prefix=""
    ):
        cls = Conv2D(
            filters=filters,
            kernel_size=(1, 1),
            activation=dense_classifier_activation,
            name="{}dense_classifier_out".format(name_prefix),
        )(in_)
        s = (self.n_periods * self.input_dims) - cls.get_shape().as_list()[1]
        out = self.crop_nodes_to_match(
            node1=ZeroPadding2D(padding=[[s // 2, s // 2 + s % 2], [0, 0]])(cls),
            node2=in_reshaped,
        )
        return out

    @staticmethod
    def create_seq_modeling(
        in_,
        input_dims,
        data_per_period,
        n_periods,
        n_classes,
        transition_window,
        name_prefix="",
    ):
        cls = AveragePooling2D(
            (data_per_period, 1), name="{}average_pool".format(name_prefix)
        )(in_)
        out = Conv2D(
            filters=n_classes,
            kernel_size=(transition_window, 1),
            activation="softmax",
            kernel_regularizer=regularizers.l2(1e-5),
            padding="same",
            name="{}sequence_conv_out".format(name_prefix),
        )(cls)
        s = [-1, n_periods, input_dims // data_per_period, n_classes]
        if s[2] == 1:
            s.pop(2)  # Squeeze the dim
        out = Lambda(
            lambda x: tf.reshape(x, s),
            name="{}sequence_classification_reshaped".format(name_prefix),
        )(out)
        return out

    def init_model(self, inputs=None, create_seg_modeling=True, name_prefix=""):
        """
        Build the UNet model with the specified input image shape.
        """
        if inputs is None:
            inputs = Input(shape=[self.n_periods, self.input_dims, self.n_channels])
        reshaped = [-1, self.n_periods * self.input_dims, 1, self.n_channels]
        in_reshaped = Lambda(lambda x: tf.reshape(x, reshaped))(inputs)

        # Apply regularization if not None or 0
        kr = regularizers.l2(self.l2_reg) if self.l2_reg else None

        settings = {
            "depth": self.depth,
            "pools": self.pools,
            "filters": self.init_filters,
            "kernel_size": self.kernel_size,
            "activation": self.activation,
            "dilation": self.dilation,
            "padding": self.padding,
            "kernel_reg": kr,
            "name_prefix": name_prefix,
        }

        """
        Encoding path
        """
        enc, residual_cons, filters = self.create_encoder(in_=in_reshaped, **settings)

        """
        Decoding path
        """
        settings["filters"] = filters
        up = self.create_upsample(enc, residual_cons, **settings)

        """
        Dense class modeling layers
        """
        cls = self.create_dense_modeling(
            in_=up,
            in_reshaped=in_reshaped,
            filters=self.n_classes,
            dense_classifier_activation=self.dense_classifier_activation,
            name_prefix=name_prefix,
        )

        """
        Sequence modeling
        """
        if create_seg_modeling:
            out = self.create_seq_modeling(
                in_=cls,
                input_dims=self.input_dims,
                data_per_period=self.data_per_prediction,
                n_periods=self.n_periods,
                n_classes=self.n_classes,
                transition_window=self.transition_window,
                name_prefix=name_prefix,
            )
        else:
            out = cls

        return [inputs], [out]

    def crop_nodes_to_match(self, node1, node2):
        """
        If necessary, applies Cropping2D layer to node1 to match shape of node2
        """
        s1 = np.array(node1.get_shape().as_list())[1:-2]
        s2 = np.array(node2.get_shape().as_list())[1:-2]

        if np.any(s1 != s2):
            self.n_crops += 1
            c = (s1 - s2).astype(np.int)
            cr = np.array([c // 2, c // 2]).flatten()
            cr[self.n_crops % 2] += c % 2
            cropped_node1 = Cropping2D([list(cr), [0, 0]])(node1)
        else:
            cropped_node1 = node1
        return cropped_node1

    def log(self):
        self.logger(
            "{} Model Summary\n" "-------------------".format(__class__.__name__)
        )
        self.logger("N periods:         {}".format(self.n_periods))
        self.logger("Input dims:        {}".format(self.input_dims))
        self.logger("N channels:        {}".format(self.n_channels))
        self.logger("N classes:         {}".format(self.n_classes))
        self.logger("Kernel size:       {}".format(self.kernel_size))
        self.logger("Dilation rate:     {}".format(self.dilation))
        self.logger("CF factor:         %.3f" % self.cf)
        self.logger("Init filters:      {}".format(self.init_filters))
        self.logger("Depth:             %i" % self.depth)
        self.logger("Poolings:          {}".format(self.pools))
        self.logger("Transition window  {}".format(self.transition_window))
        self.logger("Dense activation   {}".format(self.dense_classifier_activation))
        self.logger("l2 reg:            %s" % self.l2_reg)
        self.logger("Padding:           %s" % self.padding)
        self.logger("Conv activation:   %s" % self.activation)
        self.logger("Receptive field:   %s" % self.receptive_field[0])
        self.logger("Seq length.:       {}".format(self.n_periods * self.input_dims))
        self.logger("N params:          %i" % self.count_params())
        self.logger("Input:             %s" % self.input)
        self.logger("Output:            %s" % self.output)
