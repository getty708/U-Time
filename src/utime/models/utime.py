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


def get_padding_size_stride_is_1(x, k, d):
    r""" Returns padding size which can make W_in == W_out when the stride size is 1.

    .. math::
        p = \lfloor \frac{1}{2} \{ (x-1)s - x + k + (k-1)(d-1) \} \rfloor

    Args:
        x (int): length of input/output sequence
        k (int): kernel size
        d (int): dilation size
    """
    p = int((k - 1) * d / 2)
    return p


class SingleConvBlock(nn.Module):
    """

    This block is consist from (Conv-BN) * 1.

    """

    def __init__(
        self,
        in_ch,
        out_ch,
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

        self.conv1 = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            # padding=(0, pad),
            padding=(0, 0),
            dilation=(1, dilation),
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.afunc = afunc

        self.pad = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        if self.pad is None:
            p = get_padding_size(
                x.size(-1), self.kernel_size, self.stride, self.dilation)
            self.pad = (p, p, 0, 0)

        x = F.pad(x, self.pad, 'constant', 0)
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

        if mid_ch == None:
            mid_ch = out_ch

        self.conv1 = nn.Conv2d(
            in_ch,
            mid_ch,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            # padding=(0, pad),
            padding=(0, 0),
            dilation=(1, dilation),
        )
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(
            mid_ch,
            out_ch,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            # padding=(0, pad),
            padding=(0, 0),
            dilation=(1, dilation),
        )
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.afunc = afunc

        self.pad = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        if self.pad is None:
            p = get_padding_size(
                x.size(-1), self.kernel_size, self.stride, self.dilation)
            self.pad = (p, p, 0, 0)

        x = F.pad(x, self.pad, 'constant', 0)
        x = self.afunc(self.bn1(self.conv1(x)))

        x = F.pad(x, self.pad, 'constant', 0)
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

    def __init__(self, in_ch, out_ch, pool_size):
        """
        Args:
            in_ch/out_ch (int):
                input/output channels.
            pool_size (int):
                kernel size of a pooling operation. When pool_size <= 0, pooling opperation
                will be skipped.
        """
        super().__init__()
        self.double_conv = DoubleConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=(1, pool_size))

    def forward(self, x):
        x = self.double_conv(x)
        x = self.pool(x)
        return x


class UTimeEncoder(nn.ModuleList):
    """
    Attributes:
        input_dims (int):
            number of input channels.
        out_ch_1st (int):
            number of convolutional filters of 1st convolutional filters
        depth (int):
            Number of conv blocks in encoding layer (number of 2x2 max pools)
            Note: each block doubles the filter count while halving the spatial
            dimensions of the features.

    """

    def __init__(
        self,
        in_ch,
        depth=4,
        pools=None,
    ):
        super().__init__()
        self.depth = depth
        self.in_ch = in_ch
        self.pools = pools
        filters = []  # list of output channels.

        # -- main blocks --
        l = []
        for i in range(depth):
            l.append(
                DownBlock(in_ch, in_ch * 2, pools[i]),
            )
            filters.append(in_ch * 2)
            in_ch = int(in_ch * 2)
        self.conv_blocks = nn.ModuleList(l)

        # -- bottom --
        self.bottom = DoubleConvBlock(in_ch, in_ch, dilation=1)
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

    This module returns (N, in_ch, H, W) when the input has shape of (N, in_ch, H, 2*W).

    Attributes:
        None: -

    Note:
        ``padding`` is allways set to 'same'.


    """

    def __init__(
        self,
        in_ch,
        out_ch,
        mode='bilinear',
    ):
        """
        Args:
            in_ch (int):
                number of input channels of ``x1`` (main stream).
            out_ch (int):
                input/output channels.
            mode (str):
                the upsampling algorithm: ``bilinear``(``nn.Upsample``), or 
                ``deconv``(``nn.ConvTransposed2d``, working). (Default: bilinear)

        """
        super().__init__()

        # -- Upsamplomg Layer --
        if mode == 'bilinear':
            # FIXME: Why this convoltion is needed right after up()?
            # In the implementation by milesial, the conv is not used.
            # https://github.com/milesial/Pytorch-UNet/blob/67bf11b4db4c5f2891bd7e8e7f58bcde8ee2d2db/unet/unet_parts.py#L63
            self.up = nn.Upsample(
                scale_factor=(1, 2),
                mode='bilinear',
                align_corners=True)
            self.conv0 = SingleConvBlock(
                in_ch, in_ch // 2, dilation=1)
        else:
            raise ValueError()

        # --  Double Conv Layer --
        self.double_conv = DoubleConvBlock(in_ch, out_ch)

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


class UTimeDecoder(nn.ModuleList):
    """
    Attributes:
        depth (int):
            Number of conv blocks in encoding layer (number of 2x2 max pools)
            Note: each block doubles the filter count while halving the spatial
            dimensions of the features.
        filters ([int]):
            list of the number of output channels.

    """

    def __init__(
        self,
        in_ch,
        depth=4,
        pools=None,
    ):
        """
        Args:
            in_ch/out_ch (int):
                input/output channels
            depth (int): -
            pools ([list]):
                list of kernel size which is used in pooling layers of the encoder.

        """
        super().__init__()
        self.depth = depth
        self.in_ch = in_ch
        filters = []  # list of output channels.
        assert self.depth == len(pools)

        # -- main blocks --
        l = []
        for i, pool in enumerate(reversed(pools)):
            l.append(
                UpBlock(in_ch, in_ch // 2),
            )
            filters.append(in_ch // 2)
            in_ch //= 2
        self.up_blocks = nn.ModuleList(l)

        # -- Store meta data --
        self.filters = tuple(filters)

    def forward(self, x1, x2_list):
        """
        Args:
            x1 (Tensor): input
            x2_list ([Tensor]): tensors for skipped connections.
        """
        # -- main block --
        for i in range(self.depth):
            print(f"[{i}] x1={x1.size()}, x2={x2_list[i].size()}")
            x1 = self.up_blocks[i](x1, x2_list[i])
        return x1


class DenseClassifier(nn.Module):
    """ Implementation of a dense classifier proposed in UTime.
    """

    def __init__(self, in_ch, n_classes):
        """
        Args:
            in_ch (int): the number of classes.
        """
        super().__init__()
        self.in_ch = in_ch
        self.n_classes = n_classes

        self.conv = SingleConvBlock(
            in_ch, n_classes, afunc=torch.tanh)

    def forward(self, x):
        x = self.conv(x)
        return x


class SegmentClassifier(nn.Module):
    """ Implementation of a segment clasifier proposed in UTime.
    """

    def __init__(self, in_ch, data_per_period):
        """
        Args:
            in_ch (int): the number of classes.
        """
        super().__init__()
        self.in_ch = in_ch
        self.data_per_period = data_per_period

        # average pooling and point-wise convolution
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, data_per_period))
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.avg_pool(x)
        out = F.softmax(self.conv(x), dim=1)
        return out


class UTime(nn.Module):
    """
    Input must take channel-first format (BCHW).
    This model use 2D convolutional filter with kernel size = (1 x f).

    OBS: Uses 2D operations internally with a 'dummy' axis, so that a batch
         of shape [bs, d, c] is processed as [bs, d, 1, c]. These operations
         are (on our systems, at least) currently significantly faster than
         their 1D counterparts in tf.keras.

    See also original U-net paper at http://arxiv.org/abs/1505.04597

    """

    def __init__(
        self,
        n_classes,
        batch_shape,
        in_ch_enc=16,
        depth=4,
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
        self.in_ch_enc = in_ch_enc
        self.depth = depth
        self.n_periods = batch_shape[3]
        self.n_classes = int(n_classes)
        self.pools = pools
        if len(self.pools) != self.depth:
            raise ValueError(
                "Argument 'pools' must be a single integer or a "
                "list of values of length equal to 'depth'."
            )

        # -- Model --
        # NOTE: Add input encoding layer (UNet)
        # Ref: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        self.inc = DoubleConvBlock(self.input_dims, in_ch_enc)

        self.encoder = UTimeEncoder(in_ch_enc, depth=depth, pools=pools)

        in_ch_dec = self.encoder.filters[-1]  # FIXME:
        self.decoder = UTimeDecoder(in_ch_dec, depth=depth, pools=pools)

        in_ch_dc = self.decoder.filters[-1]  # FIXME:
        self.dense_clf = DenseClassifier(
            in_ch_dc, self.n_classes)

        in_ch_sc = self.n_classes
        data_per_period = data_per_prediction
        self.segment_clf = SegmentClassifier(in_ch_sc, data_per_period)

        # raise NotImplementedError()

        # FIXME: activation func?
        # self.dense_classifier_activation = dense_classifier_activation
        # self.data_per_prediction = data_per_prediction or self.input_dims
        # if not isinstance(self.data_per_prediction, (int, np.integer)):
        #     raise TypeError("data_per_prediction must be an integer value")
        # if self.input_dims % self.data_per_prediction:
        #     raise ValueError(
        #         "'input_dims' ({}) must be evenly divisible by "
        #         "'data_per_prediction' ({})".format(
        #             self.input_dims, self.data_per_prediction
        #         )
        #     )

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

    def log(self):
        pass
        # self.logger(
        #     "{} Model Summary\n" "-------------------".format(__class__.__name__)
        # )
        # self.logger("N periods:         {}".format(self.n_periods))
        # self.logger("Input dims:        {}".format(self.input_dims))
        # self.logger("N channels:        {}".format(self.n_channels))
        # self.logger("N classes:         {}".format(self.n_classes))
        # self.logger("Kernel size:       {}".format(self.kernel_size))
        # self.logger("Dilation rate:     {}".format(self.dilation))
        # self.logger("CF factor:         %.3f" % self.cf)
        # self.logger("Init filters:      {}".format(self.init_filters))
        # self.logger("Depth:             %i" % self.depth)
        # self.logger("Poolings:          {}".format(self.pools))
        # self.logger("Transition window  {}".format(self.transition_window))
        # self.logger("Dense activation   {}".format(self.dense_classifier_activation))
        # self.logger("l2 reg:            %s" % self.l2_reg)
        # self.logger("Padding:           %s" % self.padding)
        # self.logger("Conv activation:   %s" % self.activation)
        # self.logger("Receptive field:   %s" % self.receptive_field[0])
        # self.logger("Seq length.:       {}".format(self.n_periods * self.input_dims))
        # self.logger("N params:          %i" % self.count_params())
        # self.logger("Input:             %s" % self.input)
        # self.logger("Output:            %s" % self.output)
