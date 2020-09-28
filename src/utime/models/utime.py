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


def get_padding_size(w, k, s, d):
    r""" Returns padding size to make W_in == W_out for a convolutional layer.
    
    It returns the padding size to implement padding="name" in tensorflow.
    
    .. math::
        p = \lfloor \frac{1}{2} \{ (x-1)s - x + k + (k-1)(d-1) \} \rfloor

    Args:
        w (int): length of input/output sequence
        k (int): kernel size
        s (int): stride size
        d (int): dilation size
    
    Examples:
        >>> input = torch.arange(1, 5, dtype=torch.float32).view(1, 32, 1, 100)
        >>> k, s, d = 3, 1, 1
        >>> conv = nn.Conv2d(32, 32, kernel_size=(1, k))
        >>> pad = get_padding_size(input.size(3), k, s, d)
        >>> 
        >>> input = F.pad(input, (pad, pad, 0, 0,))
        >>> input.size()
        torch.Size([1, 32, 1, 101])
        >>> output = conv(input)
        >>> output.size()
        torch.Size([1, 32, 1, 100])
    
    """
    p = int(((w-1)*s - w + k + (k-1)*(d-1)) / 2)
    return p


def get_padding_size_stride_is_1(k, d):
    r""" Returns padding size to make W_in == W_out when the stride size is 1.

    This is a special case of ``get_padding_size()``.

    .. math::
        p = \lfloor \frac{1}{2} \{ (k-1) \times d \} \rfloor

    Args:
        k (int): kernel size
        d (int): dilation size
    """
    p = int((k - 1) * d / 2)
    return p


class SingleConvBlock(nn.Module):
    """ This is a thin wrapper to compute (Conv-BN).
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
            mid_ch (int or None, optional):
                the number of channels for an ``out_ch`` of 1st conv and ``in_ch`` of
                2nd conv. If mid_ch is None, mid_ch = out_ch.
            afunc (function, optional):
                Activation function for convolution layers (Default: F.elu)
            dilation (int, optional):
                (Default: 2)
            kernel_size (int, optional):
                Kernel size for convolution layers. (Default: 5)
            padding (str, optional):
                padding algorithm: `same` or `valid`.
            stride (int, optional):
                (Default: 1)

        """
        super().__init__()

        if mid_ch == None:
            mid_ch = out_ch

        self.conv1 = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, 0),
            dilation=(1, dilation),
        )
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.afunc = afunc

        self.pad = None if padding == 'same' else 0
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
    """ This is a thin wrapper to compute (Conv-BN-Conv-BN).
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
            mid_ch (int or None, optional):
                the number of channels for an ``out_ch`` of 1st conv and ``in_ch`` of
                2nd conv. If mid_ch is None, mid_ch = out_ch.
            afunc (function, optional):
                activation function for convolution layers (Default: F.elu)
            dilation (int, optional):
                (Default: 2)
            kernel_size (int, optional):
                kernel size for convolution layers. (Default: 5)
            padding (str, optional):
                padding algorithm: `same` or `valid`.
            stride (int, optional):
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
            padding=(0, 0),
            dilation=(1, dilation),
        )
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(
            mid_ch,
            out_ch,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, 0),
            dilation=(1, dilation),
        )
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.afunc = afunc

        self.pad = None if padding == 'same' else 0
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
    """ This module provide a single donwnsampling operation for U-Time's decoder.

    Attributes:
        double_conv (nn.Module): -
        pool (nn.MaxPool2d): -

    Note:
        ``padding`` is allways set to 'same'.

    """

    def __init__(self, in_ch, out_ch, pool_size):
        """
        Args:
            in_ch/out_ch (int):
                input/output channels.
            pool_size (int):
                kernel size of a pooling.
        """
        super().__init__()
        self.double_conv = DoubleConvBlock(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=(1, pool_size))

    def forward(self, x):
        """ Returns 2 tensors (main stream and skip connection).
        """
        x_skip = self.double_conv(x)
        x = self.pool(x_skip)
        return x, x_skip

class UpBlock(nn.Module):
    """ This module provide a single upsampling operation for U-Time's encoder.
    
    This module doubles returns (N, in_ch, H, W) when the input has shape of (N, in_ch, H, 2*W).

    Attributes:
        up (nn.Upsampling or nn.ConvTransposed2d): -
        double_conv (DoubleConvBlock): -

    Examples:
        >>> x      = torch.arange(1, 5, dtype=torch.float32).view(1, 64, 1, 50)
        >>> x_skip = torch.arange(1, 5, dtype=torch.float32).view(1, 64, 1, 100)
        >>> pool_size = 4
        >>> up_block = UpBlock(64, 32, pool_size)
        >>>
        >>> y = up_block(x, x_skip)
        >>> y.size()
        torch.Size([1, 32, 1, 100])
    
    Note:
        ``padding`` is allways set to 'same'.
    
    """

    def __init__(self, in_ch, out_ch, pool_size, mode='bilinear',):
        """
        Args:
            in_ch (int):
                the number of input channels of ``x1`` (main stream) ``x2`` (skip connection).
                ``x1`` and ``x2`` should have the same number of channels.
            out_ch (int):
                input/output channels.
            pool_size (int):
                kernel_size for corresponding pooling operation.
            mode (str, optional):
                the upsampling algorithm: ``bilinear``(``nn.Upsample``), or 
                ``deconv``(``nn.ConvTransposed2d``, not implemented). (Default: bilinear)

        """
        super().__init__()

        # -- Upsamplomg Layer --
        if mode == 'bilinear':
            # FIXME: In the U-Time implementation, ``Conv2d``  is appied right after
            #        ``Upsampling``. Why this is needed.
            self.up = nn.Upsample(
                scale_factor=(1, pool_size),
                mode='bilinear',
                align_corners=True)
        else:
            raise ValueError()

        # --  Double Conv Layer --
        self.double_conv = DoubleConvBlock(in_ch * 2, out_ch)
        
    def forward(self, x1, x2):
        """ 
        Args:
            x1 (Tensor): a tensor from main stream. shape = (N, C, H, W)
            x2 (Tensor): a tensor from downsampling layer.
        """
        # -- upsampling --
        x1 = self.up(x1)
        
        # -- Concat --        
        diff_h = x2.size()[2] - x1.size()[2]
        diff_w = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                        diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x1, x2], dim=1)
        
        # -- conv --
        x = self.double_conv(x)
        return x


class UTimeEncoder(nn.ModuleList):
    """
    Attributes:
        depth (int):
            the number of ``DownBlock``.
        pools ([int]):
            list of kernel sizes for pooling.
        conv_blocks (nn.ModuleList): list of ``DownBlock``.        

    Todo:
        implement ``get_output_ch(block_index)`` and remove ``filters``.
    
    """

    def __init__(self, in_ch, pools):
        super().__init__()
        self.in_ch = in_ch
        self.depth = len(pools)
        self.pools = pools
        filters = []  # list of output channels.

        # -- main blocks --
        l = []
        for i in range(self.depth):
            l.append(DownBlock(in_ch, in_ch * 2, pools[i]))
            filters.append(in_ch * 2)
            in_ch = int(in_ch * 2)
        self.conv_blocks = nn.ModuleList(l)

        # -- bottom --
        self.bottom = DoubleConvBlock(in_ch, in_ch, dilation=1)
        filters.append(in_ch)

        # -- Store meta data --
        self.filters = tuple(filters)
        
    def forward(self, x):
        # -- donwnsampling blocks --
        skip_connections = []
        for i in range(self.depth):
            x, x_skip = self.conv_blocks[i](x)
            skip_connections.append(x_skip)
        
        # -- bottom --
        encoded = self.bottom(x)

        return encoded, skip_connections


class UTimeDecoder(nn.ModuleList):
    """
    Attributes:
        depth (int):
            the number of ``DownBlock``.
        pools ([int]):
            list of kernel sizes for pooling.
        conv_blocks (nn.ModuleList):
            list of ``DownBlock``.
        filters ([int]):
            list of the number of output channels.

    Todo:
        implement ``get_output_ch(block_index)`` and remove ``filters``.
    
    """

    def __init__(self, in_ch, pools):
        super().__init__()
        self.in_ch = in_ch
        self.depth = len(pools)
        self.pools = pools
        filters = []  # list of output channels.

        # -- main blocks --
        l = []
        for i, pool_size in enumerate(reversed(pools)):
            l.append(UpBlock(in_ch, in_ch // 2, pool_size))
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
        for i in range(self.depth):
            i_inv = (self.depth - 1) - i
            x1 = self.up_blocks[i](x1, x2_list[i_inv])
        return x1


class DenseClassifier(nn.Module):
    """ Implementation of a dense classifier proposed in UTime.

    Attributes:
        in_ch (int): -
        n_classes (int): -
        conv (SingleConvBlock): -
    """

    def __init__(self, in_ch, n_classes):
        """
        Args:
            in_ch (int): -
            n_classes (int): -
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

    Attributes:
        in_ch (int): -
        data_per_period (int): -
        avg_pool (nn.AvgPool2d): -
        conv (nn.Conv2d): -
    """

    def __init__(self, in_ch, data_per_period):
        """
        Args:
            in_ch (int): the number of input channels (= the number of classes).
        """
        super().__init__()
        self.in_ch = in_ch
        self.data_per_period = data_per_period
        
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
        input_shape,
        in_ch_enc=32,
        pools=(10, 8, 6, 4),
        data_per_period=None,
        logger=None,
        **kwargs,
    ):
        """

        Args:
            n_classes (int):
                The number of classes to model, gives the number of filters in the
                final 1x1 conv layer.
            input_shape (list):
                Giving the shape of a single sample. Input data should have CHW format.
            in_ch_enc (int, optional):
                the number of input channels for UTimeEncoder. (Default: 32)
            pools ([int]):
               list of kernel sizes for pooling operations.
            data_per_prediction (int):
               TODO
            logger (logging.Logger | ScreenLogger):
               UTime.Logger object, logging to files or screen.
        """
        super(UTime, self).__init__()

        # -- parameter validation --
        if not isinstance(pools, tuple):
            raise ValueError((
                "Argument 'pools' must be a tuple of values of length equal to 'depth',"
                + f"but got {pools}."
            ))

        # Set logger or standard print wrapper
        self.logger = logger or ScreenLogger()

        # Set various attributes
        assert len(input_shape) == 3
        self.input_shape = input_shape # format=CHW
        self.in_ch_enc = in_ch_enc
        self.n_classes = int(n_classes)
        self.pools = pools
        self.depth = len(self.pools)
        

        # -- MODEL --
        # NOTE: Add input encoding layer (UNet)
        # Ref: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
        self.inc = DoubleConvBlock(self.input_shape[0], in_ch_enc)

        self.encoder = UTimeEncoder(in_ch_enc,  pools=pools)

        in_ch_dec = self.encoder.filters[-1]  # FIXME:
        self.decoder = UTimeDecoder(in_ch_dec, pools=pools)

        self.dense_clf = DenseClassifier(in_ch_enc, self.n_classes)

        self.segment_clf = SegmentClassifier(self.n_classes, data_per_period)
        

    def forward(self, x):
        x = self.inc(x)
        (x, res) = self.encoder(x)
        x = self.decoder(x, res)
        x = self.dense_clf(x) 
        x = self.segment_clf(x)
        return x
    
    def log(self):
        pass
