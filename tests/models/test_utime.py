import pprint

import numpy as np
import pytest
import torch

from utime.models.utime import (DoubleConvBlock,
                                EncoderBlock,
                                UpBlock,
                                UTimeEncoder,)


BATCH_SIZE = 10
IN_CH      = 32
OUT_CH     = 32
N_PERIODS  = 100


def test_DoubleConvBlock__01():
    batch_shape = (BATCH_SIZE, IN_CH, 1, N_PERIODS)

    # -- build --
    net = DoubleConvBlock(
        IN_CH,
        OUT_CH,
        N_PERIODS)
    net.to(torch.double)
    pprint.pprint(net)
    
    # -- forward --
    x = np.random.uniform(-1, 1, batch_shape)
    x_tensor = torch.from_numpy(x).to(dtype=torch.double)
    y = net(x_tensor)
    print(f"y: {y.size()}, {y.dtype}")

    assert y.size(0) == BATCH_SIZE
    assert y.size(1) == OUT_CH
    assert y.size(2) == 1
    assert y.size(3) == N_PERIODS



@pytest.mark.parametrize("pool_size,w_out",(
    (-1, N_PERIODS),
    (4,  N_PERIODS//4),
))
def test_EncoderBlock__01(pool_size, w_out):
    batch_shape = (BATCH_SIZE, IN_CH, 1, N_PERIODS)

    # -- build --
    net = EncoderBlock(
        IN_CH,
        OUT_CH,
        N_PERIODS,
        pool_size=pool_size)
    net.to(torch.double)
    pprint.pprint(net)
    
    # -- forward --
    x = np.random.uniform(-1, 1, batch_shape)
    x_tensor = torch.from_numpy(x).to(dtype=torch.double)
    y = net(x_tensor)
    print(f"y: {y.size()}, {y.dtype}")

    assert y.size(0) == BATCH_SIZE
    assert y.size(1) == OUT_CH
    assert y.size(2) == 1
    assert y.size(3) == w_out

    
@pytest.mark.parametrize("depth,pools,w_out",(
    (2, [4, 5], 5),
))
def test_UTimeEncoder__01(depth, pools, w_out):
    """ Build an encoder. """
    depth = 2
    pools = [4, 5]
    batch_shape = (BATCH_SIZE, IN_CH, 1, N_PERIODS)
    
    # -- build --
    net = UTimeEncoder(
        IN_CH,
        OUT_CH,
        N_PERIODS,
        depth=depth,
        pools=pools,
    )
    net.to(torch.double)
    pprint.pprint(net)

    # -- forward --
    x = np.random.uniform(-1, 1, batch_shape)
    x_tensor = torch.from_numpy(x).to(dtype=torch.double)
    (y, res) = net(x_tensor)
    print(f"y: {y.size()}, {y.dtype}")
    print(f"res: len={len(res)}, res[0]={res[0].size()}")

    assert y.size(0) == BATCH_SIZE
    assert y.size(1) == OUT_CH * (2**depth)
    assert y.size(2) == 1
    assert y.size(3) == w_out

    assert len(res) == depth
    
    
def test_UpBlock__01():
    n_period_x1 = N_PERIODS // 2
    batch_shape_x1 = (BATCH_SIZE, IN_CH, 1, n_period_x1)
    batch_shape_x2 = (BATCH_SIZE, IN_CH, 1, N_PERIODS)
    
    # -- build --
    net = UpBlock(IN_CH, OUT_CH, N_PERIODS)
    net.to(torch.double)
    pprint.pprint(net)
    
    # -- forward --
    x1 = np.random.uniform(-1, 1, batch_shape_x1)
    x2 = np.random.uniform(-1, 1, batch_shape_x2)
    x1_tensor = torch.from_numpy(x1).to(dtype=torch.double)
    x2_tensor = torch.from_numpy(x2).to(dtype=torch.double)
    y = net(x1_tensor, x2_tensor)
    print(f"x1: {x1_tensor.size()}, x2: {x2_tensor.size()}, y: {y.size()}, {y.dtype}")

    assert y.size(0) == BATCH_SIZE
    assert y.size(1) == OUT_CH
    assert y.size(2) == 1
    assert y.size(3) == N_PERIODS
