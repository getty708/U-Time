import pprint

import numpy as np
import pytest
import torch

from utime.models.utime import (SingleConvBlock,
                                DoubleConvBlock,
                                DownBlock,
                                UpBlock,
                                UTimeEncoder,
                                UTimeDecoder,
                                DenseClassifier,
                                SegmentClassifier,
                                UTime)


BATCH_SIZE = 10
IN_CH = 32
OUT_CH = 32
N_PERIODS = 1000
N_CLASSES = 10


def test_SingleConvBlock__01():
    batch_shape = (BATCH_SIZE, IN_CH, 1, N_PERIODS)

    # -- build --
    net = SingleConvBlock(
        IN_CH,
        OUT_CH)
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


def test_DoubleConvBlock__01():
    batch_shape = (BATCH_SIZE, IN_CH, 1, N_PERIODS)

    # -- build --
    net = DoubleConvBlock(
        IN_CH,
        OUT_CH)
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


@pytest.mark.parametrize("pool_size,w_out", (
    (1, N_PERIODS),
    (4,  N_PERIODS//4),
))
def test_DownBlock__01(pool_size, w_out):
    batch_shape = (BATCH_SIZE, IN_CH, 1, N_PERIODS)

    # -- build --
    net = DownBlock(
        IN_CH,
        OUT_CH,
        pool_size)
    net.to(torch.double)
    pprint.pprint(net)

    # -- forward --
    x = np.random.uniform(-1, 1, batch_shape)
    x_tensor = torch.from_numpy(x).to(dtype=torch.double)
    y, y_skip = net(x_tensor)
    print(f"y: {y.size()}, {y.dtype}")

    assert y.size(0) == BATCH_SIZE
    assert y.size(1) == OUT_CH
    assert y.size(2) == 1
    assert y.size(3) == w_out

    assert y_skip.size(0) == BATCH_SIZE
    assert y_skip.size(1) == OUT_CH
    assert y_skip.size(2) == 1
    assert y_skip.size(3) == N_PERIODS
    

@pytest.mark.parametrize("depth,pools", (
    (2, [4, 4],),
))
def test_UTimeEncoder__01(depth, pools):
    """ Build an encoder. """
    def get_w_out():
        w_out = N_PERIODS
        for p in pools:
            w_out //= p
        return w_out

    batch_shape = (BATCH_SIZE, IN_CH, 1, N_PERIODS)
    ch_out = IN_CH * (2**depth)
    w_out = get_w_out()
    

    # -- build --
    net = UTimeEncoder(
        IN_CH,
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
    assert y.size(1) == ch_out
    assert y.size(2) == 1
    assert y.size(3) == w_out

    assert len(res) == depth
    assert len(net.filters) == depth + 1
    np.testing.assert_array_equal(net.filters, [IN_CH*2, IN_CH*4, IN_CH*4])


def test_UpBlock__01():
    batch_shape_x1 = (BATCH_SIZE, IN_CH, 1, N_PERIODS // 2)
    batch_shape_x2 = (BATCH_SIZE, IN_CH, 1, N_PERIODS)

    # -- build --
    net = UpBlock(IN_CH, IN_CH // 2, 2)
    net.to(torch.double)
    pprint.pprint(net)
    
    # -- forward --
    x1 = np.random.uniform(-1, 1, batch_shape_x1)
    x2 = np.random.uniform(-1, 1, batch_shape_x2)
    x1_tensor = torch.from_numpy(x1).to(dtype=torch.double)
    x2_tensor = torch.from_numpy(x2).to(dtype=torch.double)
    y = net(x1_tensor, x2_tensor)
    print(
        f"x1: {x1_tensor.size()}, x2: {x2_tensor.size()}, y: {y.size()}, {y.dtype}")

    assert y.size(0) == BATCH_SIZE
    assert y.size(1) == IN_CH // 2
    assert y.size(2) == 1
    assert y.size(3) == N_PERIODS


@pytest.mark.parametrize("depth,pools,w_out", (
    (2, [4, 5], 5),
))
def test_UTimeDecoder__01(depth, pools, w_out):
    """ Build an encoder. """
    depth = 2
    pools = [4, 5]
    in_ch = IN_CH * 4
    n_periods = N_PERIODS // (2**3)
    batch_shape = (BATCH_SIZE, in_ch, 1, n_periods)

    # -- build --
    net = UTimeDecoder(
        in_ch,
        pools=pools,
    )
    net.to(torch.double)
    pprint.pprint(net)
    
    # -- forward --
    x1 = np.random.uniform(-1, 1, batch_shape)
    x1_tensor = torch.from_numpy(x1).to(dtype=torch.double)

    x2_list = []
    in_ch = IN_CH
    n_periods = N_PERIODS
    for i, pool_size in enumerate(pools):
        in_ch *= 2
        n_periods //= 2

        batch_shape_2 = (BATCH_SIZE, in_ch, 1, n_periods)
        x2 = np.random.uniform(-1, 1, batch_shape_2)
        print(x2.shape)
        x2_list.append(torch.from_numpy(x2).to(dtype=torch.double))        

    y = net(x1_tensor, x2_list)
    print(f"y: {y.size()}, {y.dtype}")


def test_DenseClassifier__01():
    """ Build an encoder. """
    batch_shape = (BATCH_SIZE, IN_CH, 1, N_PERIODS)

    # -- build --
    net = DenseClassifier(
        IN_CH,
        N_CLASSES,
    )
    net.to(torch.double)
    pprint.pprint(net)

    # -- forward --
    x = np.random.uniform(-1, 1, batch_shape)
    x_tensor = torch.from_numpy(x).to(dtype=torch.double)
    y = net(x_tensor)
    print(f"y: {y.size()}, {y.dtype}")

    assert y.size(0) == BATCH_SIZE
    assert y.size(1) == N_CLASSES
    assert y.size(3) == N_PERIODS


def test_SegmentClassifier__01():
    """ Build an encoder. """
    data_per_period = 10
    in_ch = N_CLASSES
    batch_shape = (BATCH_SIZE, in_ch, 1, N_PERIODS)

    # -- build --
    net = SegmentClassifier(
        in_ch,
        data_per_period,
    )
    net.to(torch.double)
    pprint.pprint(net)

    # -- forward --
    x = np.random.uniform(-1, 1, batch_shape)
    x_tensor = torch.from_numpy(x).to(dtype=torch.double)
    y = net(x_tensor)
    print(f"x: {x_tensor.size()}, {x_tensor.dtype}")
    print(f"y: {y.size()}, {y.dtype}")

    assert y.size(0) == BATCH_SIZE
    assert y.size(1) == N_CLASSES
    assert y.size(2) == 1
    assert y.size(3) == N_PERIODS // data_per_period


@pytest.mark.parametrize("pools", (
    (4, 4, 4, 4),
    (6, 4, 2),
))
def test_UTime__01(pools):
    """ Build an encoder. """
    input_dims = 6
    data_per_period = 10
    batch_shape = (BATCH_SIZE, input_dims, 1, N_PERIODS)

    # -- build --
    net = UTime(
        N_CLASSES,
        batch_shape[1:],
        pools=pools,
        data_per_period=data_per_period)
    net.to(torch.double)
    pprint.pprint(net)
    
    # -- forward --
    x = np.random.uniform(-1, 1, batch_shape)
    x_tensor = torch.from_numpy(x).to(dtype=torch.double)
    y = net(x_tensor)
    print(f"y: {y.size()}, {y.dtype}")

    assert y.size(0) == BATCH_SIZE
    assert y.size(1) == N_CLASSES
    assert y.size(2) == 1
    assert y.size(3) == N_PERIODS // data_per_period
