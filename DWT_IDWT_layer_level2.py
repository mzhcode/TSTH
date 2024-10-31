"""
自定义 pytorch 层，实现一维、二维、三维张量的 DWT 和 IDWT，未考虑边界延拓
只有当图像行列数都是偶数，且重构滤波器组低频分量长度为 2 时，才能精确重构，否则在边界处有误差。
"""
# from typing import Self
import numpy as np
import math
from torch.nn import Module
from DWT_IDWT_Functions_level2 import *
import pywt

__all__ = ['DWT_1D', 'IDWT_1D', 'DWT_2D', 'IDWT_2D', 'DWT_3D', 'IDWT_3D', 'DWT_2D_tiny']
class DWT_1D(Module):
    """
    input: the 1D data to be decomposed -- (N, C, Length)
    output: lfc -- (N, C, Length/2)
            hfc -- (N, C, Length/2)
    """
    def __init__(self, wavename):
        """
        1D discrete wavelet transform (DWT) for sequence decomposition
        用于序列分解的一维离散小波变换 DWT
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(DWT_1D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = self.input_height
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_h = matrix_h[:,(self.band_length_half-1):end]
        matrix_g = matrix_g[:,(self.band_length_half-1):end]
        if torch.cuda.is_available():
            self.matrix_low = torch.Tensor(matrix_h).cuda()
            self.matrix_high = torch.Tensor(matrix_g).cuda()
        else:
            self.matrix_low = torch.Tensor(matrix_h)
            self.matrix_high = torch.Tensor(matrix_g)

    def forward(self, input):
        """
        input_low_frequency_component = \mathcal{L} * input
        input_high_frequency_component = \mathcal{H} * input
        :param input: the data to be decomposed
        :return: the low-frequency and high-frequency components of the input data
        """
        assert len(input.size()) == 3
        self.input_height = input.size()[-1]
        self.get_matrix()
        return DWTFunction_1D.apply(input, self.matrix_low, self.matrix_high)


class IDWT_1D(Module):
    """
    input:  lfc -- (N, C, Length/2)
            hfc -- (N, C, Length/2)
    output: the original data -- (N, C, Length)
    """
    def __init__(self, wavename):
        """
        1D inverse DWT (IDWT) for sequence reconstruction
        用于序列重构的一维离散小波逆变换 IDWT
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(IDWT_1D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_high = wavelet.dec_hi
        self.band_low.reverse()
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        generating the matrices: \mathcal{L}, \mathcal{H}
        生成变换矩阵
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = self.input_height
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)
        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_h = matrix_h[:,(self.band_length_half-1):end]
        matrix_g = matrix_g[:,(self.band_length_half-1):end]
        if torch.cuda.is_available():
            self.matrix_low = torch.Tensor(matrix_h).cuda()
            self.matrix_high = torch.Tensor(matrix_g).cuda()
        else:
            self.matrix_low = torch.Tensor(matrix_h)
            self.matrix_high = torch.Tensor(matrix_g)

    def forward(self, L, H):
        """
        :param L: the low-frequency component of the original data
        :param H: the high-frequency component of the original data
        :return: the original data
        """
        assert len(L.size()) == len(H.size()) == 3
        self.input_height = L.size()[-1] + H.size()[-1]
        self.get_matrix()
        return IDWTFunction_1D.apply(L, H, self.matrix_low, self.matrix_high)


class DWT_2D_tiny(Module):
    """
    input: the 2D data to be decomposed -- (N, C, H, W)
    output -- lfc: (N, C, H/2, W/2)
              #hfc_lh: (N, C, H/2, W/2)
              #hfc_hl: (N, C, H/2, W/2)
              #hfc_hh: (N, C, H/2, W/2)
    DWT_2D_tiny only outputs the low-frequency component, which is used in WaveCNet;
    the all four components could be get using DWT_2D, which is used in WaveUNet.
    """
    def __init__(self, wavename):
        """
        2D discrete wavelet transform (DWT) for 2D image decomposition
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(DWT_2D_tiny, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)

        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, input):
        """
        input_lfc = \mathcal{L} * input * \mathcal{L}^T
        #input_hfc_lh = \mathcal{H} * input * \mathcal{L}^T
        #input_hfc_hl = \mathcal{L} * input * \mathcal{H}^T
        #input_hfc_hh = \mathcal{H} * input * \mathcal{H}^T
        :param input: the 2D data to be decomposed
        :return: the low-frequency component of the input 2D data
        """
        assert len(input.size()) == 4
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_2D_tiny.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)


class DWT_2D(Module):
    """
    input: the 2D data to be decomposed -- (N, C, H, W)
    output -- lfc: (N, C, H/2, W/2)
              hfc_lh: (N, C, H/2, W/2)
              hfc_hl: (N, C, H/2, W/2)
              hfc_hh: (N, C, H/2, W/2)
    """
    def __init__(self, wavename):
        """
        2D discrete wavelet transform (DWT) for 2D image decomposition
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(DWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)
        self.L1_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.5, detach_reset=True, backend='cupy')
        self.H1_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.5, detach_reset=True, backend='cupy')

    def get_matrix(self):
        """
        生成变换矩阵
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)

        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, input):
        """
        input_lfc = \mathcal{L} * input * \mathcal{L}^T
        input_hfc_lh = \mathcal{H} * input * \mathcal{L}^T
        input_hfc_hl = \mathcal{L} * input * \mathcal{H}^T
        input_hfc_hh = \mathcal{H} * input * \mathcal{H}^T
        :param input: the 2D data to be decomposed
        :return: the low-frequency and high-frequency components of the input 2D data
        """
        assert len(input.size()) == 4
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_2D.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1, self.L1_lif, self.H1_lif)


class IDWT_2D(Module):
    """
    input:  lfc -- (N, C, H/2, W/2)
            hfc_lh -- (N, C, H/2, W/2)
            hfc_hl -- (N, C, H/2, W/2)
            hfc_hh -- (N, C, H/2, W/2)
    output: the original 2D data -- (N, C, H, W)
    """
    def __init__(self, wavename):
        """
        2D inverse DWT (IDWT) for 2D image reconstruction
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(IDWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_low.reverse()
        self.band_high = wavelet.dec_hi
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)
        self.iL1_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.75, detach_reset=True, backend='cupy')
        self.iH1_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.75, detach_reset=True, backend='cupy')

    def get_matrix(self):
        """
        生成变换矩阵
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, LL, LH, HL, HH):
        """
        recontructing the original 2D data
        the original 2D data = \mathcal{L}^T * lfc * \mathcal{L}
                             + \mathcal{H}^T * hfc_lh * \mathcal{L}
                             + \mathcal{L}^T * hfc_hl * \mathcal{H}
                             + \mathcal{H}^T * hfc_hh * \mathcal{H}
        :param LL: the low-frequency component
        :param LH: the high-frequency component, hfc_lh
        :param HL: the high-frequency component, hfc_hl
        :param HH: the high-frequency component, hfc_hh
        :return: the original 2D data
        """
        assert len(LL.size()) == len(LH.size()) == len(HL.size()) == len(HH.size()) == 4
        self.input_height = LL.size()[-2] + HH.size()[-2]
        self.input_width = LL.size()[-1] + HH.size()[-1]
        self.get_matrix()
        return IDWTFunction_2D.apply(LL, LH, HL, HH, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1, self.iL1_lif, self.iH1_lif)


class DWT_3D(Module):
    """
    input: the 3D data to be decomposed -- (N, C, D, H, W)
    output: lfc -- (N, C, D/2, H/2, W/2)
            hfc_llh -- (N, C, D/2, H/2, W/2)
            hfc_lhl -- (N, C, D/2, H/2, W/2)
            hfc_lhh -- (N, C, D/2, H/2, W/2)
            hfc_hll -- (N, C, D/2, H/2, W/2)
            hfc_hlh -- (N, C, D/2, H/2, W/2)
            hfc_hhl -- (N, C, D/2, H/2, W/2)
            hfc_hhh -- (N, C, D/2, H/2, W/2)
    """
    def __init__(self, wavename):
        """
        3D discrete wavelet transform (DWT) for 3D data decomposition
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(DWT_3D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)
        self.L_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.35, detach_reset=True, backend='cupy')
        self.H_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.35, detach_reset=True, backend='cupy')
        self.LL_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.35, detach_reset=True, backend='cupy')
        self.LH_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.35, detach_reset=True, backend='cupy')
        self.HL_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.35, detach_reset=True, backend='cupy')
        self.HH_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.35, detach_reset=True, backend='cupy')

        self.LLL1_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.35, detach_reset=True, backend='cupy')

        self.L2_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.35, detach_reset=True, backend='cupy')
        self.H2_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.35, detach_reset=True, backend='cupy')
        self.LL2_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.35, detach_reset=True, backend='cupy')
        self.LH2_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.35, detach_reset=True, backend='cupy')
        self.HL2_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.35, detach_reset=True, backend='cupy')
        self.HH2_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.35, detach_reset=True, backend='cupy')
    def get_matrix(self):
        """
        生成变换矩阵
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]
        matrix_h_2 = matrix_h[0:(math.floor(self.input_depth / 2)), 0:(self.input_depth + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]
        matrix_g_2 = matrix_g[0:(self.input_depth - math.floor(self.input_depth / 2)),0:(self.input_depth + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_h_2 = matrix_h_2[:,(self.band_length_half-1):end]

        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        matrix_g_2 = matrix_g_2[:,(self.band_length_half-1):end]
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_low_2 = torch.Tensor(matrix_h_2).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
            self.matrix_high_2 = torch.Tensor(matrix_g_2).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_low_2 = torch.Tensor(matrix_h_2)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)
            self.matrix_high_2 = torch.Tensor(matrix_g_2)

    def forward(self, input):
        """
        :param input: the 3D data to be decomposed
        :return: the eight components of the input data, one low-frequency and seven high-frequency components
        """
        assert len(input.size()) == 5
        self.input_depth = input.size()[-3]
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        LLL1, LLH1, LHL1, LHH1, HLL1, HLH1, HHL1, HHH1 = DWTFunction_3D.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_low_2,
                                           self.matrix_high_0, self.matrix_high_1, self.matrix_high_2,
                                           self.L_lif, self.H_lif, self.LL_lif, self.LH_lif, self.HL_lif, self.HH_lif)
        
        LLL1 = self.LLL1_lif(LLL1.permute(2, 0, 1, 3, 4)).permute(1, 2, 0, 3, 4)

        self.input_depth =  LLL1.size()[-3]
        self.input_height = LLL1.size()[-2]
        self.input_width = LLL1.size()[-1]
        self.get_matrix()

        LLL2, LLH2, LHL2, LHH2, HLL2, HLH2, HHL2, HHH2 = DWTFunction_3D.apply(LLL1, self.matrix_low_0, self.matrix_low_1, self.matrix_low_2,
                                           self.matrix_high_0, self.matrix_high_1, self.matrix_high_2,
                                           self.L2_lif, self.H2_lif, self.LL2_lif, self.LH2_lif, self.HL2_lif, self.HH2_lif)
        
        high1 = torch.stack([LLH1, LHL1, LHH1, HLL1, HLH1, HHL1, HHH1], dim=1)
        high2 = torch.stack([LLH2, LHL2, LHH2, HLL2, HLH2, HHL2, HHH2], dim=1)

        return LLL2, high1, high2
        


class IDWT_3D(Module):
    """
    input:  lfc -- (N, C, D/2, H/2, W/2)
            hfc_llh -- (N, C, D/2, H/2, W/2)
            hfc_lhl -- (N, C, D/2, H/2, W/2)
            hfc_lhh -- (N, C, D/2, H/2, W/2)
            hfc_hll -- (N, C, D/2, H/2, W/2)
            hfc_hlh -- (N, C, D/2, H/2, W/2)
            hfc_hhl -- (N, C, D/2, H/2, W/2)
            hfc_hhh -- (N, C, D/2, H/2, W/2)
    output: the original 3D data -- (N, C, D, H, W)
    """
    def __init__(self, wavename):
        """
        3D inverse DWT (IDWT) for 3D data reconstruction
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(IDWT_3D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_high = wavelet.dec_hi
        self.band_low.reverse()
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)
        self.iLL1_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.35, detach_reset=True, backend='cupy')
        self.iLH1_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.35, detach_reset=True, backend='cupy')
        self.iHL1_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.35, detach_reset=True, backend='cupy')
        self.iHH1_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.35, detach_reset=True, backend='cupy')
        self.iL1_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.35, detach_reset=True, backend='cupy')
        self.iH1_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.35, detach_reset=True, backend='cupy')

        self.iLLL1_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.35, detach_reset=True, backend='cupy')

        self.iLL2_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.35, detach_reset=True, backend='cupy')
        self.iLH2_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.35, detach_reset=True, backend='cupy')
        self.iHL2_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.35, detach_reset=True, backend='cupy')
        self.iHH2_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.35, detach_reset=True, backend='cupy')
        self.iL2_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.35, detach_reset=True, backend='cupy')
        self.iH2_lif = MultiStepLIFNode(tau=1.75, v_threshold=0.35, detach_reset=True, backend='cupy')

    def get_matrix(self):
        """
        生成变换矩阵
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]
        matrix_h_2 = matrix_h[0:(math.floor(self.input_depth / 2)), 0:(self.input_depth + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]
        matrix_g_2 = matrix_g[0:(self.input_depth - math.floor(self.input_depth / 2)),0:(self.input_depth + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_h_2 = matrix_h_2[:,(self.band_length_half-1):end]

        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        matrix_g_2 = matrix_g_2[:,(self.band_length_half-1):end]
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_low_2 = torch.Tensor(matrix_h_2).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
            self.matrix_high_2 = torch.Tensor(matrix_g_2).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_low_2 = torch.Tensor(matrix_h_2)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)
            self.matrix_high_2 = torch.Tensor(matrix_g_2)

    def forward(self, LLL2, high1, high2):
        """
        :param LLL: the low-frequency component, lfc
        :param LLH: the high-frequency componetn, hfc_llh
        :param LHL: the high-frequency componetn, hfc_lhl
        :param LHH: the high-frequency componetn, hfc_lhh
        :param HLL: the high-frequency componetn, hfc_hll
        :param HLH: the high-frequency componetn, hfc_hlh
        :param HHL: the high-frequency componetn, hfc_hhl
        :param HHH: the high-frequency componetn, hfc_hhh
        :return: the original 3D input data
        """
        # assert len(LLL.size()) == len(LLH.size()) == len(LHL.size()) == len(LHH.size()) == 5
        # assert len(HLL.size()) == len(HLH.size()) == len(HHL.size()) == len(HHH.size()) == 5
        self.input_depth = LLL2.size()[-3] + high2[6].size()[-3]
        self.input_height = LLL2.size()[-2] + high2[6].size()[-2]
        self.input_width = LLL2.size()[-1] + high2[6].size()[-1]
        self.get_matrix()
        LLH2 = high2[0]
        LHL2 = high2[1]
        LHH2 = high2[2]
        HLL2 = high2[3]
        HLH2 = high2[4]
        HHL2 = high2[5]
        HHH2 = high2[6]
        LLH1 = high1[0]
        LHL1 = high1[1]
        LHH1 = high1[2]
        HLL1 = high1[3]
        HLH1 = high1[4]
        HHL1 = high1[5]
        HHH1 = high1[6]
        
        LLL1 = IDWTFunction_3D.apply(LLL2, LLH2, LHL2, LHH2, HLL2, HLH2, HHL2, HHH2,
                                     self.matrix_low_0, self.matrix_low_1, self.matrix_low_2,
                                     self.matrix_high_0, self.matrix_high_1, self.matrix_high_2,
                                     self.iLL2_lif, self.iLH2_lif, self.iHL2_lif, self.iHH2_lif, self.iL2_lif, self.iH2_lif)
        
        LLL1 = self.iLLL1_lif(LLL1.permute(2, 0, 1, 3, 4)).permute(1, 2, 0, 3, 4)

        self.input_depth = LLL1.size()[-3] + high1[6].size()[-3]
        self.input_height = LLL1.size()[-2] + high1[6].size()[-2]
        self.input_width = LLL1.size()[-1] + high1[6].size()[-1]
        self.get_matrix()

        output = IDWTFunction_3D.apply(LLL1, LLH1, LHL1, LHH1, HLL1, HLH1, HHL1, HHH1,
                                     self.matrix_low_0, self.matrix_low_1, self.matrix_low_2,
                                     self.matrix_high_0, self.matrix_high_1, self.matrix_high_2,
                                     self.iLL1_lif, self.iLH1_lif, self.iHL1_lif, self.iHH1_lif, self.iL1_lif, self.iH1_lif)
        return output


if __name__ == '__main__':
    from datetime import datetime
    from torch.autograd import gradcheck
    wavelet = pywt.Wavelet('bior1.1')
    h = wavelet.rec_lo
    g = wavelet.rec_hi
    h_ = wavelet.dec_lo
    g_ = wavelet.dec_hi
    h_.reverse()
    g_.reverse()

    """
    image_full_name = '/home/li-qiufu/Pictures/standard_test_images/lena_color_512.tif'
    image = cv2.imread(image_full_name, flags = 1)
    image = image[0:512,0:512,:]
    print(image.shape)
    height, width, channel = image.shape
    #image = image.reshape((1,height,width))
    t0 = datetime.now()
    for index in range(100):
        m0 = DWT_2D(band_low = h, band_high = g)
        image_tensor = torch.Tensor(image)
        image_tensor.unsqueeze_(dim = 0)
        print('image_re shape: {}'.format(image_tensor.size()))
        image_tensor.transpose_(1,3)
        print('image_re shape: {}'.format(image_tensor.size()))
        image_tensor.transpose_(2,3)
        print('image_re shape: {}'.format(image_tensor.size()))
        image_tensor.requires_grad = False
        LL, LH, HL, HH = m0(image_tensor)
        matrix_low_0 = torch.Tensor(m0.matrix_low_0)
        matrix_low_1 = torch.Tensor(m0.matrix_low_1)
        matrix_high_0 = torch.Tensor(m0.matrix_high_0)
        matrix_high_1 = torch.Tensor(m0.matrix_high_1)

        #image_tensor.requires_grad = True
        #input = (image_tensor.double(), matrix_low_0.double(), matrix_low_1.double(), matrix_high_0.double(), matrix_high_1.double())
        #test = gradcheck(DWTFunction_2D.apply, input)
        #print(test)
        #print(LL.requires_grad)
        #print(LH.requires_grad)
        #print(HL.requires_grad)
        #print(HH.requires_grad)
        #LL.requires_grad = True
        #input = (LL.double(), LH.double(), HL.double(), HH.double(), matrix_low_0.double(), matrix_low_1.double(), matrix_high_0.double(), matrix_high_1.double())
        #test = gradcheck(IDWTFunction_2D.apply, input)
        #print(test)

        m1 = IDWT_2D(band_low = h_, band_high = g_)
        image_re = m1(LL,LH,HL,HH)
    t1 = datetime.now()
    image_re.transpose_(2,3)
    image_re.transpose_(1,3)
    image_re_np = image_re.detach().numpy()
    print('image_re shape: {}'.format(image_re_np.shape))

    image_zero = image - image_re_np[0]
    print(np.max(image_zero), np.min(image_zero))
    print(image_zero[:,8])
    print('taking {} secondes'.format(t1 - t0))
    cv2.imshow('reconstruction', image_re_np[0]/255)
    cv2.imshow('image_zero', image_zero/255)
    cv2.waitKey(0)
    """
    """
    image_full_name = '/home/liqiufu/Pictures/standard_test_images/lena_color_512.tif'
    image = cv2.imread(image_full_name, flags = 1)
    image = image[0:512,0:512,:]
    print(image.shape)
    image_3d = np.concatenate((image, image, image, image, image, image), axis = 2)
    print(image_3d.shape)
    image_tensor = torch.Tensor(image_3d)
    #image_tensor = image_tensor.transpose(dim0 = 2, dim1 = 1)
    #image_tensor = image_tensor.transpose(dim0 = 1, dim1 = 0)
    image_tensor.unsqueeze_(dim = 0)
    image_tensor.unsqueeze_(dim = 0)
    t0 = datetime.now()
    for index in range(10):
        m0 = DWT_3D(wavename = 'haar')
        print('image_re shape: {}'.format(image_tensor.size()))
        image_tensor.requires_grad = False
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = m0(image_tensor)
        matrix_low_0 = torch.Tensor(m0.matrix_low_0)
        matrix_low_1 = torch.Tensor(m0.matrix_low_1)
        matrix_low_2 = torch.Tensor(m0.matrix_low_2)
        matrix_high_0 = torch.Tensor(m0.matrix_high_0)
        matrix_high_1 = torch.Tensor(m0.matrix_high_1)
        matrix_high_2 = torch.Tensor(m0.matrix_high_2)

        #image_tensor.requires_grad = True
        #input = (image_tensor.double(), matrix_low_0.double(), matrix_low_1.double(), matrix_low_2.double(),
        #                                matrix_high_0.double(), matrix_high_1.double(), matrix_high_2.double())
        #test = gradcheck(DWTFunction_3D.apply, input)
        #print('testing dwt3d -- {}'.format(test))
        #LLL.requires_grad = True
        #input = (LLL.double(), LLH.double(), LHL.double(), LHH.double(),
        #         HLL.double(), HLH.double(), HHL.double(), HHH.double(),
        #         matrix_low_0.double(), matrix_low_1.double(), matrix_low_2.double(),
        #         matrix_high_0.double(), matrix_high_1.double(), matrix_high_2.double())
        #test = gradcheck(IDWTFunction_3D.apply, input)
        #print('testing idwt3d -- {}'.format(test))

        m1 = IDWT_3D(wavename = 'haar')
        image_re = m1(LLL,LLH,LHL,LHH,HLL,HLH,HHL,HHH)
    t1 = datetime.now()
    image_re.squeeze_(dim = 0)
    image_re.squeeze_(dim = 0)
    #image_re.transpose_(0,1)
    #image_re.transpose_(1,2)
    image_re_np = image_re.detach().numpy()
    print('image_re shape: {}'.format(image_re_np.shape))

    image_zero = image - image_re_np[:,:,0:3]
    print(np.max(image_zero), np.min(image_zero))
    #print(image_zero[:,8,0])
    print('taking {} secondes'.format(t1 - t0))
    cv2.imshow('reconstruction', image_re_np[:,:,0:3]/255)
    cv2.imshow('image_zero', image_zero/255)
    cv2.waitKey(0)
    """

    """
    import matplotlib.pyplot as plt
    import numpy as np
    vector_np = np.array(list(range(1280)))#.reshape((128,1))

    print(vector_np.shape)
    t0 = datetime.now()
    for index in range(100):
        vector = torch.Tensor(vector_np)
        vector.unsqueeze_(dim = 0)
        vector.unsqueeze_(dim = 0)
        m0 = DWT_1D(band_low = h, band_high = g)
        L, H = m0(vector)

        #matrix_low = torch.Tensor(m0.matrix_low)
        #matrix_high = torch.Tensor(m0.matrix_high)
        #vector.requires_grad = True
        #input = (vector.double(), matrix_low.double(), matrix_high.double())
        #test = gradcheck(DWTFunction_1D.apply, input)
        #print('testing 1D-DWT: {}'.format(test))
        #print(L.requires_grad)
        #print(H.requires_grad)
        #L.requires_grad = True
        #H.requires_grad = True
        #input = (L.double(), H.double(), matrix_low.double(), matrix_high.double())
        #test = gradcheck(IDWTFunction_1D.apply, input)
        #print('testing 1D-IDWT: {}'.format(test))

        m1 = IDWT_1D(band_low = h_, band_high = g_)
        vector_re = m1(L, H)
    t1 = datetime.now()
    vector_re_np = vector_re.detach().numpy()
    print('image_re shape: {}'.format(vector_re_np.shape))

    vector_zero = vector_np - vector_re_np.reshape(vector_np.shape)
    print(np.max(vector_zero), np.min(vector_zero))
    print(vector_zero[:8])
    print('taking {} secondes'.format(t1 - t0))
    """
