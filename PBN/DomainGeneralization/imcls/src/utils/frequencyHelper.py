
import numpy as np
import torch
from scipy import signal


def fft(img):
    return np.fft.fft2(img)


def fftshift(img):
    return np.fft.fftshift(fft(img))


def ifft(img):
    return np.fft.ifft2(img)


def ifftshift(img):
    return ifft(np.fft.ifftshift(img))


def distance(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize / 2) ** 2 + (j - imageSize / 2) ** 2)
    if dis < r:
        return 1.0
    else:
        return 0


def mask_radial(img, r):
    rows, cols = img.shape
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, imageSize=rows, r=r)
    return mask


def generateSmoothKernel(data, r):
    result = np.zeros_like(data)
    [k1, k2, m, n] = data.shape
    mask = np.zeros([3, 3])
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                mask[i, j] = 1
            else:
                mask[i, j] = r
    mask = mask
    for i in range(m):
        for j in range(n):
            result[:, :, i, j] = signal.convolve2d(data[:, :, i, j], mask, boundary='symm', mode='same')
    return result


def generateDataWithDifferentFrequencies_GrayScale(Images, r):
    Images_freq_low = []
    mask = mask_radial(np.zeros([28, 28]), r)
    for i in range(Images.shape[0]):
        fd = fftshift(Images[i, :].reshape([28, 28]))
        fd = fd * mask
        img_low = ifftshift(fd)
        Images_freq_low.append(np.real(img_low).reshape([28 * 28]))

    return np.array(Images_freq_low)


def generateDataWithDifferentFrequencies_3Channel(Images, r):
    Images = Images.permute(0, 2, 3, 1)
    Images_freq_low = []
    Images_freq_high = []
    mask = mask_radial(np.zeros([Images.shape[1], Images.shape[2]]), r)
    for i in range(Images.shape[0]):
        tmp = np.zeros([Images.shape[1], Images.shape[2], 3])
        for j in range(3):
            fd = fftshift(Images[i, :, :, j])
            fd = fd * mask
            img_low = ifftshift(fd)
            tmp[:, :, j] = np.real(img_low)
        Images_freq_low.append(tmp)
        tmp = np.zeros([Images.shape[1], Images.shape[2], 3])
        for j in range(3):
            fd = fftshift(Images[i, :, :, j])
            fd = fd * (1 - mask)
            img_high = ifftshift(fd)
            tmp[:, :, j] = np.real(img_high)
        Images_freq_high.append(tmp)
    Images_freq_low, Images_freq_high = np.array(Images_freq_low), np.array(Images_freq_high)
    Images_freq_low, Images_freq_high = torch.tensor(Images_freq_low).permute(0, 3, 1, 2), torch.tensor(Images_freq_high).permute(0, 3, 1, 2)
    return Images_freq_low.to(torch.float32), Images_freq_high.to(torch.float32)


if __name__ == '__main__':

    image = torch.randn(10000, 32, 32, 3)
    a, b = generateDataWithDifferentFrequencies_3Channel(image, 4)
    pass
