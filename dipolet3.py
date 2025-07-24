import mlx.core as mx
import numpy as np
import time
from utils import continuous_dipole_kernel, imshow_3d, rmse
from scipy.io import loadmat
from dipolet import gen_dipolets_filters, transformada_dipolet

def timeit(fun):
    def wrapper(*args, **kwargs):
        tic = time.time()
        result = fun(*args, **kwargs)
        print(f'{fun.__name__} Elapsed time: {time.time() - tic}')
        return result

    return wrapper


def rmse_mlx(gt, pred, mask=None):
    if mask is None:
        mask = mx.ones_like(gt)
    masked_pred = mx.where(mask == 1, pred, 0)
    masked_gt = mx.where(mask == 1, gt, 0)
    return 100 * mx.linalg.norm(masked_pred - masked_gt) / mx.linalg.norm(masked_gt)

@timeit
def ndi_np(phase, d, iters=10, tau=2, w=None, gt=None, msk=None, hist=False):
    def sus2field(x, d):
        return np.real(np.fft.ifftn(d * np.fft.fftn(x)))

    if w is None:
        w = 1
    alpha = 1e-6
    x = np.zeros_like(phase)
    if hist:
        _hist = []

    for _ in range(iters):
        ans = x.copy()
        phi_x = sus2field(x, d)
        x = x - tau * sus2field(w * np.sin(phi_x - phase), np.conj(d)) - tau * alpha * x

        update = rmse(x, ans)
        step = {'update': update}
        print(f'Update: {update}', end='\t')
        if gt is not None and msk is not None:
            error = rmse(gt, x, msk)
            print(f'RMSE={error: .2f}', end='')
            step['error'] = error
        if hist:
            _hist.append(step)
        print('')
    if hist:
        return x, _hist
    return x


@timeit
def ndi_mlx(phase, d, iters=10, tau=2, w=None, gt=None, msk=None, hist=False):
    def sus2field(x, d):
        return mx.real(mx.fft.ifftn(d * mx.fft.fftn(x)))

    if w is None:
        w = 1
    alpha = 1e-6
    x = mx.zeros_like(phase)
    if hist:
        _hist = []

    for _ in range(iters):
        ans = x
        phi_x = sus2field(x, d)
        x = x - tau * sus2field(w * mx.sin(phi_x - phase), mx.conj(d)) - tau * alpha * x

        update = rmse_mlx(x, ans)
        step = {'update': update}
        print(f'Update: {update}', end='\t')
        if gt is not None and msk is not None:
            error = rmse_mlx(gt, x, msk)
            print(f'RMSE={error: .2f}', end='')
            step['error'] = error
        if hist:
            _hist.append(step)
        print('')
    if hist:
        return x, _hist
    return x


# %%
mask = loadmat('msk.mat')['msk']
magn = loadmat('magn.mat')['magn']
gt = loadmat('chi_cosmos.mat')['chi_cosmos']
kernel = continuous_dipole_kernel(N := mask.shape)

phase = np.real(np.fft.ifftn(kernel * np.fft.fftn(gt)))

magn = magn * mask

scale = np.pi / np.max(np.abs(phase))
signal = magn * np.exp(1j * phase * scale)
_rr = np.random.rand()

snr = 100

signal = signal + ((1. / snr) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)))

phase = np.angle(signal).astype(np.float32) / scale
phase = phase * mask
phase[N[0] // 2, N[1] // 2, 2 * N[2] // 3] = 2
phase[N[0] // 2, N[1] // 2, 2 * N[2] // 3 - 1] = -2
imshow_3d(phase, rango=(-.1, .1))

# %%

d = continuous_dipole_kernel(phase.shape)
r1 = ndi_np(phase, d, iters=17, gt=gt, msk=mask)

# %%
phase2 = mx.array(phase)
d2 = mx.array(d)
gt2 = mx.array(gt)
mask2 = mx.array(mask)

r2 = ndi_mlx(phase2, d2, iters=50, gt=gt2, msk=mask2)

# %%
imshow_3d(r1, rango=(-0.1, 0.1), angles=(-90, -90, 90))
imshow_3d(r2, rango=(-0.1, 0.1), angles=(-90, -90, 90))

#%%
bank = gen_dipolets_filters(phase.shape)
print(bank.shape)

#%%
snr = 80
phase_clean = np.real(np.fft.ifftn(kernel * np.fft.fftn(gt)))
magn = magn * mask
scale = np.pi / np.max(np.abs(phase_clean))
signal = magn * np.exp(1j * phase_clean * scale)
ruido = (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
signal = signal + ((1. / snr) * ruido)
phase = np.angle(signal).astype(np.float32) / scale
phase = phase * mask

# phase[N[0] // 2, N[1] // 2, 2 * N[2] // 3] = 2
# phase[N[0] // 2, N[1] // 2, 2 * N[2] // 3 - 1] = -2

recont = transformada_dipolet(phase, gain_a=500, gain_r=500)
for i in range(recont.shape[-1]):
    imshow_3d(bank[..., i], rango=(0, 1), angles=(-90, -90, 90))
    imshow_3d(recont[..., i], rango=(-0.05, 0.05), angles=(-90, -90, 90))

#%%

fn = np.angle(np.fft.fftn(ruido))
imshow_3d(fn, rango=(-1, 1), angles=(-90, -90, 90))


