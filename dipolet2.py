import numpy as np
import matplotlib.pyplot as plt
from utils import continuous_dipole_kernel, imshow_3d, rmse
from nibabel import load as load_nii
from numpy.fft import fftn, ifftn, ifftshift, fftshift
from scipy.io import loadmat
from functools import cache
import torch


def sigmoid(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    return 1 / (1 + np.exp(-x))


def sigmoid_window(t: np.ndarray | torch.Tensor, val: float = 0.9, gain: float = 40) -> np.ndarray | torch.Tensor:
    return 1 - sigmoid(gain * (t - val))


def gen_radial_filters(N: tuple[int, int, int], gain: float = 40, levels: int = 3) -> list[np.ndarray]:
    freq_x = fftshift(np.fft.fftfreq(N[0], d=1.0))
    freq_y = fftshift(np.fft.fftfreq(N[1], d=1.0))
    freq_z = fftshift(np.fft.fftfreq(N[2], d=1.0))
    omega_x, omega_y, omega_z = np.meshgrid(freq_x, freq_y, freq_z, indexing='ij')
    rho = np.sqrt(omega_x ** 2 + omega_y ** 2 + omega_z ** 2)

    fltrs = []
    ans = np.ones_like(rho)
    for i in range(1, levels + 1):
        aux = sigmoid_window(rho * (2 ** i), gain=gain)
        fltrs.append(ans - aux)
        ans = aux
    fltrs.append(ans)
    return fltrs


def gen_angular_filters(N: tuple[int, int, int], gain: float = 100, levels: int = 3) -> list[np.ndarray]:
    kernel = continuous_dipole_kernel(N)
    k = np.fft.fftshift(kernel)
    k2 = np.abs(k)
    ans = 0
    dipoles = []
    delta = 0.3 / (2 ** (levels - 1))

    for i in range(levels):
        new = sigmoid_window(k2, val=delta * (2 ** i), gain=gain)
        dipoles.append(new - ans)
        ans = new.copy()
    dipoles.append(1 - ans)
    return dipoles


def gen_dipolets_filters(N, fun_radial=gen_radial_filters, gain_a=100, gain_r=40, angular_levels=3, radial_levels=3):
    angulars = gen_angular_filters(N, gain=gain_a, levels=angular_levels)
    radials = fun_radial(N, gain=gain_r, levels=radial_levels)
    bank = []
    for rad_f in radials[:-1]:
        ang_cum = 0
        for ang_f in angulars:
            bank.append((rad_f * ang_f)[..., np.newaxis])
            ang_cum += ang_f
    bank.append(radials[-1][..., np.newaxis])
    bank = np.concatenate(bank, axis=-1)
    return bank


def transformada_dipolet(img, fun_radial=gen_radial_filters, angular_levels=3, radial_levels=3, gain_a=100, gain_r=40):
    N = img.shape
    bank = gen_dipolets_filters(N, fun_radial=fun_radial, angular_levels=angular_levels,
                                radial_levels=radial_levels, gain_a=gain_a, gain_r=gain_r)
    result = []
    ftimg = fftn(img)
    for i in range(bank.shape[-1]):
        aux = np.real(ifftn(ifftshift(bank[..., i]) * ftimg))
        result.append(aux[..., np.newaxis])
    result = np.concatenate(result, axis=-1)
    return result


if __name__ == '__main__':
    1/0
    # %%

    mask = loadmat('msk.mat')['msk']
    magn = loadmat('magn.mat')['magn']
    gt = loadmat('chi_cosmos.mat')['chi_cosmos']
    kernel = continuous_dipole_kernel(N := mask.shape)

    phase = np.real(ifftn(kernel * fftn(gt)))

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
    torch.cuda.init()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # %%
    def ndi(phase, iters=10, tau=2, w=None, dipole=False, dip_thr=0.15, gt=None, msk=None, hist=False):
        phase = torch.Tensor(phase).to(device)
        def sus2field(x, d):
            return torch.real(torch.fft.ifftn(d * torch.fft.fftn(x)))

        if w is None:
            w = 1
        else:
            w = torch.Tensor(w).to(device)
        alpha = 1e-6
        d = torch.Tensor(continuous_dipole_kernel(phase.shape)).to(device)
        x = torch.zeros_like(phase).to(device)

        if hist:
            _hist = []

        if dipole:
            bank = torch.Tensor(gen_dipolets_filters(x.shape, gain_a=500, gain_r=500)).to(device)

        for it in range(iters):
            ans = x.clone()
            phi_x = sus2field(x, d)
            x = x - tau * sus2field(w * torch.sin(phi_x - phase), torch.conj(d)) - tau * alpha * x

            if dipole:
                x_d = torch.real(torch.fft.ifftn(torch.fft.ifftshift(bank, dim=(0, 1, 2)) * torch.fft.fftn(x, dim=(0, 1, 2)).unsqueeze(-1), dim=(0, 1, 2)))

                x_d = torch.clip(x_d, min=-dip_thr, max=dip_thr)
                x_d[..., 0] = torch.clip(x_d[..., 0], min=-dip_thr*0.01, max=dip_thr*0.01)
                x_d[..., 1] = torch.clip(x_d[..., 1], min=-dip_thr*0.02, max=dip_thr*0.02)
                x_d[..., 2] = torch.clip(x_d[..., 2], min=-dip_thr*0.5, max=dip_thr*0.5)
                x_d[..., 3] = torch.clip(x_d[..., 3], min=-dip_thr*0.5, max=dip_thr*0.5)
                x_d[..., 4] = torch.clip(x_d[..., 4], min=-dip_thr*0.5, max=dip_thr*0.5)
                x_d[..., 5] = torch.clip(x_d[..., 5], min=-dip_thr*0.5, max=dip_thr*0.5)
                x_d[..., 6] = torch.clip(x_d[..., 6], min=-dip_thr*0.5, max=dip_thr*0.5)


                x = torch.sum(x_d, dim=-1)

            update = rmse(x.cpu().numpy(), ans.cpu().numpy())
            step = {'update': update}
            print(f'{it} - Update: {update}', end='\t')
            if gt is not None and msk is not None:
                error = rmse(gt, x.cpu().numpy(), msk)
                print(f'RMSE={error: .2f}', end='')
                step['error'] = error
            if hist:
                _hist.append(step)
            print('')
        if hist:
            return x.cpu().numpy(), _hist
        return x.cpu().numpy()


    print(1)
    recon1, hist1 = ndi(phase, iters=100, gt=gt, msk=mask, hist=True, dipole=True)
    imshow_3d(recon1, rango=(-0.1, 0.1), angles=(-90, -90, 90), title=f'NDI rmse={rmse(gt, recon1, mask):.2f}')
    imshow_3d(recon1*mask, rango=(-0.1, 0.1), angles=(-90, -90, 90), title=f'NDI rmse={rmse(gt, recon1, mask):.2f}')


    # %%
    # %%
    def ndi2(phase, iters=10, tau=2, w=None, dipole=False, dip_thr=0.15, gt=None, msk=None, hist=False):
        phase = torch.Tensor(phase).to(device)
        def sus2field(x, d):
            return torch.real(torch.fft.ifftn(d * torch.fft.fftn(x)))

        if w is None:
            w = 1
        else:
            w = torch.Tensor(w).to(device)
        alpha = 1e-6
        d = torch.Tensor(continuous_dipole_kernel(phase.shape)).to(device)
        x = torch.zeros_like(phase).to(device)

        if hist:
            _hist = []

        if dipole:
            bank = torch.Tensor(gen_dipolets_filters(x.shape, gain_a=500, gain_r=500)).to(device)

        for it in range(iters):
            ans = x.clone()
            phi_x = sus2field(x, d)

            if dipole:
                x_d = torch.real(torch.fft.ifftn(torch.fft.ifftshift(bank, dim=(0, 1, 2)) * torch.fft.fftn(x, dim=(0, 1, 2)).unsqueeze(-1), dim=(0, 1, 2)))
                x = x - tau * sus2field(w * torch.sin(phi_x - phase), torch.conj(d)) - tau * alpha * x - tau *0.5* torch.sum(x_d[..., :3], dim=-1)
            else:
                x = x - tau * sus2field(w * torch.sin(phi_x - phase), torch.conj(d)) - tau * alpha * x

            update = rmse(x.cpu().numpy(), ans.cpu().numpy())
            step = {'update': update}
            print(f'{it} - Update: {update}', end='\t')
            if gt is not None and msk is not None:
                error = rmse(gt, x.cpu().numpy(), msk)
                print(f'RMSE={error: .2f}', end='')
                step['error'] = error
            if hist:
                _hist.append(step)
            print('')
        if hist:
            return x.cpu().numpy(), _hist
        return x.cpu().numpy()


    print(1)
    recon1, hist1 = ndi2(phase, iters=500, gt=gt, msk=mask, hist=True, dipole=True)
    imshow_3d(recon1, rango=(-0.1, 0.1), angles=(-90, -90, 90), title=f'NDI rmse={rmse(gt, recon1, mask):.2f}')
    imshow_3d(recon1*mask, rango=(-0.1, 0.1), angles=(-90, -90, 90), title=f'NDI rmse={rmse(gt, recon1, mask):.2f}')

    # %%
    bank = gen_dipolets_filters(N)
    recont = transformada_dipolet(recon1, gain_a=500, gain_r=500)
    for i in range(recont.shape[-1]):
        imshow_3d(bank[..., i], rango=(0, 1), angles=(-90, -90, 90))
        imshow_3d(recont[..., i], rango=(-0.05, 0.05), angles=(-90, -90, 90))


    # %%

    def gen_radial_filters(N: tuple[int, int, int], gain: float = 40, levels: int = 3) -> list[np.ndarray]:
        freq_x = fftshift(np.fft.fftfreq(N[0], d=1.0))
        freq_y = fftshift(np.fft.fftfreq(N[1], d=1.0))
        freq_z = fftshift(np.fft.fftfreq(N[2], d=1.0))
        omega_x, omega_y, omega_z = np.meshgrid(freq_x, freq_y, freq_z, indexing='ij')
        rho = np.sqrt(omega_x ** 2 + omega_y ** 2 + omega_z ** 2)

        fltrs = []
        ans = np.ones_like(rho)
        for i in range(1, levels + 1):
            aux = sigmoid_window(rho * (2 ** i), gain=gain, val=0.9)
            fltrs.append(ans - aux)
            ans = aux
        fltrs.append(ans)
        return fltrs


    radials = gen_radial_filters(recon1.shape, levels=3)
    cum = 0
    for ang_r in radials:
        cum += ang_r
        imshow_3d(ang_r, rango=(0, 1))

    bank = gen_dipolets_filters(N)
    cum = 0
    for i in range(bank.shape[-1]):
        ang_r = bank[..., i]
        cum += ang_r
        imshow_3d(ang_r, rango=(0, 1))
    imshow_3d(cum, rango=(0, 1))

    angulars = gen_angular_filters(recon1.shape, levels=6, gain=500)
    cum = 0
    for ang_f in angulars:
        cum += ang_f
        imshow_3d(ang_f, rango=(0, 1))