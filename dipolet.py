import numpy as np
import matplotlib.pyplot as plt
from utils import continuous_dipole_kernel, imshow_3d, rmse
from nibabel import load as load_nii
from numpy.fft import fftn, ifftn, ifftshift, fftshift
from scipy.io import loadmat
from functools import cache


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def sigmoid_window(t: np.ndarray, val: float = 0.5, gain: float = 40) -> np.ndarray:
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


def gen_radial_squared_filters(N: tuple[int, int, int], gain: float = 100, levels: int = 3) -> list[np.ndarray]:
    freq_x = np.fft.fftshift(np.fft.fftfreq(N[0], d=1.0))
    freq_y = np.fft.fftshift(np.fft.fftfreq(N[1], d=1.0))
    freq_z = np.fft.fftshift(np.fft.fftfreq(N[2], d=1.0))
    omega_x, omega_y, omega_z = np.meshgrid(freq_x, freq_y, freq_z, indexing='ij')
    omega_x, omega_y, omega_z = np.abs(omega_x), np.abs(omega_y), np.abs(omega_z)
    fltrs = []
    ans = np.ones_like(omega_x)
    for i in range(1, levels + 1):
        aux = (1. * (np.sqrt(sigmoid_window(omega_x * (2 ** i), gain=gain)) > 0.5) *
               1. * (np.sqrt(sigmoid_window(omega_y * (2 ** i), gain=gain)) > 0.5) *
               np.sqrt(sigmoid_window(omega_z * (2 ** i), gain=gain)))
        fltrs.append(ans - aux)
        ans = aux
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

    # %%
    N = (160, 160, 160)
    angular_filter = gen_angular_filters(N, levels=3)
    cum = 0
    for filt in angular_filter:
        cum += filt
        imshow_3d(filt, rango=(0, 1))  # , angles=(-90, -90, 90))
    imshow_3d(cum, rango=(0, 1))  # , angles=(-90, -90, 90))

    # %%

    bank = gen_dipolets_filters(N, angular_levels=3, radial_levels=3, gain_a=100, gain_r=50)
    for i in range(bank.shape[-1]):
        imshow_3d(bank[..., i])

    img = load_nii('streaking.nii').get_fdata()
    mask = loadmat('msk.mat')['msk']

    imgf = transformada_dipolet(img * mask, angular_levels=3, radial_levels=3, gain_a=100, gain_r=50)
    for i in range(imgf.shape[-1]):
        imshow_3d(bank[..., i], rango=(0, 1), angles=(-90, -90, 90))
        imshow_3d(imgf[..., i] * mask, rango=(-0.1, 0.1), angles=(-90, -90, 90))

    recon = np.sum(imgf, axis=-1)
    print(rmse(img, recon))


    # %%

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


    angulars = gen_angular_filters(img.shape, levels=3)
    cum = 0
    for ang_f in angulars:
        cum += ang_f
        imshow_3d(ang_f, rango=(0, 1))
        # imshow_3d(np.real(ifftn(ifftshift(ang_f) * fftn(img))), rango=(-0.1, 0.1), angles=(-90, -90, 90))
    cum = 1 - cum
    imshow_3d(cum, rango=(0, 1))
    imshow_3d(np.real(ifftn(ifftshift(cum) * fftn(img))), rango=(-0.1, 0.1), angles=(-90, -90, 90))


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
            aux = sigmoid_window(rho * (2 ** i), gain=gain)
            fltrs.append(ans - aux)
            ans = aux
        fltrs.append(ans)
        return fltrs


    radials = gen_radial_filters(img.shape, levels=3)
    cum = 0
    for ang_r in radials:
        cum += ang_r
        imshow_3d(ang_r, rango=(0, 1))


    # %%
    def gen_dipolets_filters(N, fun_radial=gen_radial_filters, gain_a=100, gain_r=40, angular_levels=3,
                             radial_levels=3):
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


    bank = gen_dipolets_filters(N)
    cum = 0
    for i in range(bank.shape[-1]):
        ang_r = bank[..., i]
        cum += ang_r
        imshow_3d(ang_r, rango=(0, 1))
    imshow_3d(cum, rango=(0, 1))

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
    _m = np.abs(kernel) > 0.1
    aux = fftn(phase)
    aux[_m] = aux[_m] / kernel[_m]
    recon = np.real(ifftn(aux))
    imshow_3d(recon, rango=(-0.1, 0.1), angles=(-90, -90, 90))

    recont = transformada_dipolet(recon)
    for i in range(recont.shape[-1]):
        imshow_3d(recont[..., i], rango=(-0.1, 0.1), angles=(-90, -90, 90))
    imshow_3d(np.sum(recont, axis=-1), rango=(-0.1, 0.1), angles=(-90, -90, 90))

    # %%
    M = np.ones_like(recont)
    M[..., 0] = 0
    M[..., 1] = 0
    M[..., 2] = 0
    imshow_3d(np.sum(recont * M, axis=-1), rango=(-0.1, 0.1), angles=(-90, -90, 90))
    print(rmse(gt, np.sum(recont * M, axis=-1), mask))

    # %%
    recon2 = load_nii('streaking.nii').get_fdata()

    imshow_3d(recon2, rango=(-0.1, 0.1), angles=(-90, -90, 90))

    recont = transformada_dipolet(recon2)
    for i in range(recont.shape[-1]):
        imshow_3d(recont[..., i], rango=(-0.1, 0.1), angles=(-90, -90, 90))
    imshow_3d(np.sum(recont, axis=-1), rango=(-0.1, 0.1), angles=(-90, -90, 90))

    # %%
    M = np.ones_like(recont)
    M[..., 0] = 0
    M[..., 1] = 0
    M[..., 2] = 0
    # M[..., 3] = 0
    # M[..., 4] = 0
    # M[..., 5] = 0
    # M[..., 6] = 0
    imshow_3d(np.sum(recont * M, axis=-1), rango=(-0.1, 0.1), angles=(-90, -90, 90))
    print(rmse(gt, np.sum(recont * M, axis=-1), mask))


    # %%

    def soft_threshold(x: np.ndarray, lambda_: float) -> np.ndarray:
        if lambda_ < 0:
            raise ValueError("El umbral (lambda_) no puede ser negativo.")
        magnitude = np.maximum(0., np.abs(x) - lambda_)
        return np.sign(x) * magnitude


    def hard_threshold(x: np.ndarray, lambda_: float) -> np.ndarray:
        if lambda_ < 0:
            raise ValueError("El umbral (lambda_) no puede ser negativo.")

        mask = np.abs(x) > lambda_
        return np.where(mask, x, 0.0)


    x = np.linspace(-5, 5, 100)
    plt.figure()
    plt.plot(x, soft_threshold(x, 2))
    plt.plot(x, hard_threshold(x, 2))
    plt.show()

#%%

    def ndi(phase, iters=10, tau=2, w=None, dipole=False, dip_thr=0.15, gt=None, msk=None, hist=False):
        def sus2field(x, d):
            return np.real(ifftn(d * fftn(x)))

        if w is None:
            w = 1
        alpha = 1e-6
        d = continuous_dipole_kernel(phase.shape)
        x = np.zeros_like(phase)
        if hist:
            _hist = []

        for _ in range(iters):
            ans = x.copy()
            phi_x = sus2field(x, d)
            x = x - tau * sus2field(w * np.sin(phi_x - phase), np.conj(d)) - tau * alpha * x

            if dipole:
                x_d = transformada_dipolet(x, gain_a=500, gain_r=500)

                # print(np.max(np.abs(x_d[..., 0])))
                x_d = np.clip(x_d, a_min=-dip_thr, a_max=dip_thr)
                x_d[..., 0] = np.clip(x_d[..., 0], a_min=-dip_thr*0.01, a_max=dip_thr*0.01)
                x_d[..., 1] = np.clip(x_d[..., 1], a_min=-dip_thr*0.02, a_max=dip_thr*0.02)
                x_d[..., 2] = np.clip(x_d[..., 2], a_min=-dip_thr*0.5, a_max=dip_thr*0.5)
                x_d[..., 3] = np.clip(x_d[..., 3], a_min=-dip_thr*0.5, a_max=dip_thr*0.5)
                x_d[..., 4] = np.clip(x_d[..., 4], a_min=-dip_thr*0.5, a_max=dip_thr*0.5)
                x_d[..., 5] = np.clip(x_d[..., 5], a_min=-dip_thr*0.5, a_max=dip_thr*0.5)
                x_d[..., 6] = np.clip(x_d[..., 6], a_min=-dip_thr*0.5, a_max=dip_thr*0.5)
                # x_d[..., 4] = np.clip(x_d[..., 4], a_min=-dip_thr*0.5, a_max=dip_thr*0.5)
                # print(np.max(np.abs(x_d[..., 0])))

                x = np.sum(x_d, axis=-1)

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

    print(1)
    recon1, hist1 = ndi(phase, iters=10, gt=gt, msk=mask, hist=True)
    imshow_3d(recon1, rango=(-0.1, 0.1), angles=(-90, -90, 90), title=f'NDI rmse={rmse(gt, recon1, mask):.2f}')
    print(2)
    recon2, hist2 = ndi(phase, iters=10, gt=gt, msk=mask, dipole=True, hist=True)
    imshow_3d(recon2, rango=(-0.1, 0.1), angles=(-90, -90, 90), title=f'NDI+dipolet rmse={rmse(gt, recon2, mask):.2f}')


#%%
    fig, ax = plt.subplots(2, 1, figsize=(5, 7))
    ax[0].plot([x['update'] for x in hist1], label='ndi')
    ax[0].plot([x['update'] for x in hist2], label='ndi+dipolet')
    ax[1].plot(er1:=[x['error'] for x in hist1], label='ndi')
    ax[1].plot(np.argmin(er1), np.min(er1), 'ro')
    ax[1].plot(er2:=[x['error'] for x in hist2], label='ndi+dipolet')
    ax[1].plot(np.argmin(er2), np.min(er2), 'ro')
    ax[0].legend()
    ax[0].set_title('Update')
    ax[1].legend()
    ax[1].set_title('RMSE')
    ax[0].set_xlabel('iteraciones')
    ax[1].set_xlabel('iteraciones')
    plt.tight_layout()
    plt.show()

#%%
    recon3, hist3 = ndi(phase, iters=17, gt=gt, msk=mask, hist=True)
    imshow_3d(recon3, rango=(-0.1, 0.1), angles=(-90, -90, 90), title=f'NDI rmse={rmse(gt, recon3, mask):.2f}')

#%%
    imshow_3d((recon1-recon3), rango=(-0.05, 0.05), angles=(-90, -90, 90), title='diferencia')
    imshow_3d(np.abs(recon1-recon3), rango=(0, 0.05), angles=(-90, -90, 90), title='diferencia absoluta')

#%%
    recont = transformada_dipolet(recon2, gain_a=500, gain_r=500)
    for i in range(recont.shape[-1]):
        imshow_3d(recont[..., i], rango=(-0.05, 0.05), angles=(-90, -90, 90))