import numpy as np
import matplotlib.pyplot as plt
from utils import continuous_dipole_kernel, imshow_3d, rmse
from nibabel import load as load_nii
from numpy.fft import fftn, ifftn, ifftshift, fftshift
from scipy.io import loadmat
1/0
#%%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_window(t, val=0.5):
    return 1 - sigmoid(10 * (t - val))


def gen_radial_filters(N, levels=3):
    freq_x = np.fft.fftshift(np.fft.fftfreq(N[0], d=1.0))
    freq_y = np.fft.fftshift(np.fft.fftfreq(N[1], d=1.0))
    freq_z = np.fft.fftshift(np.fft.fftfreq(N[2], d=1.0))
    omega_x, omega_y, omega_z = np.meshgrid(freq_x, freq_y, freq_z, indexing='ij')
    rho = np.sqrt(omega_x ** 2 + omega_y ** 2 + omega_z ** 2)

    fltrs = []
    ans = sigmoid_window(rho * (2 ** 0))
    for i in range(1, levels + 1):
        aux = sigmoid_window(rho * (2 ** i))
        fltrs.append(ans - aux)
        ans = aux
    return fltrs


def gen_angular_filters(N, levels=3):
    kernel = continuous_dipole_kernel(N)
    k = np.fft.fftshift(kernel)
    ans = 0
    dipoles = []

    for i in range(levels):
        k2 = np.abs(k)
        new = sigmoid_window(k2, 0.07 * (2 ** i))
        dipoles.append(new - ans)
        ans = new
    return dipoles


def normalize_filters(filters):
    sumbank = np.sum(filters, axis=-1, keepdims=True)
    filters = filters / np.max(sumbank)
    all_space = np.ones_like(filters[..., 0:1])
    residual = all_space - sumbank
    return np.concatenate([filters, residual], axis=-1)


def gen_dipolets_filters(N):
    angulars = gen_angular_filters(N)
    radials = gen_radial_filters(N)
    bank = [a[..., np.newaxis] * r[..., np.newaxis] for a in angulars for r in radials]
    bank = np.concatenate(bank, axis=-1)
    bank = normalize_filters(bank)
    return bank


N_y = 160  # Resolución x
N_x = 160  # Resolución y
N_z = 160  # Resolución y
N = [N_x, N_y, N_z]

# %%

bank = gen_dipolets_filters(N)
print(bank.shape)

# %%
for i in range(bank.shape[-1]):
    imshow_3d(bank[..., i])

# %%

img = load_nii('streaking.nii').get_fdata()

imshow_3d(img, rango=(-0.1, 0.1), angles=(-90, -90, 90), title='imagen utilizada')

mask = loadmat('msk.mat')['msk']
magn = loadmat('magn.mat')['magn']
gt = loadmat('chi_cosmos.mat')['chi_cosmos']

print(rmse(gt, img, mask))

# %%

ftimg = fftn(img)
for i in range(bank.shape[-1]):
    aux = np.real(ifftn(ifftshift(bank[..., i]) * ftimg))
    imshow_3d(bank[..., i], title=f'filtro {i}')
    imshow_3d(aux, rango=(-0.1, 0.1), angles=(-90, -90, 90), title=f'filtro {i}')


# %%

def transformada_dipolet(img):
    N = img.shape
    bank = gen_dipolets_filters(N)
    result = []
    ftimg = fftn(img)
    for i in range(bank.shape[-1]):
        aux = np.real(ifftn(ifftshift(bank[..., i]) * ftimg))
        result.append(aux[..., np.newaxis])
    result = np.concatenate(result, axis=-1)
    return result


aux = transformada_dipolet(img)
recon = np.sum(aux, axis=-1)

# %%
from typing import List, Callable, Optional, Tuple


def soft_thresholding(x: np.ndarray, threshold: Optional[np.ndarray]) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0.0)


def fista(
        y_observed: np.ndarray,
        lambda_reg: float,
        L: float,
        n_iter: int,
        M: np.ndarray,
        W: Callable[[np.ndarray], np.ndarray],
        W_T: Callable[[np.ndarray], np.ndarray],
        H: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        H_T: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        true_image: Optional[np.ndarray] = None,
        verbose: bool = False,
        steps_return: bool = False,
) -> np.ndarray:
    if H is None:
        H = lambda x: x
    if H_T is None:
        H_T = lambda x: x

    x_k = np.zeros_like(y_observed)
    y_k_fista = x_k.copy()
    t_k = 1.0

    if L <= 0:
        raise ValueError("Lipschitz constant L must be positive.")

    if steps_return:
        errors = []

    for k in range(n_iter):
        grad_f_yk = H_T(H(y_k_fista) - y_observed)

        v_k = y_k_fista - (1.0 / L) * grad_f_yk

        x_k_1 = x_k.copy()
        x_k = W_T(soft_thresholding(W(v_k), lambda_reg * M / L))

        t_k_plus_1 = (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2)) / 2.0

        y_k_fista = x_k + ((t_k - 1.0) / t_k_plus_1) * (x_k - x_k_1)

        t_k = t_k_plus_1

        tol = rmse(x_k_1, x_k)
        if verbose:
            print(f"Iteration {k}: {tol}")
            if k == n_iter - 1 or tol < 1e-6:
                cost_f = 0.5 * np.sum((H(x_k) - y_observed) ** 2)

                cost_g = lambda_reg * np.sum(np.abs(M * W(x_k)))
                total_cost = cost_f + cost_g
                print(
                    f"Iter {k}/{n_iter}: Total Cost = {total_cost:.4f} (Fidelity = {cost_f:.4f}, Regularization = {cost_g:.4f})")
                if true_image is not None:
                    print(f"NRMSE: {rmse(true_image, x_k, mask):.5f}")

        if steps_return:
            err = rmse(true_image, x_k, mask)
            errors.append((k, err))
        if tol < 1e-6:
            break

    if steps_return:
        return x_k, errors
    return x_k


def ndi(
        y_observed: np.ndarray,
        n_iter: int,
        alpha: float = 1,
        H: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        H_T: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        true_image: Optional[np.ndarray] = None,
        verbose: bool = False,
        steps_return: bool = False
) -> np.ndarray:
    if H is None:
        H = lambda x: x
    if H_T is None:
        H_T = lambda x: x

    if steps_return:
        errors = []

    x = np.zeros_like(y_observed)

    for k in range(n_iter):
        x_1 = x.copy()
        x = x - alpha * (H_T(H(x) - y_observed))

        tol = rmse(x, x_1)
        if verbose:
            if true_image is not None:
                print(f"NRMSE: {rmse(true_image, x, mask):.5f}")
        if steps_return:
            err = rmse(x, true_image)
            errors.append((k, err))
        if tol < 1e-6:
            break
    if steps_return:
        return x, errors
    return x


def ndi2(
        y_observed: np.ndarray,
        n_iter: int,
        M: np.ndarray,
        W: Callable[[np.ndarray], np.ndarray],
        W_T: Callable[[np.ndarray], np.ndarray],
        alpha: float = 1,
        H: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        H_T: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        true_image: Optional[np.ndarray] = None,
        verbose: bool = False,
        steps_return: bool = False
) -> np.ndarray:
    if H is None:
        H = lambda x: x
    if H_T is None:
        H_T = lambda x: x

    if steps_return:
        errors = []

    x = np.zeros_like(y_observed)

    for k in range(n_iter):
        x_1 = x.copy()
        x = x - alpha * (H_T(H(x) - y_observed))

        xt = W(x)
        xt[M == 1] = (np.abs(xt[M == 1]) < 0.1) * xt[M == 1]
        x = W_T(xt)

        tol = rmse(x, x_1)
        if verbose:
            if true_image is not None:
                print(f"NRMSE: {rmse(true_image, x, mask):.5f}")
        if steps_return:
            err = rmse(x, true_image)
            errors.append((k, err))
        if tol < 1e-6:
            break
    if steps_return:
        return x, errors
    return x


# %%
kernel = continuous_dipole_kernel(N)
phase = np.real(ifftn(kernel * fftn(gt)))

magn = magn * mask

scale = np.pi / np.max(np.abs(phase))
signal = magn * np.exp(1j * phase * scale)
_rr = np.random.rand()

snr = 100

signal = signal + ((1. / snr) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)))

phase = np.angle(signal).astype(np.float32) / scale
phase = phase * mask

phase[N_x // 2, N_y // 2, N_z // 2] *= 20
phase[N_x // 2, N_y // 2 + 1, N_z // 2] *= -20

phase[N_x // 2, N_y // 2, N_z // 2 + 1] *= 20
phase[N_x // 2, N_y // 2, N_z // 2] *= -20

phase[N_x // 2, N_y // 2, N_z // 2] *= 20
phase[N_x // 2 - 1, N_y // 2 + 1, N_z // 2] *= -20

# %%

W = transformada_dipolet
W_T = lambda x: np.sum(x, axis=-1)
M = np.ones((*phase.shape, 10))
M[:, :, :, -1] = 0

H = lambda x: np.real(ifftn(kernel * fftn(x)))
H_T = lambda x: np.real(ifftn(np.conj(kernel) * fftn(x)))

# %%
recon_fista, steps = fista(phase, 0, 1, 10, M, W, W_T, H=H, H_T=H_T, true_image=gt,
                           verbose=True, steps_return=True)

imshow_3d(recon_fista, rango=(-0.1, 0.1), angles=(-90, -90, 90), title=f'recon rmse={rmse(gt, recon_fista, mask):.2f}')

# %%

recon_baseline, steps = ndi(phase, 10, H=H, H_T=H_T, true_image=gt,
                            verbose=True, steps_return=True)

imshow_3d(recon_baseline, rango=(-0.1, 0.1), angles=(-90, -90, 90),
          title=f'recon_baseline rmse={rmse(gt, recon_baseline, mask):.2f}')

# %%


recon_recon2, steps = ndi2(phase, 10, M, W, W_T, H=H, H_T=H_T, true_image=gt,
                           verbose=True, steps_return=True)

imshow_3d(recon_recon2, rango=(-0.1, 0.1), angles=(-90, -90, 90),
          title=f'recon_recon rmse={rmse(gt, recon_recon2, mask):.2f}')

# %%
trd = transformada_dipolet(recon_baseline)

for i in range(trd.shape[-1]):
    imshow_3d(trd[..., i], title=f'{i}', rango=(-0.1, 0.1), angles=(-90, -90, 90))
