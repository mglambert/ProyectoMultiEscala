import numpy as np
import matplotlib.pyplot as plt
from utils import continuous_dipole_kernel, imshow_3d, rmse
from nibabel import load as load_nii
from numpy.fft import fftn, ifftn, ifftshift, fftshift
from scipy.io import loadmat
from functools import cache

1 / 0


# %% Funciones
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_window(t, val=0.5, gain=40):
    return 1 - sigmoid(gain * (t - val))


def gen_radial_filters(N, gain=40, levels=3):
    freq_x = np.fft.fftshift(np.fft.fftfreq(N[0], d=1.0))
    freq_y = np.fft.fftshift(np.fft.fftfreq(N[1], d=1.0))
    freq_z = np.fft.fftshift(np.fft.fftfreq(N[2], d=1.0))
    omega_x, omega_y, omega_z = np.meshgrid(freq_x, freq_y, freq_z, indexing='ij')
    rho = np.sqrt(omega_x ** 2 + omega_y ** 2 + omega_z ** 2)

    fltrs = []
    ans = np.ones_like(rho)
    for i in range(1, levels + 1):
        aux = sigmoid_window(rho * (2 ** i), gain=gain)
        fltrs.append(ans - aux)
        ans = aux
    return fltrs


def gen_radial_squared_filters(N, gain=100, levels=3):
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


def gen_angular_filters(N, levels=3):
    kernel = continuous_dipole_kernel(N)
    k = np.fft.fftshift(kernel)
    ans = 0
    dipoles = []

    for i in range(levels):
        k2 = np.abs(k)
        new = sigmoid_window(k2, val=0.05 * (i + 1), gain=100)
        dipoles.append(new - ans)
        ans = new
    return dipoles


def normalize_filters(filters):
    sumbank = np.sum(filters, axis=-1, keepdims=True)
    filters = filters / np.max(sumbank)
    all_space = np.ones_like(filters[..., 0:1])
    residual = all_space - sumbank
    return np.clip(np.concatenate([filters, residual], axis=-1), 0, 1)


def gen_dipolets_filters(N, fun_radial=gen_radial_filters, gain=100, angular_levels=3, radial_levels=3):
    angulars = gen_angular_filters(N, levels=angular_levels)
    radials = fun_radial(N, gain, levels=radial_levels)
    bank = [a[..., np.newaxis] * r[..., np.newaxis] for a in angulars for r in radials]
    bank = np.concatenate(bank, axis=-1)
    bank = normalize_filters(bank)
    return bank


@cache
def transformada_dipolet(img):
    N = img.shape
    bank = gen_dipolets_filters(N, fun_radial=gen_radial_squared_filters, angular_levels=3, radial_levels=3, gain=50)
    result = []
    ftimg = fftn(img)
    for i in range(bank.shape[-1]):
        aux = np.real(ifftn(ifftshift(bank[..., i]) * ftimg))
        result.append(aux[..., np.newaxis])
    result = np.concatenate(result, axis=-1)
    return result


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
            err = rmse(true_image, x, mask)
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
        steps_return: bool = False,
        lamb_umb: float = 0.1
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
        xt[M == 1] = (np.abs(xt[M == 1]) < lamb_umb) * xt[M == 1]
        x = W_T(xt)

        tol = rmse(x, x_1)
        if verbose:
            if true_image is not None:
                print(f"NRMSE: {rmse(true_image, x, mask):.5f}")
        if steps_return:
            err = rmse(true_image, x, mask)
            errors.append((k, err))
        if tol < 1e-6:
            break
    if steps_return:
        return x, errors
    return x


# %%
N_y = 160  # Resolución x
N_x = 160  # Resolución y
N_z = 160  # Resolución y
N = [N_x, N_y, N_z]

# %%

flts_a = gen_angular_filters(N, 6)
for f in flts_a:
    imshow_3d(f, rango=(0, 1))

# %%

flts_r = gen_radial_filters(N, levels=6)
for f in flts_r:
    imshow_3d(f, rango=(0, 1))

# %%
flts_s = gen_radial_squared_filters(N, levels=4)
for f in flts_s:
    imshow_3d(f, rango=(0, 1))

# %%

fig, ax = plt.subplots(2, 3, figsize=(9, 6))

ax[0, 0].imshow(flts_a[0][N_x // 2, :, :], cmap='gray', vmin=0, vmax=1)
ax[1, 0].imshow(flts_a[1][N_x // 2, :, :], cmap='gray', vmin=0, vmax=1)
ax[0, 0].set_title('Kernel-based')
ax[0, 0].axis('off')
ax[1, 0].axis('off')
ax[0, 0].set_ylabel('Level 0')
ax[1, 0].set_ylabel('Level 1')

ax[0, 1].imshow(flts_r[0][N_x // 2, :, :], cmap='gray', vmin=0, vmax=1)
ax[1, 1].imshow(flts_r[1][N_x // 2, :, :], cmap='gray', vmin=0, vmax=1)
ax[0, 1].set_title('Radial')
ax[0, 1].axis('off')
ax[1, 1].axis('off')

ax[0, 2].imshow(flts_s[0][N_x // 2, :, :], cmap='gray', vmin=0, vmax=1)
ax[1, 2].imshow(flts_s[1][N_x // 2, :, :], cmap='gray', vmin=0, vmax=1)
ax[0, 2].set_title('Squared')
ax[0, 2].axis('off')
ax[1, 2].axis('off')

plt.tight_layout()
plt.show()

# %%

# bank = gen_dipolets_filters(N, fun_radial=gen_radial_filters, angular_levels=3, radial_levels=3, gain=100)
bank = gen_dipolets_filters(N, fun_radial=gen_radial_squared_filters, angular_levels=3, radial_levels=3, gain=100)
print(bank.shape)

# %%
for i in range(bank.shape[-1]):
    imshow_3d(bank[..., i])

# %%

fig, ax = plt.subplots(3, 4, figsize=(12, 9))
ax[0, 3].remove()
ax[1, 3].remove()

for i in range(3):
    for j in range(3):
        ax[i, j].imshow(bank[N_x // 2, :, :, 3 * i + j], cmap='gray', vmin=0, vmax=1)
        ax[i, j].axis('off')
ax[-1, -1].axis('off')
ax[-1, -1].set_title('Residual')
ax[-1, -1].imshow(bank[N_x//2, :, :, -1], cmap='gray', vmin=0, vmax=1)
plt.suptitle('Squared Dipole filters', fontsize=20)
plt.tight_layout()
plt.show()

# %%

mask = loadmat('msk.mat')['msk']
magn = loadmat('magn.mat')['magn']
gt = loadmat('chi_cosmos.mat')['chi_cosmos']

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

phase[N_x // 2, N_y // 2, N_z // 2] *= 10
phase[N_x // 2, N_y // 2 + 1, N_z // 2] *= -10

phase[N_x // 2, N_y // 2, N_z // 2 + 1] *= 10
phase[N_x // 2, N_y // 2, N_z // 2] *= -10

phase[N_x // 2, N_y // 2, N_z // 2] *= 10
phase[N_x // 2 - 1, N_y // 2 + 1, N_z // 2] *= -10

 # %%

W = transformada_dipolet
aux = W(phase)
W_T = lambda x: np.sum(x, axis=-1)
M = np.ones(aux.shape)
# M[:, :, :, -1] = 0.5

H = lambda x: np.real(ifftn(kernel * fftn(x)))
H_T = lambda x: np.real(ifftn(np.conj(kernel) * fftn(x)))

# %%
# recon_fista, steps = fista(phase, 0, 1, 10, M, W, W_T, H=H, H_T=H_T, true_image=gt,
#                            verbose=True, steps_return=True)
#
# imshow_3d(recon_fista, rango=(-0.1, 0.1), angles=(-90, -90, 90), title=f'recon rmse={rmse(gt, recon_fista, mask):.2f}')

# %%

recon_baseline, steps = ndi(phase, 50, H=H, H_T=H_T, true_image=gt,
                            verbose=True, steps_return=True)

imshow_3d(recon_baseline, rango=(-0.1, 0.1), angles=(-90, -90, 90),
          title=f'recon_baseline rmse={rmse(gt, recon_baseline, mask):.2f}')

# %%

recon_recon2, steps2 = ndi2(phase, 50, M, W, W_T, H=H, H_T=H_T, true_image=gt,
                            verbose=True, steps_return=True, lamb_umb=0.1)

imshow_3d(recon_recon2, rango=(-0.1, 0.1), angles=(-90, -90, 90),
          title=f'recon_recon rmse={rmse(gt, recon_recon2, mask):.2f}')

# %%

imshow_3d(recon_baseline - recon_recon2, rango=(-0.1, 0.1), angles=(-90, -90, 90), title=f'Baseline - dipolet')
imshow_3d(gt - recon_recon2, rango=(-0.1, 0.1), angles=(-90, -90, 90), title=f'dipolet')
imshow_3d(gt - recon_baseline, rango=(-0.1, 0.1), angles=(-90, -90, 90), title=f'Baseline')

# %%
from utils import rotation_by_permutarion

trd = transformada_dipolet(input:=recon_baseline)


fig, ax = plt.subplots(3, 4, figsize=(12, 9))
ax[1, 3].remove()

for i in range(3):
    for j in range(3):
        ax[i, j].imshow(rotation_by_permutarion(trd[N_x // 2, :, :, 3 * i + j], -90), cmap='gray', vmin=-0.08, vmax=0.08)
        ax[i, j].axis('off')

ax[0, -1].axis('off')
ax[0, -1].set_title('Input')
ax[0, -1].imshow(rotation_by_permutarion(input[N_x//2, :, :], -90), cmap='gray', vmin=-0.08, vmax=0.08)

ax[-1, -1].axis('off')
ax[-1, -1].set_title('Residual')
ax[-1, -1].imshow(rotation_by_permutarion(trd[N_x//2, :, :, -1], -90), cmap='gray', vmin=-0.08, vmax=0.08)

plt.suptitle('COSMOS with streaking', fontsize=20)
plt.tight_layout()
plt.show()





# %%

trd = transformada_dipolet(input := phase)

fig, ax = plt.subplots(3, 4, figsize=(12, 9))
ax[1, 3].remove()

for i in range(3):
    for j in range(3):
        ax[i, j].imshow(rotation_by_permutarion(trd[N_x // 2, :, :, 3 * i + j], -90), cmap='gray', vmin=-0.08,
                        vmax=0.08)
        ax[i, j].axis('off')

ax[0, -1].axis('off')
ax[0, -1].set_title('Input')
ax[0, -1].imshow(rotation_by_permutarion(input[N_x // 2, :, :], -90), cmap='gray', vmin=-0.08, vmax=0.08)

ax[-1, -1].axis('off')
ax[-1, -1].set_title('Residual')
ax[-1, -1].imshow(rotation_by_permutarion(trd[N_x // 2, :, :, -1], -90), cmap='gray', vmin=-0.08, vmax=0.08)

plt.suptitle('Local field with phase jump', fontsize=20)
plt.tight_layout()
plt.show()


# %%
# %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
phase = loadmat('data_blend.mat')['phase']
mask = loadmat('data_blend.mat')['mask']
phs_scale = loadmat('data_blend.mat')['phs_scale']
imshow_3d(phase * mask, rango=(-0.05, 0.05), angles=(-90, -90, -90), title='In-vivo local field')

# %%
trd = transformada_dipolet(input:=phase * mask)

fig, ax = plt.subplots(3, 4, figsize=(12, 7))
ax[1, 3].remove()

for i in range(3):
    for j in range(3):
        ax[i, j].imshow(rotation_by_permutarion(trd[N_x // 2, :, :, 3 * i + j], -90), cmap='gray', vmin=-0.05, vmax=0.05)
        ax[i, j].axis('off')

ax[0, -1].axis('off')
ax[0, -1].set_title('Input')
ax[0, -1].imshow(rotation_by_permutarion(input[N_x//2, :, :], -90), cmap='gray', vmin=-0.05, vmax=0.05)

ax[-1, -1].axis('off')
ax[-1, -1].set_title('Residual')
ax[-1, -1].imshow(rotation_by_permutarion(trd[N_x//2, :, :, -1], -90), cmap='gray', vmin=-0.05, vmax=0.05)

plt.suptitle('In-vivo local field', fontsize=20)
plt.tight_layout()
plt.show()

#%%

W = transformada_dipolet
aux = W(phase)
W_T = lambda x: np.sum(x, axis=-1)
M = np.ones(aux.shape)
M[:, :, :, -1] = 0
kernel = continuous_dipole_kernel(phase.shape)

H = lambda x: np.real(ifftn(kernel * fftn(x)))
H_T = lambda x: np.real(ifftn(np.conj(kernel) * fftn(x)))


recon_baseline = ndi(phase*mask, 10, H=H, H_T=H_T)

imshow_3d(recon_baseline, rango=(-0.1, 0.1), angles=(-90, -90, -90), title='In-vivo reconstruction')
#%%

trd = transformada_dipolet(input:=recon_baseline * mask)

fig, ax = plt.subplots(3, 4, figsize=(12, 7))
ax[1, 3].remove()

for i in range(3):
    for j in range(3):
        ax[i, j].imshow(rotation_by_permutarion(trd[N_x // 2, :, :, 3 * i + j], -90), cmap='gray', vmin=-0.05, vmax=0.05)
        ax[i, j].axis('off')

ax[0, -1].axis('off')
ax[0, -1].set_title('Input')
ax[0, -1].imshow(rotation_by_permutarion(input[N_x//2, :, :], -90), cmap='gray', vmin=-0.05, vmax=0.05)

ax[-1, -1].axis('off')
ax[-1, -1].set_title('Residual')
ax[-1, -1].imshow(rotation_by_permutarion(trd[N_x//2, :, :, -1], -90), cmap='gray', vmin=-0.05, vmax=0.05)

plt.suptitle('In-vivo reconstruction', fontsize=20)
plt.tight_layout()
plt.show()


# %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

mask = loadmat('msk.mat')['msk']
magn = loadmat('magn.mat')['magn']
gt = loadmat('chi_cosmos.mat')['chi_cosmos']
kernel = continuous_dipole_kernel(mask.shape)

# %%
phase = np.real(ifftn(kernel * fftn(gt)))

magn = magn * mask

scale = np.pi / np.max(np.abs(phase))
signal = magn * np.exp(1j * phase * scale)
_rr = np.random.rand()

snr = 50

signal = signal + ((1. / snr) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)))

phase = np.angle(signal).astype(np.float32) / scale
phase = phase * mask

phase[N_x // 2, N_y // 2, N_z // 2] *= 150
phase[N_x // 2, N_y // 2 + 1, N_z // 2] *= -150

phase[N_x // 3, N_y // 2, N_z // 2 + 1] *= 150
phase[N_x // 3, N_y // 2, N_z // 2] *= -150

phase[N_x // 2, N_y // 2, N_z // 3] *= 150
phase[N_x // 2 - 1, N_y // 2 + 1, N_z // 3] *= -150

phase[N_x // 2, N_y // 2-1, 2*N_z // 3] *= -150
phase[N_x // 2, N_y // 2, 2*N_z // 3] *= 150
phase[N_x // 2 - 1, N_y // 2 + 1, 2*N_z // 3] *= -150

imshow_3d(phase)

 # %%

W = transformada_dipolet
aux = W(phase)
W_T = lambda x: np.sum(x, axis=-1)
M = np.ones(aux.shape)
# M[:, :, :, -1] = 0.5

H = lambda x: np.real(ifftn(kernel * fftn(x)))
H_T = lambda x: np.real(ifftn(np.conj(kernel) * fftn(x)))

# %%

recon_baseline, steps = ndi(phase, 8, H=H, H_T=H_T, true_image=gt,
                            verbose=True, steps_return=True)

imshow_3d(recon_baseline, rango=(-0.1, 0.1), angles=(-90, -90, 90),
          title=f'recon_baseline rmse={rmse(gt, recon_baseline, mask):.2f}')

# %%

recon_recon2, steps2 = ndi2(phase, 17, M, W, W_T, H=H, H_T=H_T, true_image=gt,
                            verbose=True, steps_return=True, lamb_umb=0.12)

imshow_3d(recon_recon2, rango=(-0.1, 0.1), angles=(-90, -90, 90),
          title=f'recon_recon rmse={rmse(gt, recon_recon2, mask):.2f}')

#%%

imshow_3d(np.abs(recon_baseline-recon_recon2), rango=(0, 0.1), angles=(-90, -90, 90),
          title=f'|GD-Proposed|')
#%%

plt.figure(figsize=(7, 5))

# Extraer coordenadas
x1, y1 = zip(*steps)
x2, y2 = zip(*steps2)

# Graficar ambas líneas
plt.plot(x1, y1, label='GD')
plt.plot(x2, y2, label='Proposed')

# Encontrar el mínimo de cada línea
min_idx_1 = y1.index(min(y1))
min_idx_2 = y2.index(min(y2))

# Puntos mínimos
plt.plot(x1[min_idx_1], y1[min_idx_1], 'ro')  # punto rojo para GD
plt.plot(x2[min_idx_2], y2[min_idx_2], 'ro')  # punto rojo para Proposed

# Etiquetas con el valor mínimo (3 decimales)
plt.text(x1[min_idx_1]*1.01, y1[min_idx_1]*1.01, f'{y1[min_idx_1]:.3f}', ha='left', va='bottom', fontsize=9)
plt.text(x2[min_idx_2]*1.01, y2[min_idx_2]*1.01, f'{y2[min_idx_2]:.3f}', ha='left', va='bottom', fontsize=9)

# Título y etiquetas
plt.legend()
plt.title('RMSE vs iterations')
plt.xlabel('Iterations')
plt.ylabel('RMSE')
plt.tight_layout()
plt.show()
