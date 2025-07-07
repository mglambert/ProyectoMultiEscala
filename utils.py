import numpy as np
import matplotlib.pyplot as plt


def rotation_by_permutarion(im, angle):
    if angle == 90:
        im = im.T
    elif angle == -90:
        im = im.T
        im = im[::-1, :]
        return im
    elif angle == 180:
        im = im[:, :-1]
    return im


def vertical_plot_comp(images_list, rango=(-0.1, 0.1), rot=(0, 0, 0), fsize=5, return_img=False, images=(1, 1, 1)):
    sub_images = []
    for image in images_list:
        D, H, W = image.shape
        im1 = rotation_by_permutarion(image[D // 2, :, :], rot[0])
        im2 = rotation_by_permutarion(image[:, H // 2, :], rot[1])
        im3 = rotation_by_permutarion(image[:, :, W // 2], rot[2])
        max_width = np.max([im1.shape[1], im2.shape[1], im3.shape[1]])

        if im1.shape[1] < max_width:
            padding_width = max_width - im1.shape[1]
            pad_before = padding_width // 2
            pad_after = padding_width - pad_before
            im1 = np.pad(im1, ((0, 0), (pad_before, pad_after)), mode='constant', constant_values=0)
        rows_with_content = np.any(im1 != 0, axis=1)
        im1 = im1[rows_with_content]

        if im2.shape[1] < max_width:
            padding_width = max_width - im2.shape[1]
            pad_before = padding_width // 2
            pad_after = padding_width - pad_before
            im2 = np.pad(im2, ((0, 0), (pad_before, pad_after)), mode='constant', constant_values=0)
        rows_with_content = np.any(im2 != 0, axis=1)
        im2 = im2[rows_with_content]

        if im3.shape[1] < max_width:
            padding_width = max_width - im3.shape[1]
            pad_before = padding_width // 2
            pad_after = padding_width - pad_before
            im3 = np.pad(im3, ((0, 0), (pad_before, pad_after)), mode='constant', constant_values=0)
        rows_with_content = np.any(im3 != 0, axis=1)
        im3 = im3[rows_with_content]

        if np.sum(images) == 1:
            idx = np.argmax(images)
            im = im1 if idx == 0 else im2 if idx == 1 else im3
        elif np.sum(images) == 3:
            im = np.concatenate((im1, im2, im3), axis=0)
        else:
            _im = [x for x, i in zip([im1, im2, im3], images) if i == 1]
            im = np.concatenate(_im, axis=0)

        cols_with_content = np.any(im != 0, axis=0)
        im = im[:, cols_with_content]

        sub_images.append(im)
    im = np.concatenate(sub_images, axis=1)
    plt.figure(figsize=(fsize * len(images_list), 3 * fsize))
    plt.imshow(im, cmap='gray', vmin=rango[0], vmax=rango[1])
    plt.show()
    if return_img:
        return im


def rmse(gt, pred, mask=None):
    if mask is None:
        mask = np.ones_like(gt)
    return 100 * np.linalg.norm(pred[mask == 1] - gt[mask == 1]) / np.linalg.norm(gt[mask == 1])


def continuous_dipole_kernel(N, voxel_size=(1, 1, 1), B0_dir=(0, 0, 1)):
    rx = np.arange(-np.floor(N[0] / 2), np.ceil(N[0] / 2))
    ry = np.arange(-np.floor(N[1] / 2), np.ceil(N[1] / 2))
    rz = np.arange(-np.floor(N[2] / 2), np.ceil(N[2] / 2))

    kx, ky, kz = np.meshgrid(rx, ry, rz, indexing='ij')
    kx /= (np.max(np.abs(kx)) * voxel_size[0])
    ky /= (np.max(np.abs(ky)) * voxel_size[1])
    kz /= (np.max(np.abs(kz)) * voxel_size[2])

    k2 = kx ** 2 + ky ** 2 + kz ** 2
    # k2[k2 == 0] = np.finfo(np.float32).eps
    kernel = np.fft.ifftshift(
        1 / 3.0 - ((kx * B0_dir[0] + ky * B0_dir[1] + kz * B0_dir[2]) ** 2) / (k2 + np.finfo(np.float64).eps))
    kernel[0, 0, 0] = 0
    return kernel

def imshow_3d(image, title=None, cmap='gray', rango=None, angles=None):

    if rango is None:
        rango = (image.min(), image.max())
    if angles is None:
        angles = (0, 0, 0)

    D, H, W = image.shape

    # Create a figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Axial view (top-down)
    ax1.imshow(rotation_by_permutarion(image[D // 2, :, :], angles[0]), cmap=cmap, aspect='equal', vmin=rango[0], vmax=rango[1])
    # ax1.set_title('Axial View')
    ax1.axis('off')

    # Coronal view (front)
    ax2.imshow(rotation_by_permutarion(image[:, H // 2, :], angles[1]), cmap=cmap, aspect='equal', vmin=rango[0], vmax=rango[1])
    # ax2.set_title('Coronal View')
    ax2.axis('off')

    # Sagittal view (side)
    ax3.imshow(rotation_by_permutarion(image[:, :, W // 2], angles[2]), cmap=cmap, aspect='equal', vmin=rango[0], vmax=rango[1])
    # ax3.set_title('Sagittal View')
    ax3.axis('off')

    if title:
        fig.suptitle(title, fontsize=32)

    plt.tight_layout()
    plt.show()