import numpy as np
from scipy import signal
from typing import Tuple


def logit(value: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute the logit function with a scaling factor.

    Parameters:
        value: Input array with values between 0 and 1
        alpha: Scaling factor for the logit function

    Returns:
        The logit-transformed array
    """
    return np.log(value / (1 - value)) / alpha


def sigmoid(value: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute the sigmoid function with a scaling factor.

    Parameters:
        value: Input array
        alpha: Scaling factor for the sigmoid function

    Returns:
        The sigmoid-transformed array
    """
    return 1.0 / (1.0 + np.exp(-alpha * value))


def dyadic_upsample_1d(data: np.ndarray, factor: int = 1) -> np.ndarray:
    """
    Perform dyadic upsampling on a 1D array by inserting zeros.

    Parameters:
        data: Input 1D array to be upsampled
        factor: Upsampling factor (power of 2), e.g., factor=1 means upsampling by 2^1=2

    Returns:
        Upsampled 1D array with zeros inserted between original samples
    """
    if factor <= 0:
        return data.copy()

    step = 2**factor
    upsampled = np.zeros(data.shape[0] * step, dtype=data.dtype)
    upsampled[::step] = data

    return upsampled


def dyadic_upsample_2d(image: np.ndarray, factor: int = 1) -> np.ndarray:
    """
    Perform dyadic upsampling on a 2D array by inserting zeros.

    Parameters:
        image: Input 2D array to be upsampled
        factor: Upsampling factor (power of 2), e.g., factor=1 means upsampling by 2^1=2

    Returns:
        Upsampled 2D array with zeros inserted between original samples
    """
    if factor <= 0:
        return image.copy()

    step = 2**factor
    upsampled = np.zeros((image.shape[0] * step, image.shape[1] * step), dtype=image.dtype)
    upsampled[::step, ::step] = image

    return upsampled


def dyadic_upsample_3d(volume: np.ndarray, factor: int = 1) -> np.ndarray:
    """
    Perform dyadic upsampling on a 3D array by inserting zeros.

    Parameters:
        volume: Input 3D array to be upsampled
        factor: Upsampling factor (power of 2), e.g., factor=1 means upsampling by 2^1=2

    Returns:
        Upsampled 3D array with zeros inserted between original samples
    """
    if factor <= 0:
        return volume.copy()

    step = 2**factor
    upsampled = np.zeros((volume.shape[0] * step, volume.shape[1] * step, volume.shape[2] * step), dtype=volume.dtype)
    upsampled[::step, ::step, ::step] = volume

    return upsampled


def dyadic_upsample(data: np.ndarray, factor: int = 1) -> np.ndarray:
    """
    Perform dyadic upsampling on an array by inserting zeros.
    This function handles 1D, 2D, and 3D arrays automatically.

    Parameters:
        data: Input array to be upsampled (1D, 2D, or 3D)
        factor: Upsampling factor (power of 2), e.g., factor=1 means upsampling by 2^1=2

    Returns:
        Upsampled array with zeros inserted between original samples
    """
    if factor <= 0:
        return data.copy()

    # Determine dimensionality and call appropriate function
    ndim = len(data.shape)

    if ndim == 1:
        return dyadic_upsample_1d(data, factor)
    elif ndim == 2:
        return dyadic_upsample_2d(data, factor)
    elif ndim == 3:
        return dyadic_upsample_3d(data, factor)
    else:
        raise ValueError(f"Function not implemented for {ndim}-dimensional arrays")


def subsample(image: np.ndarray) -> np.ndarray:
    """
    Subsample a 2D image by a factor of 2 with smoothing.

    Parameters:
        image: Input 2D array to be subsampled

    Returns:
        Subsampled 2D array with half the dimensions of the input
    """
    subsampled_image = convolve2d(image, np.ones((2, 2)) / 4.0, mode="same")
    subsampled_image = subsampled_image[::2, ::2]
    return subsampled_image


def subsample3d(volume: np.ndarray) -> np.ndarray:
    """
    Subsample a 3D volume by a factor of 2 with smoothing.

    Parameters:
        volume: Input 3D array to be subsampled

    Returns:
        Subsampled 3D array with half the dimensions of the input
    """
    subsampled_volume = convolve3d(volume, np.ones((2, 2, 2)) / 8.0, mode="same")
    subsampled_volume = subsampled_volume[::2, ::2, ::2]
    return subsampled_volume


def convolve2d(data: np.ndarray, kernel: np.ndarray, mode: str = "same") -> np.ndarray:
    """
    Perform 2D convolution with rotation correction.

    Parameters:
        data: Input 2D array
        kernel: Convolution kernel
        mode: Convolution mode, default is "same"

    Returns:
        The convolved 2D array
    """
    rotated_data = np.rot90(data, 2)
    rotated_kernel = np.rot90(kernel, 2)

    result = signal.convolve2d(rotated_data, rotated_kernel, mode=mode)
    result = np.rot90(result, 2)

    return result


def convolve3d(data: np.ndarray, kernel: np.ndarray, mode: str = "same") -> np.ndarray:
    """
    Perform 3D convolution with rotation correction.

    Parameters:
        data: Input 3D array
        kernel: Convolution kernel
        mode: Convolution mode, default is "same"

    Returns:
        The convolved 3D array
    """

    result = signal.convolve(data, kernel, mode=mode)

    return result


def get_daubechies_coefficients(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get Daubechies wavelet filter coefficients of the specified order.

    Parameters:
        order: Order of the Daubechies wavelet (2, 4, 6, etc.)

    Returns:
        A tuple containing:
            - low_pass: Low-pass (scaling) filter coefficients
            - high_pass: High-pass (wavelet) filter coefficients
    """
    # Common Daubechies coefficients
    if order == 2:  # db1 is equivalent to Haar
        low_pass = np.array([0.7071067811865476, 0.7071067811865476])
        high_pass = np.array([-0.7071067811865476, 0.7071067811865476])
    elif order == 4:  # db2
        low_pass = np.array([-0.12940952255126037, 0.2241438680420134, 0.8365163037378079, 0.48296291314453416])
        high_pass = np.array([-0.48296291314453416, 0.8365163037378079, -0.2241438680420134, -0.12940952255126037])
    elif order == 6:  # db3
        low_pass = np.array(
            [
                0.03522629188570953,
                -0.08544127388202666,
                -0.13501102001025458,
                0.45987750211849154,
                0.8068915093110925,
                0.33267055295008263,
            ]
        )
        high_pass = np.array(
            [
                -0.33267055295008263,
                0.8068915093110925,
                -0.45987750211849154,
                -0.13501102001025458,
                0.08544127388202666,
                0.03522629188570953,
            ]
        )
    elif order == 8:  # db4
        low_pass = np.array(
            [
                -0.010597401785069032,
                0.0328830116668852,
                0.030841381835560764,
                -0.18703481171909309,
                -0.027983769416859854,
                0.6308807679298589,
                0.7148465705529157,
                0.2303778133088965,
            ]
        )
        high_pass = np.array(
            [
                -0.2303778133088965,
                0.7148465705529157,
                -0.6308807679298589,
                -0.027983769416859854,
                0.18703481171909309,
                0.030841381835560764,
                -0.0328830116668852,
                -0.010597401785069032,
            ]
        )
    else:
        raise ValueError(f"Daubechies wavelet of order {order} is not implemented")

    return low_pass, high_pass


def get_symlet_coefficients(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get Symlet wavelet filter coefficients of the specified order.

    Parameters:
        order: Order of the Symlet wavelet (2, 3, 4, etc.)

    Returns:
        A tuple containing:
            - low_pass: Low-pass (scaling) filter coefficients
            - high_pass: High-pass (wavelet) filter coefficients
    """
    if order == 2:
        low_pass = np.array([-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025])
        high_pass = np.array([-0.48296291314469025, 0.836516303737469, -0.22414386804185735, -0.12940952255092145])
    elif order == 3:
        low_pass = np.array(
            [
                0.035226291882100656,
                -0.08544127388224149,
                -0.13501102001039084,
                0.4598775021193313,
                0.8068915093133388,
                0.3326705529509569,
            ]
        )
        high_pass = np.array(
            [
                -0.3326705529509569,
                0.8068915093133388,
                -0.4598775021193313,
                -0.13501102001039084,
                0.08544127388224149,
                0.035226291882100656,
            ]
        )
    elif order == 4:
        low_pass = np.array(
            [
                -0.07576571478927333,
                -0.02963552764599851,
                0.49761866763201545,
                0.8037387518059161,
                0.29785779560527736,
                -0.09921954357684722,
                -0.012603967262037833,
                0.0322231006040427,
            ]
        )
        high_pass = np.array(
            [
                -0.0322231006040427,
                -0.012603967262037833,
                0.09921954357684722,
                0.29785779560527736,
                -0.8037387518059161,
                0.49761866763201545,
                0.02963552764599851,
                -0.07576571478927333,
            ]
        )

    else:
        raise ValueError(f"Symlet wavelet of order {order} is not implemented")

    return low_pass, high_pass


def get_coiflet_coefficients(order: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get Coiflet wavelet filter coefficients of the specified order.

    Parameters:
        order: Order of the Coiflet wavelet (1 or 2), default is 1

    Returns:
        A tuple containing:
            - low_pass: Low-pass (scaling) filter coefficients
            - high_pass: High-pass (wavelet) filter coefficients
    """
    if order == 1:
        low_pass = np.array(
            [
                -0.015655728135791993,
                -0.07273261951252645,
                0.3848648468648578,
                0.8525720202116004,
                0.3378976624574818,
                -0.07273261951252645,
            ]
        )
        high_pass = np.array(
            [
                0.07273261951252645,
                0.3378976624574818,
                -0.8525720202116004,
                0.3848648468648578,
                0.07273261951252645,
                -0.015655728135791993,
            ]
        )
    elif order == 2:
        low_pass = np.array(
            [
                -0.000720549445520347,
                -0.0018232088709110323,
                0.005611434819368834,
                0.02368017194684777,
                -0.05943441864643109,
                -0.07648859907828076,
                0.4170051844232391,
                0.8127236354494135,
                0.3861100668227629,
                -0.0673725547237256,
                -0.04146493678687178,
                0.01638733646320364,
            ]
        )
        high_pass = np.array(
            [
                -0.01638733646320364,
                -0.04146493678687178,
                0.0673725547237256,
                0.3861100668227629,
                -0.8127236354494135,
                0.4170051844232391,
                0.07648859907828076,
                -0.05943441864643109,
                -0.02368017194684777,
                0.005611434819368834,
                0.0018232088709110323,
                -0.000720549445520347,
            ]
        )
    else:
        raise ValueError(f"Coiflet wavelet of order {order} is not implemented")

    return low_pass, high_pass


def wavelet_decompose(
    image: np.ndarray, low_pass: np.ndarray, high_pass: np.ndarray, number_of_scales: int
) -> np.ndarray:
    """
    Perform wavelet decomposition on a 2D image.

    Parameters:
        image: Input 2D array
        low_pass: Low-pass filter coefficients
        high_pass: High-pass filter coefficients
        number_of_scales: Number of scales for the wavelet decomposition

    Returns:
        Wavelet coefficients array with shape (image.shape[0], image.shape[1], 2*number_of_scales)
    """
    low_pass_0 = low_pass.copy()

    # Initialize coefficients array
    coefficients = np.zeros((*image.shape, 2 * number_of_scales))

    # Create working copy of the image
    img_copy = image.copy()
    for scale in range(1, number_of_scales + 1):
        # Create 2D filters for horizontal and vertical directions
        h_filter = np.outer(low_pass, high_pass)
        v_filter = np.outer(high_pass, low_pass)

        # Apply convolution for horizontal and vertical high-frequency components
        h_coeffs = convolve2d(img_copy, h_filter, mode="same")
        v_coeffs = convolve2d(img_copy, v_filter, mode="same")

        # Store coefficients
        coefficients[:, :, scale - 1] = h_coeffs
        coefficients[:, :, scale + number_of_scales - 1] = v_coeffs

        low_pass, high_pass = (
            signal.convolve(dyadic_upsample(low_pass), low_pass_0, "same"),
            signal.convolve(dyadic_upsample(high_pass), low_pass_0, "same"),
        )

    return coefficients


def wavelet_decompose3d(
    volume: np.ndarray, low_pass: np.ndarray, high_pass: np.ndarray, number_of_scales: int
) -> np.ndarray:
    """
    Perform wavelet decomposition on a 3D volume.

    Parameters:
        volume: Input 3D array
        low_pass: Low-pass filter coefficients
        high_pass: High-pass filter coefficients
        number_of_scales: Number of scales for the wavelet decomposition

    Returns:
        Wavelet coefficients array with shape (volume.shape[0], volume.shape[1], volume.shape[2], 3*number_of_scales)
    """
    low_pass_0 = low_pass.copy()

    # Initialize coefficients array
    coefficients = np.zeros((*volume.shape, 3 * number_of_scales))

    # Create working copy of the volume
    volume_copy = volume.copy()
    for scale in range(1, number_of_scales + 1):
        # Create 3D filters for x, y, z directions using outer products
        x_filter = np.einsum("i,j,k->ijk", high_pass, low_pass, low_pass)
        y_filter = np.einsum("i,j,k->ijk", low_pass, high_pass, low_pass)
        z_filter = np.einsum("i,j,k->ijk", low_pass, low_pass, high_pass)

        # Apply convolution for x, y, z high-frequency components
        x_coeffs = convolve3d(volume_copy, x_filter, mode="same")
        y_coeffs = convolve3d(volume_copy, y_filter, mode="same")
        z_coeffs = convolve3d(volume_copy, z_filter, mode="same")

        # Store coefficients
        coefficients[..., scale - 1] = x_coeffs
        coefficients[..., scale + number_of_scales - 1] = y_coeffs
        coefficients[..., scale + 2 * number_of_scales - 1] = z_coeffs

        low_pass, high_pass = (
            signal.convolve(dyadic_upsample(low_pass), low_pass_0, "same"),
            signal.convolve(dyadic_upsample(high_pass), low_pass_0, "same"),
        )

    return coefficients


def my_haar_psi_numpy(
    reference_image: np.ndarray,
    distorted_image: np.ndarray,
    preprocess_with_subsampling: bool = True,
    wavelet_type: str = "db",
    wavelet_order: int = 2,
    number_of_scales: int = 3,
    C: float = 30.0,
    alpha: float = 4.2,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calculate the Perceptual Similarity Index between two images or volumes using various wavelet types.

    This function supports 2D grayscale images, 2D color images and 3D grayscale volumes. It can use different types of wavelets for the decomposition.

    Parameters:
        reference_image: The reference image or volume
        distorted_image: The distorted image or volume to compare against the reference
        preprocess_with_subsampling: Whether to subsample the images before processing, default is True
        wavelet_type: Type of wavelet to use ("db", "sym", "coif"), default is "db"
        wavelet_order: Order of the wavelet, default is 1 (db1 equivalent to haar)
        number_of_scales: Number of scales for wavelet decomposition, default is 3
        C: Stability constant for similarity calculation, default is 30.0
        alpha: Scaling factor for sigmoid and logit functions, default is 4.2

    Returns:
        A tuple containing:
            - similarity: The similarity score (higher means more similar)
            - local_similarities: Local similarity maps for each orientation
            - weights: Weights maps used for similarity calculation
    """
    # Convert to float64
    reference_image = reference_image.astype(np.float64)
    distorted_image = distorted_image.astype(np.float64)

    reference_image_y = reference_image
    distorted_image_y = distorted_image

    # Apply subsampling if requested
    if preprocess_with_subsampling:
        reference_image_y = subsample3d(reference_image_y)
        distorted_image_y = subsample3d(distorted_image_y)


    coefficients_reference_image_y = wavelet_decompose3d(reference_image_y, low_pass, high_pass, number_of_scales)
    coefficients_distorted_image_y = wavelet_decompose3d(distorted_image_y, low_pass, high_pass, number_of_scales)

    local_similarities = np.zeros((*reference_image_y.shape, 3))
    weights = np.zeros((*reference_image_y.shape, 3))


    for orientation in range(3):
        weights[..., orientation] = np.maximum(
            np.abs(coefficients_reference_image_y[..., (number_of_scales - 1) + orientation * number_of_scales]),
            np.abs(coefficients_distorted_image_y[..., (number_of_scales - 1) + orientation * number_of_scales]),
        )
        coefficients_reference_image_y_magnitude = np.abs(
            coefficients_reference_image_y[..., (orientation * number_of_scales, 1 + orientation * number_of_scales)]
        )
        coefficients_distorted_image_y_magnitude = np.abs(
            coefficients_distorted_image_y[..., (orientation * number_of_scales, 1 + orientation * number_of_scales)]
        )
        local_similarities[..., orientation] = (
            np.sum(
                (2 * coefficients_reference_image_y_magnitude * coefficients_distorted_image_y_magnitude + C)
                / (coefficients_reference_image_y_magnitude**2 + coefficients_distorted_image_y_magnitude**2 + C),
                axis=-1,
            )
            / 2
        )

    # Calculates the final score
    similarity = logit(np.sum(sigmoid(local_similarities[:], alpha) * weights[:]) / np.sum(weights[:]), alpha) ** 2

    # Returns the result
    return similarity, local_similarities, weights
