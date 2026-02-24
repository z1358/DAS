# KATANA: Simple Post-Training Robustness Using Test Time Augmentations
# https://arxiv.org/pdf/2109.08191v1.pdf
import PIL
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ColorJitter, Compose, Lambda
from numpy import random


class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, img):
        noise = torch.randn(img.size()) * self.std + self.mean
        noise = noise.to(img.device)
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Clip(torch.nn.Module):
    def __init__(self, min_val=0., max_val=1.):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, img):
        return torch.clip(img, self.min_val, self.max_val)

    def __repr__(self):
        return self.__class__.__name__ + '(min_val={0}, max_val={1})'.format(self.min_val, self.max_val)


class ColorJitterPro(ColorJitter):
    """Randomly change the brightness, contrast, saturation, and gamma correction of an image."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, gamma=0):
        super().__init__(brightness, contrast, saturation, hue)
        self.gamma = self._check_input(gamma, 'gamma')

    @staticmethod
    @torch.jit.unused
    def get_params(brightness, contrast, saturation, hue, gamma):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        if gamma is not None:
            gamma_factor = random.uniform(gamma[0], gamma[1])
            transforms.append(Lambda(lambda img: F.adjust_gamma(img, gamma_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx = torch.randperm(5)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = F.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = F.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = F.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = F.adjust_hue(img, hue_factor)

            if fn_id == 4 and self.gamma is not None:
                gamma = self.gamma
                gamma_factor = torch.tensor(1.0).uniform_(gamma[0], gamma[1]).item()
                img = img.clamp(1e-8, 1.0)  # to fix Nan values in gradients, which happens when applying gamma
                                            # after contrast
                img = F.adjust_gamma(img, gamma_factor)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        format_string += ', gamma={0})'.format(self.gamma)
        return format_string


def get_tta_transforms(img_size, gaussian_std: float=0.005, soft=False, padding_mode='edge', cotta_augs=True):
    n_pixels = img_size[0] if isinstance(img_size, (list, tuple)) else img_size

    tta_transforms = [
        Clip(0.0, 1.0),
        ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode=padding_mode),
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            interpolation=PIL.Image.BILINEAR,
            fill=0
        )
    ]
    if cotta_augs:
        tta_transforms += [transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
                           transforms.CenterCrop(size=n_pixels),
                           transforms.RandomHorizontalFlip(p=0.5),
                           GaussianNoise(0, gaussian_std),
                           Clip(0.0, 1.0)]
    else:
        tta_transforms += [transforms.CenterCrop(size=n_pixels),
                           transforms.RandomHorizontalFlip(p=0.5),
                           Clip(0.0, 1.0)]

    return transforms.Compose(tta_transforms)


import torch
import torch.fft as fft
import random
import torchvision.transforms as transforms # Keep for potential future use or type hinting

class FourierStyleTransferFDA(torch.nn.Module):
    """
    Applies Fourier Domain Adaptation (FDA) style transfer.
    Transfers the low-frequency amplitude spectrum (style) from a target
    image ('img_trg_style') to a source image ('img_src_content'),
    while preserving the source image's phase spectrum (content).

    Args:
        low_freq_ratio (float): The ratio of the smallest spatial dimension
                                to use as the radius for defining the
                                low-frequency region in the Fourier domain.
                                Example: 0.1 means 10% radius.
        eps (float): A small epsilon value added to amplitude before division
                     or log to avoid numerical instability (currently not used
                     in this direct replacement version, but good practice).
    """
    def __init__(self, low_freq_ratio=0.5, alpha=1.0, eps=1e-8):
        super().__init__()
        if not (0 < low_freq_ratio <= 0.5):
             raise ValueError("low_freq_ratio for square mask should be between 0 and 0.5.")
        self.low_freq_ratio = low_freq_ratio
        self.eps = eps # Retained for potential future use, though not needed for swap
        self.alpha = alpha

    def forward(self, img_src_content, img_trg_style):
        # Ensure inputs are 4D tensors and have the same shape
        # --- Input Validation ---
        if img_src_content.dim() != 4 or img_trg_style.dim() != 4:
            raise ValueError("Both input tensors must be 4D (B, C, H, W).")
        if img_src_content.shape[1:] != img_trg_style.shape[1:]:
             raise ValueError(f"Content and Style images must have the same C, H, W dimensions. "
                              f"Got {img_src_content.shape} and {img_trg_style.shape}")

        b_src, c, h, w = img_src_content.shape
        b_trg = img_trg_style.shape[0]
        device = img_src_content.device

        # Ensure target batch size is sufficient and trim if necessary
        if b_trg < b_src:
            raise ValueError(f"Target style batch size ({b_trg}) must be >= source content batch size ({b_src}).")
        if b_trg > b_src:
            img_trg_style = img_trg_style[:b_src] # Use only the first b_src style images

        # Compute Fast Fourier Transform (FFT) for both images
        fft_src = fft.fftshift(fft.fft2(img_src_content, dim=(-2, -1)))
        fft_trg = fft.fftshift(fft.fft2(img_trg_style, dim=(-2, -1)))

        # Extract amplitude and phase spectra
        amp_src, phase_src = torch.abs(fft_src), torch.angle(fft_src)
        amp_trg = torch.abs(fft_trg)
        # phase_trg = torch.angle(fft_trg) # Target phase is not needed

        # Create the SQUARE low-frequency mask
        half_side = int(min(h, w) * self.low_freq_ratio)
        if half_side < 1: half_side = 1 # Ensure at least 1 pixel half-side

        cy, cx = h // 2, w // 2

        # Define the boundaries of the square region
        y_start = max(0, cy - half_side)
        y_end = min(h, cy + half_side)
        x_start = max(0, cx - half_side)
        x_end = min(w, cx + half_side)

        # Create the mask: 1s inside the square, 0s outside
        low_freq_mask = torch.zeros((h, w), device=device)
        low_freq_mask[y_start:y_end, x_start:x_end] = 1.0

        # Expand dims (1, 1, H, W) for broadcasting
        low_freq_mask = low_freq_mask.unsqueeze(0).unsqueeze(0)
        # Calculate the high-frequency mask (complement)
        high_freq_mask = 1.0 - low_freq_mask

        lam = 1
        # Create the new amplitude spectrum by mixing low frequencies
        # Calculate the mixed amplitude in the low-frequency region
        amp_mixed_low = lam * amp_trg + (1 - lam) * amp_src
        # Combine: keep source high-freq amp, use mixed low-freq amp
        amp_new = amp_src * high_freq_mask + amp_mixed_low * low_freq_mask

        real_part = amp_new * torch.cos(phase_src)
        imag_part = amp_new * torch.sin(phase_src)
        fft_new = torch.complex(real_part, imag_part)

        # Apply Inverse Fast Fourier Transform (IFFT)
        img_reconstructed = fft.ifft2(fft.ifftshift(fft_new), dim=(-2, -1)).real

        # Clamp values to the valid range [0, 1]
        img_reconstructed = torch.clamp(img_reconstructed, 0.0, 1.0)

        return img_reconstructed
