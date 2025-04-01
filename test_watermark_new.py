import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os
import io
import itertools


class AdvancedWatermarkTester:
    def __init__(self, msg_decoder_path, key_str):
        self.msg_decoder = torch.jit.load(msg_decoder_path)
        self.msg_decoder.eval()
        self.key = torch.tensor([int(b) for b in key_str])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.msg_decoder.to(self.device)

    # Basic transformations (enhanced)
    def apply_jpeg_compression(self, image, quality=80):
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=quality)
        return Image.open(output)

    def apply_rotation(self, image, angle):
        return image.rotate(angle, expand=True, resample=Image.BILINEAR)

    def apply_blur(self, image, radius):
        return image.filter(ImageFilter.GaussianBlur(radius))

    def apply_noise(self, image, std):
        img_array = np.array(image)
        noise = np.random.normal(0, std * 255, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

    def apply_scale(self, image, scale):
        width, height = image.size
        return image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)

    # Additional transformations
    def apply_contrast(self, image, factor):
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    def apply_brightness(self, image, factor):
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def apply_sharpness(self, image, factor):
        enhancer = ImageEnhance.Sharpness(image)
        return enhancer.enhance(factor)

    def apply_color(self, image, factor):
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)

    def apply_perspective(self, image, distortion):
        width, height = image.size
        corners = [(0, 0), (width, 0), (width, height), (0, height)]
        new_corners = [
            (int(distortion * width), int(distortion * height)),
            (int((1 - distortion) * width), int(distortion * height)),
            (int((1 - distortion) * width), int((1 - distortion) * height)),
            (int(distortion * width), int((1 - distortion) * height))
        ]
        return image.transform(image.size, Image.PERSPECTIVE,
                               self._find_coeffs(corners, new_corners),
                               Image.BICUBIC)

    def _find_coeffs(self, pa, pb):
        matrix = []
        for p1, p2 in zip(pa, pb):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])
        A = np.matrix(matrix, dtype=np.float32)
        B = np.array(pb).reshape(8)
        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)

    # Combined attacks
    def apply_combined_attack(self, image, attacks):
        result = image
        for attack, params in attacks:
            result = attack(result, **params)
        return result

    def extract_watermark(self, image, confidence_threshold=0.5):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            decoded = self.msg_decoder(img)
            confidence = torch.sigmoid(decoded)
            pred = (confidence > confidence_threshold).float()

        return pred.cpu(), confidence.cpu()


def evaluate_image_quality(original_img, distorted_img):
    """Calculate image quality metrics with resizing to match dimensions."""
    original_array = np.array(original_img)
    distorted_array = np.array(distorted_img)

    # Get original dimensions
    orig_height, orig_width = original_array.shape[:2]

    # Resize distorted image to match original dimensions
    if distorted_array.shape != original_array.shape:
        distorted_img = Image.fromarray(distorted_array).resize((orig_width, orig_height), Image.LANCZOS)
        distorted_array = np.array(distorted_img)

    try:
        ssim_value = ssim(original_array, distorted_array, channel_axis=2)
    except:
        print(
            f"Warning: SSIM calculation failed. Original shape: {original_array.shape}, Distorted shape: {distorted_array.shape}")
        ssim_value = 0.0

    return ssim_value


# Update the test_advanced_robustness function to use this modified evaluation
def test_advanced_robustness(image_path, tester, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    original_image = Image.open(image_path)
    original_size = original_image.size
    results = {}

    # Single transformations
    single_attacks = {
        'jpeg_compression': [(30, 50, 70, 90)],
        'rotation': [(5, 15, 25, 45, 90)],
        'blur': [(1, 2, 3, 5)],
        'noise': [(0.01, 0.05, 0.1)],
        'scale': [(0.3, 0.5, 0.7, 0.9)],
        'contrast': [(0.5, 1.5, 2.0)],
        'brightness': [(0.5, 1.5, 2.0)],
        'sharpness': [(0.5, 2.0, 3.0)],
        'color': [(0.5, 1.5, 2.0)],
        'perspective': [(0.1, 0.2, 0.3)]
    }
    for attack_name, params_list in single_attacks.items():
        for params in params_list[0]:
            attack_func = getattr(tester, f'apply_{attack_name}')
            transformed = attack_func(original_image, params)

            # Resize transformed image back to original size for SSIM calculation
            if transformed.size != original_size:
                transformed_for_ssim = transformed.resize(original_size, Image.LANCZOS)
            else:
                transformed_for_ssim = transformed

            pred, conf = tester.extract_watermark(transformed)

            attack_key = f"{attack_name}_{params}"
            results[attack_key] = {
                'accuracy': (pred == tester.key).float().mean().item(),
                'confidence': conf.mean().item(),
                'ssim': evaluate_image_quality(original_image, transformed_for_ssim)
            }
            transformed.save(os.path.join(output_dir, f"{attack_key}.png"))
    combined_attacks = [
        [('jpeg_compression', {'quality': 70}), ('rotation', {'angle': 15})],
        [('blur', {'radius': 2}), ('noise', {'std': 0.05})],
        [('scale', {'scale': 0.7}), ('jpeg_compression', {'quality': 70})],
        [('rotation', {'angle': 15}), ('blur', {'radius': 2}), ('jpeg_compression', {'quality': 70})]
    ]

    for i, attack_sequence in enumerate(combined_attacks):
        transformed = original_image
        attack_name = '_'.join(a[0] for a in attack_sequence)

        for attack, params in attack_sequence:
            attack_func = getattr(tester, f'apply_{attack}')
            transformed = attack_func(transformed, **params)
            if transformed.size != original_size:
                transformed_for_ssim = transformed.resize(original_size, Image.LANCZOS)
            else:
                transformed_for_ssim = transformed

            pred, conf = tester.extract_watermark(transformed)
            results[f"combined_{i}_{attack_name}"] = {
                'accuracy': (pred == tester.key).float().mean().item(),
                'confidence': conf.mean().item(),
                'ssim': evaluate_image_quality(original_image, transformed_for_ssim)
            }
            transformed.save(os.path.join(output_dir, f"combined_{i}_{attack_name}.png"))

    # Save detailed results
    with open(os.path.join(output_dir, 'advanced_results.txt'), 'w') as f:
        f.write("Advanced Watermark Robustness Test Results\n")
        f.write("=" * 50 + "\n\n")

        # Single attacks
        f.write("Single Attacks:\n")
        f.write("-" * 30 + "\n")
        for attack, metrics in results.items():
            if not attack.startswith('combined'):
                f.write(f"{attack}:\n")
                f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"  Confidence: {metrics['confidence']:.4f}\n")
                f.write(f"  SSIM: {metrics['ssim']:.4f}\n")
                f.write("-" * 30 + "\n")

        # Combined attacks
        f.write("\nCombined Attacks:\n")
        f.write("-" * 30 + "\n")
        for attack, metrics in results.items():
            if attack.startswith('combined'):
                f.write(f"{attack}:\n")
                f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
                f.write(f"  Confidence: {metrics['confidence']:.4f}\n")
                f.write(f"  SSIM: {metrics['ssim']:.4f}\n")
                f.write("-" * 30 + "\n")

    return results


# def test_advanced_robustness(image_path, tester, output_dir):
#     os.makedirs(output_dir, exist_ok=True)
#     original_image = Image.open(image_path)
#     results = {}
#
#     # Single transformations
#     single_attacks = {
#         'jpeg_compression': [(30, 50, 70, 90)],
#         'rotation': [(5, 15, 25, 45, 90)],
#         'blur': [(1, 2, 3, 5)],
#         'noise': [(0.01, 0.05, 0.1)],
#         'scale': [(0.3, 0.5, 0.7, 0.9)],
#         'contrast': [(0.5, 1.5, 2.0)],
#         'brightness': [(0.5, 1.5, 2.0)],
#         'sharpness': [(0.5, 2.0, 3.0)],
#         'color': [(0.5, 1.5, 2.0)],
#         'perspective': [(0.1, 0.2, 0.3)]
#     }
#
#     # Test single attacks with different parameters
#     for attack_name, params_list in single_attacks.items():
#         for params in params_list[0]:
#             attack_func = getattr(tester, f'apply_{attack_name}')
#             transformed = attack_func(original_image, params)
#             pred, conf = tester.extract_watermark(transformed)
#
#             attack_key = f"{attack_name}_{params}"
#             results[attack_key] = {
#                 'accuracy': (pred == tester.key).float().mean().item(),
#                 'confidence': conf.mean().item(),
#                 'ssim': evaluate_image_quality(original_image, transformed_for_ssim)
#                 #ssim(np.array(original_image), np.array(transformed), channel_axis=2)
#             }
#             transformed.save(os.path.join(output_dir, f"{attack_key}.png"))
#
#     # Combined attacks
#     combined_attacks = [
#         [('jpeg_compression', {'quality': 70}), ('rotation', {'angle': 15})],
#         [('blur', {'radius': 2}), ('noise', {'std': 0.05})],
#         [('scale', {'scale': 0.7}), ('jpeg_compression', {'quality': 70})],
#         [('rotation', {'angle': 15}), ('blur', {'radius': 2}), ('jpeg_compression', {'quality': 70})]
#     ]
#
#     for i, attack_sequence in enumerate(combined_attacks):
#         transformed = original_image
#         attack_name = '_'.join(a[0] for a in attack_sequence)
#
#         for attack, params in attack_sequence:
#             attack_func = getattr(tester, f'apply_{attack}')
#             transformed = attack_func(transformed, **params)
#
#         pred, conf = tester.extract_watermark(transformed)
#         results[f"combined_{i}_{attack_name}"] = {
#             'accuracy': (pred == tester.key).float().mean().item(),
#             'confidence': conf.mean().item(),
#             'ssim': evaluate_image_quality(original_image, transformed_for_ssim)
#             #'ssim': ssim(np.array(original_image), np.array(transformed), channel_axis=2)
#         }
#         transformed.save(os.path.join(output_dir, f"combined_{i}_{attack_name}.png"))
#
#     # Save detailed results
#     with open(os.path.join(output_dir, 'advanced_results.txt'), 'w') as f:
#         f.write("Advanced Watermark Robustness Test Results\n")
#         f.write("=" * 50 + "\n\n")
#
#         # Single attacks
#         f.write("Single Attacks:\n")
#         f.write("-" * 30 + "\n")
#         for attack, metrics in results.items():
#             if not attack.startswith('combined'):
#                 f.write(f"{attack}:\n")
#                 f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
#                 f.write(f"  Confidence: {metrics['confidence']:.4f}\n")
#                 f.write(f"  SSIM: {metrics['ssim']:.4f}\n")
#                 f.write("-" * 30 + "\n")
#
#         # Combined attacks
#         f.write("\nCombined Attacks:\n")
#         f.write("-" * 30 + "\n")
#         for attack, metrics in results.items():
#             if attack.startswith('combined'):
#                 f.write(f"{attack}:\n")
#                 f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
#                 f.write(f"  Confidence: {metrics['confidence']:.4f}\n")
#                 f.write(f"  SSIM: {metrics['ssim']:.4f}\n")
#                 f.write("-" * 30 + "\n")
#
#     return results

# def test_directory(image_dir, msg_decoder_path, key_str):
#     """Test all images in a directory."""
#     tester = WatermarkTester(msg_decoder_path, key_str)
#
#     for img_name in os.listdir(image_dir):
#         if img_name.endswith(('.png', '.jpg', '.jpeg')):
#             print(f"\nTesting {img_name}...")
#             img_path = os.path.join(image_dir, img_name)
#             output_dir = os.path.join('results', os.path.splitext(img_name)[0])
#             results = test_image_robustness(img_path, tester, output_dir)
#
#             # Print results
#             print(f"\nResults for {img_name}:")
#             for transform, metrics in results.items():
#                 print(f"{transform:15s}: Accuracy={metrics['accuracy']:.4f}, "
#                       f"SSIM={metrics['ssim']:.4f}")


if __name__ == "__main__":
    # Your key from training
    key_str = "111010110101000001010111010011010100010000100111"

    # Initialize tester
    tester = AdvancedWatermarkTester("models/dec_48b_whit.torchscript.pt", key_str)

    # Test generated images
    image_dir = "generated_images"
    for img_name in os.listdir(image_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            print(f"\nTesting {img_name}...")
            img_path = os.path.join(image_dir, img_name)
            output_dir = os.path.join('advanced_results', os.path.splitext(img_name)[0])
            results = test_advanced_robustness(img_path, tester, output_dir)
            print(f"\nResults for {img_name}:")
            for transform, metrics in results.items():
                print(f"{transform:15s}: Accuracy={metrics['accuracy']:.4f}, "
                      f"SSIM={metrics['ssim']:.4f}")

# import torch
# import torchvision.transforms as transforms
# import torchvision.transforms.functional as TF
# from utils_model import load_model_from_config
# import os
# import io
# import numpy as np
# import cv2
# from PIL import Image, ImageFilter
# from skimage.metrics import structural_similarity as ssim
# import matplotlib.pyplot as plt
#
#
# class WatermarkTester:
#     def __init__(self, msg_decoder_path, key_str):
#         self.msg_decoder = torch.jit.load(msg_decoder_path)
#         self.msg_decoder.eval()
#         self.key = torch.tensor([int(b) for b in key_str])
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.msg_decoder.to(self.device)
#
#     def preprocess_image(self, image):
#         transform = transforms.Compose([
#             transforms.Resize(256),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#         ])
#         return transform(image).unsqueeze(0).to(self.device)
#
#     def extract_watermark(self, image):
#         with torch.no_grad():
#             img = self.preprocess_image(image)
#             decoded = self.msg_decoder(img)
#             pred = (decoded > 0).float()
#         return pred.cpu()
#
#     def get_accuracy(self, pred):
#         return (pred.squeeze() == self.key).float().mean().item()
#
#     # Enhanced transformations with quality metrics
#     def apply_jpeg_compression(self, image, quality=80):
#         output = io.BytesIO()
#         image.save(output, format='JPEG', quality=quality)
#         return Image.open(output)
#
#     def apply_rotation(self, image, angle):
#         return image.rotate(angle, expand=True)
#
#     def apply_gaussian_blur(self, image, kernel_size=5):
#         return image.filter(ImageFilter.GaussianBlur(radius=kernel_size / 2))
#
#     def apply_noise(self, image, var=0.01):
#         img_array = np.array(image)
#         noise = np.random.normal(0, var ** 0.5, img_array.shape)
#         noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
#         return Image.fromarray(noisy_img)
#
#     def apply_scaling(self, image, scale=0.5):
#         width, height = image.size
#         return image.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
#
#     def apply_translation(self, image, x_shift=10, y_shift=10):
#         return image.transform(image.size, Image.AFFINE, (1, 0, x_shift, 0, 1, y_shift))
#
#
# def evaluate_image_quality(original_img, distorted_img):
#     """Calculate image quality metrics."""
#     original_array = np.array(original_img)
#     distorted_array = np.array(distorted_img)
#
#     # Ensure images are the same size
#     if original_array.shape != distorted_array.shape:
#         distorted_array = cv2.resize(distorted_array, (original_array.shape[1], original_array.shape[0]))
#
#     mse = np.mean((original_array.astype("float") - distorted_array.astype("float")) ** 2)
#     ssim_value = ssim(original_array, distorted_array, multichannel=True)
#     return mse, ssim_value
#
#
# def visualize_transformations(original_image, transformations, output_dir):
#     """Create a grid of transformed images and save it."""
#     n = len(transformations) + 1  # +1 for original
#     cols = 4
#     rows = (n + cols - 1) // cols
#
#     plt.figure(figsize=(15, 5 * rows))
#
#     # Plot original
#     plt.subplot(rows, cols, 1)
#     plt.imshow(original_image)
#     plt.title('Original')
#     plt.axis('off')
#
#     # Plot transformations
#     for idx, (name, img) in enumerate(transformations.items(), 1):
#         plt.subplot(rows, cols, idx + 1)
#         plt.imshow(img)
#         plt.title(name)
#         plt.axis('off')
#
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'transformations_grid.png'))
#     plt.close()
#
#
# def test_image_robustness(image_path, tester, output_dir):
#     """Test watermark robustness with visualizations and saving."""
#     os.makedirs(output_dir, exist_ok=True)
#     original_image = Image.open(image_path)
#     results = {}
#     transformed_images = {}
#
#     # Define transformations with parameters
#     transformations = {
#         'jpeg_80': lambda img: tester.apply_jpeg_compression(img, 80),
#         'jpeg_50': lambda img: tester.apply_jpeg_compression(img, 50),
#         'rotation_25': lambda img: tester.apply_rotation(img, 25),
#         'rotation_90': lambda img: tester.apply_rotation(img, 90),
#         'blur_3': lambda img: tester.apply_gaussian_blur(img, 3),
#         'blur_5': lambda img: tester.apply_gaussian_blur(img, 5),
#         'noise_01': lambda img: tester.apply_noise(img, 0.01),
#         'noise_05': lambda img: tester.apply_noise(img, 0.05),
#         'scale_05': lambda img: tester.apply_scaling(img, 0.5),
#         'scale_07': lambda img: tester.apply_scaling(img, 0.7),
#         'translate': lambda img: tester.apply_translation(img, 20, 20)
#     }
#
#     # Apply transformations and collect results
#     for name, transform_fn in transformations.items():
#         # Apply transformation
#         transformed = transform_fn(original_image)
#         transformed_images[name] = transformed
#
#         # Save transformed image
#         transformed.save(os.path.join(output_dir, f'{name}.png'))
#
#         # Extract watermark and calculate accuracy
#         pred = tester.extract_watermark(transformed)
#         accuracy = tester.get_accuracy(pred)
#
#         # Calculate image quality metrics
#         mse, ssim_val = evaluate_image_quality(original_image, transformed)
#
#         results[name] = {
#             'accuracy': accuracy,
#             'mse': mse,
#             'ssim': ssim_val
#         }
#
#     # Create visualization
#     visualize_transformations(original_image, transformed_images, output_dir)
#
#     # Save results to text file
#     with open(os.path.join(output_dir, 'results.txt'), 'w') as f:
#         f.write("Watermark Robustness Test Results\n")
#         f.write("=" * 40 + "\n\n")
#         for name, metrics in results.items():
#             f.write(f"{name}:\n")
#             f.write(f"  Watermark Accuracy: {metrics['accuracy']:.4f}\n")
#             f.write(f"  MSE: {metrics['mse']:.2f}\n")
#             f.write(f"  SSIM: {metrics['ssim']:.4f}\n")
#             f.write("-" * 40 + "\n")
#
#     return results
#
#
# def test_directory(image_dir, msg_decoder_path, key_str):
#     """Test all images in a directory."""
#     tester = WatermarkTester(msg_decoder_path, key_str)
#
#     for img_name in os.listdir(image_dir):
#         if img_name.endswith(('.png', '.jpg', '.jpeg')):
#             print(f"\nTesting {img_name}...")
#             img_path = os.path.join(image_dir, img_name)
#             output_dir = os.path.join('results', os.path.splitext(img_name)[0])
#             results = test_image_robustness(img_path, tester, output_dir)
#
#             # Print results
#             print(f"\nResults for {img_name}:")
#             for transform, metrics in results.items():
#                 print(f"{transform:15s}: Accuracy={metrics['accuracy']:.4f}, "
#                       f"SSIM={metrics['ssim']:.4f}")
#
#
# if __name__ == "__main__":
#     # Your key from training
#     key_str = "111010110101000001010111010011010100010000100111"
#
#     # Test images
#     test_directory("generated_images", "models/dec_48b_whit.torchscript.pt", key_str)
