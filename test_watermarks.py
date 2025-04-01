# import torch
# from PIL import Image
# import torchvision.transforms as transforms
# from utils_model import load_model_from_config
# import os
#
#
# def extract_watermark(image_path, msg_decoder, key_str):
#     # Load and preprocess image
#     image = Image.open(image_path)
#     transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])
#     img = transform(image).unsqueeze(0)
#
#     # Extract watermark
#     with torch.no_grad():
#         decoded = msg_decoder(img)
#         pred = (decoded > 0).float()
#
#     # Compare with key
#     key = torch.tensor([int(b) for b in key_str])
#     bit_acc = (pred == key).float().mean().item()
#
#     return bit_acc, pred
#
#
# def test_images(image_dir, msg_decoder_path, key_str):
#     # Load decoder
#     msg_decoder = torch.jit.load(msg_decoder_path)
#     msg_decoder.eval()
#
#     # Test all images in directory
#     for img_name in os.listdir(image_dir):
#         if img_name.endswith(('.png', '.jpg', '.jpeg')):
#             img_path = os.path.join(image_dir, img_name)
#             bit_acc, pred = extract_watermark(img_path, msg_decoder, key_str)
#             print(f"Image: {img_name}")
#             print(f"Bit accuracy: {bit_acc:.4f}")
#             print("Predicted bits:", ''.join([str(int(b)) for b in pred.squeeze()]))
#             print("-" * 50)
#
#
# # Your key from training
# key_str = "111010110101000001010111010011010100010000100111"
#
# # Test both generated and original images
# test_images("generated_images", "models/dec_48b_whit.torchscript.pt", key_str)

import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from utils_model import load_model_from_config
import os
import io
import numpy as np


class WatermarkTester:
    def __init__(self, msg_decoder_path, key_str):
        self.msg_decoder = torch.jit.load(msg_decoder_path)
        self.msg_decoder.eval()
        self.key = torch.tensor([int(b) for b in key_str])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.msg_decoder.to(self.device)

    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return transform(image).unsqueeze(0).to(self.device)

    def extract_watermark(self, image):
        with torch.no_grad():
            img = self.preprocess_image(image)
            decoded = self.msg_decoder(img)
            pred = (decoded > 0).float()
        return pred.cpu()

    def get_accuracy(self, pred):
        return (pred.squeeze() == self.key).float().mean().item()

    # Different transformations
    def apply_jpeg_compression(self, image, quality=80):
        output = io.BytesIO()
        image.save(output, format='JPEG', quality=quality)
        return Image.open(output)

    def apply_rotation(self, image, angle):
        return image.rotate(angle, expand=True)

    def apply_crop(self, image, ratio=0.1):
        width, height = image.size
        crop_pixels_x = int(width * ratio)
        crop_pixels_y = int(height * ratio)
        return image.crop((crop_pixels_x, crop_pixels_y,
                           width - crop_pixels_x, height - crop_pixels_y))

    def apply_resize(self, image, scale=0.5):
        width, height = image.size
        new_size = (int(width * scale), int(height * scale))
        return image.resize(new_size, Image.LANCZOS)

    def apply_brightness(self, image, factor=1.5):
        return TF.adjust_brightness(image, factor)

    def apply_noise(self, image, std=0.1):
        img_array = np.array(image)
        noise = np.random.normal(0, std * 255, img_array.shape)
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

    def apply_blur(self, image, radius=2):
        return image.filter(ImageFilter.GaussianBlur(radius))


def test_image_robustness(image_path, tester):
    original_image = Image.open(image_path)
    results = {}

    # Test original
    pred = tester.extract_watermark(original_image)
    results['original'] = tester.get_accuracy(pred)

    # Test JPEG compression
    for quality in [80, 50, 30]:
        compressed = tester.apply_jpeg_compression(original_image, quality)
        pred = tester.extract_watermark(compressed)
        results[f'jpeg_q{quality}'] = tester.get_accuracy(pred)

    # Test rotations
    for angle in [5, 25, 90]:
        rotated = tester.apply_rotation(original_image, angle)
        pred = tester.extract_watermark(rotated)
        results[f'rotation_{angle}'] = tester.get_accuracy(pred)

    # Test cropping
    for ratio in [0.1, 0.2]:
        cropped = tester.apply_crop(original_image, ratio)
        pred = tester.extract_watermark(cropped)
        results[f'crop_{ratio}'] = tester.get_accuracy(pred)

    # Test resizing
    for scale in [0.3, 0.7]:
        resized = tester.apply_resize(original_image, scale)
        pred = tester.extract_watermark(resized)
        results[f'resize_{scale}'] = tester.get_accuracy(pred)

    # Test brightness
    for factor in [1.5, 2.0]:
        brightened = tester.apply_brightness(original_image, factor)
        pred = tester.extract_watermark(brightened)
        results[f'brightness_{factor}'] = tester.get_accuracy(pred)

    # Test noise
    for std in [0.1, 0.2]:
        noisy = tester.apply_noise(original_image, std)
        pred = tester.extract_watermark(noisy)
        results[f'noise_{std}'] = tester.get_accuracy(pred)

    # Test blur
    for radius in [2, 4]:
        blurred = tester.apply_blur(original_image, radius)
        pred = tester.extract_watermark(blurred)
        results[f'blur_{radius}'] = tester.get_accuracy(pred)

    return results


def test_directory(image_dir, msg_decoder_path, key_str):
    tester = WatermarkTester(msg_decoder_path, key_str)

    all_results = {}
    for img_name in os.listdir(image_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            print(f"\nTesting {img_name}...")
            img_path = os.path.join(image_dir, img_name)
            results = test_image_robustness(img_path, tester)

            # Print results for this image
            print(f"\nResults for {img_name}:")
            for transform, accuracy in results.items():
                print(f"{transform:15s}: {accuracy:.4f}")

            all_results[img_name] = results

    # Print average results across all images
    print("\nAverage results across all images:")
    avg_results = {}
    for transform in all_results[list(all_results.keys())[0]].keys():
        avg = np.mean([results[transform] for results in all_results.values()])
        print(f"{transform:15s}: {avg:.4f}")


if __name__ == "__main__":
    # Your key from training
    key_str = "111010110101000001010111010011010100010000100111"

    # Test both generated and original images
    test_directory("generated_images", "models/dec_48b_whit.torchscript.pt", key_str)