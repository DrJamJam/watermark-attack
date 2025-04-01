import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from utils_model import load_model_from_config
from omegaconf import OmegaConf


class WatermarkDetector:
    def __init__(self, msg_decoder_path, key_str, confidence_threshold=0.7):
        self.msg_decoder = torch.jit.load(msg_decoder_path)
        self.msg_decoder.eval()
        self.key = torch.tensor([int(b) for b in key_str])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.msg_decoder.to(self.device)
        self.confidence_threshold = confidence_threshold

    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return transform(image).unsqueeze(0).to(self.device)

    def detect_watermark(self, image_path):
        # Load and preprocess image
        image = Image.open(image_path)
        img_tensor = self.preprocess_image(image)

        # Extract watermark
        with torch.no_grad():
            decoded = self.msg_decoder(img_tensor)
            confidence = torch.sigmoid(decoded)
            pred = (confidence > self.confidence_threshold).float()

        # Calculate accuracy compared to known key
        bit_accuracy = (pred.cpu().squeeze() == self.key).float().mean().item()

        # Get confidence scores
        confidence_mean = confidence.mean().item()
        confidence_std = confidence.std().item()

        # Determine if watermarked based on accuracy threshold
        is_watermarked = bit_accuracy > 0.8  # You can adjust this threshold

        return {
            'is_watermarked': is_watermarked,
            'bit_accuracy': bit_accuracy,
            'confidence_mean': confidence_mean,
            'confidence_std': confidence_std,
            'predicted_bits': pred.cpu().squeeze().tolist(),
            'confidence_scores': confidence.cpu().squeeze().tolist()
        }


def test_directory(image_dir, detector):
    """Test all images in a directory for watermarks."""
    results = {}

    for img_name in os.listdir(image_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, img_name)
            print(f"\nAnalyzing {img_name}...")

            result = detector.detect_watermark(img_path)
            results[img_name] = result

            # Print detailed results
            print(f"Watermark detected: {result['is_watermarked']}")
            print(f"Bit accuracy: {result['bit_accuracy']:.4f}")
            print(f"Confidence mean: {result['confidence_mean']:.4f}")
            print(f"Confidence std: {result['confidence_std']:.4f}")

    return results


def save_results(results, output_path):
    """Save detection results to a file."""
    with open(output_path, 'w') as f:
        f.write("Watermark Detection Results\n")
        f.write("=" * 50 + "\n\n")

        for img_name, result in results.items():
            f.write(f"Image: {img_name}\n")
            f.write(f"Watermark detected: {result['is_watermarked']}\n")
            f.write(f"Bit accuracy: {result['bit_accuracy']:.4f}\n")
            f.write(f"Confidence mean: {result['confidence_mean']:.4f}\n")
            f.write(f"Confidence std: {result['confidence_std']:.4f}\n")
            f.write("-" * 50 + "\n")


if __name__ == "__main__":
    # Your key from training
    key_str = "111010110101000001010111010011010100010000100111"

    # Initialize detector
    detector = WatermarkDetector(
        msg_decoder_path="models/dec_48b_whit.torchscript.pt",
        key_str=key_str,
        confidence_threshold=0.7
    )

    # Test a single image
    test_image = "path/to/test/image.png"
    if os.path.exists(test_image):
        result = detector.detect_watermark(test_image)
        print("\nSingle image test results:")
        print(f"Watermark detected: {result['is_watermarked']}")
        print(f"Bit accuracy: {result['bit_accuracy']:.4f}")
        print(f"Confidence mean: {result['confidence_mean']:.4f}")

    # Test a directory of images
    test_dir = "generated_images"
    if os.path.exists(test_dir):
        print("\nTesting directory of images...")
        results = test_directory(test_dir, detector)
        save_results(results, "watermark_detection_results.txt")