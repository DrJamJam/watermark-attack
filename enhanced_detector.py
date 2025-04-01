import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from collections import Counter


class EnhancedWatermarkDetector:
    def __init__(self, msg_decoder_path, key_str):
        self.msg_decoder = torch.jit.load(msg_decoder_path)
        self.msg_decoder.eval()
        self.key = torch.tensor([int(b) for b in key_str])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.msg_decoder.to(self.device)
        self.window_size = 256

    def preprocess_image(self, image):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return transform(image).unsqueeze(0).to(self.device)

    def adaptive_threshold(self, confidence_scores):
        """Calculate adaptive threshold based on confidence distribution"""
        mean = confidence_scores.mean()
        std = confidence_scores.std()
        return mean + 0.5 * std

    def error_correction(self, predicted_bits, confidence_scores):
        """Apply error correction using neighboring bits"""
        # Convert tensor to numpy for easier manipulation
        corrected = predicted_bits.cpu().numpy()
        confidence_scores = confidence_scores.cpu().numpy()
        conf_threshold = np.mean(confidence_scores)

        # First pass: correct isolated errors
        for i in range(1, len(corrected) - 1):
            if confidence_scores[i] < conf_threshold:
                neighbors = [corrected[i - 1], corrected[i + 1]]
                if neighbors[0] == neighbors[1]:  # Only correct if neighbors agree
                    corrected[i] = neighbors[0]

        # Second pass: look at larger windows for error clusters
        window_size = 5
        for i in range(window_size // 2, len(corrected) - window_size // 2):
            window = corrected[i - window_size // 2:i + window_size // 2 + 1]
            if np.sum(window) >= window_size - 1:  # Most bits are 1
                corrected[i] = 1
            elif np.sum(window) <= 1:  # Most bits are 0
                corrected[i] = 0

        return torch.tensor(corrected, device=self.device)

    def sliding_window_detection(self, image):
        """Extract watermark using multiple overlapping windows"""
        width, height = image.size
        stride = self.window_size // 2
        predictions = []
        confidences = []

        for i in range(0, max(1, width - self.window_size + 1), stride):
            for j in range(0, max(1, height - self.window_size + 1), stride):
                window = image.crop((i, j, min(i + self.window_size, width),
                                     min(j + self.window_size, height)))
                window = window.resize((self.window_size, self.window_size))

                with torch.no_grad():
                    window_tensor = self.preprocess_image(window)
                    decoded = self.msg_decoder(window_tensor)
                    confidence = torch.sigmoid(decoded)
                    predictions.append((confidence > 0.5).float().cpu().squeeze())
                    confidences.append(confidence.cpu().squeeze())

        # Combine predictions
        stacked_predictions = torch.stack(predictions)
        stacked_confidences = torch.stack(confidences)

        # Weight predictions by their confidence
        weighted_pred = (stacked_predictions * stacked_confidences).sum(dim=0)
        weighted_pred = weighted_pred / stacked_confidences.sum(dim=0)

        return weighted_pred, stacked_confidences.mean(dim=0)

    def detect_watermark(self, image_path):
        image = Image.open(image_path)

        # Get predictions from sliding windows
        weighted_pred, avg_confidence = self.sliding_window_detection(image)

        # Apply adaptive threshold
        adaptive_thresh = self.adaptive_threshold(avg_confidence)
        initial_pred = (weighted_pred > adaptive_thresh).float()

        # Apply error correction
        corrected_pred = self.error_correction(initial_pred, avg_confidence)

        # Calculate accuracy with original and corrected predictions
        initial_accuracy = (initial_pred == self.key).float().mean().item()
        corrected_accuracy = (corrected_pred.cpu() == self.key).float().mean().item()

        return {
            'initial_accuracy': initial_accuracy,
            'corrected_accuracy': corrected_accuracy,
            'initial_bits': initial_pred.tolist(),
            'corrected_bits': corrected_pred.cpu().tolist(),
            'confidence_mean': avg_confidence.mean().item(),
            'confidence_std': avg_confidence.std().item(),
            'adaptive_threshold': adaptive_thresh.item(),
            'is_watermarked': corrected_accuracy > 0.7
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

            print(f"Initial accuracy: {result['initial_accuracy']:.4f}")
            print(f"Corrected accuracy: {result['corrected_accuracy']:.4f}")
            print(f"Confidence mean: {result['confidence_mean']:.4f}")
            print(f"Confidence std: {result['confidence_std']:.4f}")
            print(f"Adaptive threshold: {result['adaptive_threshold']:.4f}")

            print("\nBit patterns:")
            print("Initial:   ", ''.join(str(int(b)) for b in result['initial_bits']))
            print("Corrected: ", ''.join(str(int(b)) for b in result['corrected_bits']))
            print("Expected:  ", ''.join(str(int(b)) for b in detector.key.tolist()))

            # Show matches
            matches = ['1' if c == e else '0'
                       for c, e in zip(result['corrected_bits'], detector.key.tolist())]
            print("Matches:   ", ''.join(matches))

    return results


def save_results(results, output_path):
    """Save detection results to a file."""
    with open(output_path, 'w') as f:
        f.write("Watermark Detection Results\n")
        f.write("=" * 50 + "\n\n")

        for img_name, result in results.items():
            f.write(f"Image: {img_name}\n")
            f.write(f"Initial accuracy: {result['initial_accuracy']:.4f}\n")
            f.write(f"Corrected accuracy: {result['corrected_accuracy']:.4f}\n")
            f.write(f"Confidence mean: {result['confidence_mean']:.4f}\n")
            f.write(f"Confidence std: {result['confidence_std']:.4f}\n")
            f.write(f"Adaptive threshold: {result['adaptive_threshold']:.4f}\n")
            #
            # f.write(f"Watermark detected: {result['is_watermarked']}\n")
            # f.write(f"Bit accuracy: {result['bit_accuracy']:.4f}\n")
            # f.write(f"Confidence mean: {result['confidence_mean']:.4f}\n")
            # f.write(f"Confidence std: {result['confidence_std']:.4f}\n")
            f.write("-" * 50 + "\n")


if __name__ == "__main__":
    key_str = "111010110101000001010111010011010100010000100111"

    detector = EnhancedWatermarkDetector(
        msg_decoder_path="models/dec_48b_whit.torchscript.pt",
        key_str=key_str
    )

    test_dir = "generated_images"
    if os.path.exists(test_dir):
        print("\nTesting directory of images...")
        results = test_directory(test_dir, detector)
        save_results(results, "enhanced_watermark_detection_results.txt")
