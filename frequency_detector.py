import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import numpy as np
import os
from scipy import fftpack


class FrequencyDomainDetector:
    def __init__(self, msg_decoder_path, key_str):
        self.msg_decoder = torch.jit.load(msg_decoder_path)
        self.msg_decoder.eval()
        self.key = torch.tensor([int(b) for b in key_str])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.msg_decoder.to(self.device)

    def frequency_analysis(self, image):
        # Convert to YCbCr and extract Y channel
        ycbcr = image.convert('YCbCr')
        y, _, _ = ycbcr.split()
        y_np = np.array(y)

        # Apply 2D FFT
        fft = fftpack.fft2(y_np)
        fft_shift = fftpack.fftshift(fft)
        magnitude = np.abs(fft_shift)
        phase = np.angle(fft_shift)

        # Create feature vector from frequency components
        features = np.concatenate([
            magnitude.flatten()[:100],  # Use first 100 magnitude components
            phase.flatten()[:100]  # Use first 100 phase components
        ])

        return torch.tensor(features).float()

    def preprocess_image(self, image):
        # Enhanced preprocessing
        enhancer = ImageEnhance.Contrast(image)
        enhanced = enhancer.enhance(1.5)

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return transform(enhanced).unsqueeze(0).to(self.device)

    def detect_watermark(self, image_path):
        image = Image.open(image_path)

        # Get frequency domain features
        freq_features = self.frequency_analysis(image)

        # Get spatial domain features
        spatial_features = self.preprocess_image(image)

        with torch.no_grad():
            # Extract watermark
            decoded = self.msg_decoder(spatial_features)
            confidence = torch.sigmoid(decoded)

            # Adjust confidence based on frequency features
            freq_weight = 0.3
            adjusted_confidence = (1 - freq_weight) * confidence + \
                                  freq_weight * (freq_features.mean() / freq_features.std())

            # Get predictions
            pred = (adjusted_confidence > 0.5).float().cpu()

        # Calculate accuracy
        accuracy = (pred.squeeze() == self.key).float().mean().item()

        return {
            'accuracy': accuracy,
            'predicted_bits': pred.squeeze().tolist(),
            'confidence': adjusted_confidence.cpu().mean().item(),
            'is_watermarked': accuracy > 0.7
        }


def test_directory(detector, image_dir):
    for img_name in os.listdir(image_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            print(f"\nAnalyzing {img_name}...")
            img_path = os.path.join(image_dir, img_name)
            result = detector.detect_watermark(img_path)

            print(f"Accuracy: {result['accuracy']:.4f}")
            print(f"Confidence: {result['confidence']:.4f}")
            print("\nBit patterns:")
            print("Predicted:", ''.join(str(int(b)) for b in result['predicted_bits']))
            print("Expected: ", ''.join(str(int(b)) for b in detector.key.tolist()))


if __name__ == "__main__":
    key_str = "111010110101000001010111010011010100010000100111"
    detector = FrequencyDomainDetector("models/dec_48b_whit.torchscript.pt", key_str)
    test_directory(detector, "generated_images")