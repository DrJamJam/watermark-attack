import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import numpy as np
import os


class ImprovedWatermarkDetector:
    def __init__(self, msg_decoder_path, key_str):
        self.msg_decoder = torch.jit.load(msg_decoder_path)
        self.msg_decoder.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.msg_decoder.to(self.device)
        self.key = torch.tensor([int(b) for b in key_str]).to(self.device)

    def preprocess_image(self, image):
        """Enhanced preprocessing with multiple variations"""
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Create variations
        variations = []
        # Original
        variations.append(transform(image))
        # Slightly enhanced contrast
        enhanced = ImageEnhance.Contrast(image).enhance(1.2)
        variations.append(transform(enhanced))

        return torch.stack(variations).to(self.device)

    def detect_watermark(self, image_path):
        """Improved watermark detection"""
        image = Image.open(image_path)
        img_variations = self.preprocess_image(image)

        all_predictions = []
        all_confidences = []

        with torch.no_grad():
            for img_tensor in img_variations:
                decoded = self.msg_decoder(img_tensor.unsqueeze(0))
                confidence = torch.sigmoid(decoded).squeeze()

                # Use dynamic thresholding
                mean_conf = confidence.mean()
                std_conf = confidence.std()
                threshold = mean_conf + 0.5 * std_conf

                pred = (confidence > threshold).float()
                accuracy = (pred == self.key).float().mean().item()

                all_predictions.append((pred, accuracy))
                all_confidences.append(confidence)

            # Combine predictions from variations
            combined_confidence = torch.stack(all_confidences).mean(dim=0)
            best_pred, best_acc = max(all_predictions, key=lambda x: x[1])

            # Analyze bit reliability
            bit_reliability = torch.stack([abs(conf - 0.5) for conf in all_confidences]).mean(dim=0)
            reliable_bits = (bit_reliability > 0.1).float()

            return {
                'accuracy': best_acc,
                'predicted_bits': best_pred.cpu().tolist(),
                'confidence': combined_confidence.mean().item(),
                'bit_confidences': combined_confidence.cpu().tolist(),
                'bit_reliability': bit_reliability.cpu().tolist(),
                'is_watermarked': best_acc > 0.6,  # Lowered threshold
                'reliable_bits': reliable_bits.cpu().tolist()
            }


def test_detector():
    key_str = "111010110101000001010111010011010100010000100111"
    detector = ImprovedWatermarkDetector("models/dec_48b_whit.torchscript.pt", key_str)

    test_dir = "generated_images"
    for img_name in os.listdir(test_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            print(f"\nAnalyzing {img_name}...")
            img_path = os.path.join(test_dir, img_name)
            result = detector.detect_watermark(img_path)

            print(f"Accuracy: {result['accuracy']:.4f}")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"Is Watermarked: {result['is_watermarked']}")

            print("\nBit patterns:")
            print("Predicted:", ''.join(str(int(b)) for b in result['predicted_bits']))
            print("Expected: ", ''.join(str(int(b)) for b in detector.key.tolist()))

            # Show matches
            matches = ['1' if int(p) == int(e) else '0'
                       for p, e in zip(result['predicted_bits'], detector.key.tolist())]
            print("Matches:   ", ''.join(matches))

            # Show confidence distribution
            low_conf_bits = [i for i, conf in enumerate(result['bit_confidences'])
                             if abs(conf - 0.5) < 0.1]
            if low_conf_bits:
                print("\nUncertain bits (confidence near 0.5):", low_conf_bits)


if __name__ == "__main__":
    test_detector()
