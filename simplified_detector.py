import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os


class SimpleWatermarkDetector:
    def __init__(self, msg_decoder_path, key_str):
        self.msg_decoder = torch.jit.load(msg_decoder_path)
        self.msg_decoder.eval()
        self.key = torch.tensor([int(b) for b in key_str])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.msg_decoder.to(self.device)
        self.key = torch.tensor([int(b) for b in key_str]).to(self.device)

    def preprocess_image(self, image):
        """Basic image preprocessing"""
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return transform(image).unsqueeze(0).to(self.device)

    def detect_watermark(self, image_path):
        """Simple watermark detection"""
        image = Image.open(image_path)
        img_tensor = self.preprocess_image(image)

        with torch.no_grad():
            # Get decoder output
            decoded = self.msg_decoder(img_tensor)
            confidence = torch.sigmoid(decoded).squeeze()

            # Get predictions with different thresholds
            thresholds = [0.45, 0.48, 0.5, 0.52, 0.55]
            predictions = []
            for threshold in thresholds:
                pred = (confidence > threshold).float()
                pred = pred.to(self.device)
                accuracy = (pred == self.key).float().mean().item()
                predictions.append((pred, accuracy))

            # Select best prediction
            best_pred, best_acc = max(predictions, key=lambda x: x[1])

            return {
                'accuracy': best_acc,
                'predicted_bits': best_pred.cpu().tolist(),
                'confidence': confidence.mean().item(),
                'bit_confidences': confidence.cpu().tolist(),
                'is_watermarked': best_acc > 0.7,
            }


def test_detector():
    key_str = "111010110101000001010111010011010100010000100111"
    detector = SimpleWatermarkDetector("models/dec_48b_whit.torchscript.pt", key_str)

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