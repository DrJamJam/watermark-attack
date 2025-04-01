from hybrid_detector import HybridDetector
import os


def test_detector():
    key_str = "111010110101000001010111010011010100010000100111"
    detector = HybridDetector("models/dec_48b_whit.torchscript.pt", key_str)

    test_dir = "generated_images"
    for img_name in os.listdir(test_dir):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            print(f"\nAnalyzing {img_name}...")
            img_path = os.path.join(test_dir, img_name)
            result = detector.detect_watermark(img_path)

            print(f"Accuracy: {result['accuracy']:.4f}")
            print(f"Overall Confidence: {result['confidence']:.4f}")
            print(f"Frequency Confidence: {result['freq_confidence']:.4f}")
            print(f"Pattern Confidence: {result['pattern_confidence']:.4f}")
            print(f"Reliability: {result['reliability']:.4f}")

            print("\nBit patterns:")
            print("Predicted:", ''.join(str(int(b)) for b in result['predicted_bits']))
            print("Expected: ", ''.join(str(int(b)) for b in detector.key.tolist()))

            # Show matches
            matches = ['1' if int(p) == int(e) else '0'
                       for p, e in zip(result['predicted_bits'], detector.key.tolist())]
            print("Matches:   ", ''.join(matches))


if __name__ == "__main__":
    test_detector()