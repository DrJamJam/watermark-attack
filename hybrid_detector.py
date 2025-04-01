import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import numpy as np
import cv2
from scipy import fftpack
import os
from itertools import product
from sklearn.ensemble import VotingClassifier
from itertools import product
from scipy.stats import skew, kurtosis
# Import the detection utils
from detection_utils import calculate_frequency_features, calculate_texture_features


class HybridDetector:
    def __init__(self, msg_decoder_path, key_str):
        self.msg_decoder = torch.jit.load(msg_decoder_path)
        self.msg_decoder.eval()
        self.key = torch.tensor([int(b) for b in key_str])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.msg_decoder.to(self.device)
        self.window_size = 256

    def advanced_frequency_analysis(self, image):
        # Convert to different color spaces
        gray = np.array(image.convert('L'))
        ycrcb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2YCrCb)

        all_features = []
        # Multi-scale FFT
        for scale in [1.0, 0.5, 0.25]:
            scaled_gray = cv2.resize(gray, None, fx=scale, fy=scale)
            fft = np.fft.fft2(scaled_gray)
            fft_shift = np.fft.fftshift(fft)

            # Use detection_utils for frequency features
            freq_features = calculate_frequency_features(fft_shift)
            all_features.append(freq_features)

            # Also calculate for Y channel of YCrCb
            y_channel = ycrcb[:, :, 0]
            scaled_y = cv2.resize(y_channel, None, fx=scale, fy=scale)
            y_fft = np.fft.fft2(scaled_y)
            y_fft_shift = np.fft.fftshift(y_fft)
            y_freq_features = calculate_frequency_features(y_fft_shift)
            all_features.append(y_freq_features)

        # Concatenate all features
        combined_features = np.concatenate(all_features)
        return torch.tensor(combined_features, dtype=torch.float32)


    def local_pattern_analysis(self, image):
        # Convert to grayscale
        gray = np.array(image.convert('L'))

        # Use detection_utils for texture features
        texture_features = calculate_texture_features(gray)

        # Calculate for different scales
        scales = [0.75, 0.5]
        for scale in scales:
            scaled_gray = cv2.resize(gray, None, fx=scale, fy=scale)
            scaled_features = calculate_texture_features(scaled_gray)
            texture_features = np.concatenate([texture_features, scaled_features])

        return torch.tensor(texture_features, dtype=torch.float32).to(self.device)


    def enhanced_preprocessing(self, image):
        variations = []
        transforms_list = []

        # Basic transforms
        transforms_list.append(('original', transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])))

        # Color space variations
        for color_space in ['L', 'YCbCr', 'LAB']:
            transforms_list.append((f'{color_space}', transforms.Compose([
                transforms.Resize(256),
                transforms.Lambda(lambda x: x.convert(color_space)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])))

        # Enhancement variations
        enhancement_ops = {
            'contrast': (ImageEnhance.Contrast, [0.8, 1.2]),
            'brightness': (ImageEnhance.Brightness, [0.9, 1.1]),
            'sharpness': (ImageEnhance.Sharpness, [0.8, 1.2])
        }

        for op_name, (enhancer, values) in enhancement_ops.items():
            for value in values:
                transforms_list.append((f'{op_name}_{value}', transforms.Compose([
                    transforms.Lambda(lambda x: enhancer(x).enhance(value)),
                    transforms.Resize(256),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])))

        # Apply all transforms
        for name, transform in transforms_list:
            try:
                variations.append(transform(image))
            except Exception as e:
                print(f"Warning: Transform {name} failed: {str(e)}")

        return torch.stack([v for v in variations if v is not None]).to(self.device)

    def adaptive_threshold_detection(self, confidence_scores):
        # Multiple threshold schemes
        threshold_schemes = [
            np.linspace(0.45, 0.55, 5),
            np.linspace(0.48, 0.52, 3),
            np.linspace(0.49, 0.51, 2)
        ]

        predictions = []
        weights = []

        for scheme in threshold_schemes:
            for threshold in scheme:
                pred = (confidence_scores > threshold).float()
                conf_distance = abs(confidence_scores - threshold)
                weight = 1.0 / (1.0 + conf_distance)

                predictions.append(pred)
                weights.append(weight)

        # Weighted voting
        weighted_pred = torch.stack(predictions) * torch.stack(weights)
        final_pred = (weighted_pred.sum(dim=0) / sum(weights) > 0.5).float()

        return final_pred

    def detect_watermark(self, image_path):
        image = Image.open(image_path)

        # Get features using both detection_utils functions
        freq_features = self.advanced_frequency_analysis(image)
        pattern_features = self.local_pattern_analysis(image).to(self.device)
        variations = self.enhanced_preprocessing(image)

        # Process each variation
        all_predictions = []
        all_confidences = []

        with torch.no_grad():
            # Spatial domain detection
            for var in variations:
                try:
                    decoded = self.msg_decoder(var.unsqueeze(0))
                    confidence = torch.sigmoid(decoded)
                    pred = self.adaptive_threshold_detection(confidence)

                    # Ensure consistent dimensions
                    if pred.dim() == 2:
                        pred = pred.squeeze(0)  # Remove batch dimension if present

                    # Only append valid predictions
                    if pred.numel() == 48:  # Check if prediction has correct size
                        all_predictions.append(pred)
                        all_confidences.append(confidence.squeeze())
                except Exception as e:
                    print(f"Warning: Error in processing variation: {str(e)}")

            # Process frequency and pattern features
            try:
                pattern_features = pattern_features.to(self.device)
                freq_features = freq_features.to(self.device)

                feature_confidence = torch.sigmoid(torch.mean(pattern_features))
                freq_confidence = torch.sigmoid(torch.mean(freq_features))

                combined_features = torch.cat([freq_features, pattern_features])
                feature_based_confidence = torch.sigmoid(combined_features.mean())
                feature_based_pred = torch.full((48,), feature_based_confidence.item(),
                                                device=self.device)

                all_predictions.append(feature_based_pred)
                all_confidences.append(feature_based_confidence)
            except Exception as e:
                print(f"Warning: Error in feature processing: {str(e)}")

        # Check if we have valid predictions
        if not all_predictions:
            raise ValueError("No valid predictions were generated")

        # Ensure all predictions have the same shape
        all_predictions = [p.view(-1) for p in all_predictions]  # Flatten all predictions

        # Combine all predictions
        combined_pred = torch.stack(all_predictions).mean(dim=0)
        final_pred = (combined_pred > 0.5).float()

        # Calculate metrics
        accuracy = (final_pred.cpu() == self.key).float().mean().item()
        confidence = torch.mean(torch.stack([c.mean() for c in all_confidences])).item()
        reliability = 1.0 - abs(confidence - 0.75)

        return {
            'accuracy': accuracy,
            'predicted_bits': final_pred.cpu().numpy().tolist(),
            'confidence': confidence,
            'reliability': reliability,
            'freq_confidence': freq_confidence.item(),
            'pattern_confidence': feature_confidence.item(),
            'is_watermarked': accuracy > 0.7 and reliability > 0.6,
            'num_variations_processed': len(all_predictions)
        }
    # def detect_watermark(self, image_path):
    #     image = Image.open(image_path)
    #
    #     # Get features using both detection_utils functions
    #     freq_features = self.advanced_frequency_analysis(image)
    #     pattern_features = self.local_pattern_analysis(image).to(self.device)
    #     variations = self.enhanced_preprocessing(image)
    #
    #     # Process each variation
    #     all_predictions = []
    #     all_confidences = []
    #
    #     with torch.no_grad():
    #         # Spatial domain detection
    #         for var in variations:
    #             try:
    #                 decoded = self.msg_decoder(var.unsqueeze(0))
    #                 confidence = torch.sigmoid(decoded)
    #                 pred = self.adaptive_threshold_detection(confidence)
    #
    #                 # Only append valid predictions
    #                 if pred.numel() > 0:  # Check if prediction is not empty
    #                     all_predictions.append(pred)
    #                     all_confidences.append(confidence)
    #             except Exception as e:
    #                 print(f"Warning: Error in processing variation: {str(e)}")
    #
    #         # Process frequency and pattern features
    #         try:
    #             pattern_features = pattern_features.to(self.device)
    #             freq_features = freq_features.to(self.device)
    #
    #             feature_confidence = torch.sigmoid(torch.mean(pattern_features))
    #             freq_confidence = torch.sigmoid(torch.mean(freq_features))
    #
    #             combined_features = torch.cat([freq_features, pattern_features])
    #             feature_based_confidence = torch.sigmoid(combined_features.mean())
    #             feature_based_pred = torch.full((48,), feature_based_confidence.item(),
    #                                             device=self.device)  # Create 48-length tensor
    #
    #             all_predictions.append(feature_based_pred)
    #             all_confidences.append(feature_based_confidence)
    #         except Exception as e:
    #             print(f"Warning: Error in feature processing: {str(e)}")
    #
    #     # Check if we have valid predictions
    #     if not all_predictions:
    #         raise ValueError("No valid predictions were generated")
    #
    #     # Combine all predictions
    #     combined_pred = torch.stack([p for p in all_predictions if p.numel() == 48]).mean(dim=0)
    #     final_pred = (combined_pred > 0.5).float()
    #
    #     # Calculate metrics
    #     accuracy = (final_pred.cpu() == self.key).float().mean().item()
    #     confidence = torch.mean(torch.stack([c.mean() for c in all_confidences])).item()
    #     reliability = 1.0 - abs(confidence - 0.75)
    #
    #     return {
    #         'accuracy': accuracy,
    #         'predicted_bits': final_pred.cpu().numpy().tolist(),
    #         'confidence': confidence,
    #         'reliability': reliability,
    #         'freq_confidence': freq_confidence.item(),
    #         'pattern_confidence': feature_confidence.item(),
    #         'is_watermarked': accuracy > 0.7 and reliability > 0.6,
    #         'num_variations_processed': len(all_predictions)
    #     }
    # def detect_watermark(self, image_path):
    #     image = Image.open(image_path)
    #
    #     # Get features using both detection_utils functions
    #     freq_features = self.advanced_frequency_analysis(image)  # Already on self.device
    #     pattern_features = self.local_pattern_analysis(image).to(self.device)  # Move to device
    #     variations = self.enhanced_preprocessing(image)
    #
    #     # Process each variation
    #     all_predictions = []
    #     all_confidences = []
    #
    #     with torch.no_grad():
    #         # Spatial domain detection
    #         for var in variations:
    #             decoded = self.msg_decoder(var.unsqueeze(0))
    #             confidence = torch.sigmoid(decoded)
    #             pred = self.adaptive_threshold_detection(confidence)
    #
    #             all_predictions.append(pred)
    #             all_confidences.append(confidence)
    #
    #         # Process frequency and pattern features
    #         pattern_features = pattern_features.to(self.device)
    #         freq_features = freq_features.to(self.device)
    #
    #         feature_confidence = torch.sigmoid(torch.mean(pattern_features))
    #         freq_confidence = torch.sigmoid(torch.mean(freq_features))
    #
    #         combined_features = torch.cat([freq_features, pattern_features])
    #         feature_based_confidence = torch.sigmoid(combined_features.mean())
    #         feature_based_pred = (feature_based_confidence > 0.5).float()
    #
    #         all_predictions.append(feature_based_pred)
    #         all_confidences.append(feature_based_confidence)
    #
    #     # Combine all predictions
    #     combined_pred = torch.stack(all_predictions).mean(dim=0)
    #     final_pred = (combined_pred > 0.5).float()
    #
    #     # Calculate metrics
    #     accuracy = (final_pred.cpu() == self.key).float().mean().item()
    #     confidence = torch.stack(all_confidences).mean().item()
    #     reliability = 1.0 - abs(confidence - 0.75)
    #
    #     return {
    #         'accuracy': accuracy,
    #         'predicted_bits': final_pred.cpu().numpy().tolist(),
    #         'confidence': confidence,
    #         'reliability': reliability,
    #         'freq_confidence': freq_confidence.item(),
    #         'pattern_confidence': feature_confidence.item(),
    #         'is_watermarked': accuracy > 0.7 and reliability > 0.6
    #     }
