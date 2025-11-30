import cv2
import numpy as np
import random
from pathlib import Path

class ImageAugmenter:
    def __init__(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
    
    def rotate_image(self, image, angle_range=15):
        """Rotate image by random angle"""
        angle = random.uniform(-angle_range, angle_range)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (w, h))
        return rotated
    
    def flip_image(self, image, flip_probability=0.5):
        """Randomly flip image left-right (NOT up-down)"""
        if random.random() < flip_probability:
            return cv2.flip(image, 1)  # 1 = horizontal flip
        return image
    
    def adjust_brightness(self, image, brightness_range=0.3):
        """Adjust brightness"""
        factor = random.uniform(1.0 - brightness_range, 1.0 + brightness_range)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 2] = hsv[:, :, 2] * factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    
    def add_noise(self, image, noise_intensity=0.05):
        """Add Gaussian noise"""
        noise = np.random.normal(0, noise_intensity * 255, image.shape)
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy
    
    def augment(self, image):
        """Apply random augmentations"""
        if random.random() < 0.5:
            image = self.rotate_image(image, angle_range=15)
        if random.random() < 0.5:
            image = self.flip_image(image)
        if random.random() < 0.5:
            image = self.adjust_brightness(image)
        if random.random() < 0.3:
            image = self.add_noise(image, noise_intensity=0.02)
        
        return image

def test_augmentation():
    """Test augmentation on sample image"""
    # Load sample image
    img_path = 'data/processed/train_images/0001.jpg'
    
    if not Path(img_path).exists():
        print(f"❌ Image not found: {img_path}")
        return
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    print(f"✅ Loaded image: {img_path}")
    print(f"   Shape: {img.shape}")
    
    # Apply augmentations
    augmenter = ImageAugmenter()
    
    augmented_images = []
    for i in range(5):
        aug_img = augmenter.augment(img.copy())
        augmented_images.append(aug_img)
        
        # Save
        save_path = f'outputs/augmentation_test_{i}.jpg'
        cv2.imwrite(save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
        print(f"   ✅ Saved augmented image {i+1}: {save_path}")
    
    print(f"\n✅ Augmentation test complete!")
    print(f"   5 augmented samples saved to outputs/")

if __name__ == '__main__':
    test_augmentation()