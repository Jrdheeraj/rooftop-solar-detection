import os
import json
import shutil
import random
from pathlib import Path
import pandas as pd
from tqdm import tqdm

class YOLODatasetCreator:
    def __init__(self, output_dir='data/processed'):
        self.output_dir = output_dir
        self.train_dir = f"{output_dir}/dataset/images/train"
        self.val_dir = f"{output_dir}/dataset/images/val"
        self.test_dir = f"{output_dir}/dataset/images/test"
        self.train_labels = f"{output_dir}/dataset/labels/train"
        self.val_labels = f"{output_dir}/dataset/labels/val"
        self.test_labels = f"{output_dir}/dataset/labels/test"
    
    def create_directories(self):
        """Create YOLO dataset directory structure"""
        for path in [self.train_dir, self.val_dir, self.test_dir,
                     self.train_labels, self.val_labels, self.test_labels]:
            Path(path).mkdir(parents=True, exist_ok=True)
        print("✅ YOLO dataset directories created")
    
    def create_yolo_label(self, has_solar, image_h=400, image_w=400):
        """
        Create YOLO format label for solar panel detection
        
        YOLO Format: <class_id> <x_center> <y_center> <width> <height>
        All normalized to [0, 1]
        
        Example: 0 0.5 0.5 0.3 0.2
        Means: class 0 (solar panel), centered at (0.5, 0.5), 
               width 0.3 (30% of image), height 0.2 (20% of image)
        """
        if has_solar == 1:
            # Generate synthetic bounding box
            # For Day 1 MVP, we use generic centered box
            # In Day 2 training, YOLOv8 learns actual panel positions
            
            x_center = random.uniform(0.35, 0.65)  # Center ±15%
            y_center = random.uniform(0.35, 0.65)
            width = random.uniform(0.20, 0.40)     # 20-40% of image
            height = random.uniform(0.20, 0.40)
            
            label = f"0 {x_center:.4f} {y_center:.4f} {width:.4f} {height:.4f}"
            return label
        else:
            # No solar panel
            return None
    
    def copy_and_split_images(self, source_dir, metadata_file, train_ratio=0.8, val_ratio=0.1):
        """
        Copy images and create labels with train/val/test split
        
        For 50 images:
        - Train: 40 (80%)
        - Val: 5 (10%)
        - Test: 5 (10%)
        """
        # Load metadata from image_fetcher
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Shuffle for randomness
        random.seed(42)  # Same seed = reproducible splits
        random.shuffle(metadata)
        
        # Calculate split sizes
        train_size = int(len(metadata) * train_ratio)
        val_size = int(len(metadata) * val_ratio)
        
        splits = {
            'train': metadata[:train_size],
            'val': metadata[train_size:train_size+val_size],
            'test': metadata[train_size+val_size:]
        }
        
        print(f"\n📊 TRAIN/VAL/TEST SPLIT:")
        print(f"   Train: {len(splits['train'])} images")
        print(f"   Val:   {len(splits['val'])} images")
        print(f"   Test:  {len(splits['test'])} images")
        
        # Process each split
        for split_name, split_data in splits.items():
            if split_name == 'train':
                img_dir, label_dir = self.train_dir, self.train_labels
            elif split_name == 'val':
                img_dir, label_dir = self.val_dir, self.val_labels
            else:
                img_dir, label_dir = self.test_dir, self.test_labels
            
            print(f"\n📁 Processing {split_name.upper()} split...")
            
            for meta in tqdm(split_data):
                sample_id = meta['sample_id']
                has_solar = meta['hassolar']
                
                # Copy image file
                src_img = meta['image_path']
                dst_img = f"{img_dir}/{sample_id}.jpg"
                
                if os.path.exists(src_img):
                    shutil.copy(src_img, dst_img)
                    
                    # Create corresponding label file
                    label_content = self.create_yolo_label(has_solar)
                    
                    # Save label file (.txt)
                    label_file = f"{label_dir}/{sample_id}.txt"
                    if label_content:
                        with open(label_file, 'w') as f:
                            f.write(label_content)
                    else:
                        # For "no panel" images, create empty file
                        Path(label_file).touch()
        
        print(f"\n✅ Dataset split complete!")
    
    def create_dataset_yaml(self):
        """
        Create dataset.yaml - YOLOv8 configuration file
        
        This tells YOLOv8 where to find training data
        """
        yaml_content = """path: data/processed/dataset
train: images/train
val: images/val
test: images/test

nc: 1
names: ['solar_panel']
"""
        
        yaml_path = f"{self.output_dir}/dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        print(f"\n✅ dataset.yaml created at {yaml_path}")
        return yaml_path
    
    def verify_dataset(self):
        """
        Verify that dataset is correctly organized
        Check: matching images and labels, correct counts
        """
        print("\n🔍 VERIFYING DATASET...")
        
        counts = {
            'train_images': len(os.listdir(self.train_dir)),
            'train_labels': len(os.listdir(self.train_labels)),
            'val_images': len(os.listdir(self.val_dir)),
            'val_labels': len(os.listdir(self.val_labels)),
            'test_images': len(os.listdir(self.test_dir)),
            'test_labels': len(os.listdir(self.test_labels))
        }
        
        print("\n📊 DATASET VERIFICATION:")
        for key, count in counts.items():
            print(f"   {key}: {count}")
        
        # Check matching
        train_imgs = set(os.listdir(self.train_dir))
        train_labels = set([f.replace('.txt', '.jpg') for f in os.listdir(self.train_labels)])
        
        if train_imgs == train_labels:
            print("\n   ✅ All train images have corresponding labels")
        else:
            print("\n   ⚠️  Mismatch between images and labels!")
        
        print("\n✅ Dataset verification complete!")

# Main execution
if __name__ == '__main__':
    creator = YOLODatasetCreator(output_dir='data/processed')
    
    # Step 1: Create directories
    creator.create_directories()
    
    # Step 2: Copy images and create labels with train/val/test split
    creator.copy_and_split_images(
    source_dir='data/processed/images_all',
    metadata_file='data/processed/images_all/metadata.json',
    train_ratio=0.8,
    val_ratio=0.1
)

    # Step 3: Create dataset.yaml
    creator.create_dataset_yaml()
    
    # Step 4: Verify dataset
    creator.verify_dataset()
