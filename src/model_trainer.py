import os
import yaml
import torch
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SolarPanelTrainer:
    def __init__(self, config_path='config/training_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device(
            f"cuda:{self.config['device']}" 
            if torch.cuda.is_available() and self.config['device'] != 'cpu'
            else 'cpu'
        )
        logger.info(f"Using device: {self.device}")
    
    def download_pretrained_model(self):
        logger.info("Downloading YOLOv8 model...")
        model = YOLO('yolov8m.pt')
        logger.info("✓ Model downloaded")
        return model
    
    def train_model(self):
        model = YOLO('yolov8m.pt')
        logger.info("="*60)
        logger.info("STARTING TRAINING")
        logger.info("="*60)
        
        results = model.train(
            data='config/dataset.yaml',
            epochs=self.config['epochs'],
            imgsz=self.config['imgsz'],
            batch=self.config['batch'],
            patience=self.config['patience'],
            device=0 if str(self.device).startswith('cuda') else 'cpu',
            optimizer=self.config['optimizer'],
            lr0=self.config['lr0'],
            lrf=self.config['lrf'],
            momentum=self.config['momentum'],
            weight_decay=self.config['weight_decay'],
            augment=True,
            mosaic=self.config['augmentation']['mosaic'],
            fliplr=self.config['augmentation']['fliplr'],
            flipud=self.config['augmentation']['flipud'],
            degrees=self.config['augmentation']['degrees'],
            translate=self.config['augmentation']['translate'],
            scale=self.config['augmentation']['scale'],
            hsv_h=self.config['augmentation']['hsv_h'],
            hsv_s=self.config['augmentation']['hsv_s'],
            hsv_v=self.config['augmentation']['hsv_v'],
            verbose=True,
            save=True,
            project='runs/detect',
            name='solar_panel_v1'
        )
        
        logger.info("✓ Training complete!")
        return model, results
    
    def export_model(self, model_path='runs/detect/solar_panel_v1/weights/best.pt'):
        logger.info(f"Loading best model...")
        best_model = YOLO(model_path)
        
        best_model.export(format='pt')
        logger.info("✓ Exported to .pt")
        
        os.makedirs('models', exist_ok=True)
        import shutil
        shutil.copy(model_path, 'models/solar_panel_detector.pt')
        logger.info("✓ Copied to models/solar_panel_detector.pt")
    
    def evaluate_model(self, model_path='runs/detect/solar_panel_v1/weights/best.pt'):
        logger.info("Evaluating model...")
        model = YOLO(model_path)
        metrics = model.val()
        
        logger.info(f"\nmAP50: {metrics.box.map50:.4f}")
        logger.info(f"mAP50-95: {metrics.box.map:.4f}")
        logger.info(f"Precision: {metrics.box.p[0]:.4f}")
        logger.info(f"Recall: {metrics.box.r[0]:.4f}")
        
        precision = metrics.box.p[0]
        recall = metrics.box.r[0]
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        logger.info(f"F1 Score: {f1:.4f}\n")
        
        return metrics
    
    def run(self):
        try:
            self.download_pretrained_model()
            model, results = self.train_model()
            self.export_model()
            self.evaluate_model()
            logger.info("\n✅ TRAINING COMPLETE!")
        except Exception as e:
            logger.error(f"Training failed: {e}", exc_info=True)
            raise

if __name__ == '__main__':
    trainer = SolarPanelTrainer()
    trainer.run()