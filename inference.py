"""
Clean Inference Script for GWHD Wheat Detection
================================================
NO CLI PROMPTS - Just edit the paths below and run!

Draws clean bounding boxes (no confidence scores) and saves counts to txt files
"""

from ultralytics import YOLO
import cv2
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np


# ==================== CONFIGURATION ====================
# EDIT THESE THREE PATHS BEFORE RUNNING:

MODEL_PATH = "best.pt"
INPUT_DIR = "uploads"
OUTPUT_DIR = "results"

# Detection settings
CONF_THRESHOLD = 0.25   # Lower = more detections, Higher = fewer false positives
IOU_THRESHOLD = 0.45    # NMS threshold

# Box visualization
BOX_COLOR = (0, 0, 255)  # Green in BGR format
BOX_THICKNESS = 2        # Line thickness for boxes

# =======================================================


class CleanWheatDetector:
    """Wheat head detector with clean visualization"""
    
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize the detector
        
        Args:
            model_path: Path to trained YOLOv8 model (.pt file)
            conf_threshold: Confidence threshold (0-1)
            iou_threshold: IoU threshold for NMS (0-1)
        """
        print(f"Loading model from {model_path}...")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        print("✓ Model loaded successfully!")
    
    def draw_clean_boxes(self, image, boxes, color=(0, 255, 0), thickness=2):
        """
        Draw bounding boxes without confidence scores
        
        Args:
            image: Input image (numpy array)
            boxes: Bounding boxes in xyxy format
            color: Box color in BGR (default: green)
            thickness: Box line thickness
            
        Returns:
            Image with drawn boxes
        """
        img_with_boxes = image.copy()
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, thickness)
        
        return img_with_boxes
    
    def detect_single(self, image_path, output_dir="inference_results", 
                     box_color=(0, 255, 0), box_thickness=2):
        """
        Detect wheat heads in a single image with clean visualization
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
            box_color: Box color in BGR (default: green)
            box_thickness: Box line thickness
            
        Returns:
            dict: Detection results
        """
        image_path = Path(image_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run prediction
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        confidences = result.boxes.conf.cpu().numpy()
        
        # Read original image
        image = cv2.imread(str(image_path))
        
        # Draw clean boxes (no confidence labels)
        img_with_boxes = self.draw_clean_boxes(image, boxes, box_color, box_thickness)
        
        # Save image with boxes
        image_dir = Path(output_dir) / "visualizations"
        image_dir.mkdir(parents=True, exist_ok=True)
        output_image_path = image_dir / f"{image_path.stem}{image_path.suffix}"
        cv2.imwrite(str(output_image_path), img_with_boxes)
        
        # Save count to txt file
        counts_dir = Path(output_dir) / "counts"
        counts_dir.mkdir(parents=True, exist_ok=True)

        count_file = counts_dir / f"{image_path.stem}.txt"
        with open(count_file, 'w') as f:
            #f.write(f"Image: {image_path.name}\n")
            f.write(f"{len(boxes)}\n")
        
        # Save detailed results (boxes + confidences) to separate txt file
        details_file = output_dir / f"{image_path.stem}_details.txt"
        with open(details_file, 'w') as f:
            f.write(f"Image: {image_path.name}\n")
            f.write(f"Total detections: {len(boxes)}\n")
            f.write(f"Confidence threshold: {self.conf_threshold}\n")
            f.write(f"\nDetailed Results:\n")
            f.write(f"{'Box':<6} {'X1':<8} {'Y1':<8} {'X2':<8} {'Y2':<8} {'Confidence':<12}\n")
            f.write("-" * 60 + "\n")
            
            for idx, (box, conf) in enumerate(zip(boxes, confidences), 1):
                x1, y1, x2, y2 = box
                f.write(f"{idx:<6} {x1:<8.1f} {y1:<8.1f} {x2:<8.1f} {y2:<8.1f} {conf:<12.4f}\n")
        
        detection_result = {
            'image_name': image_path.name,
            'num_detections': len(boxes),
            'output_image': str(output_image_path),
            'count_file': str(count_file),
            'details_file': str(details_file)
        }
        
        return detection_result
    
    def detect_batch(self, image_dir, output_dir="inference_results", 
                    box_color=(0, 0, 255), box_thickness=2,
                    extensions=['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']):
        """
        Detect wheat heads in multiple images with clean visualization
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save outputs
            box_color: Box color in BGR
            box_thickness: Box line thickness
            extensions: Image file extensions to process
            
        Returns:
            list: Detection results for all images
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(list(image_dir.glob(ext)))
        
        if not image_files:
            print(f"⚠️  No images found in {image_dir}")
            return []
        
        print(f"Found {len(image_files)} images")
        
        all_results = []
        
        # Process each image
        for img_path in tqdm(image_files, desc="Processing images"):
            try:
                result = self.detect_single(
                    img_path, 
                    output_dir=output_dir,
                    box_color=box_color,
                    box_thickness=box_thickness
                )
                all_results.append(result)
            except Exception as e:
                print(f"Error processing {img_path.name}: {e}")
                continue
        
        # Create summary files
        self._create_summary(all_results, output_dir)
        
        return all_results
    
    def _create_summary(self, results, output_dir):
        """Create summary files for batch detection"""
        output_dir = Path(output_dir)
        
        # Summary txt file
        summary_file = output_dir / "detection_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("GWHD Wheat Head Detection - Summary\n")
            f.write("="*70 + "\n\n")
            
            total_detections = sum(r['num_detections'] for r in results)
            avg_detections = total_detections / len(results) if results else 0
            
            f.write(f"Total images processed: {len(results)}\n")
            f.write(f"Total wheat heads detected: {total_detections}\n")
            f.write(f"Average per image: {avg_detections:.2f}\n")
            f.write(f"Min detections: {min(r['num_detections'] for r in results) if results else 0}\n")
            f.write(f"Max detections: {max(r['num_detections'] for r in results) if results else 0}\n")
            f.write(f"\n" + "="*70 + "\n")
            f.write(f"{'Image Name':<50} {'Count':<10}\n")
            f.write("="*70 + "\n")
            
            for result in sorted(results, key=lambda x: x['num_detections'], reverse=True):
                f.write(f"{result['image_name']:<50} {result['num_detections']:<10}\n")
        
        print(f"\n✓ Summary saved to {summary_file}")
        
        # CSV file for easy analysis
        csv_file = output_dir / "detection_results.csv"
        df = pd.DataFrame([
            {
                'image_name': r['image_name'],
                'wheat_head_count': r['num_detections']
            }
            for r in results
        ])
        df = df.sort_values('wheat_head_count', ascending=False)
        df.to_csv(csv_file, index=False)
        
        print(f"✓ CSV results saved to {csv_file}")
        
        # Statistics
        print(f"\n" + "="*70)
        print("Detection Statistics:")
        print("="*70)
        print(f"Total images: {len(results)}")
        print(f"Total detections: {sum(r['num_detections'] for r in results)}")
        print(f"Average per image: {avg_detections:.2f}")
        print(f"Min: {min(r['num_detections'] for r in results) if results else 0}")
        print(f"Max: {max(r['num_detections'] for r in results) if results else 0}")
        print("="*70)


def main():
    """Main execution function - automatically runs batch inference"""
    
    print("="*70)
    print("GWHD Wheat Head Detection - Clean Inference")
    print("="*70)
    
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Input: {INPUT_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Confidence threshold: {CONF_THRESHOLD}")
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"\n❌ Model not found at {MODEL_PATH}")
        print("Please update MODEL_PATH in the script to point to your trained model")
        print("\nTo train: python gwhd_yolov8_training.py")
        return
    
    # Check if input directory exists
    if not Path(INPUT_DIR).exists():
        print(f"\n❌ Input directory not found at {INPUT_DIR}")
        print("Please update INPUT_DIR in the script to point to your images folder")
        return
    
    # Initialize detector
    detector = CleanWheatDetector(
        model_path=MODEL_PATH,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD
    )
    
    # Run batch detection automatically
    print("\n" + "="*70)
    print("Running Batch Detection")
    print("="*70)
    
    results = detector.detect_batch(
        INPUT_DIR, 
        output_dir=OUTPUT_DIR,
        box_color=BOX_COLOR,
        box_thickness=BOX_THICKNESS
    )
    
    if results:
        print(f"\n✓ Batch detection complete!")
        print(f"  Processed: {len(results)} images")
        print(f"  Results saved to: {OUTPUT_DIR}/")
        print(f"\nOutput files:")
        print(f"  • {OUTPUT_DIR}/detection_summary.txt - Overall summary")
        print(f"  • {OUTPUT_DIR}/detection_results.csv - CSV format results")
        print(f"  • {OUTPUT_DIR}/*_detected.png - Images with boxes")
        print(f"  • {OUTPUT_DIR}/*_count.txt - Individual counts")
        print(f"  • {OUTPUT_DIR}/*_details.txt - Detailed box coordinates")
    else:
        print(f"\n❌ No images were processed successfully")


if __name__ == "__main__":
    main()