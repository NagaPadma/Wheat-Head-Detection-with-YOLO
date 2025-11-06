# GWHD Wheat Head Detection with YOLOv8

Complete training pipeline for detecting wheat heads in the Global Wheat Head Detection (GWHD) dataset using YOLOv8.

## 📁 Project Structure

```
project/
├── gwhd_yolov8_training.py    # Main training script
├── inference.py                # Inference script for predictions
├── quick_start_guide.py        # Quick setup instructions
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

Or manually:
```bash
pip install ultralytics opencv-python pandas numpy matplotlib pillow pyyaml tqdm --break-system-packages
```

### 2. Organize Your Data

Place all your GWHD images in a single folder:
```
your_data/
├── images/
│   ├── 4563856cc6d75c670eafd86d5eb7245fbe8f273c28f9e36f7c6aaf097c7ce423.png
│   ├── a2a15938845d9812de03bd44799c4b1bf856a8ad11752e81c94dc8d138515021.png
│   └── ...
└── csvs/
    ├── competition_train.csv
    ├── competition_val.csv
    └── competition_test.csv
```

### 3. Configure the Script

Edit `gwhd_yolov8_training.py` and set your images directory:

```python
# Find this line (around line 27):
IMAGES_DIR = None

# Change it to:
IMAGES_DIR = "/path/to/your/images"
```

### 4. Run Training

```bash
python gwhd_yolov8_training.py
```

## ⚙️ Configuration Options

In `gwhd_yolov8_training.py`, you can adjust these parameters in the `Config` class:

| Parameter | Default | Options | Description |
|-----------|---------|---------|-------------|
| `MODEL_SIZE` | `"m"` | n, s, m, l, x | Model size (n=fastest, x=most accurate) |
| `IMG_SIZE` | `1024` | 640, 1024, 1280 | Training image size |
| `BATCH_SIZE` | `16` | 4, 8, 16, 32 | Batch size (reduce if GPU OOM) |
| `EPOCHS` | `100` | 50-200 | Number of training epochs |
| `PATIENCE` | `15` | 10-30 | Early stopping patience |

### Model Size Recommendations

- **YOLOv8n**: Fastest, lowest accuracy (~2-3 hours training)
- **YOLOv8s**: Fast, good accuracy (~3-4 hours training)
- **YOLOv8m**: ⭐ **Recommended** - Best balance (~4-6 hours training)
- **YOLOv8l**: Slower, high accuracy (~6-8 hours training)
- **YOLOv8x**: Slowest, highest accuracy (~8-12 hours training)

## 📊 Training Process

The script will:

1. ✅ Convert CSV annotations to YOLO format
2. ✅ Create proper directory structure
3. ✅ Visualize sample training images
4. ✅ Train YOLOv8 model with optimal settings
5. ✅ Evaluate on validation set
6. ✅ Run inference on test samples
7. ✅ Save all results and weights

### Expected Output Structure

```
gwhd_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
├── runs/
│   └── wheat_detection/
│       ├── weights/
│       │   ├── best.pt          # ⭐ Best model
│       │   └── last.pt
│       ├── results.png           # Training curves
│       ├── confusion_matrix.png
│       ├── PR_curve.png
│       └── ...
├── predictions/                  # Sample predictions
└── sample_visualization.png      # Training sample viz
```

## 🎯 Using the Trained Model

### Option 1: Use the Inference Script

```bash
python inference.py
```

Then choose:
- Detect on a single image
- Detect on a directory of images
- Quick count (no visualization)

### Option 2: Use in Your Code

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('gwhd_dataset/runs/wheat_detection/weights/best.pt')

# Single image prediction
results = model.predict('wheat_field.jpg', conf=0.25)

# Get detections
for result in results:
    boxes = result.boxes.xyxy      # Bounding boxes [x1, y1, x2, y2]
    confidences = result.boxes.conf # Confidence scores
    
    print(f"Detected {len(boxes)} wheat heads")
    
    # Visualize
    annotated = result.plot()
    cv2.imwrite('output.jpg', annotated)
```

### Option 3: Use the WheatDetector Class

```python
from inference import WheatDetector

# Initialize
detector = WheatDetector(
    model_path='gwhd_dataset/runs/wheat_detection/weights/best.pt',
    conf_threshold=0.25
)

# Detect in single image
result = detector.detect('wheat_image.jpg')
print(f"Found {result['num_detections']} wheat heads")

# Batch detection
results = detector.detect_batch('test_images/', output_dir='results/')

# Quick count
count = detector.count_wheat_heads('wheat_image.jpg')
```

## 📈 Monitoring Training

Training metrics are saved in real-time:
- **results.png**: Loss curves, mAP, precision, recall
- **TensorBoard**: `tensorboard --logdir gwhd_dataset/runs`

### Key Metrics to Watch

- **mAP@0.5**: Main metric for wheat head detection (target: >0.85)
- **mAP@0.5:0.95**: Overall detection quality (target: >0.65)
- **Precision**: How many detections are correct (target: >0.90)
- **Recall**: How many wheat heads are found (target: >0.85)

## 🔧 Troubleshooting

### GPU Out of Memory

Reduce batch size:
```python
BATCH_SIZE = 8  # or even 4
```

Or reduce image size:
```python
IMG_SIZE = 640
```

### Model Not Learning

- Increase epochs: `EPOCHS = 150`
- Try a larger model: `MODEL_SIZE = "l"`
- Check data quality and labels

### Too Many False Positives

Lower confidence threshold during inference:
```python
results = model.predict('image.jpg', conf=0.35)  # Higher threshold
```

### Missing Wheat Heads (Low Recall)

- Use larger image size: `IMG_SIZE = 1280`
- Lower confidence threshold: `conf=0.20`
- Train longer or use larger model

## 🎓 Understanding GWHD Data Format

The CSV files contain:
- `image_name`: Image filename
- `BoxesString`: Bounding boxes in format "x1 y1 x2 y2;x1 y1 x2 y2;..."
- `domain`: Dataset source (e.g., "Arvalis_1")

Example:
```
image_name,BoxesString,domain
image.png,99 692 160 764;641 27 697 115,Arvalis_1
```

The script automatically converts this to YOLO format:
```
# YOLO format: class_id x_center y_center width height (normalized)
0 0.516 0.728 0.122 0.144
0 0.669 0.071 0.112 0.176
```

## 💡 Best Practices

1. **Start Simple**: Begin with YOLOv8m and default settings
2. **Monitor Training**: Watch the results.png for overfitting
3. **Validate Often**: Check predictions on validation set
4. **Tune Threshold**: Adjust confidence threshold based on your use case
5. **Data Quality**: Ensure images are good quality and labels are accurate

## 📚 Additional Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [GWHD Dataset Paper](http://www.global-wheat.com/)
- [Ultralytics GitHub](https://github.com/ultralytics/ultralytics)

## 🐛 Common Issues

**Q: "IMAGES_DIR not set" error**  
A: Edit the script and set `Config.IMAGES_DIR = "/path/to/images"`

**Q: Training is very slow**  
A: Reduce image size (640) or batch size (8), or use smaller model (YOLOv8s)

**Q: How to resume training?**  
A: Use `model.train(resume=True)` with the last.pt checkpoint

**Q: Can I use CPU only?**  
A: Yes, but it will be much slower. Set `device='cpu'` in training parameters

## 📧 Support

For issues or questions:
1. Check the troubleshooting section
2. Review YOLOv8 documentation
3. Check GitHub issues on ultralytics/ultralytics

## 📝 License

This code is provided as-is for use with the GWHD dataset. Please cite the GWHD paper if you use this for research.

---

**Happy wheat head detecting! 🌾**
