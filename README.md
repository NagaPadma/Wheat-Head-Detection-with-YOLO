## Folder Structure
 
```
project/
├── templates/
├── uploads/
├── results/
├── app.py
├── inference.py
└── best.pt
```
 
### Directory Descriptions
 
- **templates/** - Contains HTML templates for the web interface
- **uploads/** - Directory for storing uploaded files
- **results/** - Directory where output/results are saved
- **app.py** - Main application file (Flask/Django web server)
- **inference.py** - Inference script for running predictions
- **best.pt** - Pre-trained model weights
 
## How to Run
 
### 1. Run the Web Application
 
First, start the web application:
 
```bash
python app.py
```
 
This will launch the web server. The application should be accessible at `http://localhost:5000` (or the port specified in your app.py).
 
### 2. Run Inference
 
After the web application is running, you can perform inference:
 
```bash
python inference.py
```
 
## Prerequisites
 
- Python 3.x
- Required dependencies (install via `pip install -r requirements.txt` if available)
- Pre-trained model file (`best.pt`) must be present in the root directory
 
## Notes
 
- Ensure all three subdirectories (`templates/`, `uploads/`, `results/`) exist before running the application
- The `best.pt` file contains the pre-trained model and is required for inference
- Run `app.py` before running `inference.py` for proper functionality
 
 