from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import subprocess
import threading

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
CHECKPOINT_PATH = 'checkpoints/detector_epoch_38.pth'
ENCODER_PATH = 'weights/FoMo4Wheat_giant.pth'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_inference(image_path, output_folder):
    """
    Run the inference script on the uploaded image
    """
    try:
        print(f"Starting inference for: {image_path}")
        
        # Build the command
        cmd = [
            'python', 'inference.py',
            #'--checkpoint', CHECKPOINT_PATH,
            #'--encoder', ENCODER_PATH,
            #'--image', image_path,
            #'--output', output_folder,
            #'--visualize'
        ]
        
        # Set environment variables to use UTF-8 encoding
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONLEGACYWINDOWSSTDIO'] = '0'
        
        # On Windows, use chcp 65001 to set UTF-8 code page
        if os.name == 'nt':
            cmd = ['cmd', '/c', 'chcp 65001 >nul && python'] + cmd[1:]
        
        # Run the inference
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env=env,
            encoding='utf-8',
            errors='replace'
        )
        
        print(f"Inference completed successfully")
        print(f"Output: {result.stdout}")
        
        if result.stderr:
            print(f"Stderr: {result.stderr}")
        
        # Get the original filename and extension
        original_filename = os.path.basename(image_path)
        original_name, original_ext = os.path.splitext(original_filename)
        
        visualizations_folder = os.path.join(output_folder, 'visualizations')
        
        # Look for the _pred file with any extension
        possible_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']
        pred_path = None
        
        for ext in possible_extensions:
            potential_path = os.path.join(visualizations_folder, f"{original_name}_pred{ext}")
            if os.path.exists(potential_path):
                pred_path = potential_path
                break
        
        if pred_path:
            # Rename to original filename with original extension
            final_path = os.path.join(visualizations_folder, original_filename)
            
            # Remove old file if exists
            if os.path.exists(final_path):
                os.remove(final_path)
            
            # Rename the _pred file to the original filename
            os.rename(pred_path, final_path)
            print(f"Renamed {os.path.basename(pred_path)} to {original_filename}")
        else:
            print(f"Warning: No _pred file found for {original_name}")
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Inference failed with error: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False
    except Exception as e:
        print(f"Error running inference: {e}")
        return False

def run_inference_async(image_path, output_folder):
    """
    Run inference in a separate thread to avoid blocking the upload response
    """
    thread = threading.Thread(target=run_inference, args=(image_path, output_folder))
    thread.daemon = True
    thread.start()

@app.route('/')
def home():
    return render_template('upload_image_camera.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return 'No file part', 400
    
    file = request.files['image']
    
    if file.filename == '':
        return 'No selected file', 400
    
    if file and allowed_file(file.filename):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"Image received and saved to: {filepath}")
        
        # Run inference asynchronously
        run_inference_async(filepath, app.config['RESULTS_FOLDER'])
        
        # Check if result already exists (in case of re-upload)
        result_path = os.path.join(app.config['RESULTS_FOLDER'], 'visualizations', filename)
        has_result = os.path.exists(result_path)
        
        return jsonify({
            'message': 'Image uploaded successfully! Processing...',
            'filename': filename,
            'has_result': has_result
        }), 200
    
    return 'Invalid file type', 400

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Route to serve result images from visualizations folder
@app.route('/results/<filename>')
def result_file(filename):
    visualizations_path = os.path.join(app.config['RESULTS_FOLDER'], 'visualizations')
    return send_from_directory(visualizations_path, filename)

# Route to check if result exists in visualizations folder
@app.route('/check-result/<filename>')
def check_result(filename):
    result_path = os.path.join(app.config['RESULTS_FOLDER'], 'visualizations', filename)
    return jsonify({'has_result': os.path.exists(result_path)})

@app.route('/counts/<filename>')
def count_file(filename):
    # Change .png/.jpg extension to .txt
    name_without_ext = os.path.splitext(filename)[0]
    txt_filename = f"{name_without_ext}.txt"
    counts_path = os.path.join(app.config['RESULTS_FOLDER'], 'counts')
    return send_from_directory(counts_path, txt_filename)

if __name__ == '__main__':
    # Verify required files exist
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"WARNING: Checkpoint file not found: {CHECKPOINT_PATH}")
    if not os.path.exists(ENCODER_PATH):
        print(f"WARNING: Encoder file not found: {ENCODER_PATH}")
    
    app.run(host='0.0.0.0', port=5000, debug=True)