
from flask import Flask, render_template, request, jsonify, send_file
import os
import json
import uuid
from werkzeug.utils import secure_filename
import shutil
from datetime import datetime
import glob

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output_videos'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}
EXERCISE_TYPES = ['bicep_curl', 'lateral_raise', 'squat', 'jumping_jack']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html', exercise_types=EXERCISE_TYPES)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file selected'}), 400
        
        file = request.files['video']
        exercise_type = request.form.get('exercise_type')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not exercise_type or exercise_type not in EXERCISE_TYPES:
            return jsonify({'error': 'Invalid exercise type'}), 400
        
        if file and allowed_file(file.filename):
            session_id = str(uuid.uuid4())[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = secure_filename(file.filename)
            filename = f"{timestamp}_{session_id}_{filename}"
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)

            session_output_dir = os.path.join(app.config['OUTPUT_FOLDER'], f"session_{session_id}")
            os.makedirs(session_output_dir, exist_ok=True)

            original_video_copy = os.path.join(session_output_dir, f"original_{filename}")
            shutil.copy2(upload_path, original_video_copy)

            results = analyze_exercise_video(exercise_type, upload_path, session_output_dir)

            results['original_video'] = original_video_copy

            static_results_dir = os.path.join('static', 'results', f"session_{session_id}")
            shutil.copytree(session_output_dir, static_results_dir, dirs_exist_ok=True)

            web_results = prepare_web_results(results, session_id)

            print("Returning video URLs:", web_results.get('output_video'), web_results.get('original_video'))

            return jsonify({
                'success': True,
                'session_id': session_id,
                'results': web_results
            })
        
        return jsonify({'error': 'Invalid file format'}), 400

    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

def prepare_web_results(results, session_id):
    """Convert file paths to web-accessible URLs"""
    web_results = results.copy()
    base_url = f"/static/results/session_{session_id}"

    if 'original_video' in results:
        original_video_filename = os.path.basename(results['original_video'])
        web_results['original_video'] = f"{base_url}/{original_video_filename}"

    if 'output_video' in results:
        video_filename = os.path.basename(results['output_video'])
        web_results['output_video'] = f"{base_url}/{video_filename}"

    if 'log_file' in results:
        log_filename = os.path.basename(results['log_file'])
        web_results['log_file'] = f"{base_url}/{log_filename}"

    if 'graphs_directory' in results:
        web_results['graphs_directory'] = f"{base_url}/graphs"
        graph_files = []
        local_graph_dir = results['graphs_directory']
        if os.path.exists(local_graph_dir):
            for file in os.listdir(local_graph_dir):
                if file.endswith('.png'):
                    graph_files.append({
                        'name': file.replace('_', ' ').replace('.png', '').title(),
                        'path': f"{base_url}/graphs/{file}"
                    })
            web_results['graph_files'] = graph_files

    return web_results

@app.route('/results/<session_id>')
def show_results(session_id):
    """Serve results HTML view with embedded video and graphs"""
    results_dir = os.path.join('static', 'results', f"session_{session_id}")
    summary_files = glob.glob(os.path.join(results_dir, '*_summary.json'))

    if not summary_files:
        return "Results not found", 404

    with open(summary_files[0], 'r') as f:
        results = json.load(f)

    web_results = prepare_web_results(results, session_id)
    return render_template('results.html', results=web_results, session_id=session_id)

@app.route('/download/<session_id>/<filename>')
def download_file(session_id, filename):
    file_path = os.path.join('static', 'results', f"session_{session_id}", filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return "File not found", 404

def analyze_exercise_video(exercise_type, video_path, output_dir):
    from integrated import analyze_exercise_video as actual_analyzer
    return actual_analyzer(exercise_type, video_path, output_dir)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
