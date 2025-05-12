import os
from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import openai
from PIL import Image
import requests
from io import BytesIO
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

def generate_image(prompt):
    """Generate image using DALL-E"""
    try:
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response['data'][0]['url']
        return image_url
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

def process_excel(file_path):
    """Process Excel file and return data"""
    try:
        df = pd.read_excel(file_path)
        return df.to_dict('records')
    except Exception as e:
        print(f"Error processing Excel: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and file.filename.endswith('.xlsx'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Process Excel file
        data = process_excel(file_path)
        if data:
            return jsonify({'message': 'File uploaded successfully', 'data': data})
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt')
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    image_url = generate_image(prompt)
    if image_url:
        return jsonify({'image_url': image_url})
    
    return jsonify({'error': 'Failed to generate image'}), 500

@app.route('/replace', methods=['POST'])
def replace_image():
    data = request.json
    image_url = data.get('image_url')
    old_image_path = data.get('old_image_path')
    
    if not image_url or not old_image_path:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    try:
        # Download the new image
        response = requests.get(image_url)
        new_image = Image.open(BytesIO(response.content))
        
        # Save the new image
        new_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'new_image.png')
        new_image.save(new_image_path)
        
        return jsonify({'message': 'Image replaced successfully', 'new_image_path': new_image_path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 