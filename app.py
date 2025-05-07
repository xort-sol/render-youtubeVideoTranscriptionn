from flask import Flask, request, jsonify
from flask_cors import CORS
import yt_dlp
import assemblyai as aai
import os
import tempfile
import shutil
from dotenv import load_dotenv
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure AssemblyAI
aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY', 'b22ca2f6671b4976b7109b6b48f18fc7')

def download_audio(youtube_url, cookies=None):
    """Download audio from YouTube URL with optional cookies"""
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, 'audio')
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'extract_audio': True,
        'audio_format': 'mp3',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }],
        'quiet': True,
    }
    
    # Add cookies if provided
    if cookies:
        ydl_opts['cookiefile'] = cookies
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        return f"{output_path}.mp3"
    except Exception as e:
        shutil.rmtree(temp_dir)
        raise e

@app.route('/', methods=['GET'])
def root():
    """Root endpoint for health check"""
    return jsonify({
        'status': 'active',
        'timestamp': int(time.time()),
        'cookie_support': {
            'available': True,
            'count': 0
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Render"""
    return jsonify({'status': 'healthy'}), 200

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        logger.info("Received transcription request")
        data = request.json
        youtube_url = data.get('url')
        cookies = data.get('cookies')
        
        logger.info(f"Processing URL: {youtube_url}")
        
        if not youtube_url:
            logger.error("No YouTube URL provided")
            return jsonify({'error': 'YouTube URL is required'}), 400
        
        # Create temporary file for cookies if provided
        cookie_file = None
        if cookies:
            logger.info("Cookies provided, creating temporary file")
            cookie_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
            cookie_file.write(cookies)
            cookie_file.close()
        
        try:
            # Download audio
            logger.info("Downloading audio...")
            audio_file = download_audio(youtube_url, cookie_file.name if cookie_file else None)
            logger.info("Audio downloaded successfully")
            
            # Transcribe
            logger.info("Starting transcription...")
            transcriber = aai.Transcriber()
            transcript = transcriber.transcribe(audio_file)
            
            if transcript.status == aai.TranscriptStatus.error:
                logger.error(f"Transcription error: {transcript.error}")
                return jsonify({'error': transcript.error}), 500
            
            logger.info("Transcription completed successfully")
            
            # Clean up
            os.remove(audio_file)
            if cookie_file:
                os.remove(cookie_file.name)
            shutil.rmtree(os.path.dirname(audio_file))
            
            return jsonify({
                'success': True,
                'transcript': transcript.text,
                'segments': [
                    {
                        'text': segment.text,
                        'start': segment.start,
                        'end': segment.end,
                        'confidence': segment.confidence
                    }
                    for segment in transcript.segments
                ]
            })
            
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            # Clean up on error
            if cookie_file:
                os.remove(cookie_file.name)
            return jsonify({'error': str(e)}), 500
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 