# 🎬 Deepfake Detection System

A **completely free, open-source** deepfake detection system using state-of-the-art pre-trained models. No paid dependencies, no API keys required!

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

## 🌟 Features

✅ **Image Deepfake Detection** - Detect AI-generated and manipulated images
✅ **Video Deepfake Detection** - Frame-by-frame analysis with timestamp reports
✅ **Audio Deepfake Detection** - Identify AI-synthesized speech
✅ **REST API** - FastAPI endpoints for easy integration
✅ **Web Interface** - Beautiful drag-and-drop UI
✅ **CLI Tool** - Command-line interface for batch processing
✅ **100% Free** - All models and libraries are open-source
✅ **GPU/CPU Support** - Automatic device detection
✅ **Docker Ready** - One-command deployment

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip or conda
- 4GB RAM minimum (8GB+ recommended)
- GPU optional but recommended

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Neer229/Deepfake-proect.git
cd Deepfake-proect

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Download pre-trained models (automatic on first run)
python -c "from app.models.image_detector import ImageDetector; ImageDetector()"
```

## 📖 Usage

### Web Interface (Easiest)

```bash
python -m uvicorn app.main:app --reload
```

Open browser to: **http://localhost:8000**

### REST API

```bash
# Start API server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Detect deepfake in image
curl -X POST -F "file=@image.jpg" http://localhost:8000/api/image/detect

# Detect deepfake in video
curl -X POST -F "file=@video.mp4" http://localhost:8000/api/video/detect

# Detect deepfake in audio
curl -X POST -F "file=@audio.wav" http://localhost:8000/api/audio/detect
```

### Command Line Interface

```bash
# Detect image
python cli/deepfake_cli.py detect-image path/to/image.jpg

# Detect video
python cli/deepfake_cli.py detect-video path/to/video.mp4

# Detect audio
python cli/deepfake_cli.py detect-audio path/to/audio.wav

# Batch detection
python cli/deepfake_cli.py detect-all path/to/folder/

# Benchmark performance
python cli/deepfake_cli.py benchmark
```

## 🏗️ Project Structure

```
Deepfake-proect/
├── app/
│   ├── main.py                 # FastAPI application
│   ├── models/
│   │   ├── image_detector.py   # Image detection module
│   │   ├── video_detector.py   # Video detection module
│   │   └── audio_detector.py   # Audio detection module
│   ├── routes/
│   │   ├── image.py            # Image detection endpoints
│   │   ├── video.py            # Video detection endpoints
│   │   └── audio.py            # Audio detection endpoints
│   ├── utils/
│   │   ├── config.py           # Configuration settings
│   │   └── file_handler.py     # File processing utilities
│   └── static/
│       ├── index.html          # Web UI
│       └── style.css           # Styling
├── cli/
│   └── deepfake_cli.py         # CLI application
├── tests/
│   ├── test_image.py
│   ├── test_video.py
│   └── test_audio.py
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose configuration
└── README.md                   # This file
```

## 🤖 Detection Models

### Image Detection
- **MesoNet-4**: Lightweight CNN designed for deepfake detection
- **Xception**: Transfer learning model trained on FaceForensics++
- **EfficientNet**: Efficient neural network architecture

Models sourced from:
- [Hugging Face Model Hub](https://huggingface.co/models?search=deepfake)
- [TrueMedia.org Open Source](https://github.com/truemediaorg/ml-models)

### Video Detection
- Frame-by-frame analysis using image detection models
- Configurable frame sampling for speed optimization
- Temporal analysis for consistency

### Audio Detection
- Speech manipulation detection using RawNet2
- Spectrogram analysis
- Multiple frequency analysis

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up

# Or build manually
docker build -t deepfake-detector .
docker run -p 8000:8000 deepfake-detector
```

Access at: http://localhost:8000

## 📊 API Endpoints

### Image Detection
```
POST /api/image/detect
Content-Type: multipart/form-data
Body: file (image file)

Response:
{
    "filename": "image.jpg",
    "is_deepfake": false,
    "confidence": 0.95,
    "model_used": "mesonet",
    "processing_time": 0.234
}
```

### Video Detection
```
POST /api/video/detect
Content-Type: multipart/form-data
Body: file (video file)

Response:
{
    "filename": "video.mp4",
    "is_deepfake": true,
    "average_confidence": 0.78,
    "frames_analyzed": 60,
    "detections": [
        {"frame": 10, "confidence": 0.85},
        {"frame": 20, "confidence": 0.72}
    ],
    "processing_time": 15.234
}
```

### Audio Detection
```
POST /api/audio/detect
Content-Type: multipart/form-data
Body: file (audio file)

Response:
{
    "filename": "audio.wav",
    "is_deepfake": false,
    "confidence": 0.88,
    "processing_time": 5.123
}
```

## 📈 Performance

On NVIDIA GPU (RTX 3080):
- **Image Detection**: ~50-100ms per image
- **Video Detection**: ~2-5 min per minute of video
- **Audio Detection**: ~3-5s per minute of audio

On CPU (Intel i7):
- **Image Detection**: ~500-1000ms per image
- **Video Detection**: ~15-30 min per minute of video
- **Audio Detection**: ~30-50s per minute of audio

## ⚙️ Configuration

Edit `app/utils/config.py` to customize:

```python
# Detection thresholds
CONFIDENCE_THRESHOLD = 0.5
DEEPFAKE_THRESHOLD = 0.6

# Video processing
VIDEO_FRAME_SAMPLING = 5      # Every 5th frame
VIDEO_MAX_FRAMES = 300        # Maximum frames to process

# File upload limits
MAX_IMAGE_SIZE = 50 * 1024 * 1024    # 50MB
MAX_VIDEO_SIZE = 500 * 1024 * 1024   # 500MB
MAX_AUDIO_SIZE = 100 * 1024 * 1024   # 100MB
```

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_image.py -v

# Run with coverage
python -m pytest tests/ --cov=app
```

## 📚 Datasets Used for Training

Models were trained on:
- **FaceForensics++** - 1000+ original videos with manipulations
- **DFDC** - Thousands of deepfake videos
- **Celeb-DF** - High-quality celebrity deepfakes
- **DeeperForensics-1.0** - 60,000+ videos

## 💡 Tips for Best Results

1. **Image Quality**: Higher resolution images yield better results
2. **Video Quality**: 720p or higher recommended
3. **Audio Quality**: 16-bit 44.1kHz or higher
4. **Batch Processing**: Use CLI for multiple files
5. **GPU Usage**: Enable GPU for 10-20x faster processing
6. **Thresholds**: Adjust confidence thresholds based on your use case

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional detection models
- Performance optimizations
- More output formats
- Additional video codecs support
- Ensemble detection methods

## ⚠️ Disclaimer

This tool is for research and educational purposes. It may produce false positives or false negatives. Always use in combination with other verification methods for critical decisions.

## 📄 License

MIT License - See LICENSE file for details

## 🔗 Resources

- [GitHub - Deepfake Detection Awesome List](https://www.libhunt.com/topic/deepfake-detection)
- [Hugging Face - Deepfake Models](https://huggingface.co/models?search=deepfake)
- [TrueMedia Open Source Release](https://www.truemedia.org/post/open-source-release)
- [FaceForensics++ Dataset](https://github.com/ondyari/FaceForensics)

## 📞 Support

- 📖 Check the [Documentation](./docs/)
- 🐛 Open an [Issue](https://github.com/Neer229/Deepfake-proect/issues)
- 💬 Start a [Discussion](https://github.com/Neer229/Deepfake-proect/discussions)

---

**Made with ❤️ using free, open-source tools**
**No costs. No API keys. Just pure detection power.