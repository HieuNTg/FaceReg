# FaceReg - Real-Time Face Recognition System

A face recognition application built with **MTCNN** + **InceptionResnetV1** (FaceNet) and deployed via **Streamlit**. Supports image, video, and real-time face recognition with dynamic face enrollment.

## Architecture

```
Input (Image / Video / Camera)
        |
        v
  +-----------+       +-------------------+       +------------------+
  |   MTCNN   | ----> | InceptionResnetV1 | ----> | Euclidean Match  |
  | Detection |       |  512-d Embedding  |       | Threshold = 2.0  |
  +-----------+       +-------------------+       +------------------+
        |                                                |
   Bounding Box                                    Name + Confidence
```

## Features

| Feature | Description |
|---------|-------------|
| Image Recognition | Upload an image and identify faces |
| Video Recognition | Process MP4 videos with face detection overlay |
| Real-Time Recognition | Live camera feed recognition |
| Face Enrollment | Train new faces via video upload or webcam |
| Unknown Detection | Identifies unknown persons (distance > 2.0) |

## Tech Stack

- **Deep Learning:** PyTorch, FaceNet-PyTorch (MTCNN + InceptionResnetV1)
- **Web Framework:** Streamlit, Streamlit-WebRTC
- **Computer Vision:** OpenCV
- **ML Evaluation:** Scikit-learn (SVM comparison in notebooks)
- **Deployment:** Docker

## Performance

| Metric | Value |
|--------|-------|
| Accuracy | **97%** on 31-class celebrity dataset |
| Embedding Dim | 512-d vectors |
| Pre-training | VGGFace2 |
| Train/Test Split | 70% / 30% |
| Dataset | 2,562 images across 31 classes ([Kaggle](https://www.kaggle.com/datasets/vasukipatel/face-recognition-dataset)) |

## Project Structure

```
FaceReg/
├── app.py                          # Entry point
├── src/
│   ├── core/
│   │   └── face_engine.py          # Face detection, encoding & matching
│   └── ui/
│       ├── styles.py               # Custom CSS styling
│       └── pages.py                # Streamlit page functions
├── models/
│   ├── vector_face.pth             # Pre-trained face embeddings
│   └── name_face.npy               # Face labels (31 classes)
├── notebooks/
│   ├── euclidean_distance.ipynb    # Euclidean distance approach
│   └── svm_classifier.ipynb        # SVM classifier approach
├── assets/                          # Static files (images, gifs)
├── experiments/
│   └── app_webrtc.py               # WebRTC experimental version
├── Dockerfile
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# Clone the repository
git clone https://github.com/HieuNTg/FaceReg.git
cd FaceReg

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Docker

```bash
docker build -t facereg .
docker run -p 8501:8501 facereg
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

## Training Notebooks

| Notebook | Approach | Details |
|----------|----------|---------|
| `notebooks/euclidean_distance.ipynb` | Euclidean Distance | Distance-based matching with averaged embeddings |
| `notebooks/svm_classifier.ipynb` | SVM Classifier | Support Vector Machine on 512-d embeddings |

## Pipeline

![Pipeline](https://github.com/HieuNTg/FaceReg/assets/96096473/a4edc90e-5d74-42d4-bc74-90d17968a229)

## Demo

![Result 1](https://github.com/HieuNTg/FaceReg/assets/96096473/1e420dbd-0df3-4cfd-97f6-58e2117a348f)
![Result 2](https://github.com/HieuNTg/FaceReg/assets/96096473/31cd7a58-d2e0-49ed-93a9-a6867d42a946)

## Future Improvements

- **Liveness Detection** - Anti-spoofing with face mesh / blink detection
- **Multi-Face Recognition** - Detect and identify all faces in a frame simultaneously
- **Cosine Similarity** - Alternative distance metric for better embedding comparison
- **Attendance System** - Automated check-in/check-out with reporting
- **Edge Deployment** - ONNX/TensorRT conversion for Raspberry Pi / Jetson Nano
- **Vector Database** - Milvus/Pinecone for scalable face search
