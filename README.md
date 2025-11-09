# VERISIGNX 🔐

**AI-Driven Signature Verification for Fraud-Resistant Cheque Processing**

VERISIGNX is an advanced machine learning system designed to detect forged signatures on cheques using computer vision, OCR, and SVM classification. The system combines multiple detection techniques with a robust verification pipeline to ensure high accuracy in signature authentication.

![VERISIGNX](https://img.shields.io/badge/VERISIGNX-AI%20Signature%20Verification-blue)
![Python](https://img.shields.io/badge/Python-3.7+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 Features

- **Multi-Stage Detection Pipeline**: OCR-based signature extraction, Line Sweep algorithm, and Connected Components analysis
- **Advanced Feature Extraction**: SIFT descriptors, contour features, geometric properties, and statistical measures
- **SVM-Based Classification**: Trained LinearSVC model for genuine vs. forged signature detection
- **Modern Web Interface**: Beautiful Flask-based web application with real-time processing visualization
- **Comprehensive Dataset**: IDRBT Cheque Image Dataset with 100+ cheque samples
- **High Accuracy**: Robust feature engineering and machine learning pipeline

## 📁 Project Structure

```
VERISIGNX/
│
├── Code/
│   ├── Detection/
│   │   ├── OCR/                    # OCR-based signature detection
│   │   │   ├── OCR.py
│   │   │   └── OCR_Results/
│   │   ├── Line Sweep/             # Line sweep algorithm
│   │   │   ├── LineSweep.py
│   │   │   └── LineSweep_Results/
│   │   └── Connected Components/  # Connected components analysis
│   │       ├── ConnectedComponents.py
│   │       ├── UnionArray.py
│   │       └── ConnectedComponents_Results/
│   │
│   └── Verification/
│       └── SVM/                    # SVM-based verification system
│           ├── app.py              # Flask web application
│           ├── train.py            # Model training script
│           ├── test.py             # Model testing script
│           ├── run.py              # Model evaluation script
│           ├── features.py         # Feature extraction
│           ├── preprocess.py       # Image preprocessing
│           ├── model.pkl           # Trained SVM model
│           ├── Data/
│           │   ├── genuine/        # Genuine signature samples
│           │   ├── forged/         # Forged signature samples
│           │   └── origin/         # Original test images
│           ├── templates/
│           │   └── index.html     # Web interface
│           └── static/
│               └── style.css       # Styling
│
└── Dataset/
    └── IDRBT_Cheque_Image_Dataset/  # Cheque image dataset
        └── [100+ cheque images]
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Tesseract OCR ([Download here](https://github.com/UB-Mannheim/tesseract/wiki))
- Git LFS (for handling large files) - Required for cloning repository with datasets

### Quick Setup Scripts

We provide automated setup scripts for easy Git configuration:

**Windows (PowerShell):**
```powershell
.\setup_git.ps1
```

**Linux/macOS (Bash):**
```bash
bash setup_git.sh
```

For detailed upload instructions, see [UPLOAD_INSTRUCTIONS.md](UPLOAD_INSTRUCTIONS.md).

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/KnightShan/VERISIGNX.git
   cd VERISIGNX
   ```

2. **Install Git LFS** (for large files)
   ```bash
   git lfs install
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Tesseract OCR**
   - Windows: Install Tesseract and update the path in `Code/Detection/OCR/OCR.py` (line 12)
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

### Required Python Packages

Create a `requirements.txt` file with:
```
flask>=2.0.0
numpy>=1.21.0
opencv-python>=4.5.0
opencv-contrib-python>=4.5.0
Pillow>=8.0.0
pytesseract>=0.3.8
scikit-learn>=0.24.0
scipy>=1.7.0
scikit-image>=0.18.0
matplotlib>=3.4.0
```

## 📖 Usage

### 1. Detection Pipeline

#### OCR-Based Signature Detection
```bash
cd Code/Detection/OCR
python OCR.py
```
Extracts signature regions from cheque images using OCR to locate "please sign above" text.

#### Line Sweep Algorithm
```bash
cd Code/Detection/Line\ Sweep
python LineSweep.py
```
Applies line sweep algorithm to detect signature boundaries.

#### Connected Components Analysis
```bash
cd Code/Detection/Connected\ Components
python ConnectedComponents.py
```
Uses connected components to identify and extract signature regions.

### 2. Model Training

Train the SVM classifier:
```bash
cd Code/Verification/SVM
python train.py
```
This will:
- Extract SIFT descriptors from training images
- Build a visual vocabulary using k-means clustering
- Extract contour and geometric features
- Train a LinearSVC classifier
- Save the model as `model.pkl`

### 3. Model Testing

Test the trained model:
```bash
cd Code/Verification/SVM
python test.py
```
Evaluates the model on test data and displays accuracy metrics.

### 4. Web Application

Run the Flask web application:
```bash
cd Code/Verification/SVM
python app.py
```
Then open your browser and navigate to `http://localhost:5000`

**Features:**
- Upload cheque signature images
- Real-time processing visualization
- Automatic signature extraction
- AI-powered verification results
- Beautiful, modern UI

## 🔬 Technical Details

### Detection Methods

1. **OCR Detection**: Uses Tesseract OCR to identify "please" and "above" keywords, then calculates signature region based on text positions.

2. **Line Sweep**: Scans image rows and columns to find the first and last occurrence of signature pixels, creating a bounding box.

3. **Connected Components**: Implements Union-Find data structure to identify connected pixel regions and selects the largest component as the signature.

### Feature Extraction

The system extracts 512 features per signature:

- **SIFT Descriptors (500 features)**: Visual vocabulary histogram using k-means clustering
- **Contour Features (12 features)**:
  - Aspect ratio
  - Hull area / Bounding area ratio
  - Contour area / Bounding area ratio
  - White pixel ratio
  - Centroid (x, y)
  - Eccentricity
  - Solidity
  - Skew (x, y)
  - Kurtosis (x, y)

### Machine Learning Model

- **Algorithm**: Linear Support Vector Classifier (LinearSVC)
- **Features**: 512-dimensional feature vector (500 SIFT + 12 contour)
- **Preprocessing**: StandardScaler normalization
- **Output**: Binary classification (1 = Forged, 2 = Genuine)

## 📊 Dataset

The project uses the **IDRBT Cheque Image Dataset** containing:
- 100+ cheque images in TIFF format
- Various cheque formats and signatures
- Suitable for training and testing signature verification models

**Dataset Structure:**
```
Dataset/
└── IDRBT_Cheque_Image_Dataset/
    ├── Cheque_083654.tif
    ├── Cheque_083655.tif
    └── ...
```

## 🎯 Results

The system achieves high accuracy in distinguishing between genuine and forged signatures through:
- Multi-stage detection pipeline ensuring accurate signature extraction
- Comprehensive feature engineering capturing both visual and geometric properties
- Robust SVM classifier trained on diverse signature samples

## 👥 Contributors

- **Shantanu Maity** - Lead Developer ([LinkedIn](https://www.linkedin.com/in/shantanumaity))
- **Deepak Shajan** - Co-Developer ([LinkedIn](https://www.linkedin.com/in/dk-dsa-logics/))

**Supervised by:** Prof. Sandipan Maiti

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- IDRBT for providing the cheque image dataset
- OpenCV and scikit-learn communities for excellent libraries
- Tesseract OCR for text recognition capabilities

## 📧 Contact

For questions, issues, or contributions, please open an issue on GitHub or contact the developers.

---

**Note**: This project is for educational and research purposes. Always verify important documents through official channels.

