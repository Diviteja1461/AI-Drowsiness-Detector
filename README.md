# AI Drowsiness Detector for Drivers

## Overview
This project implements an AI-based driver drowsiness detection system using Machine Learning (ML) and Deep Learning (DL) techniques. The system classifies a driver's facial image as **Drowsy** or **Non Drowsy**.

A key contribution of this project is the identification and correction of **data leakage** in the standard random split approach. A **person-independent split** is used to ensure honest evaluation — where the test set contains faces of completely unseen people.

---

## University
**Technische Hochschule Deggendorf — Campus Cham**
Machine Learning and Deep Learning in Production and Logistics (SS26)
Supervisor: Dr. Sunil P. Survaiya

## Group Members
| Name | Matriculation Number |
|---|---|
| Divi Teja Dimmiti | 22409514 |
| Adi Dev Anil Seena | 12669951 |
| Ajay Venkatesh | 22526093 |
| Shilpa Golla | 22401043 |

---

## Dataset
**Driver Drowsiness Dataset (DDD)**
- Source: [Kaggle](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd)
- Total Images: 41,793
- Classes: Drowsy (22,348) and Non Drowsy (19,445)
- Participants: 27 unique people
- Image Size: 227 × 227 RGB

---

## Data Split Strategy

### Random Split (80/10/10)
Images are randomly divided into train, val, and test sets. The same person's face appears in all splits — causing **data leakage**. Results shown for reference only.

### Person-Independent Split (Honest Evaluation)
Images are split by participant ID — each person appears in only one split.

| Split | People | Images |
|---|---|---|
| Train | A–P (16 people) | 21,266 |
| Validation | Q–W (7 people) | 7,926 |
| Test | X, Y, ZA, ZB (4 people) | 9,967 |

---

## Notebook Structure
**File:** `AI_Drowsiness_Detector.ipynb`

| Section | Description |
|---|---|
| 1. Import Libraries | All required libraries imported once |
| 2. Global Configuration | Dataset path, device, person split |
| 3. EDA | Class distribution, sample images, image properties |
| 4. ML Baseline — Random Split | SVM + Random Forest using ResNet18 features |
| 5. Person-Independent Dataset Class | Custom PyTorch Dataset by participant ID |
| 6. MobileNetV2 | Transfer learning, person-independent split, fine-tuning |
| 7. InceptionResNetV1 (VGGFace2) | Face-specific pretrained model, fine-tuning |
| 8. Final Results Comparison | Summary of all model results |
| 9. Grad-CAM Visualization | Explainability — which facial regions model focuses on |

---

## Models

### Machine Learning (Random Split — Reference Only)
| Model | Feature Extractor | Test Accuracy |
|---|---|---|
| Random Forest | ResNet18 (ImageNet) | 99.95% |
| SVM | ResNet18 (ImageNet) | 100.00% |

*Note: High accuracy due to data leakage — not realistic performance.*

### Deep Learning (Person-Independent Split — Honest Evaluation)
| Model | Pretrained On | Val Accuracy | Test Accuracy |
|---|---|---|---|
| MobileNetV2 — Initial | ImageNet (1.2M images) | 64.72% | **66.02%** |
| MobileNetV2 — Fine-Tuned | ImageNet | 68.42% | 64.88% |
| InceptionResNetV1 — Initial | VGGFace2 (3.3M faces) | 81.66% | 59.18% |
| InceptionResNetV1 — Fine-Tuned | VGGFace2 | 81.66% | 59.18% |

**Best Model: MobileNetV2 — 66.02% test accuracy (person-independent split)**

---

## Grad-CAM Visualization
Gradient-weighted Class Activation Mapping (Grad-CAM) applied on MobileNetV2 to visualize which facial regions the model focuses on when predicting drowsiness. The model primarily focuses on the eye, nose and mouth regions — consistent with real drowsiness indicators.

---

## Streamlit Demo App
An interactive web application for real-time drowsiness detection.

**Live Demo:** https://ai-drowsiness-detector-ckpn37rcdmvwc765wwxojx.streamlit.app/

**Run locally:**
1. Run the notebook `AI_Drowsiness_Detector.ipynb` completely to generate `best_mobilenet_16_7_4.pth`
2. Place `best_mobilenet_16_7_4.pth` in the same folder as `app.py`
3. Run the app:

```bash
streamlit run app.py
```

**Features:**
- Upload a driver face image
- Get instant Drowsy / Non Drowsy prediction
- Confidence scores for both classes
- About page with dataset and model information

---

## Key Findings
- Random split gives 99%+ accuracy due to data leakage — not realistic
- Person-independent split gives honest evaluation on completely unseen faces
- MobileNetV2 achieves best test accuracy — 66.02%
- InceptionResNetV1 (VGGFace2) achieves best validation accuracy — 81.66%
- Fine-tuning did not improve test accuracy for either model
- Dataset limitation: only 27 participants, 4 test people — small test set causes variance
- A larger dataset with 100+ participants would significantly improve cross-person accuracy

---

## Installation
```bash
pip install torch torchvision facenet-pytorch scikit-learn matplotlib seaborn streamlit opencv-python jupyter
```

---

## Project Structure

AI-Drowsiness-Detector/

├── AI_Drowsiness_Detector.ipynb   # Main notebook — all models and results

├── app.py                          # Streamlit demo application

├── best_mobilenet_16_7_4.pth      # Best MobileNetV2 model weights

├── requirements.txt                # Python dependencies for Streamlit Cloud

└── README.md                       # Project documentation


**Note:** Dataset is not included due to size limits.
- Download dataset from Kaggle link above
- Model weights are generated by running the notebook

---

## Environment
- MacBook Air M2 — Apple MPS GPU backend
- Python 3.12
- PyTorch 2.12.0
- torchvision 0.27.0
- facenet-pytorch 2.6.0
- streamlit 1.58.0
