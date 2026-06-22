import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #F1F5F9;
}

.block-container { padding: 0 !important; max-width: 100% !important; }
#MainMenu, footer, header { visibility: hidden; }

.top-header {
    background: linear-gradient(120deg, #1A2B5C 60%, #2563EB 100%);
    padding: 32px 48px 28px 48px;
    color: white;
}
.top-header .badge {
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.top-header h1 {
    font-size: 34px;
    font-weight: 700;
    margin: 0 0 8px 0;
    line-height: 1.2;
}
.top-header .sub { font-size: 14px; opacity: 0.75; font-weight: 300; }
.top-header .meta {
    margin-top: 16px;
    font-size: 12px;
    opacity: 0.55;
    border-top: 1px solid rgba(255,255,255,0.15);
    padding-top: 14px;
}

.main-content { padding: 32px 48px; }

.card {
    background: white;
    border-radius: 12px;
    padding: 24px;
    border: 1px solid #E2E8F0;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

.result-box { border-radius: 10px; padding: 20px 24px; margin-bottom: 16px; }
.result-drowsy { background: #FEF2F2; border-left: 4px solid #DC2626; }
.result-alert { background: #F0FDF4; border-left: 4px solid #16A34A; }
.result-label {
    font-size: 11px; font-weight: 600; letter-spacing: 1px;
    text-transform: uppercase; margin-bottom: 4px; opacity: 0.6;
}
.result-title { font-size: 24px; font-weight: 700; margin-bottom: 6px; }
.result-desc { font-size: 13px; opacity: 0.75; line-height: 1.5; }

.info-pill {
    display: inline-block;
    background: #EFF6FF; color: #1D4ED8;
    border-radius: 6px; padding: 4px 12px;
    font-size: 12px; font-weight: 500;
    margin-right: 8px; margin-bottom: 8px;
}

.sec-title { font-size: 15px; font-weight: 600; color: #1A2B5C; margin-bottom: 4px; }
.sec-sub { font-size: 12px; color: #94A3B8; margin-bottom: 16px; }

.img-cap {
    text-align: center; font-size: 11px; font-weight: 600;
    color: #94A3B8; letter-spacing: 0.8px;
    text-transform: uppercase; margin-top: 8px;
}

.stat-row { display: flex; gap: 12px; margin-bottom: 12px; }
.stat-box {
    flex: 1; background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px; padding: 14px 16px;
}
.stat-label {
    font-size: 11px; color: #94A3B8; font-weight: 500;
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;
}
.stat-value { font-size: 15px; font-weight: 600; color: #1A2B5C; }

.page-footer {
    background: #1E293B; color: rgba(255,255,255,0.5);
    padding: 20px 48px; font-size: 12px; margin-top: 32px;
    display: flex; justify-content: space-between;
}
</style>
""", unsafe_allow_html=True)

# ── Load Model ──
@st.cache_resource
def load_model():
    m = models.mobilenet_v2(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(m.last_channel, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)
    )
    m.load_state_dict(torch.load(
        "/Users/divitejadimmiti/Desktop/Ml_pro/best_mobilenet_16_7_4.pth",
        map_location="cpu"
    ))
    m.eval()
    return m

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ── Grad-CAM ──
class GradCAM:
    def __init__(self, model, layer):
        self.gradients = None
        self.activations = None
        layer.register_forward_hook(lambda m,i,o: setattr(self,'activations',o.detach()))
        layer.register_full_backward_hook(lambda m,gi,go: setattr(self,'gradients',go[0].detach()))
        self.model = model

    def generate(self, tensor):
        out = self.model(tensor)
        cls = out.argmax(1).item()
        self.model.zero_grad()
        out[0, cls].backward()
        pg = self.gradients.mean([0,2,3])
        act = self.activations[0].clone()
        for i in range(act.shape[0]):
            act[i] *= pg[i]
        hm = act.mean(0).cpu().numpy()
        hm = np.maximum(hm, 0)
        if hm.max() > 0: hm /= hm.max()
        return hm, cls

gc = GradCAM(model, model.features[-1])

def run_gradcam(img):
    t = transform(img).unsqueeze(0)
    t.requires_grad = True
    hm, cls = gc.generate(t)
    arr = np.array(img.resize((224,224)))
    hm_r = cv2.resize(hm, (224,224))
    hm_c = cv2.cvtColor(cv2.applyColorMap(np.uint8(255*hm_r), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    ov = cv2.addWeighted(arr, 0.6, hm_c, 0.4, 0)
    return hm_c, ov

def predict(img):
    t = transform(img).unsqueeze(0)
    with torch.no_grad():
        out = model(t)
        p = torch.softmax(out, 1)[0]
        cls = p.argmax().item()
    return cls, p[cls].item()*100, p

# ── HEADER ──
st.markdown("""
<div class="top-header">
    <div class="badge">AI Safety System</div>
    <h1>Driver Drowsiness Detection</h1>
    <p class="sub">Upload a driver face image to instantly detect signs of drowsiness using deep learning.</p>
    <div class="meta">
        Technische Hochschule Deggendorf — Campus Cham &nbsp;&nbsp;|&nbsp;&nbsp;
        Machine Learning and Deep Learning in Production and Logistics (SS26) &nbsp;&nbsp;|&nbsp;&nbsp;
        Supervisor: Dr. Sunil P. Survaiya
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="main-content">', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Detection", "About"])

with tab1:
    col_left, col_right = st.columns([1, 1.4], gap="large")

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Upload Image</div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-sub">JPG, JPEG or PNG — driver face photo</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_container_width=True)
            w, h = image.size
            st.markdown(f"""
            <div class="stat-row" style="margin-top:14px;">
                <div class="stat-box">
                    <div class="stat-label">Width</div>
                    <div class="stat-value">{w}px</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Height</div>
                    <div class="stat-value">{h}px</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Format</div>
                    <div class="stat-value">{uploaded.name.split('.')[-1].upper()}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:40px 20px;color:#CBD5E1;">
                <div style="font-size:14px;font-weight:500;color:#94A3B8;">No image uploaded yet</div>
                <div style="font-size:12px;color:#CBD5E1;margin-top:4px;">Upload a face image to begin</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="sec-title">Model Information</div>
            <div class="sec-sub">Details about the prediction model</div>
            <div>
                <span class="info-pill">MobileNetV2</span>
                <span class="info-pill">Transfer Learning</span>
                <span class="info-pill">ImageNet</span>
            </div>
            <div style="margin-top:12px;">
                <div class="stat-row">
                    <div class="stat-box">
                        <div class="stat-label">Test Accuracy</div>
                        <div class="stat-value">66.02%</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Split Strategy</div>
                        <div class="stat-value">Person-Independent</div>
                    </div>
                </div>
                <div class="stat-row">
                    <div class="stat-box">
                        <div class="stat-label">Trainable Params</div>
                        <div class="stat-value">2,009,794</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Input Size</div>
                        <div class="stat-value">224 x 224</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        if uploaded:
            with st.spinner("Analyzing image..."):
                pred_class, conf, probs = predict(image)
                heatmap, overlay = run_gradcam(image)

            drowsy_pct = probs[0].item() * 100
            alert_pct  = probs[1].item() * 100

            # Result
            if pred_class == 0:
                st.markdown("""
                <div class="result-box result-drowsy">
                    <div class="result-label">Detection Result</div>
                    <div class="result-title" style="color:#DC2626;">Drowsy Detected</div>
                    <div class="result-desc" style="color:#7F1D1D;">
                        The driver shows signs of drowsiness.
                        Immediate action is recommended — please pull over and rest.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-box result-alert">
                    <div class="result-label">Detection Result</div>
                    <div class="result-title" style="color:#16A34A;">Driver is Alert</div>
                    <div class="result-desc" style="color:#14532D;">
                        No signs of drowsiness detected.
                        The driver appears to be awake and alert.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Confidence scores
            st.markdown("""
            <div class="card">
                <div class="sec-title">Confidence Scores</div>
                <div class="sec-sub">Probability distribution across both classes</div>
            </div>
            """, unsafe_allow_html=True)

            col_d, col_a = st.columns(2)
            with col_d:
                st.metric(label="Drowsy", value=f"{drowsy_pct:.1f}%")
            with col_a:
                st.metric(label="Non Drowsy", value=f"{alert_pct:.1f}%")

            st.markdown(f"**Drowsy**")
            st.progress(drowsy_pct / 100)
            st.markdown(f"**Non Drowsy**")
            st.progress(alert_pct / 100)

            # Grad-CAM
            st.markdown("""
            <div class="card">
                <div class="sec-title">Grad-CAM Visualization</div>
                <div class="sec-sub">
                    Gradient-weighted Class Activation Mapping — shows which facial
                    regions influenced the model prediction.
                </div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3, gap="small")
            with c1:
                st.image(image.resize((224,224)), use_container_width=True)
                st.markdown('<div class="img-cap">Original</div>', unsafe_allow_html=True)
            with c2:
                st.image(heatmap, use_container_width=True)
                st.markdown('<div class="img-cap">Heatmap</div>', unsafe_allow_html=True)
            with c3:
                st.image(overlay, use_container_width=True)
                st.markdown('<div class="img-cap">Overlay</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="card" style="text-align:center;padding:80px 40px;">
                <div style="font-size:15px;font-weight:600;color:#94A3B8;margin-bottom:8px;">
                    Awaiting image upload
                </div>
                <div style="font-size:13px;color:#CBD5E1;">
                    Upload a driver face image on the left to see the detection result,
                    confidence scores, and Grad-CAM visualization here.
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("""
        <div class="card">
            <div class="sec-title">About This Project</div>
            <div class="sec-sub">AI Drowsiness Detector for Drivers</div>
            <p style="font-size:13px;color:#475569;line-height:1.7;">
                This system detects driver drowsiness from facial images using
                MobileNetV2 — a lightweight deep learning model pretrained on ImageNet
                and fine-tuned on the Driver Drowsiness Dataset (DDD).
            </p>
            <p style="font-size:13px;color:#475569;line-height:1.7;">
                The project uses a <strong>person-independent split</strong> to avoid
                data leakage — ensuring the test set contains completely unseen faces
                for an honest evaluation.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="sec-title">Dataset</div>
            <div class="sec-sub">Driver Drowsiness Dataset (DDD) — Kaggle</div>
            <div class="stat-row">
                <div class="stat-box">
                    <div class="stat-label">Total Images</div>
                    <div class="stat-value">41,793</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Participants</div>
                    <div class="stat-value">27 people</div>
                </div>
            </div>
            <div class="stat-row">
                <div class="stat-box">
                    <div class="stat-label">Drowsy</div>
                    <div class="stat-value">22,348</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Non Drowsy</div>
                    <div class="stat-value">19,445</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
            <div class="sec-title">Model Results</div>
            <div class="sec-sub">Person-Independent Split (16/7/4)</div>
            <div class="stat-row">
                <div class="stat-box">
                    <div class="stat-label">Random Forest</div>
                    <div class="stat-value">99.95% <span style="font-size:11px;color:#94A3B8;">(random split)</span></div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">SVM</div>
                    <div class="stat-value">100.00% <span style="font-size:11px;color:#94A3B8;">(random split)</span></div>
                </div>
            </div>
            <div class="stat-row">
                <div class="stat-box">
                    <div class="stat-label">MobileNetV2</div>
                    <div class="stat-value" style="color:#2563EB;">66.02%</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">InceptionResNetV1</div>
                    <div class="stat-value">59.18%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="sec-title">Group Members</div>
            <div class="sec-sub">Machine Learning and Deep Learning — SS26</div>
            <table style="width:100%;font-size:13px;border-collapse:collapse;">
                <tr style="border-bottom:1px solid #F1F5F9;">
                    <td style="padding:8px 0;color:#475569;font-weight:500;">Divi Teja Dimmiti</td>
                    <td style="padding:8px 0;color:#94A3B8;text-align:right;">22409514</td>
                </tr>
                <tr style="border-bottom:1px solid #F1F5F9;">
                    <td style="padding:8px 0;color:#475569;font-weight:500;">Adi Dev Anil Seena</td>
                    <td style="padding:8px 0;color:#94A3B8;text-align:right;">12669951</td>
                </tr>
                <tr style="border-bottom:1px solid #F1F5F9;">
                    <td style="padding:8px 0;color:#475569;font-weight:500;">Ajay Venkatesh</td>
                    <td style="padding:8px 0;color:#94A3B8;text-align:right;">22526093</td>
                </tr>
                <tr>
                    <td style="padding:8px 0;color:#475569;font-weight:500;">Shilpa Golla</td>
                    <td style="padding:8px 0;color:#94A3B8;text-align:right;">22401043</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="page-footer">
    <span>Driver Drowsiness Detection System — THD Campus Cham — SS26</span>
    <span>Supervisor: Dr. Sunil P. Survaiya</span>
</div>
""", unsafe_allow_html=True)