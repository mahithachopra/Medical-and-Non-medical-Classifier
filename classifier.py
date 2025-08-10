"""
Medical vs Non-Medical Image Classifier with Streamlit Web UI
- Upload a PDF or enter a URL
- Extract images and classify them using CLIP zero-shot
- Display thumbnails with labels and confidence bars
"""

import os
import io
import urllib.parse
from typing import List, Tuple

import torch
from PIL import Image
import requests
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import clip
import streamlit as st

# ------------------------ Helpers -------------------------

def download_image(url: str):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")

def extract_images_from_url(page_url: str, max_images: int = 50):
    result = []
    resp = requests.get(page_url, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")
    img_tags = soup.find_all("img")
    for tag in img_tags:
        if len(result) >= max_images:
            break
        src = tag.get("src") or tag.get("data-src")
        if not src:
            continue
        src = urllib.parse.urljoin(page_url, src)
        try:
            img = download_image(src)
            result.append((img, src))
        except:
            continue
    return result

def extract_images_from_pdf(pdf_bytes, max_images: int = 50):
    result = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for pno in range(len(doc)):
        page = doc[pno]
        image_list = page.get_images(full=True)
        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            try:
                im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                result.append((im, f"page{pno+1}-image{img_index+1}"))
            except:
                continue
        pix = page.get_pixmap(dpi=150)
        im = Image.open(io.BytesIO(pix.tobytes())).convert("RGB")
        result.append((im, f"page{pno+1}-render"))
        if len(result) >= max_images:
            break
    return result

# ------------------------ CLIP Classifier -------------------------
class ZeroShotMedicalClassifier:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)
        self.model.eval()
        self.medical_prompts = [
            "a medical image", "an X-ray image", "an MRI scan", "a CT scan",
            "an ultrasound image", "a radiology image", "a microscope medical image"
        ]
        self.non_medical_prompts = [
            "a photograph of nature", "a landscape photograph", "a photo of a building",
            "a photo of animals", "a non-medical photograph", "a regular color photograph"
        ]
        with torch.no_grad():
            texts = self.medical_prompts + self.non_medical_prompts
            tokenized = clip.tokenize(texts).to(self.device)
            text_embeddings = self.model.encode_text(tokenized)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            self.med_emb = text_embeddings[: len(self.medical_prompts)].mean(dim=0, keepdim=True)
            self.nonmed_emb = text_embeddings[len(self.medical_prompts):].mean(dim=0, keepdim=True)
            self.med_emb = self.med_emb / self.med_emb.norm()
            self.nonmed_emb = self.nonmed_emb / self.nonmed_emb.norm()

    def classify(self, pil_img: Image.Image):
        image_input = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_emb = self.model.encode_image(image_input)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            med_score = (img_emb @ self.med_emb.T).item()
            nonmed_score = (img_emb @ self.nonmed_emb.T).item()
        label = "medical" if med_score >= nonmed_score else "non-medical"
        return label, med_score, nonmed_score

# ------------------------ Streamlit UI -------------------------
st.set_page_config(page_title="Medical vs Non-Medical Classifier", layout="wide")
st.title("🩺 Medical vs Non-Medical Image Classifier")

classifier = ZeroShotMedicalClassifier()

option = st.radio("Select Input Type", ["URL", "PDF Upload"])
images = []

if option == "URL":
    url_input = st.text_input("Enter webpage URL")
    if url_input and st.button("Extract & Classify"):
        with st.spinner("Extracting images from URL..."):
            images = extract_images_from_url(url_input)

elif option == "PDF Upload":
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf_file and st.button("Extract & Classify"):
        with st.spinner("Extracting images from PDF..."):
            images = extract_images_from_pdf(pdf_file.read())

if images:
    st.subheader(f"Found {len(images)} images")
    cols = st.columns(3)
    for idx, (img, src) in enumerate(images):
        label, med_score, nonmed_score = classifier.classify(img)
        with cols[idx % 3]:
            st.image(img, caption=f"{label} ({src})")
            st.progress(med_score if label == "medical" else nonmed_score)
            st.caption(f"Medical score: {med_score:.3f}, Non-medical score: {nonmed_score:.3f}")
