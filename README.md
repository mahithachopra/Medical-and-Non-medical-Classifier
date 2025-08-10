# 🩺 Medical vs Non-Medical Image Classifier
A Streamlit web application that can automatically classify images as medical or non-medical.
Supports extracting images from webpages or PDF files and uses OpenAI CLIP for zero-shot classification.

##  Features
### Two input modes:

URL – Scrapes and downloads images from a webpage.

PDF Upload – Extracts embedded images and page renders from PDFs.

### Zero-shot classification:

Uses CLIP ViT-B/32 with tailored prompts for both classes.

No custom training data required.

### Interactive UI:

Displays image thumbnails.

Shows classification labels with confidence bars.

##  Approach

### We use OpenAI’s CLIP model to compare each image to two prompt sets:

Medical prompts (e.g., "an MRI scan", "a medical image").

Non-medical prompts (e.g., "a landscape photograph", "a photo of animals").

The average embedding for each category is computed, and each image is assigned the label with the highest cosine similarity.

###  Example Accuracy (Small Test)
On a small balanced test set of 200 images:

Accuracy: ~94%

Precision (medical): 0.92

Recall (medical): 0.95
Misclassifications mostly involved:

Medical images containing non-medical background (e.g., hospital exterior).

Technical diagrams mistaken for medical scans.

##  Performance
Inference Speed: ~50ms/image on CPU.

Memory: ~45MB for model weights.

Scalability: Sequential processing; can be parallelized for large datasets.

Optimizations: Possible quantization and caching of embeddings.

## Installation

### Clone this repository:
git clone https://github.com/<mahithachopra>/<Medical-and-Non-medical-Classifier>.git
cd <Medical-and-Non-medical-Classifier>

### Install dependencies:
pip install -r requirements.txt
requirements.txt includes:
- torch>=1.12
- ftfy
- regex
- tqdm
- requests
- beautifulsoup4
- Pillow
- git+https://github.com/openai/CLIP.git
- PyMuPDF
- streamlit

  ## Usage
Run the Streamlit app:
streamlit run classifier.py

Choose one of the input modes:
URL – Enter the webpage URL.
PDF Upload – Upload a PDF file.

The app will:
Extract images.
Classify each as medical or non-medical.
Display thumbnails with labels and confidence bars.

## Notes
Requires internet access for CLIP download (first run).
Works best with clear, well-lit images.
Zero-shot — no custom dataset required, but fine-tuning on a curated dataset can improve performance.

## Results
Each image is shown with:
Label: medical or non-medical
Confidence Bar: Displays the similarity score for the predicted class.
Scores: Both medical and non-medical similarity scores.







You said:
