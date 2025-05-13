# 🧠 Sinhala A/L IT Flowchart Generator

> This project takes an image of a Sinhala A/L Information Technology (IT) flowchart question, translates it to English, understands the logic using a fine-tuned GPT-2 model, and generates a visual flowchart.

---

## ✨ Features

- 📷 Upload Sinhala IT flowchart question images
- 🔠 Uses OCR to read Sinhala text
- 🌐 Translates Sinhala text to English using Google Cloud Translate API
- 🧠 Fine-tuned GPT2 model understands the translated question
- 🔄 Generates a structured flowchart from the AI output
- 🖼️ Web interface built with Streamlit
- 💾 Download or view the generated flowchart

---

## 🛠️ Tech Stack

- 🐍 **Language**: Python
- 📓 **Notebooks**: Jupyter (`.ipynb`) for fine-tuning GPT-2
- 🧠 **AI Model**: GPT2LMHeadModel (Hugging Face Transformers)
- 📝 **OCR**: Tesseract OCR (Sinhala language support)
- 🌐 **Translation**: Google Cloud Translate API
- 🌟 **Interface**: Streamlit
- 🔧 **Other Libraries**: OpenCV, Graphviz, Pillow

---

## 🚀 Getting Started

```bash
# Clone the repository
https://github.com/wihanga-dilantha/Flowchart-Generate-using-sinhala-AL-IT-questions.git

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Model Training
  # you can train the model using model_training.ipynb (I used google colab to train the model)

# Translator - Get the google cloud translator authentication json  (paid tool) and add it to the google_auth\auth.json
  # you can also use the python library for that but the cloud translator seems more accurate

# Run the Streamlit app
streamlit run with_steps.py
   # for step by step flowchart generation (you can use this to understand the process, or debug purposes)
streamlit run simplified_interfaces.py
   # for automatic flowchart generation (this file hides the step by step generation)
