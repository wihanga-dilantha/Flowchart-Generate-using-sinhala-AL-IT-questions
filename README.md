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
git clone https://github.com/your-username/sinhala-flowchart-generator.git

# Enter the folder
cd sinhala-flowchart-generator

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
