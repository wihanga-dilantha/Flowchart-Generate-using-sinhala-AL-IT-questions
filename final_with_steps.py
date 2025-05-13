import streamlit as st
from PIL import Image
import pytesseract
from google.cloud import translate_v2 as translate
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from graphviz import Digraph

# Set up authentication using the service account key
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'google_auth\\auth.json'

# Initialize the Translation client
client = translate.Client()

# Function to process the uploaded image and extract text
def extract_text_from_image(image):
    gray_image = image.convert('L')
    text = pytesseract.image_to_string(gray_image, lang='sin')
    return text

# Function to translate text from Sinhala to English
def translate_text(text, target_language='en'):
    result = client.translate(text, target_language=target_language)
    return result['translatedText']

# Function to load the GPT-2 model
def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    return model

# Function to load the GPT-2 tokenizer
def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer

# Function to generate text using the GPT-2 model
def generate_text(model_path, sequence, max_length=230):
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'question: {sequence}', return_tensors='pt')
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    return tokenizer.decode(final_outputs[0], skip_special_tokens=True)

# Function to extract the relevant part of the generated text
def output_select(get_text):
    start_marker = "# Add nodes"
    end_marker = "---"
    start_index = get_text.find(start_marker)
    end_index = get_text.find(end_marker)
    if start_index != -1 and end_index != -1:
        selected_text = get_text[start_index + len(start_marker):end_index].strip()
        return selected_text
    else:
        return "Markers not found"

# Function to generate a flowchart from the selected text
def generate_flowchart(selected_text):
    output_path = "output_data/flowchart001"
    dot = Digraph()

    lines = selected_text.strip().split('\n')

    for line in lines:
        if line.startswith('dot.node'):
            parts = line.split('(')[1].split(')')[0]
            node_info = parts.split(',')
            node_id = node_info[0].strip().strip("'")
            node_label = node_info[1].strip().strip("'")
            node_shape = node_info[2].strip().split('=')[-1].strip().strip("'")
            dot.node(node_id, node_label, shape=node_shape)

    for line in lines:
        if line.startswith('dot.edge'):
            parts = line.split('(')[1].split(')')[0]
            edge_info = parts.split(',')
            from_node = edge_info[0].strip().strip("'")
            to_node = edge_info[1].strip().strip("'")
            label = edge_info[2].strip().split('=')[-1].strip().strip("'") if len(edge_info) > 2 else None
            dot.edge(from_node, to_node, label=label)

    dot.render(output_path, format='png')
    return output_path

# Streamlit UI
st.title("Image to Flowchart Generator")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ''

if 'edited_text' not in st.session_state:
    st.session_state.edited_text = ''

if 'translated_text' not in st.session_state:
    st.session_state.translated_text = ''

if 'generated_text' not in st.session_state:
    st.session_state.generated_text = ''

if 'selected_text' not in st.session_state:
    st.session_state.selected_text = ''

model_path = "model/"

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    if st.button("Extract Text"):
        st.session_state.extracted_text = extract_text_from_image(image)
        st.session_state.edited_text = st.session_state.extracted_text
    
    st.text_area("Extracted Text:", st.session_state.extracted_text, height=200)
    
    if st.button("Translate Text"):
        st.session_state.translated_text = translate_text(st.session_state.edited_text)
    
    st.text_area("Translated Text:", st.session_state.translated_text, height=200)
    
    if st.button("Generate Text"):
        with st.spinner("Generating text..."):
            st.session_state.generated_text = generate_text(model_path, st.session_state.translated_text)
            st.session_state.selected_text = output_select(st.session_state.generated_text)
    
    st.text_area("Generated Text:", st.session_state.generated_text, height=200)
    st.text_area("Selected Text:", st.session_state.selected_text, height=200)
    
    if st.button("Generate Flowchart"):
        with st.spinner("Generating flowchart..."):
            output_path = generate_flowchart(st.session_state.selected_text)
            st.image(f'{output_path}.png')
else:
    st.warning("Please upload an image to continue.")
