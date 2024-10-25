import whisper
import gradio as gr
from groq import Groq
from deep_translator import GoogleTranslator
from diffusers import StableDiffusionPipeline
import os
import torch
import openai
from huggingface_hub import InferenceApi
from PIL import Image
import requests
import io
import time
 
# Set up Groq API key
api_key = os.getenv("g_key")
client = Groq(api_key=api_key)
 
# Hugging Face API details for image generation
key = os.getenv("h_key")
API_URL = "https://api-inference.huggingface.co/models/Artples/LAI-ImageGeneration-vSDXL-2"
headers = {"Authorization": f"Bearer {key}"}
 
 
# Function for querying image generation with retries
def query_image_generation(payload, max_retries=5):
    for attempt in range(max_retries):
        response = requests.post(API_URL, headers=headers, json=payload)
 
        if response.status_code == 503:
            print(f"Model is still loading, retrying... Attempt {attempt + 1}/{max_retries}")
            estimated_time = min(response.json().get("estimated_time", 60), 60)
            time.sleep(estimated_time)
            continue
 
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            print(f"Response: {response.text}")
            return None
 
        return response.content
 
    print(f"Failed to generate image after {max_retries} attempts.")
    return None
 
# Function for generating an image from text
def generate_image(prompt):
    image_bytes = query_image_generation({"inputs": prompt})
 
    if image_bytes is None:
        return None
 
    try:
        image = Image.open(io.BytesIO(image_bytes))  # Opening the image from bytes
        return image
    except Exception as e:
        print(f"Error: {e}")
        return None
 
 
# Updated function for text generation using the new API structure
def generate_creative_text(prompt):
    chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "user", "content":prompt}
                ],
                model="llama-3.2-90b-text-preview"
            )
    chatbot_response = chat_completion.choices[0].message.content
    return chatbot_response
 
 
def process_audio(audio_path, image_option, creative_text_option):
    if audio_path is None:
        return "Please upload an audio file.", None, None, None
 
    # Step 1: Transcribe audio
    try:
        with open(audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), file.read()),
                model="whisper-large-v3",
                language="ta",
                response_format="verbose_json",
            )
        tamil_text = transcription.text
    except Exception as e:
        return f"An error occurred during transcription: {str(e)}", None, None, None
    # Step 2: Translate Tamil to English
    try:
        translator = GoogleTranslator(source='ta', target='en')
        translation = translator.translate(tamil_text)
    except Exception as e:
        return tamil_text, f"An error occurred during translation: {str(e)}", None, None
 
    # Step 3: Generate creative text (if selected)
    creative_text = None
    if creative_text_option == "Generate Creative Text":
        creative_text = generate_creative_text(translation)
 
    # Step 4: Generate image (if selected)
    image = None
    if image_option == "Generate Image":
        image = generate_image(translation)
        if image is None:
            return tamil_text, translation, creative_text, f"An error occurred during image generation"
 
    return tamil_text, translation, creative_text, image      
 
 
# Create Gradio interface
with gr.Blocks(theme=gr.themes.Base()) as iface:
    gr.Markdown("# Audio Transcription, Translation, Image & Creative Text Generation")
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Upload Audio File")
            image_option = gr.Dropdown(["Generate Image", "Skip Image"], label="Image Generation", value="Generate Image")
            creative_text_option = gr.Dropdown(["Generate Creative Text", "Skip Creative Text"], label="Creative Text Generation", value="Generate Creative Text")
            submit_button = gr.Button("Process Audio")
        with gr.Column():
            tamil_text_output = gr.Textbox(label="Tamil Transcription")
            translation_output = gr.Textbox(label="English Translation")
            creative_text_output = gr.Textbox(label="Creative Text")
            image_output = gr.Image(label="Generated Image")
    submit_button.click(
        fn=process_audio,
        inputs=[audio_input, image_option, creative_text_option],
        outputs=[tamil_text_output, translation_output, creative_text_output, image_output]
    )
 
# Launch the interface
iface.launch()
