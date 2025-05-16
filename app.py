import io
import torch
import librosa
import requests
import tempfile
import numpy as np
import gradio as gr
import soundfile as sf
from transformers import AutoModel

# Function to load reference audio from URL
def load_audio_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        audio_data, sample_rate = sf.read(io.BytesIO(response.content))
        return sample_rate, audio_data
    return None, None

def synthesize_speech(text, ref_audio, ref_text):
    if ref_audio is None or ref_text.strip() == "":
        return "Error: Please provide a reference audio and its corresponding text."
    
    # Ensure valid reference audio input
    if isinstance(ref_audio, tuple) and len(ref_audio) == 2:
        sample_rate, audio_data = ref_audio
    else:
        return "Error: Invalid reference audio input."
    
    # Save reference audio directly without resampling
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        sf.write(temp_audio.name, audio_data, samplerate=sample_rate, format='WAV')
        temp_audio.flush()
    
    audio = model(text, ref_audio_path=temp_audio.name, ref_text=ref_text)
             
    # Normalize output and save
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    return 24000, audio


# Load TTS model
repo_id = "ai4bharat/IndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device", device)
model = model.to(device)

# Example Data (Multiple Examples)
EXAMPLES = [
    {
        "label": "PAN_F (Happy)",
        "audio_url": "https://github.com/AI4Bharat/IndicF5/raw/refs/heads/main/prompts/PAN_F_HAPPY_00002.wav",
        "ref_text": "ਇੱਕ ਗ੍ਰਾਹਕ ਨੇ ਸਾਡੀ ਬੇਮਿਸਾਲ ਸੇਵਾ ਬਾਰੇ ਦਿਲੋਂਗਵਾਹੀ ਦਿੱਤੀ ਜਿਸ ਨਾਲ ਸਾਨੂੰ ਅਨੰਦ ਮਹਿਸੂਸ ਹੋਇਆ।"
    },
    {
        "label": "TAM_F (Happy)",
        "audio_url": "https://github.com/AI4Bharat/IndicF5/raw/refs/heads/main/prompts/TAM_F_HAPPY_00001.wav",
        "ref_text": "நான் நெனச்ச மாதிரியே அமேசான்ல பெரிய தள்ளுபடி வந்திருக்கு..."
    },
    {
        "label": "MAR_F (WIKI)",
        "audio_url": "https://github.com/AI4Bharat/IndicF5/raw/refs/heads/main/prompts/MAR_F_WIKI_00001.wav",
        "ref_text": "दिगंतराव्दारे अंतराळ कक्षेतला कचरा चिन्हित करण्यासाठी प्रयत्न केले जात आहे."
    },
    {
        "label": "MAR_M (WIKI)",
        "audio_url": "https://github.com/AI4Bharat/IndicF5/raw/refs/heads/main/prompts/MAR_M_WIKI_00001.wav",
        "ref_text": "या प्रथाला एकोणीसशे पंचातर ईसवी पासून भारतीय दंड संहिताची धारा चारशे अठ्ठावीस..."
    },
    {
        "label": "KAN_F (Happy)",
        "audio_url": "https://github.com/AI4Bharat/IndicF5/raw/refs/heads/main/prompts/KAN_F_HAPPY_00001.wav",
        "ref_text": "ನಮ್‌ ಫ್ರಿಜ್ಜಲ್ಲಿ ಕೂಲಿಂಗ್‌ ಸಮಸ್ಯೆ ಆಗಿ ನಾನ್‌ ಭಾಳ ದಿನದಿಂದ ಒದ್ದಾಡ್ತಿದ್ದೆ..."
    },
]


# Preload audio
for ex in EXAMPLES:
    sr, ad = load_audio_from_url(ex["audio_url"])
    ex["sample_rate"] = sr
    ex["audio_data"] = ad

SPEAKER_OPTIONS = {ex["label"]: ex for ex in EXAMPLES}

# On speaker select
def on_speaker_selected(speaker_label):
    ex = SPEAKER_OPTIONS[speaker_label]
    return (ex["sample_rate"], ex["audio_data"]), ex["ref_text"]

# Define Gradio interface with layout adjustments
with gr.Blocks() as iface:
    gr.Markdown(
        """
        # **IndicF5**
<<<<<<< HEAD
        **Supported languages**:  
        **Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu.**  
=======

        **Supported languages**:  
        **Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu.**  
        
>>>>>>> 88f10a81f57efc7d545895510d9ab03989008903
        """
    )
    
    with gr.Row():
        with gr.Column():
            speaker_dropdown = gr.Dropdown(
                choices=list(SPEAKER_OPTIONS.keys()),
                label="Select Reference Speaker"
            )
            text_input = gr.Textbox(label="Text to Synthesize", lines=3)
            ref_audio_input = gr.Audio(type="numpy", label="Reference Prompt Audio")
            ref_text_input = gr.Textbox(label="Text in Reference Prompt Audio", lines=2)
            submit_btn = gr.Button("🎤 Generate Speech", variant="primary")
        
        with gr.Column():
            output_audio = gr.Audio(label="Generated Speech", type="numpy")
    
    # Event: Load only reference audio & text
    speaker_dropdown.change(
        fn=on_speaker_selected,
        inputs=[speaker_dropdown],
        outputs=[ref_audio_input, ref_text_input]
    )

    submit_btn.click(
        synthesize_speech, 
        inputs=[text_input, ref_audio_input, ref_text_input], 
        outputs=[output_audio]
    )


iface.launch(share=True)