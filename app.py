import io
import torch
import librosa
import requests
import tempfile
import numpy as np
import gradio as gr
import soundfile as sf
from transformers import AutoModel
from faster_whisper import WhisperModel

# Function to load reference audio from URL
def load_audio_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        audio_data, sample_rate = sf.read(io.BytesIO(response.content))
        return sample_rate, audio_data
    return None, None

# Synthesize speech using IndicF5
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

# Transcribe reference audio using Faster-Whisper
def transcribe_ref_audio(ref_audio):
    if ref_audio is None:
        return "Error: Please provide a reference audio to transcribe."

    if isinstance(ref_audio, tuple) and len(ref_audio) == 2:
        sample_rate, audio_data = ref_audio
    else:
        return "Error: Invalid reference audio input."

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        sf.write(temp_audio.name, audio_data, samplerate=sample_rate)
        segments, _ = whisper_model.transcribe(temp_audio.name)

    transcription = " ".join([seg.text for seg in segments])
    return transcription

# Load TTS model
repo_id = "ai4bharat/IndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load Faster-Whisper transcription model
whisper_model = WhisperModel("large-v3", device=device.type, compute_type="int8")

print("Device:", device)

# Example Data (Multiple Examples)
EXAMPLES = [
    {
        "label": "Anjali",
        "audio_url": "https://github.com/Darshit999/indicF5/raw/main/voices/female/anjali.mp3",
        "ref_text": "क्या हाल है चाई का कप उठाओ और बैठ कर सुनो बातें जब इस आवाज में होंगी तो सुकून सा लगेगा"
    },
    {
        "label": "Anika",
        "audio_url": "https://github.com/Darshit999/indicF5/raw/main/voices/female/anika.mp3",
        "ref_text": " छोटी गुडिया ने जैसे ही जादूई दर्वाजा खोला, एक चमकदार दुनिया उसके सामने थी। उसने कहा, क्या यहां सच में बोलने वाले खरगोश हैं।"
    },
    {
        "label": "Diya",
        "audio_url": "https://github.com/Darshit999/indicF5/raw/main/voices/female/diya.mp3",
        "ref_text": " जो कहानी मैं सुनाने वाली हूँ, उसकी कोई तस्वीर नहीं, कोई गवाह नहीं, सिर्फ एक डर, जो अब तक मेरे अंदर जिन्दा है।"
    },
    {
        "label": "Jessica",
        "audio_url": "https://github.com/Darshit999/indicF5/raw/main/voices/female/jessica.mp3",
        "ref_text": "जो व्यक्ति चक्कर खा रहा होता है, वह समझता है कि दुनिया घूम रही है।"
    },
    {
        "label": "Monika",
        "audio_url": "https://github.com/Darshit999/indicF5/raw/main/voices/female/monika.mp3",
        "ref_text": "ईलेवन लेब्स की सबसे सफल भारतिय आवाज अब हिंदी में, तो आईए कुछ बहतरीन बनाते हैं।"
    },
    {
        "label": "Muskaan",
        "audio_url": "https://github.com/Darshit999/indicF5/raw/main/voices/female/muskaan.mp3",
        "ref_text": "अरे बैठो न, आराम से कहानी सुनते हैं, बातों बातों में सब समझ आ जाएगा"
    },
    {
        "label": "Meher",
        "audio_url": "https://github.com/Darshit999/indicF5/raw/main/voices/female/meher.mp3",
        "ref_text": "वो दर्वाजा धीरे धीरे खुला. अंदर अंधेरा था. लेकिन अजीब ये था कि किसी ने दर्वाजा खोला नहीं था."
    },
    
    {
        "label": "Kunal",
        "audio_url": "https://github.com/Darshit999/indicF5/raw/main/voices/male/kunal.mp3",
        "ref_text": "अगर आप उस समय मुस्कुराते हैं जब आपके आसपास कोई नहीं होता, तो आप वास्तव में इसका मतलब समझते हैं।"
    },
    {
        "label": "Yatin",
        "audio_url": "https://github.com/Darshit999/indicF5/raw/main/voices/male/yatin.mp3",
        "ref_text": "एक गहरा रहस्य हवा मै तैर रहा था, हर सांस के साथ करीब आता हुआ, और किसी को नही पता था की अगली पल क्या लेकर आएगा।"
    },
    {
        "label": "Neeraj",
        "audio_url": "https://github.com/Darshit999/indicF5/raw/main/voices/male/neeraj.wav",
        "ref_text": "भारतिय जीवन धारा में जिन ग्रंथों का महत्व पूर्ण स्थान है। उनमें पुरान भक्ती ग्रंथों के रूप में बहुत महत्व पूर्ण माने जाते हैं।"
    },
    {
        "label": "Rudra",
        "audio_url": "https://github.com/Darshit999/indicF5/raw/main/voices/male/rudra.mp3",
        "ref_text": "कितनी अजीब बात है कि आजकल लोगों ने किताबें पढ़ना छोड़ ही दिया है। अब किताबें अगर सुननी ही हैं, तो अच्छी आवाज में सुनो।"
    },
    {
        "label": "Viraj",
        "audio_url": "https://github.com/Darshit999/indicF5/raw/main/voices/male/viraj.mp3",
        "ref_text": "कुछ तो है जो छिपा हुआ है। हवा में हलकी सी गड़-गड़ा हट जैसे कोई राज खुलने वाला हो। क्या आपने वो महसूस किया?"
    },
    {
        "label": "Vikas",
        "audio_url": "https://github.com/Darshit999/indicF5/raw/main/voices/male/vikas.mp3",
        "ref_text": "अरे हा, मुझे वो दिन अच्छी तरह याद है, सब लोग इतने खुश थे और माहोल भी बहोत खास था।"
    },
    {
        "label": "Arjun",
        "audio_url": "https://github.com/Darshit999/indicF5/raw/main/voices/male/arjun.mp3",
        "ref_text": "आपको परिवर्तन को नियम के रूप में स्वागत करना चाहिए, लेकिन अपने शासक के रूप में नहीं."
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
        **Supported languages**:  
        **Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu.**  
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
            transcribe_btn = gr.Button("📝 Transcribe Reference Audio")
        
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
    
    transcribe_btn.click(
        transcribe_ref_audio,
        inputs=[ref_audio_input],
        outputs=[ref_text_input]
    )

iface.launch(share=True)