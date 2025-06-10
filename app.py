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

# Global buffer to store accepted segments
final_audio_buffer = []
final_sample_rate = 24000  # default sample rate for TTS output

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
    
    if isinstance(ref_audio, tuple) and len(ref_audio) == 2:
        sample_rate, audio_data = ref_audio
    else:
        return "Error: Invalid reference audio input."
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        sf.write(temp_audio.name, audio_data, samplerate=sample_rate, format='WAV')
        temp_audio.flush()
    
    audio = model(text, ref_audio_path=temp_audio.name, ref_text=ref_text)
    
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

# Append generated audio to final track
def add_to_final_track(generated_audio):
    if generated_audio is None:
        return "Error: No generated audio to add.", None

    sample_rate, audio_data = generated_audio
    if audio_data is not None:
        final_audio_buffer.append(audio_data)

    if final_audio_buffer:
        combined_audio = np.concatenate(final_audio_buffer)
        return "Audio added to final track ✅", (final_sample_rate, combined_audio)
    else:
        return "No audio in final track.", None

# Clear the final track
def clear_final_track():
    final_audio_buffer.clear()
    return "Final track cleared.", None

# Load TTS model
repo_id = "ai4bharat/IndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load Faster-Whisper model
whisper_model = WhisperModel("large-v3", device=device.type, compute_type="int8")

# Load reference examples
EXAMPLES = [
    {
        "label": "Anjali",
        "audio_url": "https://github.com/Darshit999/indicF5/raw/main/voices/female/anjali.mp3",
        "ref_text": "क्या हाल है चाई का कप उठाओ और बैठ कर सुनो बातें जब इस आवाज में होंगी तो सुकून सा लगेगा"
    },
    {
        "label": "Arjun",
        "audio_url": "https://github.com/Darshit999/indicF5/raw/main/voices/male/arjun.mp3",
        "ref_text": "आपको परिवर्तन को नियम के रूप में स्वागत करना चाहिए, लेकिन अपने शासक के रूप में नहीं."
    },
    # Add more examples if needed...
]

for ex in EXAMPLES:
    sr, ad = load_audio_from_url(ex["audio_url"])
    ex["sample_rate"] = sr
    ex["audio_data"] = ad

SPEAKER_OPTIONS = {ex["label"]: ex for ex in EXAMPLES}

def on_speaker_selected(speaker_label):
    ex = SPEAKER_OPTIONS[speaker_label]
    return (ex["sample_rate"], ex["audio_data"]), ex["ref_text"]

# Gradio Interface
with gr.Blocks() as iface:
    gr.Markdown(
        """
        # **IndicF5**
        **Supported languages**:  
        Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu  
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
            add_btn = gr.Button("✅ Add to Final Track")
            final_status = gr.Textbox(label="Status", interactive=False)
            final_audio_output = gr.Audio(label="Final Track Preview", type="numpy")
            clear_btn = gr.Button("🔄 Clear Final Track")

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

    add_btn.click(
        add_to_final_track,
        inputs=[output_audio],
        outputs=[final_status, final_audio_output]
    )

    clear_btn.click(
        clear_final_track,
        outputs=[final_status, final_audio_output]
    )

iface.launch(share=True)
