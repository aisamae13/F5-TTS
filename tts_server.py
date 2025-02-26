from flask import Flask, request, send_file
import torch
import f5_tts
from f5_tts.api import F5TTS
import soundfile as sf
import io
import os
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Define paths for each accent's checkpoint and vocab
CHECKPOINT_MAP = {
    "en_AU": "C:/Users/Alyssa/F5-TTS/ckpts/test_au/pretrained_model_1200000_au.pt",
    "en_US": "C:/Users/Alyssa/F5-TTS/ckpts/test_us/pretrained_model_1200000_us.pt",
    "en_GB": "C:/Users/Alyssa/F5-TTS/ckpts/test_gb/pretrained_model_1200000_gb.pt"
}

VOCAB_MAP = {
    "en_AU": "C:/Users/Alyssa/F5-TTS/data/test_au_pinyin/vocab.txt",
    "en_US": "C:/Users/Alyssa/F5-TTS/data/test_us_pinyin/vocab.txt",
    "en_GB": "C:/Users/Alyssa/F5-TTS/data/test_gb_pinyin/vocab.txt"
}

models = {}

def load_model(locale):
    if locale not in models:
        checkpoint_path = CHECKPOINT_MAP.get(locale, CHECKPOINT_MAP["en_US"])
        vocab_path = VOCAB_MAP.get(locale, VOCAB_MAP["en_US"])  # Default to US vocab if not found

        if not os.path.exists(checkpoint_path):
            logging.error(f"Checkpoint file not found: {checkpoint_path}")
            return None
        if not os.path.exists(vocab_path):
            logging.warning(f"Vocab file not found: {vocab_path}, using default or empty vocab")
            vocab_path = ""  # Fall back to empty or default vocab

        # Initialize F5TTS with the checkpoint and vocab
        model = F5TTS(
            model_type="F5-TTS",  # Use F5-TTS model
            ckpt_file=checkpoint_path,  # Load the specific checkpoint
            vocab_file=vocab_path,  # Use the specific vocab for the accent
            ode_method="euler",  # Default ODE method
            use_ema=True,  # Use EMA (exponential moving average) model
            vocoder_name="vocos",  # Use vocos vocoder (ensure it’s installed)
            local_path="C:/Users/Alyssa/F5-TTS/ckpts/vocos-mel-24khz",
            device=None  # Let F5TTS detect the device (CUDA/CPU)
        )
        models[locale] = model
    return models[locale]

@app.route('/')
def home():
    return "F5-TTS Server is running!"

@app.route('/tts', methods=['POST'])
def generate_tts():
    try:
        data = request.json
        text = data.get('text', '')  # Kunin ang text mula sa Flutter app
        locale = data.get('locale', 'en_US')  # Kunin ang accent (e.g., 'en_AU', 'en_US', 'en_GB')

        if not text:
            return {"error": "No text given"}, 400

        model = load_model(locale)
        if model is None:
            return {"error": "Model failed to load"}, 500

        # Use the infer method to generate audio
        with torch.no_grad():
            voice_id = locale.split('_')[1].lower()  # e.g., 'au', 'us', 'gb'
            # Set reference audio file based on locale
            ref_file = None
            if locale == "en_AU":
                ref_file = "C:/Users/Alyssa/F5-TTS/data/test_au_pinyin/wavs/segment_1.wav"  # Use any segment, e.g., segment_1
            elif locale == "en_US":
                ref_file = "C:/Users/Alyssa/F5-TTS/data/test_us_pinyin/wavs/segment_18.wav"  # Use a different segment, e.g., segment_5
            elif locale == "en_GB":
                ref_file = "C:/Users/Alyssa/F5-TTS/data/test_gb_pinyin/wavs/segment_10.wav"  # Use another segment, e.g., segment_10

            wav, sr, _ = model.infer(
                ref_file=ref_file,
                ref_text="",  # Maaari itong iwanang blangko o idagdag ang reference text kung kinakailangan
                gen_text=text,
                show_info=lambda x: logging.info(x),
                progress=None,
                target_rms=0.1,
                cross_fade_duration=0.15,
                sway_sampling_coef=-1,
                cfg_strength=2,
                nfe_step=32,
                speed=1.0,
                fix_duration=None,
                remove_silence=False,
                file_wave=None,
                file_spect=None
            )

        # Ensure wav is a numpy array or tensor, convert to numpy if needed
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()

        # Save audio to a buffer and send it
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, wav, sr, format='wav')  # Use the sample rate from infer (16000 Hz)
        audio_buffer.seek(0)

        return send_file(
            audio_buffer,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='output.wav'
        )

    except Exception as e:
        logging.error(f"Error generating TTS: {e}")
        return {"error": str(e)}, 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)