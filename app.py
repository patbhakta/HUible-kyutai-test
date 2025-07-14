import gradio as gr
import numpy as np
import sphn
import tempfile
import torch
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import TTSModel

# --- 1. Global Model Loading (Updated Links) ---
# Correct Hugging Face repository links
MODEL_REPO = "kyutai/tts-1.6b-en_fr"
VOICE_REPO = "kyutai/tts-voices"

print("Initializing Text-to-Speech model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    # Load model from the updated repository
    checkpoint_info = CheckpointInfo.from_hf_repo(MODEL_REPO)
    tts_model = TTSModel.from_checkpoint_info(
        checkpoint_info, n_q=32, temp=0.6, device=device
    )
    # Set the voice repo on the loaded model
    tts_model.voice_repo = VOICE_REPO
    MODEL_LOADED = True
    print("Model loaded successfully.")
except Exception as e:
    MODEL_LOADED = False
    print(f"FATAL: Error loading model: {e}")
    def tts_model():
        raise RuntimeError("TTS Model could not be loaded. Please check your setup and dependencies.")

# --- 2. Voice Options ---
# A curated list of available voices from the Expresso dataset in the correct repo
VOICE_OPTIONS = {
    
    "💬 Default": "expresso/ex01-ex02_default_001_channel1_168s.wav",
    
    # Expressive Styles
    "😊 Happy": "expresso/ex03-ex01_happy_001_channel1_334s.wav",
    "😡 Angry": "expresso/ex03-ex01_angry_001_channel1_201s.wav",
    "😲 Awe": "expresso/ex03-ex01_awe_001_channel1_1323s.wav",
    "😌 Calm": "expresso/ex03-ex01_calm_001_channel1_1143s.wav",
    "🤔 Confused": "expresso/ex03-ex01_confused_001_channel1_909s.wav",
    "😍 Desire": "expresso/ex03-ex01_desire_004_channel1_545s.wav",
    "🤢 Disgusted": "expresso/ex03-ex01_disgusted_004_channel1_170s.wav",
    "😂 Laughing": "expresso/ex03-ex01_laughing_001_channel1_188s.wav",
    "😒 Sarcastic": "expresso/ex03-ex01_sarcastic_001_channel1_435s.wav",
    "😴 Sleepy": "expresso/ex03-ex01_sleepy_001_channel1_619s.wav",

    # Delivery Styles
    "🗣 Enunciated": "expresso/ex01-ex02_enunciated_001_channel1_432s.wav",
    "⏩ Fast": "expresso/ex01-ex02_fast_001_channel1_104s.wav",
    "📢 Projected": "expresso/ex01-ex02_projected_001_channel1_46s.wav",
    "🤫 Whisper": "expresso/ex01-ex02_whisper_001_channel1_579s.wav",
    "🎤 Narration": "expresso/ex03-ex02_narration_001_channel1_674s.wav",

    # Emotional Interactions
    "😢 Sad-Sympathetic": "expresso/ex03-ex02_sad-sympathetic_001_channel1_454s.wav",
    "🙏 Sympathetic-Sad": "expresso/ex03-ex02_sympathetic-sad_008_channel1_215s.wav",

    # Role-Based Voices
    "🐶 Animal to AnimalDir": "expresso/ex03-ex02_animal-animaldir_002_channel2_89s.wav",
    "👶 Child to ChildDir": "expresso/ex03-ex02_child-childdir_001_channel1_291s.wav",
    "🧒 ChildDir to Child": "expresso/ex03-ex02_childdir-child_004_channel1_308s.wav",

    # Miscellaneous
    "🤐 Nonverbal": "expresso/ex03-ex01_nonverbal_001_channel2_37s.wav",

    # Additional Expressive Files
    "😄 Happy (Channel 2)": "expresso/ex03-ex01_happy_001_channel2_257s.wav",
    "😠 Angry (Channel 2)": "expresso/ex03-ex01_angry_001_channel2_181s.wav",
    "😮 Awe (Channel 2)": "expresso/ex03-ex01_awe_001_channel2_1290s.wav",
    "🧘 Calm (Channel 2)": "expresso/ex03-ex01_calm_001_channel2_1081s.wav",
    "😕 Confused (Channel 2)": "expresso/ex03-ex01_confused_001_channel2_816s.wav",
    "😍 Desire (Channel 2)": "expresso/ex03-ex01_desire_004_channel2_580s.wav",
    "😒 Sarcastic (Channel 2)": "expresso/ex03-ex01_sarcastic_001_channel2_491s.wav",
    "😴 Sleepy (Channel 2)": "expresso/ex03-ex01_sleepy_001_channel2_662s.wav",
    "😂 Laughing (Channel 2)": "expresso/ex03-ex01_laughing_002_channel2_232s.wav",

    # Extra Narration & Delivery
    "🎤 Narration (Channel 2)": "expresso/ex03-ex02_narration_002_channel2_1136s.wav",
    "📢 Projected (Channel 2)": "expresso/ex01-ex02_projected_002_channel2_248s.wav",
    "🤫 Whisper (Channel 2)": "expresso/ex01-ex02_whisper_001_channel2_717s.wav",
    "💬 Default (Channel 2)": "expresso/ex01-ex02_default_001_channel2_198s.wav",
    "⏩ Fast (Channel 2)": "expresso/ex01-ex02_fast_001_channel2_73s.wav",
    "🗣 Enunciated (Channel 2)": "expresso/ex01-ex02_enunciated_001_channel2_354s.wav",

    # New Categories from ex04
    "😡 Angry (ex04)": "expresso/ex04-ex02_angry_001_channel1_119s.wav",
    "😲 Awe (ex04)": "expresso/ex04-ex02_awe_001_channel1_982s.wav",
    "😌 Calm (ex04 Channel 1)": "expresso/ex04-ex02_calm_002_channel1_480s.wav",
    "😌 Calm (ex04 Channel 2)": "expresso/ex04-ex02_calm_001_channel2_336s.wav",
    "🤔 Confused (ex04)": "expresso/ex04-ex02_confused_001_channel1_499s.wav",
    "😍 Desire (ex04)": "expresso/ex04-ex02_desire_001_channel1_657s.wav",
    "🤢 Disgusted (ex04)": "expresso/ex04-ex02_disgusted_004_channel1_169s.wav",
    "😂 Laughing (ex04)": "expresso/ex04-ex02_laughing_001_channel1_147s.wav",
    "😒 Sarcastic (ex04)": "expresso/ex04-ex02_sarcastic_001_channel1_519s.wav",
    "😴 Bored (ex04)": "expresso/ex04-ex02_bored_001_channel1_254s.wav",
    "😨 Fearful (ex04)": "expresso/ex04-ex02_fearful_001_channel1_316s.wav",
    "🙂 Happy (ex04)": "expresso/ex04-ex02_happy_001_channel1_118s.wav",
    "🤐 Nonverbal (ex04)": "expresso/ex04-ex02_nonverbal_004_channel1_18s.wav",
    "🗣 Enunciated (ex04)": "expresso/ex04-ex02_enunciated_001_channel1_496s.wav",
    "📢 Projected (ex04)": "expresso/ex04-ex03_projected_001_channel1_192s.wav",
    "⏩ Fast (ex04)": "expresso/ex04-ex03_fast_001_channel1_208s.wav",
    "🤫 Whisper (ex04)": "expresso/ex04-ex03_whisper_001_channel1_198s.wav",
}

# --- 3. Core Synthesis Function ---
def generate_speech(text: str, voice_choice: str):
    """
    Generates audio from text using the pre-loaded TTS model.
    """
    if not MODEL_LOADED:
        raise gr.Error("Model is not loaded. Cannot generate audio.")
    if not text.strip():
        raise gr.Error("Input text cannot be empty.")

    print(f"Generating speech for text: '{text[:30]}...' with voice: '{voice_choice}'")

    entries = tts_model.prepare_script([text], padding_between=1)
    voice_path = tts_model.get_voice_path(VOICE_OPTIONS[voice_choice])
    condition_attributes = tts_model.make_condition_attributes(
        [voice_path], cfg_coef=2.0
    )

    result = tts_model.generate([entries], [condition_attributes])

    with tts_model.mimi.streaming(1), torch.no_grad():
        pcms = []
        for frame in result.frames[tts_model.delay_steps :]:
            pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
            pcms.append(np.clip(pcm[0, 0], -1, 1))
        
        if not pcms:
            raise gr.Error("Audio generation failed. The model did not produce any audio frames.")

        pcm_data = np.concatenate(pcms, axis=-1)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
        sphn.write_wav(fp.name, pcm_data, tts_model.mimi.sample_rate)
        print(f"Audio saved to temporary file: {fp.name}")
        return fp.name


# --- 4. Gradio Interface Definition ---
def create_gradio_app():
    """Creates and returns the Gradio web interface."""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🗣️ HUible Text-to-Speech Demo
            Enter some text, choose a voice style, and click "Generate" to create the audio.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Input Text",
                    lines=5,
                    placeholder="Hey, what's goin on?",
                )
                voice_input = gr.Dropdown(
                    label="Voice Selection",
                    choices=list(VOICE_OPTIONS.keys()),
                    value="💬 Default", # Default voice
                )
                generate_button = gr.Button("Generate Audio", variant="primary")

            with gr.Column(scale=1):
                audio_output = gr.Audio(label="Generated Speech", type="filepath")

        generate_button.click(
            fn=generate_speech,
            inputs=[text_input, voice_input],
            outputs=audio_output,
            api_name="tts"
        )

        gr.Markdown(
            f"""
            ---
            *Model: [{MODEL_REPO}](https://huggingface.co/{MODEL_REPO}) • Voices: [{VOICE_REPO}](https://huggingface.co/{VOICE_REPO})*
            "Pat Bhakta - 2025"
            """
        )
    return demo

# --- 5. Main Execution ---
if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(share=True)
