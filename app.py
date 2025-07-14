# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "moshi==0.2.7",
#     "torch",
#     "sphn",
#     "sounddevice",
#     "gradio"
# ]
# You might need portaudio as well - sudo apt install portaudio19-dev
#
# uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 moshi sphn sounddevice gradio
# ///

import gradio as gr
import numpy as np
import sphn
import tempfile
import torch
import platform
import subprocess
import sys
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import TTSModel

# --- Enhanced Device Detection ---
def get_optimal_device():
    """
    Detects and returns the best available device for computation.
    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    device_info = {}
    
    # Check CUDA availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_info['type'] = 'CUDA'
        device_info['name'] = torch.cuda.get_device_name(0)
        device_info['memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        print(f"✅ CUDA detected: {device_info['name']} ({device_info['memory']})")
        return device, device_info
    
    # Check MPS (Apple Silicon) availability
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_info['type'] = 'MPS'
        device_info['name'] = 'Apple Silicon GPU'
        device_info['memory'] = 'Shared system memory'
        print(f"✅ MPS detected: Apple Silicon GPU acceleration enabled")
        return device, device_info
    
    # Fallback to CPU
    device = torch.device("cpu")
    device_info['type'] = 'CPU'
    device_info['name'] = platform.processor() or 'Unknown CPU'
    device_info['memory'] = 'System RAM'
    print(f"⚠️  Using CPU: {device_info['name']}")
    return device, device_info

def check_gpu_libraries():
    """Check if proper GPU libraries are installed"""
    print("\n🔍 Checking GPU library installation...")
    
    # Check PyTorch installation
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version (PyTorch): {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")
    
    # Check MPS
    if hasattr(torch.backends, 'mps'):
        print(f"MPS available: {torch.backends.mps.is_available()}")
        if torch.backends.mps.is_available():
            print("MPS backend is built and available")
    
    # Check system info
    print(f"Platform: {platform.system()} {platform.release()}")
    
    # NVIDIA GPU check on Linux/Windows
    if platform.system() in ['Linux', 'Windows']:
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("✅ nvidia-smi available - NVIDIA driver installed")
            else:
                print("❌ nvidia-smi not found or failed")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ nvidia-smi not available")

def optimize_model_for_device(model, device):
    """Apply device-specific optimizations"""
    if device.type == 'cuda':
        # Enable CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        print("✅ CUDA optimizations enabled")
    elif device.type == 'mps':
        # MPS-specific optimizations
        print("✅ MPS optimizations enabled")
    
    return model

# --- 1. Global Model Loading (Updated Links) ---
# Correct Hugging Face repository links
MODEL_REPO = "kyutai/tts-1.6b-en_fr"
VOICE_REPO = "kyutai/tts-voices"

print("=" * 60)
print("🚀 Initializing Text-to-Speech System")
print("=" * 60)

# Enhanced device detection
check_gpu_libraries()
device, device_info = get_optimal_device()

print(f"\n📱 Selected device: {device} ({device_info['type']})")
print(f"💾 Memory: {device_info['memory']}")

# Global variables
tts_model = None
MODEL_LOADED = False

def load_model():
    """Load the TTS model with proper error handling"""
    global tts_model, MODEL_LOADED

    try:
        print(f"\n📥 Loading model from {MODEL_REPO}...")
        checkpoint_info = CheckpointInfo.from_hf_repo(MODEL_REPO)
        
        tts_model = TTSModel.from_checkpoint_info(
            checkpoint_info, 
            n_q=32, 
            temp=0.6, 
            device=device
        )
        
        tts_model = optimize_model_for_device(tts_model, device)
        tts_model.voice_repo = VOICE_REPO

        MODEL_LOADED = True
        print("✅ Model loaded successfully!")
        print(f"🎯 Model loaded on device: {device}")
        return True

    except Exception as e:
        MODEL_LOADED = False
        print(f"❌ FATAL: Error loading model: {e}")
        print("💡 Troubleshooting tips:")
        print("   1. Check internet connection for model download")
        print("   2. Verify dependencies are installed: pip install torch moshi sphn")
        print("   3. For CUDA: Install CUDA-compatible PyTorch")
        print("   4. For Apple Silicon: Use PyTorch with MPS support")
        print("   5. Try clearing cache: rm -rf ~/.cache/huggingface/")
        print(f"   6. Full error: {str(e)}")
        
        tts_model = None
        return False

# Try to load model on startup
print("\n🔄 Attempting to load model...")
load_success = load_model()

# --- 2. Voice Options (keeping your existing options) ---
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

    # Additional options (keeping your full list)
    "😄 Happy (Channel 2)": "expresso/ex03-ex01_happy_001_channel2_257s.wav",
    "😠 Angry (Channel 2)": "expresso/ex03-ex01_angry_001_channel2_181s.wav",
    "😮 Awe (Channel 2)": "expresso/ex03-ex01_awe_001_channel2_1290s.wav",
    "🧘 Calm (Channel 2)": "expresso/ex03-ex01_calm_001_channel2_1081s.wav",
    "😕 Confused (Channel 2)": "expresso/ex03-ex01_confused_001_channel2_816s.wav",
    "😍 Desire (Channel 2)": "expresso/ex03-ex01_desire_004_channel2_580s.wav",
    "😒 Sarcastic (Channel 2)": "expresso/ex03-ex01_sarcastic_001_channel2_491s.wav",
    "😴 Sleepy (Channel 2)": "expresso/ex03-ex01_sleepy_001_channel2_662s.wav",
    "😂 Laughing (Channel 2)": "expresso/ex03-ex01_laughing_002_channel2_232s.wav",
    "🎤 Narration (Channel 2)": "expresso/ex03-ex02_narration_002_channel2_1136s.wav",
    "📢 Projected (Channel 2)": "expresso/ex01-ex02_projected_002_channel2_248s.wav",
    "🤫 Whisper (Channel 2)": "expresso/ex01-ex02_whisper_001_channel2_717s.wav",
    "💬 Default (Channel 2)": "expresso/ex01-ex02_default_001_channel2_198s.wav",
    "⏩ Fast (Channel 2)": "expresso/ex01-ex02_fast_001_channel2_73s.wav",
    "🗣 Enunciated (Channel 2)": "expresso/ex01-ex02_enunciated_001_channel2_354s.wav",
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

# --- 3. Enhanced Core Synthesis Function ---
def generate_speech(text: str, voice_choice: str):
    """
    Generates audio from text using the pre-loaded TTS model with proper device handling.
    """
    global tts_model, MODEL_LOADED
    
    # Check if model is loaded, try to reload if not
    if not MODEL_LOADED or tts_model is None:
        print("🔄 Model not loaded, attempting to reload...")
        gr.Info("Model not loaded, attempting to reload...")
        
        if not load_model():
            error_msg = "❌ Model failed to load. Please check your internet connection and dependencies."
            print(error_msg)
            raise gr.Error(error_msg)
    
    if not text.strip():
        raise gr.Error("Input text cannot be empty.")

    print(f"\n🎙️ Generating speech...")
    print(f"📝 Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
    print(f"🎭 Voice: {voice_choice}")
    print(f"🖥️ Device: {device} ({device_info['type']})")

    try:
        # Prepare the script
        entries = tts_model.prepare_script([text], padding_between=1)
        
        # Get voice path and prepare conditions
        voice_path = tts_model.get_voice_path(VOICE_OPTIONS[voice_choice])
        condition_attributes = tts_model.make_condition_attributes(
            [voice_path], cfg_coef=2.0
        )

        # Generate with proper device context
        with torch.no_grad():
            if device.type == 'cuda':
                torch.cuda.empty_cache()  # Clear GPU cache
            
            result = tts_model.generate([entries], [condition_attributes])

            # Process the audio frames
            with tts_model.mimi.streaming(1):
                pcms = []
                for i, frame in enumerate(result.frames[tts_model.delay_steps:]):
                    if i % 10 == 0:  # Progress indicator
                        print(f"🔄 Processing frame {i}...")
                    
                    # Ensure frame is on correct device
                    if frame.device != device:
                        frame = frame.to(device)
                    
                    pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
                    pcms.append(np.clip(pcm[0, 0], -1, 1))
                
                if not pcms:
                    raise gr.Error("Audio generation failed. The model did not produce any audio frames.")

                pcm_data = np.concatenate(pcms, axis=-1)
                print(f"✅ Generated {len(pcm_data)} audio samples")

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            sphn.write_wav(fp.name, pcm_data, tts_model.mimi.sample_rate)
            print(f"💾 Audio saved to: {fp.name}")
            return fp.name

    except Exception as e:
        print(f"❌ Error during generation: {e}")
        raise gr.Error(f"Speech generation failed: {str(e)}")
    finally:
        # Clean up GPU memory if using CUDA
        if device.type == 'cuda':
            torch.cuda.empty_cache()

# --- 4. Enhanced Gradio Interface ---
def create_gradio_app():
    """Creates and returns the Gradio web interface with device info."""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        # Status indicator
        status_msg = "✅ Model Loaded Successfully" if MODEL_LOADED else "❌ Model Failed to Load"
        status_color = "green" if MODEL_LOADED else "red"
        
        gr.Markdown(
            f"""
            # 🗣️ HUible Text-to-Speech Demo
            
            **System Info:**
            - 🖥️ Device: {device_info['type']} ({device_info['name']})
            - 💾 Memory: {device_info['memory']}
            - 🔧 PyTorch: {torch.__version__}
            - 📊 Status: <span style="color: {status_color}">{status_msg}</span>
            
            Enter some text, choose a voice style, and click "Generate" to create the audio.
            
            {"" if MODEL_LOADED else "⚠️ **Note:** Model failed to load. The app will attempt to reload when you generate audio."}
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
                    value="💬 Default",
                )
                
                with gr.Row():
                    generate_button = gr.Button("Generate Audio", variant="primary")
                    reload_button = gr.Button("Reload Model", variant="secondary")

            with gr.Column(scale=1):
                audio_output = gr.Audio(label="Generated Speech", type="filepath")

        # Add device status indicator
        gr.Markdown(
            f"""
            **Current Status:** 
            {"✅ GPU Acceleration Active" if device.type in ['cuda', 'mps'] else "⚠️ Using CPU (Consider GPU for faster generation)"}
            """
        )

        # Event handlers
        def reload_model_handler():
            """Handle model reload button click"""
            gr.Info("Reloading model...")
            success = load_model()
            if success:
                gr.Info("✅ Model reloaded successfully!")
                return "✅ Model Ready"
            else:
                gr.Warning("❌ Model reload failed. Check console for details.")
                return "❌ Model Failed"

        generate_button.click(
            fn=generate_speech,
            inputs=[text_input, voice_input],
            outputs=audio_output,
            api_name="tts"
        )
        
        reload_button.click(
            fn=reload_model_handler,
            outputs=None
        )

        gr.Markdown(
            f"""
            ---
            
            **Troubleshooting:**
            - If model fails to load, check your internet connection and try "Reload Model"
            - For CUDA issues: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
            - For dependencies: `pip install moshi sphn sounddevice gradio`
            
            *Model: [{MODEL_REPO}](https://huggingface.co/{MODEL_REPO}) • Voices: [{VOICE_REPO}](https://huggingface.co/{VOICE_REPO})*
            """
        )
    return demo

# --- 5. Main Execution ---
if __name__ == "__main__":
    print("=" * 60)
    print("🌐 Starting Gradio Interface")
    print("=" * 60)
    
    app = create_gradio_app()
    
    # Launch with better error handling and no share by default
    try:
        print("🚀 Launching Gradio interface...")
        print("📱 Access the app at: http://localhost:7860")
        print("🌍 To enable sharing, set share=True in launch()")
        
        # Launch locally first (more reliable)
        app.launch(
            share=False,  # Set to True only if you need public sharing
            server_name="0.0.0.0",  # Allow access from other devices on network
            server_port=7860,
            show_error=True,
            quiet=False
        )
        
    except Exception as e:
        print(f"❌ Failed to launch Gradio: {e}")
        print("💡 Trying alternative launch method...")
        
        # Fallback launch method
        try:
            app.launch(
                share=False,
                inbrowser=True,
                show_error=True
            )
        except Exception as e2:
            print(f"❌ Alternative launch also failed: {e2}")
            print("🔧 Please check your Gradio installation: pip install --upgrade gradio")
