import sys, os
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import warnings
import onnxruntime as ort

# Fix text encoding by using proper UTF-8 strings
text = "甘景元。他在旧书店找到一本泛黄的笔记本，封面上写着一个熟悉的名字。他愣住了，指尖不自觉地摩挲着那几个字。"
prompt = "希望你以后能够做的比我还好呦。"

# Configure ONNX Runtime global settings
ort.set_default_logger_severity(3)  # Reduce logging verbosity

# If using CUDA, set the global provider options
if 'CUDAExecutionProvider' in ort.get_available_providers():
    ort.set_default_logger_severity(3)
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
else:
    providers = ['CPUExecutionProvider']

# Initialize CosyVoice2 with supported parameters
cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False)
prompt_speech_16k = load_wav('zero_shot_prompt.wav', 16000)

# Suppress deprecation warnings if desired
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

for i, j in enumerate(cosyvoice.inference_zero_shot(text, prompt, prompt_speech_16k, stream=True)):
    output_dir = 'pretrained_models'
    os.makedirs(output_dir, exist_ok=True)

    # Save the wav file to the pretrained_models directory
    torchaudio.save(os.path.join(output_dir, 'zero_shot_{}.wav'.format(i)), j['tts_speech'], cosyvoice.sample_rate)