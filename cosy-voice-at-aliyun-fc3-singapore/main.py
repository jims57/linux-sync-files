import sys, os
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import warnings
import onnxruntime as ort

# Fix text encoding by using proper UTF-8 strings
text = "陈四，你好。十四年前我创办小米的时候，讲过一句话。我说当台风来的时候连猪都会飞，讲的是大家有像猪一样的态度，你就可以成功。把握机遇的重要性非常重要。"
prompt = "希望你以后能够做的比我还好呦。"

# Configure ONNX Runtime settings before model loading
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.log_severity_level = 3  # Reduce logging verbosity
session_options.enable_cpu_mem_arena = False  # Reduce memory usage
session_options.enable_mem_pattern = False    # Reduce memory usage

# If using CUDA
providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider'
]

# Update your CosyVoice2 initialization
cosyvoice = CosyVoice2(
    'pretrained_models/CosyVoice2-0.5B',
    load_jit=True,
    load_onnx=True,  # Enable ONNX
    load_trt=False,
    onnx_session_options=session_options,
    onnx_providers=providers
)
prompt_speech_16k = load_wav('zero_shot_prompt.wav', 16000)

# Suppress deprecation warnings if desired
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

for i, j in enumerate(cosyvoice.inference_zero_shot(text, prompt, prompt_speech_16k, stream=False)):
    output_dir = 'pretrained_models'
    os.makedirs(output_dir, exist_ok=True)

    # Save the wav file to the pretrained_models directory
    torchaudio.save(os.path.join(output_dir, 'zero_shot_{}.wav'.format(i)), j['tts_speech'], cosyvoice.sample_rate)