import sys, os
sys.path.append('third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio
import warnings

# Fix text encoding by using proper UTF-8 strings
text = "陈四，你好。十四年前我创办小米的时候，讲过一句话。我说当台风来的时候连猪都会飞，讲的是大家有像猪一样的态度，你就可以成功。把握机遇的重要性非常重要。"
prompt = "希望你以后能够做的比我还好呦。"

cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B', load_jit=True, load_onnx=False, load_trt=False)
prompt_speech_16k = load_wav('zero_shot_prompt.wav', 16000)

# Suppress deprecation warnings if desired
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

for i, j in enumerate(cosyvoice.inference_zero_shot(text, prompt, prompt_speech_16k, stream=False)):
    output_dir = 'pretrained_models'
    os.makedirs(output_dir, exist_ok=True)

    # Save the wav file to the pretrained_models directory
    torchaudio.save(os.path.join(output_dir, 'zero_shot_{}.wav'.format(i)), j['tts_speech'], cosyvoice.sample_rate)