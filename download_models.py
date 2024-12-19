from modelscope import snapshot_download
import os

# 创建文件夹（如果不存在）
output_dir = 'pretrained_models'
os.makedirs(output_dir, exist_ok=True)

# 下载模型
print("开始下载模型...")
models = {
    'CosyVoice2-0.5B': 'iic/CosyVoice2-0.5B',
    'CosyVoice-300M': 'iic/CosyVoice-300M',
    'CosyVoice-300M-25Hz': 'iic/CosyVoice-300M-25Hz',
    'CosyVoice-300M-SFT': 'iic/CosyVoice-300M-SFT',
    'CosyVoice-300M-Instruct': 'iic/CosyVoice-300M-Instruct',
    'CosyVoice-ttsfrd': 'iic/CosyVoice-ttsfrd',
}

for model_name, model_id in models.items():
    print(f"正在下载 {model_name} ...")
    snapshot_download(model_id, local_dir=os.path.join(output_dir, model_name))
    print(f"{model_name} 下载完成！")

print("所有模型下载完成！")
