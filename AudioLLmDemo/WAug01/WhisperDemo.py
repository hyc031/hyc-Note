import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import WhisperProcessor, WhisperForConditionalGeneration

import warnings
warnings.filterwarnings("ignore", module="transformers")

model_id = "openai/whisper-large-v3"  # 先验证链路，显存足够再换 large-v3
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=dtype,
).to(device)
model.eval()

# wav, sr = sf.read("../Data/AISHELL-1/sample/audio/sample_20.wav", dtype="float32")

# 个人做测试的一些文件
# /home/r618/data/chy/hycapi/AudioLLmDemo/Data/AISHELL-1/SelfData

'''
一开始传入 20 30 35 120s wav文件发现都可以转录
检查发现经截断 基本超出30s的音频被自动截断
'''
# 
wav, sr = sf.read("../Data/AISHELL-1/SelfData/merged_120s.wav", dtype="float32")

# 转单声道
if wav.ndim == 2:
    wav = wav.mean(axis=1)

# 重采样到 16 kHz
if sr != 16000:
    wav = resample_poly(wav, 16000, sr).astype(np.float32)

inputs = processor(
    wav,
    sampling_rate=16000,
    return_tensors="pt",
    return_attention_mask=True,
)

input_features = inputs.input_features.to(device=device, dtype=dtype)

print("input_features:", input_features.shape)

with torch.inference_mode():
    ids = model.generate(
        input_features=input_features,
        language="zh",
        task="transcribe",
    )

text = processor.batch_decode(ids, skip_special_tokens=True)
print(text)


