import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import pipeline
import warnings
warnings.filterwarnings("ignore", module="transformers")
# 原来的 WhisperDemo 中 传入超过30s的音频会自动截断 没有实现完整的翻译
# 这里做一些操作, 将长音频切块, 让模型自动 batch 处理. 

# ========== 1. 加载模型（保持不变） ==========
model_id = "openai/whisper-large-v3"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=dtype
).to(device)
model.eval()

# ========== 2. 读取音频 ==========
wav, sr = sf.read("../Data/AISHELL-1/SelfData/merged_120s.wav", dtype="float32")

# 转单声道
if wav.ndim == 2:
    wav = wav.mean(axis=1)

# 重采样到 16 kHz
if sr != 16000:
    wav = resample_poly(wav, 16000, sr).astype(np.float32)

# ========== 3. 切分为 4 段，每段 30 秒 ==========
sr = 16000
chunk_sec = 30
chunk_samples = chunk_sec * sr  # 480000
stride_samples = 5 * sr   # 80000（重叠 5 秒）

# 只取前 120 秒（避免尾部多余音频导致第五段）


# 检查发现第2 3 4段内容识别错误 原因如下:
# 但此时发现  模型不会跨段翻译 如果一句话落在30s的边界上 两段内容都识别错误
# 改用 HF 的 pipeline 内部做带重叠的 分块 + 边界合并
chunks = []
for i in range(0, len(wav), chunk_samples - stride_samples):
    chunk = wav[i : i + chunk_samples]
    if len(chunk) < chunk_samples:
        chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
    chunks.append(chunk)

print(f"共生成 {len(chunks)} 个 chunk")  # 120s 音频应该是 5 个（最后一段可能很短）

# 批量推理
inputs = processor(chunks, sampling_rate=16000, return_tensors="pt")
input_features = inputs.input_features.to(device=device, dtype=dtype)

with torch.inference_mode():
    ids = model.generate(
        input_features=input_features,
        language="zh",
        task="transcribe",
    )

texts = processor.batch_decode(ids, skip_special_tokens=True)

# 简单合并（每段取非重叠部分，避免重复）
# 第 0 段取全部，后续每段跳过前 5 秒对应的文本（粗略处理）
full_text = texts[0]
for t in texts[1:]:
    # 简单去重：如果下一段开头和上一段结尾有重叠，这里只是粗暴拼接
    # 精确合并需要基于 token 时间戳，建议直接用 pipeline
    full_text += t

print(full_text)




# pipe = pipeline(
#     "automatic-speech-recognition",
#     model="openai/whisper-large-v3",
#     torch_dtype=dtype,
#     device=device,
# )

# # chunk_length_s=30: 每 30 秒一块
# # stride_length_s=5:  块之间重叠 5 秒，消除边界截断
# result = pipe(
#     "../Data/AISHELL-1/SelfData/merged_120s.wav",
#     chunk_length_s=30,
#     stride_length_s=5,
#     batch_size=1,
#     return_timestamps=True,
# )

# print(result["text"])  # 完整的 120 秒转录