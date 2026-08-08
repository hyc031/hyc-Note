import torch, torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

model_id = "openai/whisper-large-v3"      # 显存紧就先用 small / medium 验证链路
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.float16
).to("cuda")

wav, sr = torchaudio.load("test.wav", backend="soundfile")
wav = torchaudio.functional.resample(wav, sr, 16000)   # Whisper 固定 16kHz
inputs = processor(wav.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")

print(inputs.input_features.shape)        # 重点：打出来看清楚
ids = model.generate(inputs.input_features.half().to("cuda"), language="zh")
print(processor.batch_decode(ids, skip_special_tokens=True))