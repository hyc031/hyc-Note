import os
import sys
import torch
import numpy as np
import soundfile as sf
import librosa
import evaluate
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ==================== 1. 加载模型 ====================
# 默认评估全量微调模型；也支持命令行传入其他路径，如：
#   python eval_whisper.py outputs/whisper-medium-aishell-lora/final_merged
model_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/whisper-medium-aishell-mini/final"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
processor = WhisperProcessor.from_pretrained(model_path)
model.eval()

TARGET_SR = 16000

# ==================== 2. 重新加载验证数据 ====================
def load_aishell_from_local(audio_root, transcript_path, max_samples=None):
    samples = []
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            utt_id = parts[0]
            text = "".join(parts[1:])
            spk_id = utt_id[6:11]
            wav_path = os.path.join(audio_root, spk_id, f"{utt_id}.wav")
            if os.path.exists(wav_path):
                samples.append({"audio_path": wav_path, "sentence": text})
            if max_samples and len(samples) >= max_samples:
                break
    return samples

dev_list = load_aishell_from_local(
    audio_root="../Data/AISHELL-1/data_aishell/wav/dev",
    transcript_path="../Data/AISHELL-1/data_aishell/transcript/aishell_transcript_v0.8.txt",
    max_samples=200,
)

# ==================== 3. 推理 + 计算 WER ====================
metric = evaluate.load("wer")
predictions = []
references = []

for item in tqdm(dev_list, desc="评估中"):
    # 读取音频
    wav, sr = sf.read(item["audio_path"], dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        wav = librosa.resample(y=wav, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)
    
    # 提取特征
    inputs = processor(wav, sampling_rate=TARGET_SR, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    
    # generate
    with torch.inference_mode():
        pred_ids = model.generate(
            input_features,
            language="zh",
            task="transcribe",
        )
    
    pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    predictions.append(pred_text)
    references.append(item["sentence"])

# 计算 WER
wer = 100 * metric.compute(predictions=predictions, references=references)
print(f"\n最终 WER: {wer:.2f}%")

# 打印几条对比看看
print("\n前 5 条对比:")
for i in range(min(5, len(predictions))):
    print(f"  参考: {references[i]}")
    print(f"  预测: {predictions[i]}")
    print()