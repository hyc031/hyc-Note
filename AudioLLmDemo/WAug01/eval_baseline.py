"""
基线 WER 评估脚本
==================
目的：在与 eval_whisper.py 完全相同的 200 条 AISHELL-1 dev 数据上，
评估【未经微调的原版】 openai/whisper-medium，得到基线 WER。

用法：
    python eval_baseline.py

产出：
    1. 终端打印基线 WER（与 eval_whisper.py 的微调后 WER 对比，得到「降了多少」）
    2. outputs/baseline_predictions.txt —— 逐条保存 参考/预测，
       供后续 before/after 错误分析使用（配合微调模型的预测结果对比）

注意（保证对比公平）：
    - forced_decoder_ids / suppress_tokens 的设置与 train_whisper2.py 训练时一致
    - dev 取样逻辑、条数（200）、generate 参数与 eval_whisper.py 完全一致
"""

import os
import sys
import torch
import numpy as np
import soundfile as sf
import librosa
import evaluate
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# ==================== 1. 加载原版模型（基线） ====================
# 默认评估原版 whisper-medium；也支持传参评估其他模型，如：
#   python eval_baseline.py outputs/whisper-medium-aishell-lora/final_merged
model_path = sys.argv[1] if len(sys.argv) > 1 else "openai/whisper-medium"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = WhisperForConditionalGeneration.from_pretrained(model_path).to(device)
processor = WhisperProcessor.from_pretrained(model_path)

# 与训练脚本保持一致的解码配置，保证基线与微调模型在同等条件下生成
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.eval()

TARGET_SR = 16000

# ==================== 2. 加载 dev 数据（与 eval_whisper.py 完全一致） ====================
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
                samples.append({"utt_id": utt_id, "audio_path": wav_path, "sentence": text})
            if max_samples and len(samples) >= max_samples:
                break
    return samples


dev_list = load_aishell_from_local(
    audio_root="../Data/AISHELL-1/data_aishell/wav/dev",
    transcript_path="../Data/AISHELL-1/data_aishell/transcript/aishell_transcript_v0.8.txt",
    max_samples=200,
)
print(f"dev 样本数: {len(dev_list)}")

# ==================== 3. 推理 + 计算 WER ====================
metric = evaluate.load("wer")
predictions = []
references = []
utt_ids = []

for item in tqdm(dev_list, desc="基线评估中"):
    wav, sr = sf.read(item["audio_path"], dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        wav = librosa.resample(y=wav, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)

    inputs = processor(wav, sampling_rate=TARGET_SR, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with torch.inference_mode():
        pred_ids = model.generate(
            input_features,
            language="zh",
            task="transcribe",
        )

    pred_text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
    predictions.append(pred_text)
    references.append(item["sentence"])
    utt_ids.append(item["utt_id"])

wer = 100 * metric.compute(predictions=predictions, references=references)
print(f"\n模型: {model_path}")
print(f"WER: {wer:.2f}%")

# ==================== 4. 保存逐条结果，供错误分析 ====================
os.makedirs("outputs", exist_ok=True)
# 输出文件名按模型区分：原版 -> baseline_predictions.txt，其他模型按目录名命名
if model_path == "openai/whisper-medium":
    out_path = "outputs/baseline_predictions.txt"
else:
    tag = os.path.basename(os.path.dirname(model_path)) or os.path.basename(model_path)
    out_path = f"outputs/{tag}_predictions.txt"

with open(out_path, "w", encoding="utf-8") as f:
    f.write(f"# 模型: {model_path}\n")
    f.write(f"# WER: {wer:.2f}%\n")
    for uid, ref, hyp in zip(utt_ids, references, predictions):
        f.write(f"{uid}\n")
        f.write(f"  参考: {ref}\n")
        f.write(f"  预测: {hyp}\n")
        # 错误分析时可直接按行对比：参考==预测 即该条识别正确
        f.write(f"  {'OK' if ref == hyp else 'ERR'}\n\n")

print(f"逐条预测已保存到 {out_path}")

# 打印几条对比看看
print("\n前 5 条对比:")
for i in range(min(5, len(predictions))):
    print(f"  参考: {references[i]}")
    print(f"  预测: {predictions[i]}")
    print()
