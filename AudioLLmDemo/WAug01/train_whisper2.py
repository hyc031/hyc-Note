import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import librosa
import soundfile as sf
import numpy as np
import evaluate
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# ==================== 1. 加载模型 ====================
model_id = "openai/whisper-medium"
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.enable_input_require_grads()

TARGET_SR = 16000

# ==================== 2. 加载数据 ====================
def load_aishell_from_local(audio_root, transcript_path):
    samples = []
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            utt_id = parts[0]
            text = "".join(parts[1:])

            spk_id = utt_id[6:11]  # BAC009S0002W0122 -> S0002
            wav_path = os.path.join(audio_root, spk_id, f"{utt_id}.wav")

            if os.path.exists(wav_path):
                samples.append({"audio_path": wav_path, "sentence": text})
    return samples


train_list = load_aishell_from_local(
    audio_root="../Data/AISHELL-1/data_aishell/wav/train",
    transcript_path="../Data/AISHELL-1/data_aishell/transcript/aishell_transcript_v0.8.txt",
)
dev_list = load_aishell_from_local(
    audio_root="../Data/AISHELL-1/data_aishell/wav/dev",
    transcript_path="../Data/AISHELL-1/data_aishell/transcript/aishell_transcript_v0.8.txt",
)

print(f"Train: {len(train_list)} 条, Dev: {len(dev_list)} 条")


# 简单测试做一个小数据集，避免训练时间过长
train_list = train_list[:1000]
dev_list = dev_list[:200]
# ==================== 3. 手动预处理（for 循环 + tqdm，避免 map 卡住） ====================
def preprocess_list(data_list, desc="预处理"):
    """逐条读取、重采样、提取特征，带异常捕获"""
    processed = []
    for item in tqdm(data_list, desc=desc):
        try:
            wav, sr = sf.read(item["audio_path"], dtype="float32")
        except Exception as e:
            print(f"\n跳过损坏文件: {item['audio_path']}, 错误: {e}")
            continue

        # 转单声道
        if wav.ndim == 2:
            wav = wav.mean(axis=1)

        # 显式重采样
        if sr != TARGET_SR:
            wav = librosa.resample(y=wav, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)

        # 提取 log-Mel
        input_features = processor.feature_extractor(
            wav, sampling_rate=TARGET_SR
        ).input_features[0]

        # 编码文本
        labels = processor.tokenizer(item["sentence"]).input_ids

        processed.append({
            "input_features": input_features,
            "labels": labels,
        })
    return processed


# 处理训练集和验证集
print("开始处理训练集...")
train_processed = preprocess_list(train_list, desc="Train")
print(f"训练集成功处理: {len(train_processed)} 条")

print("开始处理验证集...")
dev_processed = preprocess_list(dev_list, desc="Dev")
print(f"验证集成功处理: {len(dev_processed)} 条")

# 构建 Dataset
mini_train = Dataset.from_list(train_processed)
mini_dev = Dataset.from_list(dev_processed)

# ==================== 4. Data Collator ====================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# ==================== 5. WER 指标 ====================
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# ==================== 6. 训练参数 ====================
args = Seq2SeqTrainingArguments(
    output_dir="outputs/whisper-medium-aishell-mini",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=50,
    max_steps=2000,
    fp16=True,
    gradient_checkpointing=True,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    logging_steps=25,
    report_to="none",
    predict_with_generate=True,
    generation_max_length=225,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=mini_train,
    eval_dataset=mini_dev,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# ==================== 7. 训练 ====================
trainer.train()
trainer.save_model("outputs/whisper-medium-aishell-mini/final")
processor.save_pretrained("outputs/whisper-medium-aishell-mini/final")