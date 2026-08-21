"""
Whisper LoRA 微调脚本（PEFT 版）
=================================
目的：与 train_whisper2.py（全量微调）在【完全相同的数据】上做对照实验，
     补齐 modifwork.md 要求的 LoRA 实践（Week 2 补课内容）。

与 train_whisper2.py 的关系：
    - 数据相同：AISHELL-1，train 前 1000 条 / dev 前 200 条
    - 训练配置尽量对齐（batch、学习率、步数、fp16、gradient checkpointing）
    - 区别仅在于：只训练注入的 LoRA 低秩矩阵，冻结原模型全部权重

产出：
    1. outputs/whisper-medium-aishell-lora/        —— LoRA adapter（很小，几十 MB）
    2. outputs/whisper-medium-aishell-lora/final_merged —— adapter 合并回原模型的完整权重
       （可直接用 eval_whisper.py 评估：
         python eval_whisper.py outputs/whisper-medium-aishell-lora/final_merged）
    3. 终端打印 可训练参数量 / 峰值显存 —— 用于写进 results.md 对比表

依赖：peft（work.md Week 0 已装：pip install peft）
"""

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
from peft import LoraConfig, get_peft_model, TaskType

# ==================== 1. 加载模型 + 注入 LoRA ====================
model_id = "openai/whisper-medium"
processor = WhisperProcessor.from_pretrained(model_id)
model = WhisperForConditionalGeneration.from_pretrained(model_id)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.enable_input_require_grads()   # gradient checkpointing + LoRA 时必须开

# LoRA 配置
# - r=16：低秩矩阵的秩，常用 8~32，先取中间值
# - target_modules：Whisper 注意力层的 q/v 投影是微调 ASR 的常见选择
# - 只动 decoder 也可以（lora_target_modules 作用于全模型的 q_proj/v_proj，
#   encoder 和 decoder 的同名层都会被注入）
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)

# 打印可训练参数量 —— LoRA vs 全量的核心对比数字，记到 results.md
model.print_trainable_parameters()

TARGET_SR = 16000

# ==================== 2. 加载数据（与 train_whisper2.py 完全一致） ====================
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

# 与 train_whisper2.py 相同的小数据集，保证对照公平
train_list = train_list[:1000]
dev_list = dev_list[:200]

# ==================== 3. 手动预处理（与 train_whisper2.py 一致） ====================
def preprocess_list(data_list, desc="预处理"):
    """逐条读取、重采样、提取特征，带异常捕获"""
    processed = []
    for item in tqdm(data_list, desc=desc):
        try:
            wav, sr = sf.read(item["audio_path"], dtype="float32")
        except Exception as e:
            print(f"\n跳过损坏文件: {item['audio_path']}, 错误: {e}")
            continue

        if wav.ndim == 2:
            wav = wav.mean(axis=1)

        if sr != TARGET_SR:
            wav = librosa.resample(y=wav, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)

        input_features = processor.feature_extractor(
            wav, sampling_rate=TARGET_SR
        ).input_features[0]

        labels = processor.tokenizer(item["sentence"]).input_ids

        processed.append({
            "input_features": input_features,
            "labels": labels,
        })
    return processed


print("开始处理训练集...")
train_processed = preprocess_list(train_list, desc="Train")
print(f"训练集成功处理: {len(train_processed)} 条")

print("开始处理验证集...")
dev_processed = preprocess_list(dev_list, desc="Dev")
print(f"验证集成功处理: {len(dev_processed)} 条")

mini_train = Dataset.from_list(train_processed)
mini_dev = Dataset.from_list(dev_processed)

# ==================== 4. Data Collator（与 train_whisper2.py 一致） ====================
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

# ==================== 5. WER 指标（与 train_whisper2.py 一致） ====================
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# ==================== 6. 训练参数（与 train_whisper2.py 对齐，仅换输出目录） ====================
args = Seq2SeqTrainingArguments(
    output_dir="outputs/whisper-medium-aishell-lora",
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

# ==================== 7. 训练（记录峰值显存） ====================
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()

trainer.train()

# LoRA adapter 保存（只存低秩矩阵，体积很小）
trainer.save_model("outputs/whisper-medium-aishell-lora")
processor.save_pretrained("outputs/whisper-medium-aishell-lora")

if torch.cuda.is_available():
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n峰值显存占用: {peak_gb:.2f} GB")   # 与全量微调对比的数字

# ==================== 8. 合并导出完整模型，供 eval_whisper.py 直接评估 ====================
# 加载 best checkpoint（load_best_model_at_end=True，trainer.model 已是最佳权重）
print("\n合并 LoRA 权重到原模型并导出...")
merged_model = model.merge_and_unload()
merged_path = "outputs/whisper-medium-aishell-lora/final_merged"
merged_model.save_pretrained(merged_path)
processor.save_pretrained(merged_path)
print(f"合并模型已保存到 {merged_path}")

print("\n训练完成。接下来评估：")
print("  python eval_whisper.py outputs/whisper-medium-aishell-lora/final_merged")
print("基线对照：")
print("  python eval_baseline.py")
