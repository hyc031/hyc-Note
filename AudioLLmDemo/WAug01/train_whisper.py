import os
import glob
import numpy as np
import soundfile as sf
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
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

# ==================== 2. 加载数据 ====================
sample_dir = "../Data/AISHELL-1/sample/audio"
transcript_path = "../Data/AISHELL-1/data_aishell/transcript/aishell_transcript_v0.8.txt"

text_map = {}
if os.path.exists(transcript_path):
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                text_map[parts[0]] = "".join(parts[1:])

wav_files = sorted(glob.glob(os.path.join(sample_dir, "*.wav")))
print(f"找到 {len(wav_files)} 个 wav 文件")

# ==================== 3. 手动预处理（带进度条，单条异常不会卡死全部） ====================
processed = []
for wav_path in tqdm(wav_files, desc="预处理音频"):
    utt_id = os.path.splitext(os.path.basename(wav_path))[0]
    text = text_map.get(utt_id, f"sample_{utt_id}")
    
    try:
        wav, sr = sf.read(wav_path, dtype="float32")
    except Exception as e:
        print(f"\n跳过损坏文件: {wav_path}, 错误: {e}")
        continue
    
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != 16000:
        new_len = int(len(wav) * 16000 / sr)
        wav = np.interp(np.linspace(0, len(wav), new_len), np.arange(len(wav)), wav).astype(np.float32)
    
    # 提取特征
    input_features = processor.feature_extractor(
        wav, sampling_rate=16000
    ).input_features[0]
    
    # 编码标签
    labels = processor.tokenizer(text).input_ids
    
    processed.append({
        "input_features": input_features,
        "labels": labels,
    })

print(f"成功处理 {len(processed)} 条数据")

# 直接构建 Dataset，不需要 map
dataset = Dataset.from_list(processed)
dataset = dataset.train_test_split(test_size=0.2)
train_ds = dataset["train"]
eval_ds = dataset["test"]

# ==================== 4. Data Collator（不变） ====================
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

# ==================== 5. 训练参数 ====================
args = Seq2SeqTrainingArguments(
    output_dir="outputs/whisper-sample-test",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,
    learning_rate=5e-6,
    warmup_steps=10,
    max_steps=200,
    fp16=True,
    gradient_checkpointing=True,
    eval_strategy="steps",           # ← 改这里
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    logging_steps=10,
    report_to="none",
    predict_with_generate=False,
    load_best_model_at_end=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    # tokenizer=processor.feature_extractor,
)

# ==================== 6. 训练 & 保存 ====================
print("开始训练...")
trainer.train()

trainer.save_model("outputs/whisper-sample-test/final")
processor.save_pretrained("outputs/whisper-sample-test/final")
print("训练完成，模型已保存")

# ==================== 7. Generate 测试 ====================
print("\n开始 generate 测试...")

ckpt_path = "outputs/whisper-sample-test/final"
model_test = WhisperForConditionalGeneration.from_pretrained(ckpt_path).to(
    "cuda" if torch.cuda.is_available() else "cpu"
)
processor_test = WhisperProcessor.from_pretrained(ckpt_path)

test_sample = eval_ds[0]
# input_features = torch.tensor(test_sample["input_features"]).unsqueeze(0).to(model_test.device)
# with torch.inference_mode():
#     predicted_ids = model_test.generate(
#         input_features,
#         language="zh",
#         task="transcribe",
#     )

## 将 attention_mask 传入 generate，避免 padding 部分被识别为文本
input_features = torch.tensor(test_sample["input_features"]).unsqueeze(0).to(model_test.device)
attention_mask = torch.ones_like(input_features[:, 0, :])  # (1, 3000) 全1，表示无padding
with torch.inference_mode():
    predicted_ids = model_test.generate(
        input_features,
        attention_mask=attention_mask,  # ← 加上这行
        language="zh",
        task="transcribe",
    )



transcription = processor_test.batch_decode(predicted_ids, skip_special_tokens=True)[0]
print(f"Generate 结果: {transcription}")

label_ids = [t for t in test_sample["labels"] if t != -100]
reference = processor_test.tokenizer.decode(label_ids, skip_special_tokens=True)
print(f"真实标签: {reference}")