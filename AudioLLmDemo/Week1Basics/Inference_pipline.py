"""
File: inference_pipeline.py
Desc: Week 1 总结练习 - Qwen2.5-0.5B 指令遵循能力批量推理
      流程：加载数据 -> 构造 Prompt -> 批量推理 -> 简单统计评测
"""

# 将零散的内容完整串联

import torch
import json
from transformers import pipeline, AutoTokenizer, AutoModel, DataCollatorForLanguageModeling, AutoModelForCausalLM
from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
import logging
from datetime import datetime

# config & logging 

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "yahma/alpaca-cleaned"
MAX_LENGTH = 512  # 输入长度限制
BATCH_SIZE = 4    # 根据 24G 显存调整，0.5B 模型 3090 上跑 4-8 都可以
OUTPUT_FILE = "./results/predictions.jsonl"
METRIC_LOG = "./results/metrics.log"

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(METRIC_LOG, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def construct_qwen_prompt(instruction, input_text=""):
    """
    Qwen 2.5 的 Chat Template 构造函数
    Ref: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
    """
    if input_text.strip():
        # 如果有输入内容，将其作为上下文
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    return prompt

def main():
    # 2. 加载 Tokenizer 和 Model (Load Model)
    logger.info(f"Loading model: {model_id} ...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Qwen 系列通常需要设置 padding_side，否则 batch 推理可能出错
    tokenizer.padding_side = 'left' 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda:1"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,  # 设置数据格式
        device_map=device,           # 指定gpu 
        trust_remote_code=True
    )
    model.eval() # 切换到评估模式
    logger.info(f"Model loaded on device: {model.device}")
    logger.info(f"Model dtype: {model.dtype}")

    # 3. 加载与预处理数据集 (Load Dataset)
    logger.info(f"Loading dataset: {DATASET_NAME} ...")
    dataset = load_dataset(DATASET_NAME, split="train[:100]") # 只取前100条测试

    def preprocess_function(examples):
        # 批量处理函数
        prompts = [
            construct_qwen_prompt(inst, inp) 
            for inst, inp in zip(examples['instruction'], examples['input'])
        ]
        # Tokenize
        tokenized = tokenizer(
            prompts,
            max_length=MAX_LENGTH,
            padding=True,       # 动态 padding 到 batch 内最长
            truncation=True,
            return_tensors="pt", # 返回 PyTorch 张量
            # 注意：这里不设置 return_tensors，交给 DataCollator 处理更标准
        )
        return tokenized

    # 4. 构建 Dataloader (Batch Inference)
    logger.info("Building DataLoader...")
    # 移除不必要的列，并设置格式
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    # 将 tensor 移到 GPU 上（或者在 dataloader loop 里做）
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        # num_workers=2 # 如果数据量大可以开多进程，这里数据小默认0
    )

    # 5. 推理与评测 (Inference + Evaluation)
    logger.info("Starting inference loop...")
    total_tokens = 0
    prediction_count = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        with torch.no_grad(): # 关键：关闭梯度计算
            for batch_idx, batch in enumerate(dataloader):
                # 将 batch 移到模型所在的 device
                batch = {k: v.to(model.device) for k, v in batch.items()}
                
                # 前向传播，获取 logits
                # 注意：对于生成任务，通常我们直接用 generate() 而不是 forward()
                # 因为 forward 只算一步，generate 才能解码出一整段话
                generated_ids = model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_new_tokens=128,      # 生成内容长度限制
                    temperature=0.7,         # 适当随机性
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                # 解码生成的文本
                # 只解码新生成的部分（可选，这里简单解码全部）
                decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                # 简单评测：计算生成文本的 token 数量总和
                # 这里简单用生成 ID 的长度来算
                batch_generated_len = (generated_ids.size(1) - batch['input_ids'].size(1))
                total_tokens += batch_generated_len
                prediction_count += len(decoded_preds)

                # 保存结果 (简单存档)
                for pred in decoded_preds:
                    f_out.write(json.dumps({"prediction": pred}, ensure_ascii=False) + "\n")

                logger.info(f"Batch {batch_idx+1} done. Generated {batch_generated_len} tokens.")

    # 6. 最终评测结果 (Simple Evaluation)
    avg_gen_len = total_tokens / prediction_count if prediction_count > 0 else 0
    final_result = {
        "timestamp": datetime.now().isoformat(),
        "total_samples": prediction_count,
        "avg_generated_tokens": round(avg_gen_len, 2),
        "model": model_id,
        "dataset_sample": DATASET_NAME + " (first 100)"
    }

    logger.info(f" Inference completed. Results saved to {OUTPUT_FILE}")
    logger.info(f"📊 Final Metrics: {final_result}")

    # 顺便写入 metric log
    with open(METRIC_LOG, "a") as f:
        f.write(f"Final Metrics: {final_result}\n")

if __name__ == "__main__":
    main()