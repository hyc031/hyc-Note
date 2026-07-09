# SFT 将大模型变得 "能听懂指令"

# 在做Pretrain 时, 将全部内容放到model 中进行训练
# SFT 表明那一部分是问题(instruction), 那一部分是回答(output)
# 一些名词:  prompt masking; loss masking 


from trl import SFTTrainer
from datasets import load_dataset

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    train_dataset=load_dataset("yahma/alpaca-cleaned", split="train"),
)
trainer.train()

