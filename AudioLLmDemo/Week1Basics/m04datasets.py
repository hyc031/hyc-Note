# datasets 库+ 数据处理
# HF生态 + 推理全流程
# goal:  掌握 `datasets` 的加载、查看、清洗、切分、格式转换，并把一个原始公开数据集处理成可用于 SFT 微调的数据格式。


from datasets import DatasetDict, load_dataset
from transformers import AutoTokenizer


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "yahma/alpaca-cleaned"
SAVE_DIR = "./data/alpaca_sft_qwen_day4"
SYSTEM_PROMPT = "你是一个严谨、简洁、可靠的 AI 助手。"
MAX_LEN = 1024
DEBUG_SIZE = 1000


def is_valid(example):  # example 为字典属性 有两个key: instruction, output
    instruction = example["instruction"].strip() #  strip() 用于移除字符串头尾指定的字符
    # eg : 
    # str = "00000003210Runoob01230000000";  ->   
    # print str.strip( '0' );  # 去除首尾字符 0 -> 3210Runoob0123
    # 
    output = example["output"].strip()
    return len(instruction) > 0 and len(output) >= 10
    # 指令长度大于 0 并且回答文本长达大于 10 才算有效样本 否则返回False 丢弃样本


def build_messages(example):
    instruction = example["instruction"].strip()
    input_text = example["input"].strip()
    output = example["output"].strip()
    '''
    原始样本结构(Alpaca 标准三字段)
        instruction: 用户主指令 / 问题
        input: 可选补充上下文(很多样本为空)
        output: 模型标准答案(assistant 回复)
    '''
    user_content = instruction if not input_text else f"{instruction}\n\n{input_text}"
    # 三元判断: 构造用户侧完整提示文本 
    # if not input_text：如果清洗后的 input 是空字符串（无补充上下文）->  用户内容直接只用 instruction
    # 否则（有补充输入）-> 指令和补充文本用两个换行分隔拼接：指令\n\n补充内容

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    raw = load_dataset(DATASET_NAME, split="train")
    print("raw:", raw)
    print("columns:", raw.column_names)
    print("sample:", raw[0])

    ds = raw.shuffle(seed=42).select(range(min(DEBUG_SIZE, len(raw))))
    ds = ds.filter(is_valid)
    ds = ds.map(build_messages)

    def apply_template(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    ds = ds.map(apply_template)

    def add_length(example):
        tokenized = tokenizer(example["text"], add_special_tokens=False) 
        # tokenzier后 返回一个字典 包含input_dis, 和其它的key 
        return {"num_tokens": len(tokenized["input_ids"])}

    ds = ds.map(add_length)

    lengths = ds["num_tokens"]
    print("length min:", min(lengths))
    print("length max:", max(lengths))
    print("length avg:", sum(lengths) / len(lengths))
    
    ds = ds.filter(lambda x: x["num_tokens"] <= MAX_LEN)
    split_ds = ds.train_test_split(test_size=0.05, seed=42)

    keep_columns = ["text", "messages", "num_tokens"]
    processed = {}
    for split, split_data in split_ds.items():
        remove_cols = [col for col in split_data.column_names if col not in keep_columns]
        processed[split] = split_data.remove_columns(remove_cols)

    processed = DatasetDict(processed)
    processed.save_to_disk(SAVE_DIR)

    print("saved to:", SAVE_DIR)
    print(processed)
    print("final sample text:\n", processed["train"][0]["text"])


if __name__ == "__main__":
    main()
