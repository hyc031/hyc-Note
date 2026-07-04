# 第一周小结



## 分词器 tokenizer

seq 输入到 model 中， 经 tokenizer 后，将原来的文本序列转换为 token id



eg:

```python
name = "xxxxx...."
# 单序列
seq = "HuggingFace is actually quite straightforward once you understand the underlying PyTorch tensors!" 

tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer(seq, return_tensors="pt")
```

seq 经tokenizer 处理后返回 的 input 不是一个单Tensor 量 ， *HF将其封装成 一个  BatchEncoding 的对象.*

BatchEncoding  是 Python dict 的子类.

BatchEncoding 自带以下键值:

​              input_ids  -> 每个 token 在词表中的数字索引, shape [batch_size, seq_len]

​              attention_mask -> 取值: 1 = 真实 token, 0 = 填充 pad 字符, shape [batch_size, seq_len]

​              token_type_ids (双句子任务专用) -> 0:第一句话 token, 1:第二句话 token

​           				其它可选字段(需要进行设置)

​              overflowing_tokens -> 超长截断后溢出的 token 序列(return_overflowing_tokens=True)

​              offset_mapping   -> fast 分词器专属，每个 token 对应原文本字符起止坐标（做实体抽取必备）

​              special_tokens_mask-> 区分普通 token 和特殊 token(CLS/SEP/PAD  )     



##  *apply_chat_template*

apply_chat_template 创建一个对话的模型  *按特定格式拼接 System/User 角色* 



```python

messages = [
    {"role": "system", "content": "你是一个幽默的 AI 助手。"},
    {"role": "user", "content": "请用一句话解释什么是“自回归(Auto-regressive)”?"}
    
]
# apply_chat_template 会自动插入 <|im_start|> 等特殊 token
prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True # 在末尾加上助手回答的起始符  <|im_start|>assistant
)

```

对话中涉及到的一些参数: 

`temperature  =T`    *温度T大于1，让低概率词也有机会冒泡, 温度T小于1 输出更稳定。*

`top_p = p`	       *核采样，截断尾部极低概率的噪音词*

`top_k  = k`              *保留前K个高概率词进行轮盘赌*



## dataset load 

这一部分主要参考 HF 官方文档 datasets内容 [数据集 --- Datasets](https://huggingface.co/docs/datasets/index)



## 简单运行例子

```python
# config setting 
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
DATASET_NAME = "yahma/alpaca-cleaned"
MAX_LENGTH = 512  # 输入长度限制
BATCH_SIZE = 4    # 根据 24G 显存调整，0.5B 模型 3090 上跑 4-8 都可以
OUTPUT_FILE = "./results/predictions.jsonl"
METRIC_LOG = "./results/metrics.log"
```

```python
# logging 将运行状态（如加载进度、每个 batch 的处理情况）同时输出到控制台并保存到 result文件中
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(METRIC_LOG, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
```



```python
def construct_qwen_prompt(instruction, input_text=""):
    if input_text.strip():
        # 如果有输入内容，将其作为上下文
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
    return prompt
```

使用Qwen 模型构造 Prompt， 拼接 Instruction 和 Input



`tokenizer.padding_side = 'left'`  左填充, 请款如下:

```markdown
# eg:
seq1: "你好" 
seq2: "今天天气真不错啊" 
首先要进行对齐，然后进行处理，如果右填充
["你", "好", "0", "0", "0", "0", "0", "0"]
["今", "天", "天", "气", "真", "不", "错", "啊"]
```





