'''
完成一些 encode\ decode\ padding\ truncation操作
参考连接 
huggingface.co/learn/nlp-course/    # HF 官方 ch2 讲解了Tokenizer 边界情况(Batching, Padding)
huggingface.co/blog/how-to-generate
'''

# 构建 Qwen2.5  0.5B模型
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "Qwen/Qwen2.5-0.5B-Instruct"

print("="*10 + " 1. 加载模型与 Tokenizer " + "="*10)
# 文本生成任务，必须设为左填充 left padding
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

# 现代大模型标配：使用 bfloat16 加载，极致省显存且防溢出
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)
print("模型加载成功！设备:", model.device)

'''
    # 模型下载 Qwen2.5
'''

print("\n" + "="*10 + " 2. Chat Template 组装 " + "="*10)
# Instruct 模型不能直接喂纯文本，必须按特定格式拼接 System/User 角色
messages = [
    {"role": "system", "content": "你是一个幽默的 AI 助手。"},
    # {"role": "user", "content": "请用一句话解释什么是“自回归(Auto-regressive)”?"}
    {"role": "user", "content": "请你帮我介绍一下米哈游(Mihoyo)公司"}
]
# apply_chat_template 会自动插入 <|im_start|> 等特殊 token
prompt = tokenizer.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True # 在末尾加上助手回答的起始符  <|im_start|>assistant
)
print("组装后的 Prompt:\n", prompt)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # 按照特定格式拼接 system和user   一些特殊 Token： <|im_start|>, <|im_end|>
    # 
    # apply_chat_template --> 将 字符串进行拼接 此时 messages: <|im_start|>system\n你是....<|im_end|> --> prompt
    #
    # print(inputs.shape)  我想查看inputs 的维度, 发现返回的 inputs 并不是Tensor(是一个字典 dict), 
    #                      HF将其封装成 一个  BatchEncoding 的对象.
    # 
'''
BatchEncoding 是 transformers 库分词器(tokenizer)的标准返回对象, 是 Python dict 的子类.
                专门封装单条 / 批量文本分词后的全部模型输入数据.
BatchEncoding 自带以下键值:
                            input_ids  -> 每个 token 在词表中的数字索引, shape [batch_size, seq_len]
                            attention_mask -> 取值: 1 = 真实 token, 0 = 填充 pad 字符, shape [batch_size, seq_len]
                            token_type_ids (双句子任务专用) -> 0:第一句话 token, 1:第二句话 token
                        其它可选字段(需要进行设置)
                            overflowing_tokens -> 超长截断后溢出的 token 序列(return_overflowing_tokens=True)
                            offset_mapping     -> fast 分词器专属，每个 token 对应原文本字符起止坐标（做实体抽取必备）
                            special_tokens_mask-> 区分普通 token 和特殊 token(CLS/SEP/PAD  )         
'''

#print(inputs.keys())

'''
KeysView({'input_ids': tensor([[151644,   8948,    198,  56568, 101909, 108460,   9370,  15235,  54599,
            102,  44934,   1773, 151645,    198, 151644,    872,    198, 112720,
         108965, 109432,  72261,  98671,  82894,   3189,   6996,  52378,      8,
          73218, 151645,    198, 151644,  77091,    198]], device='cuda:1'), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:1')})

parameter:  input_ids -> 真正的词元 ID 张量
            attention_mask -> 注意力掩码张量
         
'''


print("\n" + "="*10 + " 3. 手写: model.forward 自回归循环 (Greedy) " + "="*10)
# 手动拿 logits 进行 decode
generated_ids = inputs.input_ids.clone()
# 方便内存管理 重新开辟新内存 generated_ids 与原先的 input_ids 互不干扰 (保护原 prompt )

# 获取 由 tokenizer 后的 inputs 中的 input_ids 

print("AI 正在逐字思考: ", end="", flush=True)
with torch.no_grad():
    for _ in range(10): # 手动循环 X 步，模拟大模型 "生成文本"
        # 1. 前向传播，获取当前所有 token 的输出
        outputs = model(input_ids=generated_ids)  # generated_ids  shape:[bsz, seq_len]
        '''
        model中: input --> Embedding --> Transformer+FFN --> LM Head
        generated_ids 经 Embedding 后 -> [bsz, seq_len, h_dim] 经TF&FFN 后 --> [bsz, seq_len, h_dim] 
                      最后经 LM Head (其实是一个大的 线性全连接层(Linear Layer)) h_dim 与 W∈R^(h_dim*vab_s) 相乘
                      最终返回 [bsz, seq_len, vab_s]  # Qwen vocab_size 大小为: 151936.
        '''

        # 2. 切片：只取序列中最后一个 token 的 logits 
        # 维度变化: [batch_size, seq_len, vocab_size] -> [batch_size, vocab_size]
        next_token_logits = outputs.logits[:, -1, :]
        
        # 3. 贪心解码：直接取概率最大的词的 ID
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # 4. 把预测出的新词 ID 拼接到原序列后面，进入下一轮循环！
        generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
        
        # 实时打印刚刚生成的那个词 (跳过前面的 prompt)
        new_word = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
        print(new_word, end="", flush=True)
print("... (手动截断)")


print("\n\n" + "="*10 + " 4. 工业级 API：体验生成参数 " + "="*10)
# 对应 temperature, top_k, top_p
temperatures = 0.2 # 自己修改温度值.
print(f"使用 generate() 并开启采样 (Temperature={temperatures} ):")
'''
Temperature 的相关设置  $$ p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)} $$  
            当 $T = 1$ 时，是标准的 Softmax, 当 $T > 1 时，低概率词被强行提拔，模型开始“发散思维”(甚至胡言乱语)
Top-K: 强行砍掉概率排名在 $K$ 名之后的词，只在前 $K$ 个词里重新分配概率并采样.

Top-P:按照概率从大到小累加,当累加概率刚超过 P (如 0.9)时,直接截断.
        这比 Top-K 更聪明, 能根据当前语境的确定性动态调整候选池的大小.
'''


with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,      # 开启采样，不再是死板的 argmax
        temperature=temperatures,     # 温度大于1，让低概率词也有机会冒泡, 温度小于1 输出更稳定。
        top_p=0.9,           # 核采样，截断尾部极低概率的噪音词
        top_k=50             # 保留前50个高概率词进行轮盘赌
    )
# model.generate 调用transformers 库中封装好的方法进行 文本生成.
# 截取新生成的部分（切掉 Prompt）
response_ids = outputs[0][inputs.input_ids.shape[-1]:]
final_response = tokenizer.decode(response_ids, skip_special_tokens=True)
print(final_response)

