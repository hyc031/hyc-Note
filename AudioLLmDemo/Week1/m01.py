# 一个情感分析的小案例, 主要测试环境是否没问题.
# 使用AutoTokenizer 和 AutoModel.
import torch
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# 使用最经典的轻量级情感分类模型
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
text = "HuggingFace is actually quite straightforward once you understand the underlying PyTorch tensors!" # --> positive 
#text = "Today has been absolutely terrible!"  # --> negative 

# print("="*10 + " 1. Pipeline 极简测试 " + "="*10)
# #  "用 pipeline() 跑一个任务"
# classifier = pipeline("sentiment-analysis", model=model_name)
# result = classifier(text)
# print(f"Pipeline 预测结果: {result}\n")
# # 对应的 预测 Text 文本情感 positive / negative

# print("="*10 + " 2. Tokenizer 手动分词 " + "="*10)
# 加载分词器，并将文本转为 PyTorch 张量 (return_tensors="pt")
tokenizer = AutoTokenizer.from_pretrained(model_name)
inputs = tokenizer(text, return_tensors="pt")

# print("Input IDs (词元索引):", inputs["input_ids"])
# print("Attention Mask (注意力掩码):", inputs["attention_mask"], "\n")

'''
将上文中的 text 使用Tokenizer 进行编码(text1 = "HuggingFace ....", text2 = "Today ....")
text1 --> tensor([[  101, 17662, 12172,  2003,  2941,  3243, 19647,  2320,  2017,  3305,
          1996, 10318,  1052, 22123,  2953,  2818, 23435,  2015,   999,   102]])    # 20个 Token 
对应的 mask tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])  # 20个

text2 --> tensor([[ 101, 2651, 2038, 2042, 7078, 6659,  999,  102]])

其中 101:[CLS], 102:[SEP]
Mask [1] 保证句子有效文本  分配注意力权重
'''


# print("="*10 + " 3. AutoModel (基础主干网络) " + "="*10)
# #  "用 AutoModel 手动加载"
# base_model = AutoModel.from_pretrained(model_name)
# with torch.no_grad():  # 推理时必须关梯度，省显存
#     base_outputs = base_model(**inputs)

# # 原生 AutoModel 只输出隐藏状态 (Hidden States)，没有分类结果
# print("基础模型输出的张量形状 (Batch, Seq_Len, Hidden_Dim):")
# print(base_outputs.last_hidden_state.shape, "\n")
'''
输出 Tensor shape -->  [bsz, seq_len, h_dim]   #经典 Transformer 中的shape形状.
'''

print("="*10 + " 4. AutoModelForSequenceClassification (带分类头) " + "="*10)
#  (Base Model + Linear Layer)
cls_model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 打印模型结构与 config 
# print(cls_model.config) 
# print(cls_model)        
'''
对应的输出一些config 信息 如: dropout\ activation \ dim \ h_dim ....
cls_model : 对应输出 一些网络层信息.
'''

with torch.no_grad():
    cls_outputs = cls_model(**inputs)

# 获取分类的 Logits (未经过 Softmax 的原始得分)
logits = cls_outputs.logits
print("输出的 Logits:", logits)
'''
logits 返回一个 Tensor , 后续 argmax计算最终预测情感信息 0(NEGATIVE) / 1(POSITIVE)  
'''


# 用 PyTorch 基础操作求出预测类别
predicted_class_id = torch.argmax(logits, dim=-1).item()
predicted_label = cls_model.config.id2label[predicted_class_id]
print(f"最终手动推理解码的标签: {predicted_label}")