"""
KV Cache 对比实验  代码由cc提示完成
核心认识：之前看不到加速，是因为测试"开销受限"而非"计算受限"——
          两种方法都做 N 次串行前向,0.5B 小模型每步的固定开销(Python循环/kernel启动/
          device_map hook) 远大于 cache 能省下的计算，于是差距被淹没。
两个修复：
  1) 去掉 device_map="auto"，直接 .to(device)，砍掉 accelerate hook 开销；
  2) 加大 batch size,让每步计算量盖过固定开销,cache 的优势才显现。
权威演示：以库自带 model.generate(use_cache=True/False) 的批量对比为准。
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

model_id = "Qwen/Qwen2.5-0.5B-Instruct"

# 手动选一张卡，不用 device_map="auto"
device = "cuda:1" if torch.cuda.is_available() else "cpu"

print("=" * 10 + " 加载模型 " + "=" * 10)
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device)
model.eval()
print("模型加载成功，设备:", device)

messages = [
    {"role": "system", "content": "你是一个乐于助人的 AI 助手。"},
    {"role": "user", "content": "请详细介绍一下大语言模型的推理加速技术。"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def make_batch(b):
    return tokenizer([prompt] * b, return_tensors="pt", padding=True).to(device)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    sync(); t0 = time.perf_counter(); out = fn(); sync()
    return time.perf_counter() - t0, out


# warmup
_ = timed(lambda: model.generate(**make_batch(1), max_new_tokens=8,
                                 do_sample=False, use_cache=True))

# ============================================================
# 权威演示：库自带 generate，随 batch 增大观察 cache 优势出现
# ============================================================
print("\n" + "=" * 64)
print("【model.generate：不同 batch 下 use_cache 的影响】")
print(f"{'batch':<8}{'gen长度':<8}{'cache=False(s)':<16}{'cache=True(s)':<16}{'加速比':<8}")
print("=" * 64)
for b in [1, 8, 32]:
    bi = make_batch(b)
    for n in [200, 400]:
        t_f, _ = timed(lambda: model.generate(**bi, max_new_tokens=n,
                                              do_sample=False, use_cache=False))
        t_t, _ = timed(lambda: model.generate(**bi, max_new_tokens=n,
                                              do_sample=False, use_cache=True))
        print(f"{b:<8}{n:<8}{t_f:<16.3f}{t_t:<16.3f}{(t_f/t_t):<8.2f}x")
print("=" * 64)
print("batch 越大，加速比越明显 —— 计算盖过固定开销后,cache 才显")


# ============================================================
# 手写循环（去掉 device_map 后，一致性应恢复）—— 教学用
# ============================================================
@torch.no_grad()
def gen_no_cache(input_ids, n):
    g = input_ids.clone()
    for _ in range(n):
        logit = model(input_ids=g).logits[:, -1, :]
        g = torch.cat([g, logit.argmax(-1, keepdim=True)], dim=-1)
    return g


@torch.no_grad()
def gen_with_cache(input_ids, n):
    g = input_ids.clone()
    past = DynamicCache()
    cur = input_ids
    cache_position = torch.arange(input_ids.shape[1], device=device)
    for _ in range(n):
        out = model(input_ids=cur, past_key_values=past,
                    use_cache=True, cache_position=cache_position)
        past = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        g = torch.cat([g, nxt], dim=-1)
        cur = nxt
        cache_position = cache_position[-1:] + 1
    return g


print("\n【手写循环一致性校验（去掉 device_map 后应一致）】")
inp = make_batch(1)
a = gen_no_cache(inp.input_ids, 60)
b = gen_with_cache(inp.input_ids, 60)
print("  结果:", "✅ 一致" if torch.equal(a, b) else "❌ 不一致")
print("\n生成示例:\n",
      tokenizer.decode(b[0][inp.input_ids.shape[-1]:], skip_special_tokens=True))