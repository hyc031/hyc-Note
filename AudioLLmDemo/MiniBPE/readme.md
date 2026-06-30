# BPE(Byte Pair Encoding) 
是Tokenizer 常用方法, 使用UTF-8编码,将出现次数最多的相邻字节(best_pair)合并, 生成一个新的编码,如此反复操作。

github: https://github.com/karpathy/minbpe
base.py: 定义Tokenizer基类,提供一些工具函数。
basic.py: 一个基础版的实现,已经包含了BPE的最主要逻辑: BasicTokenizer
regex.py: 在BasicTokenizer的基础上,添加了基于「正则表达式」的pre-tokenization逻辑,
另外添加了specail tokens:RegexTokenizer.已经是一个比较完整的BPE分词器的了.

