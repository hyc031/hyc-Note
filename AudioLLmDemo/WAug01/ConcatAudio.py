import os
import glob
import numpy as np
import soundfile as sf

def merge_to_30s(audio_files, target_sec=30, sr=16000):
    """
    将多个短音频拼接成接近 target_sec 的长片段
    audio_files: 音频文件路径列表，如 ["data/a.wav", "data/b.wav"]
    返回: numpy 数组 (长度 = target_sec * sr)
    """
    target_samples = target_sec * sr
    merged = []
    total = 0
    
    for f in audio_files:
        if not os.path.exists(f):
            print(f"警告: 文件不存在，跳过: {f}")
            continue
            
        wav, file_sr = sf.read(f)
        
        # 如果采样率不一致，简单重采样（可选）
        if file_sr != sr:
            # 这里用简单的线性插值重采样，生产环境建议用 librosa
            wav = np.interp(
                np.linspace(0, len(wav), int(len(wav) * sr / file_sr)),
                np.arange(len(wav)),
                wav
            )
        
        # 如果是立体声，转单声道
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        
        remaining = target_samples - total
        if len(wav) > remaining:
            wav = wav[:remaining]  # 只取需要的部分，不直接 break
        
        merged.append(wav)
        total += len(wav)
        
        if total >= target_samples:
            break
    
    result = np.concatenate(merged) if merged else np.array([])
    
    # 不足 30 秒用零填充
    if len(result) < target_samples:
        result = np.pad(result, (0, target_samples - len(result)))
    
    return result[:target_samples]


# ========== 使用示例 ==========

# 1. audio_files 的写法示例

# 写法 A：手动列出具体文件
audio_files = [
    "../Data/AISHELL-1/SelfData/Demo1.wav",
    "../Data/AISHELL-1/SelfData/Demo1.wav",
    "../Data/AISHELL-1/SelfData/Demo1.wav",
]

# 写法 B：用 glob 批量读取某个文件夹下的所有 wav
# audio_dir = "/mnt/agents/upload/audio"
# audio_files = sorted(glob.glob(os.path.join(audio_dir, "*.wav")))
# 结果如: ['/mnt/agents/upload/audio/001.wav', '/mnt/agents/upload/audio/002.wav', ...]

# 写法 C：混合写法（指定文件夹 + 过滤）
# audio_files = sorted(glob.glob("/mnt/agents/upload/**/*.wav", recursive=True))


# 2. 调用函数生成 30 秒音频
# target_sec 应该保证在30s 之内, 这里设置XX 观察报错情况.
audio_120s = merge_to_30s(audio_files, target_sec=120, sr=16000)

# 3. 保存到指定路径（这才是输出路径）../Data/AISHELL-1/SelfData/Demo1.wav
output_path = "../Data/AISHELL-1/SelfData/merged_120s.wav"
sf.write(output_path, audio_120s, samplerate=16000)
print(f"已保存到: {output_path}")