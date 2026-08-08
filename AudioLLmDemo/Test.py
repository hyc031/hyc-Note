from datasets import load_dataset
import soundfile as sf

ds = load_dataset(
    "hf-internal-testing/librispeech_asr_dummy",
    "clean",
    split="validation",
)

item = ds[0]
audio = item["audio"]

print(item["text"])
print(audio.keys())

sf.write(
    "test.wav",
    audio["array"],
    audio["sampling_rate"],
)

print("已保存 test.wav")