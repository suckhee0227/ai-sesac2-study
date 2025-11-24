import sounddevice as sd
import soundfile as sf
import torch
import torchaudio
import pandas as pd
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# -------------------------
# 1. 오디오 녹음
# -------------------------
def record_audio(filename="output.wav", duration=8, samplerate=44100):
    """마이크로부터 오디오를 녹음하고 WAV 파일로 저장"""
    print(f"🎙️ {duration}초 동안 녹음을 시작합니다... 말씀해주세요!")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    sf.write(filename, recording, samplerate)
    print(f"💾 오디오 저장 완료: {filename}")
    return filename

# -------------------------
# 2. Whisper 모델 불러오기
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/whisper-large-v3"

print("📥 Whisper 모델을 불러오는 중입니다...")
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
processor = AutoProcessor.from_pretrained(model_id)
print("✅ Whisper 모델 로드 완료!")

# -------------------------
# 3. 음성 → 텍스트 변환
# -------------------------
def transcribe(wav_file):
    # 오디오 읽기
    audio_input, sr = sf.read(wav_file)
    
    # 1채널 텐서 변환 (float32 유지)
    audio_tensor = torch.tensor(audio_input.T, dtype=torch.float32)
    
    # 16kHz로 리샘플링
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        audio_tensor = resampler(audio_tensor)
    
    # 모델 입력: (float32 tensor, sampling_rate=16000)
    inputs = processor(audio_tensor, sampling_rate=16000, return_tensors="pt").to(device)

    # 텍스트 생성
    with torch.no_grad():
        generated_ids = model.generate(**inputs)
    
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

# -------------------------
# 4. CSV 저장
# -------------------------
def save_to_csv(text, csv_filename="transcription.csv"):
    df = pd.DataFrame([{"text": text}])
    df.to_csv(csv_filename, index=False, sep="|", encoding='utf-8-sig')
    print(f"📂 CSV 저장 완료: {csv_filename}")
    print(df)

# -------------------------
# 5. 실행부
# -------------------------
if __name__ == "__main__":
    # 1) 8초 녹음
    wav_file = record_audio("my_recording.wav", duration=8)
    
    # 2) 녹음된 오디오 → 텍스트 변환
    print("\n📝 음성을 텍스트로 변환하는 중...")
    text = transcribe(wav_file)
    
    # 3) 변환된 텍스트 출력
    print("\n🎤 인식 결과:\n", text)
    
    # 4) CSV 저장
    save_to_csv(text, "my_recording.csv")
