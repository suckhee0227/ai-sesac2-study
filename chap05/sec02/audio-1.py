import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from openai import OpenAI
from dotenv import load_dotenv
import os

# -------------------------
# 1. 환경 변수 로드
# -------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# -------------------------
# 2. 환경 변수 확인
# -------------------------
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY 가 설정되지 않았습니다. .env 파일을 확인하세요.")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("❌ SUPABASE 설정이 누락되었습니다. .env 파일을 확인하세요.")

print(f"✅ OPENAI_API_KEY 로드 완료")
print(f"✅ SUPABASE_URL: {SUPABASE_URL}")
print(f"✅ SUPABASE_KEY: {'*' * 8} (보안상 일부만 표시)")

# -------------------------
# 3. OpenAI 클라이언트 초기화
# -------------------------
client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------
# 4. 오디오 녹음 기능
# -------------------------
def record_audio(filename="output.wav", duration=5, samplerate=44100):
    """마이크로부터 오디오를 녹음하고 WAV 파일로 저장"""
    print(f"🎙️ {duration}초 동안 녹음을 시작합니다...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=2, dtype='int16')
    sd.wait()  # 녹음 끝날 때까지 대기
    write(filename, samplerate, recording)
    print(f"💾 오디오 저장 완료: {filename}")

# -------------------------
# 5. 실행 (테스트용)
# -------------------------
if __name__ == "__main__":
    record_audio("output.wav", duration=5)
