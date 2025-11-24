import os
from supabase import create_client, Client
from dotenv import load_dotenv

# .env 파일에서 모든 환경 변수 불러오기
load_dotenv()

# --- 1. 데이터 준비 ---
# Supabase에 넣을 데이터를 딕셔너리 형태로 준비합니다.
token_data = {
    "completion_tokens": 139,
    "prompt_tokens": 32,
    "total_tokens": 171,
    "model": "gpt-4o-2024-08-06",
    "created": 1758681263
}


try:
    # --- 2. Supabase Client Creation ---
    url: str = os.environ.get("SUPABASE_URL")
    key: str = os.environ.get("SUPABASE_API_KEY")
    supabase: Client = create_client(url, key)

    # --- 3. Data Insertion (Corrected) ---
    # The dictionary's key 'log_data' now matches your column name.
    # The value is the entire token_data dictionary.
    data, count = supabase.table('api_logs').insert({'log_data': token_data}).execute()

    print("✅ The data was successfully saved to the 'log_data' column in Supabase.")
    print(f"Saved Data: {data[1]}")

except Exception as e:
    print(f"❌ An error occurred: {e}")

