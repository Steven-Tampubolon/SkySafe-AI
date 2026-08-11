"""
ai_layer/translator.py — Lapisan AI translasi SkySafe AI (role: Petani, Minggu 1).

PRINSIP KERAS (dari skysafe-ai-prompt-templates.md):
AI di lapisan ini TIDAK PERNAH menghitung/mengoreksi angka. Semua skor sudah
final dari modul deterministik (ingestion + scoring). Tugas AI murni
menerjemahkan ke bahasa natural sesuai role. Kalau LLM mengubah angka kunci
di output (confidence_label, dll) dibanding input asli -> ditolak & fallback
ke template statis. Ini pengaman trust utama sistem.
"""

import os
import json
import logging

import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("skysafe.ai_layer")
logging.basicConfig(level=logging.INFO)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
REQUEST_TIMEOUT = 20
MAX_RETRIES = 1  # 1x retry kalau validasi gagal, setelah itu fallback statis

OUTPUT_REQUIRED_KEYS = [
    "headline", "plain_explanation", "recommended_action",
    "confidence_label", "why_confidence", "source_citation",
]


class TranslationError(Exception):
    """Dilempar hanya untuk error konfigurasi (mis. API key kosong) atau role tak dikenal."""
    pass


MASTER_SYSTEM_PROMPT = """\
Anda adalah lapisan penerjemah SkySafe AI. Tugas Anda HANYA menerjemahkan data
dan skor dampak cuaca antariksa yang SUDAH DIHITUNG secara deterministik
menjadi bahasa yang mudah dipahami sesuai peran pengguna yang diberikan.

ANDA TIDAK BOLEH:
- Menghitung ulang, mengoreksi, atau mengubah angka/skor/index yang diberikan
  di input. Semua angka adalah fakta yang sudah final.
- Membuat prediksi, angka, atau klaim baru yang tidak ada di data input.
- Memberi jaminan mutlak ("pasti aman", "pasti gagal", "dijamin normal").
- Mengarang atau memodifikasi nama sumber data.
- Menyembunyikan atau meremehkan skor risiko tinggi demi nada yang "enak dibaca".

ANDA WAJIB:
- Selalu sertakan skor/index mentah dan nama sumber resmi dalam penjelasan.
- Menyesuaikan istilah dan fokus dengan peran pengguna yang diberikan.
- Menggunakan bahasa tenang, konkret, dan actionable — bukan menakut-nakuti.
- Menyatakan tingkat keyakinan (confidence_level) dan alasan singkatnya.
- Jika data bersifat FORECAST (bukan real-time terukur), nyatakan itu secara
  eksplisit dan jangan sampaikan seolah-olah kejadian pasti terjadi.
- Jika impact_label = "Kritis", arahkan pengguna untuk mengecek sumber resmi
  secara langsung (link disediakan di input) — jangan beri false reassurance.

FORMAT OUTPUT: selalu balas HANYA dalam JSON valid sesuai skema yang diberikan
di setiap prompt peran. Jangan tambahkan teks di luar JSON.
"""

ROLE_TEMPLATE_PETANI = """\
PERAN: Petani atau operator alat pertanian presisi (traktor RTK-GPS, drone
sprayer, mesin tanam/panen otomatis).

FOKUS PENJELASAN:
- Dampak ke akurasi GPS untuk tanam presisi, semprot terarah, dan panen
  otomatis. Gunakan satuan familiar (pergeseran dalam cm/meter) HANYA jika
  informasi itu tersedia di gps_impact_score — jangan mengarang angka cm.
- Jika gps_impact_label "Sedang" ke atas, sarankan alternatif praktis
  (mis. tunda operasi presisi, gunakan mode manual, cek ulang titik acuan).

NADA: praktis, hangat, hindari jargon teknis cuaca antariksa. Bayangkan Anda
menjelaskan ke petani yang tidak familiar istilah "geomagnetik" atau "Kp-index" —
sebut istilah itu sekali saja untuk konteks, lalu fokus ke dampak praktisnya.

SKEMA OUTPUT (WAJIB, balas HANYA JSON ini, tanpa teks lain):
{{
  "headline": "1 kalimat pendek, langsung ke inti dampak",
  "plain_explanation": "2-3 kalimat, bahasa awam sesuai peran",
  "recommended_action": "1-2 kalimat, konkret dan bisa langsung dilakukan",
  "confidence_label": "diambil langsung dari input, jangan diubah",
  "why_confidence": "1 kalimat alasan tingkat keyakinan",
  "source_citation": "nama sumber + url, diambil langsung dari input"
}}

DATA INPUT:
{input_json}
"""

ROLE_TEMPLATES = {
    "petani": ROLE_TEMPLATE_PETANI,
    # role lain (surveyor, radio_amatir, umum) ditambahkan Minggu 2
}


def _get_groq_api_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise TranslationError("GROQ_API_KEY tidak ditemukan di .env")
    return key


def _build_static_fallback(data: dict) -> dict:
    """
    Template statis non-AI berbasis label saja — dipakai kalau Groq API gagal
    atau outputnya tidak valid/tidak lolos validasi. Trust Panel TIDAK BOLEH
    kosong walau AI gagal total (sesuai catatan integrasi dokumen prompt).
    """
    gps_label = data.get("gps_impact_label", "Tidak diketahui")
    confidence = data.get("confidence_level", "Tidak diketahui")
    source = data.get("source_name", "Sumber tidak diketahui")
    url = data.get("source_url", "")

    return {
        "headline": f"Dampak GPS saat ini: {gps_label}",
        "plain_explanation": (
            f"Berdasarkan data dari {source}, tingkat dampak terhadap akurasi "
            f"GPS berada pada level '{gps_label}'. Data ini bersifat "
            f"{data.get('data_type', 'tidak diketahui')}."
        ),
        "recommended_action": (
            "Cek sumber resmi untuk detail lebih lanjut sebelum mengambil "
            "keputusan operasional."
        ),
        "confidence_label": confidence,
        "why_confidence": data.get("confidence_reason", "Tidak ada penjelasan tambahan."),
        "source_citation": f"{source} — {url}".strip(" —"),
        "_is_fallback": True,
    }


def _validate_output(output: dict, data: dict) -> bool:
    """
    Validasi-balik: pastikan semua key wajib ada DAN confidence_label di
    output sama persis dengan confidence_level di input. Ini pengaman supaya
    LLM tidak diam-diam mengubah angka/label kunci (lihat "Catatan Integrasi"
    di dokumen prompt template).
    """
    if not isinstance(output, dict):
        return False

    for key in OUTPUT_REQUIRED_KEYS:
        if key not in output or not isinstance(output[key], str) or not output[key].strip():
            logger.warning(f"Validasi gagal: key '{key}' hilang/kosong di output LLM.")
            return False

    expected_confidence = data.get("confidence_level")
    if output["confidence_label"].strip() != str(expected_confidence).strip():
        logger.warning(
            f"Validasi gagal: confidence_label output "
            f"({output['confidence_label']!r}) != confidence_level input "
            f"({expected_confidence!r})."
        )
        return False

    return True


def _call_groq(system_prompt: str, user_prompt: str) -> dict:
    api_key = _get_groq_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    body = resp.json()

    content = body["choices"][0]["message"]["content"]
    return json.loads(content)


def call_translation_layer(role: str, data: dict) -> dict:
    """
    Terjemahkan data + skor deterministik menjadi rekomendasi bahasa natural
    sesuai role, lewat Groq API. Selalu validasi-balik output terhadap input;
    kalau gagal (tidak valid JSON, key hilang, atau angka kunci berubah),
    retry sekali, lalu fallback ke template statis non-AI.

    Args:
        role: salah satu dari ROLE_TEMPLATES (Minggu 1 baru ada "petani").
        data: dict sesuai skema input dokumen prompt template (kp_index,
              gps_impact_label, confidence_level, source_name, dst).

    Returns:
        dict sesuai skema output Trust Panel. Ada key tambahan
        "_is_fallback": True/False untuk menandai asal output.
    """
    if role not in ROLE_TEMPLATES:
        raise TranslationError(f"Role '{role}' belum didukung di Minggu 1: {list(ROLE_TEMPLATES.keys())}")

    template = ROLE_TEMPLATES[role]
    user_prompt = template.format(input_json=json.dumps(data, ensure_ascii=False, indent=2))

    attempts = 0
    last_error = None

    while attempts <= MAX_RETRIES:
        attempts += 1
        try:
            output = _call_groq(MASTER_SYSTEM_PROMPT, user_prompt)
        except (requests.exceptions.RequestException, KeyError, json.JSONDecodeError) as e:
            logger.warning(f"Percobaan {attempts} gagal: {e}")
            last_error = e
            continue

        if _validate_output(output, data):
            output["_is_fallback"] = False
            return output

        logger.warning(f"Percobaan {attempts}: output tidak lolos validasi.")

    logger.error(f"Semua percobaan gagal (terakhir: {last_error}). Pakai fallback statis.")
    return _build_static_fallback(data)