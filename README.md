# Türkiye Cumhuriyeti Anayasası (Expert AI & Repo)

Bu depo, Türkiye Cumhuriyeti Anayasası'nın (1982) tüm değişiklikleri içeren en güncel metnini yapılandırılmış bir formatta sunar ve bu metin üzerinde çalışan gelişmiş bir **RAG (Retrieval-Augmented Generation)** hukuk asistanı içerir.

## Öne Çıkan Özellikler

- **Anayasa AI Asistanı**: Ollama (Llama3) ve BGE-M3 embedding modelleriyle çalışan, %100 yerel ve gizlilik odaklı hukuk asistanı.
- **Sıfır Halüsinasyon (Zero-Hallucination)**: 3 katmanlı prompt mimarisi ve bağımsız doğrulayıcı (validator) modeli ile sadece anayasa bağlamına sadık cevaplar.
- **Hibrit Arama**: BM25 (anahtar kelime) ve ChromaDB (vektörel) algoritmalarını birleştiren yüksek hassasiyetli erişim mekanizması.
- **Yapılandırılmış Veri**: Tüm anayasa hem Markdown hem de hiyerarşik JSON formatında sunulmaktadır.

## Depo Yapısı

- `backend/`: FastAPI tabanlı RAG motoru ve API uç noktaları.
- `frontend/`: AI asistanı için web arayüzü.
- `docs/`: Anayasanın bölümlere ayrılmış Markdown (.md) metinleri.
- `data/`: Anayasa maddelerinin hiyerarşik JSON veri seti.
- `scripts/`: Veri işleme ve temel arama araçları.

## Kurulum ve Kullanım

### 1. Gereksinimler

- Python 3.10+
- [Ollama](https://ollama.com/) (Llama3 modelinin yüklü olması gerekir: `ollama run llama3`)

### 2. Kurulum

```bash
# Depoyu klonlayın
git clone [https://github.com/AybarsBarut/Anayasa_Ai.git](https://github.com/AybarsBarut/Anayasa_Ai.git)
cd Anayasa_Ai

# Sanal ortam oluşturun ve aktif edin
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Bağımlılıkları yükleyin
pip install -r requirements.txt

### UYARI!
Temelde bir yapay zeka modeli olduğu için hep doğru cevap vereceği yanılgısına kapılmayıp asıl sorularınızı yetkili kişilere soramnız tavsiye edilir (Avukatlar,Hakimler,Savcılar...vb).
