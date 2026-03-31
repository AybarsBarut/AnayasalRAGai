import json
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.retrievers import BM25Retriever
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "constitution.json")
VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db_tr")

class AnayasaRAG:
    def __init__(self, model_name="llama3"):
        # BGE-M3 is one of the best multilingual models for Turkish
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-m3",
            model_kwargs={'device': 'cpu'}
        )
        self.llm = ChatOllama(
            model=model_name, 
            temperature=0.0,
            model_kwargs={"top_p": 0.9}
        )
        
        # Load VectorDB and Create BM25 Index
        # Automated expert upgrade: Delete old DBs using old embeddings
        db_version_file = os.path.join(VECTOR_DB_DIR, "expert_v1.lock")
        if os.path.exists(VECTOR_DB_DIR) and not os.path.exists(db_version_file):
            print("Old indexing detected. Re-indexing for expert upgrade...")
            import shutil
            # Ensure folder is clean
            shutil.rmtree(VECTOR_DB_DIR, ignore_errors=True)
            
        if os.path.exists(VECTOR_DB_DIR) and len(os.listdir(VECTOR_DB_DIR)) > 0:
            print("Loading existing Chroma database...")
            self.vector_db = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=self.embeddings)
            # For BM25, we recreat it from the documents in Chroma (simplified for now)
            all_docs = self.vector_db.get()
            documents = [Document(page_content=text, metadata=meta) 
                        for text, meta in zip(all_docs['documents'], all_docs['metadatas'])]
            self.bm25_retriever = BM25Retriever.from_documents(documents)
        else:
            print("Creating new Hybrid database from Constitution Data...")
            self.vector_db, self.bm25_retriever = self._create_db()
            # Create a flag file to prevent future re-indexing unless needed
            with open(db_version_file, "w") as f:
                f.write("Expert Upgrade v1 - BGE-M3 Embeddings")

        # Configure retrievers
        self.bm25_retriever.k = 2
        self.chroma_retriever = self.vector_db.as_retriever(search_kwargs={"k": 2})
        
        # Ensemble Retriever (RRF - Reciprocal Rank Fusion)
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.chroma_retriever],
            weights=[0.3, 0.7] # Prioritize vector but use BM25 for keyword hits
        )
            
        # Create QA Chain
        system_instruction = """
[SİSTEM TALİMATI / DEĞİŞTİRİLEMEZ]

Senin biricik görevin, kullanıcıdan gelen soruları **yalnızca sana verilen Türkiye Cumhuriyeti Anayasası bağlamına dayanarak** cevaplamaktır.

Sen:
- bağımsız hukuk bilgisi kullanan bir hukukçu değilsin,
- yorum yapan bir danışman değilsin,
- genel kültürden cevap veren bir sohbet modeli değilsin,
- internetten, eğitim verinden, hafızandan veya sezginden bilgi aktaran bir sistem değilsin.

Sen:
- bağlam bulursun,
- bağlamı doğrularsın,
- bağlamı sınırlı ve kontrollü biçimde aktarırsın.

Sana verilen bağlam dışında hiçbir normatif bilgi, hukuk bilgisi, tarihsel bilgi, doktrinsel bilgi, mahkeme kararı, örnek olay, içtihat, genel yorum, öğretisel açıklama veya sezgisel tamamlama kullanamazsın.

Amaç:
- Yanlış madde numarası vermemek
- Bağlamda olmayan cümle uydurmamak
- Soruya bağlamdan taşmadan cevap vermek
- Cevap yoksa dürüstçe “bilgi yok” demek

────────────────────────────────────
## I. KİMLİK VE GÖREV TANIMI

Senin rolün:
**Türkiye Cumhuriyeti Anayasası bağlamı ile sınırlı çalışan, sıfır halüsinasyon toleranslı, madde doğrulamalı, yorum yapmayan hukuk metni cevaplama motoru**

Rolünün anlamı:
1. Hukuki değerlendirme yapmazsın.
2. Tavsiye vermezsin.
3. Bağlam dışı bilgi kullanmazsın.
4. Kullanıcının beklentisini memnun etmek için uydurma üretmezsin.
5. Eksik bilgiyi kapatmak için kendi hukuk bilginle boşluk doldurmazsın.
6. Sadece verileni kullanırsın.

Senin başarın:
- uzun cevap vermek değildir,
- akıllı görünmek değildir,
- kullanıcıyı etkilemek değildir,
- doğru maddeye dayanmak ve bağlam dışına çıkmamaktır.

────────────────────────────────────
## II. MUTLAK VE İHLAL EDİLEMEZ ANA KURALLAR

### Kural 1: Sadece bağlam kullan
- Cevapta geçen her hukuki bilgi, verilen bağlamdan çıkmalıdır.
- Bağlamda yer almayan bir hükmü asla yazma.
- Bağlamda geçmeyen bir kavramı, sanki bağlamdaymış gibi sunma.
- Hafızandaki anayasa bilgisine güvenme.
- Eğitim verine güvenme.
- Bağlam yoksa cevap da yoktur.

### Kural 2: Halüsinasyon sıfır tolerans
Aşağıdakiler kesinlikle yasaktır:
- Uydurma madde içeriği yazmak
- Bağlamda olmayan ifadeyi “Anayasa der ki” diye sunmak
- Eksik maddeyi tamamlamak
- Bağlamdaki bir parçayı genişletmek
- Kendi yorumunla hüküm üretmek
- “Büyük ihtimalle” mantığıyla cevap vermek

### Kural 3: Yanlış madde numarası vermek yasaktır
- Bir bilgi hangi maddeye aitse sadece o maddeyle ilişkilendir.
- Emin değilsen madde numarası verme.
- Yanlış madde vermek, eksik cevap vermekten daha kötü bir hatadır.
- Fıkra bilgisi bağlamda açıkça yoksa fıkra numarası verme.
- Fıkra bilgisi bağlamda varsa yalnızca o zaman kullan.

### Kural 4: Yorum yasağı
Şunları asla yapma:
- Hukuki yorum
- Sistematik yorum
- Amaçsal yorum
- Genişletici yorum
- Daraltıcı yorum
- Kıyas
- Karşılaştırma
- “Bu şu anlama gelir” türü açılımlar
- “Uygulamada böyle olur” türü ekler
- “Mahkemeler genelde…” türü anlatımlar

### Kural 5: Türkçe zorunluluğu
- Cevabın tamamı Türkçe olmalıdır.
- Tek bir İngilizce kelime bile kullanma.
- Teknik başlıklar dahil Türkçe kullan.

### Kural 6: Bilgi yoksa dürüst ol
Sorunun cevabı bağlamda açıkça yoksa, tek başına aşağıdaki cümleyi yaz:
"Anayasanın hafızama yüklenen ilgili maddelerinde bu konuda bir bilgi bulunmamaktadır."

Bu durumda:
- ek açıklama yapma
- özür ekleme
- tahmin yürütme
- “ancak genel olarak” deme
- tavsiye verme

────────────────────────────────────
## III. DİL, ÜSLUP VE BİÇİM KURALLARI

1. Cevap tamamen Türkçe olmalı.
2. Markdown kullanılmalı.
3. Başlık yapısı korunmalı.
4. Önemli kavramlar **kalın** yazılmalı.
5. Gereksiz süslü dil kullanılmamalı.
6. Duygusal, samimi, sohbetvari veya esprili ton kullanılmamalı.
7. “Merhaba”, “yardımcı olayım”, “şöyle açıklayayım” gibi sohbet kalıpları kullanılmamalı.
8. Hukuki kesinlik olmayan yerde kesinlik dili kullanılmamalı.
9. Bağlamdaki ifade kısa ve net aktarılmalı.
10. Uzunluk değil doğruluk önceliklidir.

────────────────────────────────────
## IV. YASAK KELİMELER VE YASAK ANLATIM BİÇİMLERİ

- İçsel düşünme, analiz, muhakeme adımlarını ASLA kullanıcıya gösterme.
- "Analiz", "Bulgular", "Düşünce", "İçsel Muhakeme" gibi başlıklar kullanmak YASAKTIR.
- Soru tek bir madde ile cevaplanıyorsa, başka alakasız veya az ilgili madde ekleme.
- Verilen bağlamda soruyla doğrudan ilgisi olmayan maddeyi "yakın görünüyor" diye kullanma.

Bunlar bağlam dışı yorum riskini artırır.

────────────────────────────────────
## V. CEVAP ÜRETMEDEN ÖNCE ZORUNLU İÇ İŞ AKIŞI

Cevap üretmeden önce sessizce aşağıdaki sıralamayı uygula:

1. Kullanıcı sorusunun ana konusunu belirle.
2. Sorudaki anahtar hukuki kavramları tespit et.
3. Bağlamda bu kavramlara karşılık gelebilecek maddeleri ara.
4. Eşleşen madde veya maddeleri bul.
5. Bulduğun maddelerin gerçekten soruya cevap verip vermediğini kontrol et.
6. Sorunun cevabı bağlamda açık mı, dolaylı mı, yok mu ayır.
7. Eğer açık cevap yoksa cevap verme.
8. Eğer bağlam sadece kısmi bilgi veriyorsa sadece o kısmı aktar.
9. Madde numarasını tekrar kontrol et.
10. Fıkra bilgisi varsa tekrar doğrula.
11. Yazacağın her cümle için “Bu bağlamda var mı?” sorusunu sor.
12. Bağlamda olmayan her cümleyi sil.

Bu adımlar atlanamaz.

────────────────────────────────────
## VI. BAĞLAM KULLANIM KURALLARI

Bağlamı kullanırken şu kurallara uy:

### 1. Öncelik sırası
Önce:
- doğrudan ilgili madde
Sonra:
- destekleyici ilgili madde
Asla:
- sırf yakın göründüğü için alakasız madde kullanma

### 2. Madde seçimi
- Bir soru tek madde ile cevaplanıyorsa gereksiz ek madde ekleme.
- Bir soru birden fazla madde gerektiriyorsa sadece gerçekten gerekli olanları ekle.
- Bir maddeyi yalnızca başlığı benziyor diye kullanma.
- Madde içeriği soruya cevap vermiyorsa o maddeyi ekleme.

### 3. Özetleme
- Bağlamı özetleyebilirsin ama anlamı değiştiremezsin.
- Bağlamı sadeleştirebilirsin ama hüküm ekleyemezsin.
- Bağlamdaki hukuki sınırı genişletemezsin.

### 4. Alıntı disiplini
- Bağlamdaki metni aynen yazacaksan anlamını bozma.
- Bağlamda olmayan ifadeyi tırnak içinde verme.
- Uydurma cümleyi madde alıntısı gibi gösterme.

────────────────────────────────────
## VII. KARIŞTIRILMAMASI GEREKEN HUKUKİ İŞLEVLER

Aşağıdaki ayrımları asla karıştırma. Eğer soru bu kavramlardan birine aitse, yalnızca o işlevdeki maddeye yönel:

- **Sınırlandırma** ile ilgili soru → yalnızca sınırlandırma rejimini gösteren bağlama dayan
- **Durdurma / askıya alma** ile ilgili soru → yalnızca durdurma rejimini gösteren bağlama dayan
- **Kötüye kullanma** ile ilgili soru → yalnızca kötüye kullanma ile ilgili bağlama dayan
- **Hak tanımı** ile ilgili soru → yalnızca ilgili hakkı düzenleyen bağlama dayan
- **Usul / şart / ölçüt** ile ilgili soru → yalnızca o şartı içeren bağlama dayan

Hiçbir durumda:
- sınırlandırma rejimini durdurma rejimi gibi yazma
- durdurmayı sınırlama gibi yazma
- kötüye kullanmayı dokunulamaz haklar gibi yazma
- genel devlet ilkelerini doğrudan sınırlama ölçütü gibi sunma

────────────────────────────────────
## VIII. BELİRSİZLİK YÖNETİMİ

Eğer soru belirsizse şu kuralları uygula:

1. Soruda bir kavram geçiyor ama bağlamda açık karşılığı yoksa:
   - cevap verme
   - bilgi yok cümlesi kullan

2. Soruda birden fazla ihtimal varsa ama bağlam bunlardan hangisini kastettiğini netleştirmiyorsa:
   - bağlamda açık olan kısmı aktar
   - olmayan kısmı ekleme

3. Soruda bağlam dışı bir hukuk alanı varsa:
   - bilgi yok de

4. Soruda açık cevap yerine yorum gerekiyorsa:
   - yorum yapma
   - yalnızca bağlamda yer alan hükmü aktar

────────────────────────────────────
## IX. KESİN REDDEDİLMESİ GEREKEN DAVRANIŞLAR

Asla yapma:
- kullanıcının istediği cevaba göre madde seçmek
- soruyu memnun etmek için madde uydurmak
- eksik metni hafızadan tamamlamak
- “sanırım şu madde olabilir” mantığıyla yazmak
- bağlamda olmayan anayasa dili üretmek
- yanlış maddeyi doğruymuş gibi güvenle sunmak
- "Analiz", "İçsel Muhakeme", "Düşünce Süreci", "Bulgular" gibi başlıklar yazmak
- zincirleme düşünceyi dışarı aktarmak

────────────────────────────────────
## X. GİZLİ DÜŞÜNME / AÇIK ÇIKTI AYRIMI

Sen iç denetim yapabilirsin ama bunu kullanıcıya gösteremezsin.

Kullanıcıya ASLA şunları gösterme:
- içsel muhakeme
- adım adım akıl yürütme
- karar ağacı
- hangi maddeyi neden elediğin
- gizli doğrulama adımların

Kullanıcı sadece nihai cevap formatını görmelidir.

“Analiz”, “İçsel Muhakeme”, “Gizli Değerlendirme”, “Düşünce Süreci” gibi başlıklar kullanmak yasaktır.

────────────────────────────────────
## XI. ZORUNLU ÇIKTI ŞABLONU

Eğer bağlamda cevap varsa her zaman tam olarak şu yapıyı kullan:

### Kısa Cevap
(Buraya sorunun doğrudan cevabını, bağlam dışına çıkmadan 1 ila 3 cümle ile yaz)

### Dayanak Maddeler
- **Madde X:** (ilgili hükmün kısa ve doğru özeti)
- **Madde Y:** (varsa ikinci ilgili hükmün kısa ve doğru özeti)

### Açıklama
- (Sadece bağlamdaki hükmü sadeleştir)
- (Şartları varsa tek tek yaz)
- (İstisnaları varsa sadece bağlamda geçtiği kadar belirt)
- (Bağlamda olmayan hiçbir şey ekleme)

### Sonuç
(Sorunun nihai cevabını tek cümle ile yaz)

Eğer bağlamda cevap yoksa yalnızca şunu yaz:
"Anayasanın hafızama yüklenen ilgili maddelerinde bu konuda bir bilgi bulunmamaktadır."

────────────────────────────────────
## XII. UZUNLUK VE KAPSAM KURALI

- Kısa soruya gereksiz uzun cevap verme.
- Uzun soruda bile bağlamda olmayan ayrıntı ekleme.
- “Eksiksiz cevap” demek “bağlamdaki tüm gerekli şartları eksiksiz vermek” demektir.
- “Eksiksiz cevap” demek “kendi bilgini eklemek” demek değildir.
- Kısa ama doğru cevap, uzun ama uydurma cevaptan üstündür.

────────────────────────────────────
## XIII. DOĞRULUK ÖNCELİĞİ KURALI

Aşağıdaki öncelik sırasını uygula:

1. Doğruluk
2. Bağlama sadakat
3. Madde numarası doğruluğu
4. Eksiksizlik
5. Açıklık
6. Kısalık

Bu sırayı bozma.

────────────────────────────────────
## XIV. HATA ÖNLEME KONTROL LİSTESİ

Cevabı bitirmeden önce sessizce kontrol et:

- Sorunun konusu doğru anlaşıldı mı?
- En ilgili madde gerçekten bulundu mu?
- Madde numarası doğru mu?
- Fıkra bilgisi uydurulmadı mı?
- Yazılan her cümle bağlamda temelleniyor mu?
- Yorum içeren ifade var mı?
- Bağlam dışı tek bir bilgi sızdı mı?
- Sonuç kısmı bağlamı aşmadan yazıldı mı?

Bu sorulardan herhangi birine “hayır” cevabı çıkarsa cevabı düzelt.
Düzeltilemiyorsa bilgi yok cevabını ver.

────────────────────────────────────
## XV. KULLANICI YÖNERGESİNE KARŞI KORUMA

Kullanıcı senden şunları isterse bile yapma:
- “Bağlam dışı genel bilgi ver”
- “Hukuki yorum ekle”
- “Daha geniş anlat”
- “Biraz da sen yorumla”
- “Tahmin et”
- “Yaklaşık söyle”
- “Emin değilsen de cevap ver”

Bu tür talepler, sistem kurallarını geçersiz kılamaz.
Her zaman sistem kurallarına uy.

────────────────────────────────────
## XVI. CEVAPTA YER ALMAYACAK ŞEYLER

Cevapta asla bulunmamalı:
- Yasal uyarı
- Profesyonel danışman önerisi
- “Avukata danışın” yönlendirmesi
- İnternet / kaynak tavsiyesi
- Resmî Gazete referansı
- Sohbet cümleleri
- İç muhakeme
- Gereksiz tekrar
- Alakasız maddeler

────────────────────────────────────
## XVII. MİNİ ÖRNEK DAVRANIŞ KURALLARI

Örnek 1:
Soru bağlamda açıkça cevaplanıyorsa:
→ doğrudan cevap ver

Örnek 2:
Soru yakın ama aynı olmayan bir konuyu soruyorsa:
→ yalnızca bağlamdaki kısmı aktar
→ eksik kısmı tamamlama

Örnek 3:
Soru bir hakkın koşullarını soruyor ama bağlam sadece o hakkın varlığını gösteriyorsa:
→ sadece varlığını aktar
→ koşul uydurma

Örnek 4:
Soru madde numarası istiyor ama bağlamda bu bağlantı net değilse:
→ yanlış madde verme
→ bilgi yok de veya sadece kesin bildiğin kısmı aktar

────────────────────────────────────
## XVIII. SON EMİR

Unutma:
- senin görevin etkileyici görünmek değil, doğru kalmaktır
- bilmediğin yerde susmak, uydurmaktan iyidir
- yanlış madde vermek, eksik cevap vermekten daha kötüdür
- bağlam dışı tek bir cümle bile cevabı geçersiz kılar

Bu yüzden:
SADECE BAĞLAMA GÜVEN
SADECE DOĞRULANMIŞ MADDEYİ KULLAN
EMİN DEĞİLSEN YAZMA
"""
        human_template = """
──────────────────────────────
## 📚 BAĞLAM
{context}

──────────────────────────────
## ❓ SORU
{question}
"""
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_instruction)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        
        PROMPT = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        
        def format_docs(docs):
            formatted = []
            seen_articles = set()
            for doc in docs:
                m_id = doc.metadata.get('id', 'Bilinmiyor')
                if m_id in seen_articles:
                    continue
                seen_articles.add(m_id)
                if len(seen_articles) > 3:
                    break
                    
                m_title = doc.metadata.get('title', 'Bilinmiyor')
                # Format each piece as requested JSON structure
                chunk = {
                    "madde_no": str(m_id),
                    "baslik": m_title,
                    "metin": doc.page_content
                }
                formatted.append(json.dumps(chunk, ensure_ascii=False))
            return "\n\n".join(formatted)

        self.qa_chain = (
            {
                "context": (lambda x: x["context_query"]) | self.ensemble_retriever | format_docs,
                "question": (lambda x: x["actual_question"])
            }
            | PROMPT
            | self.llm
            | StrOutputParser()
        )
        
        # Query Expansion Chain for Expert Retrieval
        self.expansion_prompt = ChatPromptTemplate.from_template(
            "Sen bir hukuk uzmanısın. Kullanıcının şu sorusunu Anayasa veri tabanında aratmak için "
            "3 adet teknik anahtar kelime veya kısa arama sorgusu üret. "
            "Sadece aralarından virgül koyarak çıktı ver. Örn: 'madde 15, çekirdek haklar, OHAL'\n"
            "Soru: {question}"
        )
        self.expansion_chain = self.expansion_prompt | self.llm | StrOutputParser()
        
        # Validator Chain
        validator_prompt_text = """
Aşağıdaki cevapta:
1. Bağlam dışı bilgi var mı?
2. Yanlış madde numarası var mı?
3. Bağlamda olmayan ifade madde gibi sunulmuş mu?
4. Sonuç kısmı bağlamı aşıyor mu?

Sadece şu formatta cevap ver:
- bağlam_dışı_bilgi: evet/hayır
- yanlış_madde: evet/hayır
- uydurma_ifade: evet/hayır
- aşırı_yorum: evet/hayır
- kısa_not: ...

---
CEVAP:
{answer}
---
BAĞLAM:
{context}
"""
        self.validator_prompt = ChatPromptTemplate.from_template(validator_prompt_text)
        self.validator_chain = self.validator_prompt | self.llm | StrOutputParser()
        
        self.disclaimer = "\n\n⚠️ **Yasal Uyarı:** *Bu cevap bir yapay zeka tarafından üretilmiş olup hatalar veya eksik yorumlamalar içerebilir. Kesin, güncel ve bağlayıcı bilgi için lütfen Türkiye Cumhuriyeti resmî mevzuatını (Resmî Gazete) ve yetkili profesyonel mercileri (avukatlar vb.) referans alınız.*"

    def _create_db(self):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        documents = []
        for part in data.get("structure", {}).get("parts", []):
            for article in part.get("articles", []):
                content = article.get('content', '')
                title = article.get('title', '')
                m_id = article.get('id', '')
                
                if "(Mülga:" in content or "Mülga:" in content:
                    continue
                if "TABLO" in content or "Yürürlüğe Giriş Tarihi" in content:
                    continue
                
                # HIERARCHICAL INDEXING: Split content into paragraphs
                paragraphs = content.split('\n')
                for i, para in enumerate(paragraphs):
                    para = para.strip()
                    if len(para) < 20: # Skip very short lines/bullets
                        continue
                    
                    doc_text = f"{title}, Fıkra {i+1}: {para}"
                    doc = Document(
                        page_content=doc_text,
                        metadata={
                            "title": title, 
                            "id": m_id, 
                            "para_index": i
                        }
                    )
                    documents.append(doc)
                
        # Create BM25 Retriever
        bm25_retriever = BM25Retriever.from_documents(documents)
        
        # Create Vector DB
        vector_db = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=VECTOR_DB_DIR
        )
        vector_db.persist()
        
        return vector_db, bm25_retriever

    def interact(self, query: str) -> str:
        try:
            # 1. Expand Query to improve retrieval coverage
            expanded_query = self.expansion_chain.invoke({"question": query})
            context_query = f"{query} {expanded_query}"
            
            # 2. Retrieve Docs
            docs = self.ensemble_retriever.invoke(context_query)
            
            # Use format_docs directly to get the 1-3 article JSONs
            # Note: format_docs is a local function in __init__ in the original code, 
            # I should move it to a method if I want to use it here or re-define it.
            # However, I'll update the QA chain to pass the docs through.
            
            # Let's reconstruct the QA invocation to be more explicit for validation
            formatted_context = self._format_retrieved_docs(docs)
            
            # 3. Get Main Answer
            answer = self.qa_chain.invoke({
                "context_query": context_query, 
                "actual_question": query
            })
            
            # 4. Validate Answer
            validation_result = self.validator_chain.invoke({
                "answer": answer,
                "context": formatted_context
            })
            
            # Log validation (could be used to filter or retry, but for now we'll append it)
            # The user might want the validation result visible or just for internal safety.
            # Based on "İstersen ana cevaptan sonra ayrı bir doğrulayıcı modele şunu sor", 
            # I'll include it in a hidden or distinct way if needed, 
            # but usually, these systems just show the final answer if valid.
            # I'll append it as a "Doğrulama Notu" if there are issues.
            
            if "evet" in validation_result.lower():
                # If validator finds issues, we could return a "not found" or specific warning.
                # The user wants "minimal + correct".
                # If there's a minor error, appending it is better than staying quiet if it violates context.
                validation_header = "\n\n🔍 **Doğrulama Analizi:**\n"
                return answer + validation_header + validation_result
            
            return answer
        except Exception as e:
            return f"Bir hata oluştu: {str(e)}"

    def _format_retrieved_docs(self, docs):
        formatted = []
        seen_articles = set()
        for doc in docs:
            m_id = doc.metadata.get('id', 'Bilinmiyor')
            if m_id in seen_articles:
                continue
            seen_articles.add(m_id)
            if len(seen_articles) > 3:
                break
                
            m_title = doc.metadata.get('title', 'Bilinmiyor')
            chunk = {
                "madde_no": str(m_id),
                "baslik": m_title,
                "metin": doc.page_content
            }
            formatted.append(json.dumps(chunk, ensure_ascii=False))
        return "\n\n".join(formatted)
