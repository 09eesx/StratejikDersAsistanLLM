# 🧠 AÖF için Üretken Yapay Zeka Tabanlı Stratejik Öğrenme Ekosistemi

**Proje:** Büyük Dil Modelleri Final Projesi
**Hazırlayan:** Esma Elifsu Cerit
**Öğrenci No:** 220212003
**Ders:** Yapay Zeka Mühendisliği – Büyük Dil Modelleri
**Ders Hocası:** Dr. Öğr. Gör. Murat Şimşek

---

## 📌 Proje Tanımı

Bu proje, Anadolu Üniversitesi Açıköğretim Fakültesi (AÖF) öğrencilerinin akademik başarılarını artırmayı hedefleyen hibrit bir yapay zeka sistemidir. Sistem, öğrencilerin devasa ders kaynakları arasında hangi konuların kritik olduğunu belirlemelerine, sınav stratejisi geliştirmelerine ve öğrenme süreçlerini optimize etmelerine yardımcı olur.

**Problem:**

* Açıköğretim öğrencileri öğretim üyeleri ile senkronize ve interaktif bir iletişim kuramaz.
* Tüm müfredatı aynı ağırlıkta çalışmak zorunda kalırlar.
* Bilgi asimetrisi sınav başarısını riske atar.

**Amaç:**

* Ders kitaplarını, çıkmış soruları ve örnek çözümleri bütünleşik bir zeka ile analiz eden bir karar destek sistemi geliştirmek.
* Öğrenciye sadece bilgi vermekle kalmayıp sınav stratejisi de sunmak.

**Hedef:**

* Modüler ve ölçeklenebilir bir agent takımı ile öğrenciye kişisel rehberlik sağlamak.
* Çalışma süresini optimize etmek, potansiyel sınav sorularını öngörmek ve sık yapılan hataları tespit etmek.

---

## 🏗 Teknik Mimari

### 1. Sistem Bileşenleri

* **Geliştirme Ortamı:** Google Colab, Python
* **GPU Kullanımı:** Tesla T4 (LLM inference & embedding)
* **CPU Kullanımı:** Veri ön işleme, FAISS indeksleme, RegEx ayrıştırma

**Neden RAG (Retrieval-Augmented Generation)?**

* LLM’ler özel müfredat ve güncel sınav içeriklerine dair sınırlı bilgiye sahiptir.
* RAG, modelin yanlışa düşme riskini azaltır ve cevabı güvenilir kaynaklara dayandırır.

**Veri Seti Stratejisi:**

* 2022-2025 yılları arası tüm çıkmış sorular.
* Ders kitapları (PDF) ve örnek soru çözümleri.
* Triangulation: Akademik gerçeklik + Geçmiş sınav + Uygulama pratiği.

---

### 2. Modüler Projeler

#### **Proje 1: Çıkmış Soru Analiz Modülü**

* **Amaç:** Yapılandırılmamış sınav metinlerini stratejik verilere dönüştürmek.
* **Yöntem:** NLP tabanlı RegEx ayrıştırma + Zero-Shot Chain of Thought.
* **Model:** Llama 3 (Ollama)
* **Özellikler:**

  * JSON formatında hatasız çıktı.
  * Soru/şık ayrıştırması %100 doğrulukta.
  * Kavramsal etiketleme ve ünitelere atama.
 
<img width="1172" height="583" alt="image" src="https://github.com/user-attachments/assets/6d0e6e62-d69f-4c02-a8a0-032c08ff6015" />

<img width="1184" height="783" alt="image" src="https://github.com/user-attachments/assets/16c1287f-a3e6-4e01-a5a9-2f4a9a96b3fe" />

#### **Proje 2: RAG Tabanlı Akademik Bilgi Erişim Sistemi**

* **Amaç:** Ders kitaplarını interaktif bir kütüphaneye dönüştürmek.
* **Yöntem:** Vektör tabanlı anlamsal arama (Semantic Search).
* **Model:** Llama 3
* **Algoritma:** Cosine Similarity (Kosinüs Benzerliği)
* **Bileşenler:**

  * `PyPDFLoader`: PDF → Metin
  * `RecursiveCharacterTextSplitter`: Chunking
  * `sentence-transformers/MiniLM`: Embedding
  * `FAISS`: Hızlı arama

#### **Proje 3: Veri Güdümlü Öğrenci Koçu ve Planlayıcı**

* **Amaç:** Tüm verileri sentezleyerek öğrenciye özel strateji üretmek.
* **Yöntem:** Çok kanallı veri sentezleme ve otonom skorlama.
* **Model:** ChatOllama (Llama 3)
* **Algoritma:** Kişiselleştirilmiş Strateji Skorlaması

  ```
  Skor = (Frekans × 1.5) + (Ünite Yayılımı × 1.2) + (Tuzak Ağırlığı)
  ```
* **Bileşenler:** Dataclasses, PriorityAgent, StudyPlannerAgent
* **Çıktı:** Öğrenciye öncelikli üniteler, kritik kavramlar ve çalışma planı sunar.

---

## 🔹 Örnek Çıktılar

### Sosyal Politika – Kişisel Çalışma Stratejisi

| Sıra | Kavram          | Skor | Üniteler         | Risk       | Gerekçe                                              |
| ---- | --------------- | ---- | ---------------- | ---------- | ---------------------------------------------------- |
| 1    | Sosyal Sigorta  | 65.4 | 7 farklı ünitede | 🚨 Tuzaklı | 36 soruda sorulmuş, öğrenciler sıklıkla hata yapıyor |
| 2    | Bismarck        | 54.9 | 7 farklı ünitede | 🚨 Tuzaklı | 29 soruda sorulmuş, öğrenciler sıklıkla hata yapıyor |
| 3    | Sanayi Devrimi  | 18.0 | 5 farklı ünitede | 🚨 Tuzaklı | 6 soruda sorulmuş, öğrenciler sıklıkla hata yapıyor  |
| 4    | Sosyal Politika | 12.6 | 3 farklı ünitede | 🚨 Tuzaklı | 4 soruda sorulmuş, öğrenciler sıklıkla hata yapıyor  |

---

## 🧩 Kullanım Akışı

![ChatGPT Image 29 Ara 2025 04_45_59](https://github.com/user-attachments/assets/3490a599-17c2-40b4-a703-b00379f9ca16)

**Açıklama:**

1. **Veri Toplama:** Çıkmış sorular, ders kitapları, özetler ve tuzak kavramlar.
2. **Embedding & FAISS:** Tüm metinler sayısal vektörlere dönüştürülür ve FAISS indeksine eklenir.
3. **LLM Analizi:** ChatOllama modeli ile stratejik kavram analizi.
4. **Skorlama & Önceliklendirme:** Kavramlar frekans, ünite dağılımı ve tuzak ağırlığına göre skorlanır.
5. **Çalışma Planı:** Öğrenciye kişiselleştirilmiş, kritik kavramlara dayalı çalışma planı sunulur.

---

## ⚙️ Kurulum ve Çalıştırma

```bash
# Gerekli paketlerin kurulumu
pip install sentence-transformers faiss-cpu langchain_ollama tqdm

# Google Drive bağlantısı (Colab için)
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Örnek çalışma
from student_guidance_agent import run_student_guidance_agent

run_student_guidance_agent()
```

---

## 🏆 Sonuç ve Değer

* Öğrenciler, hangi konuları öncelikli çalışması gerektiğini görür.
* Tuzak noktaları vurgulanır.
* Sınav stratejisine dayalı rehberlik sunulur.
* Modüler ve ölçeklenebilir yapı sayesinde diğer AÖF derslerine adapte edilebilir.
* RAG yöntemi ile LLM’in halüsinasyon riski minimize edilir.

---

## 🔮 Gelecek Yaklaşım

* Multi-Agent mimarisine geçiş (Agno/Phidata framework)
* Gerçek zamanlı web dashboard (Streamlit)
* Tüm AÖF bölümlerine açılabilir bir sistem
* Kişiselleştirilmiş Öğrenme Yolları (PLP) ile eğitimde demokratikleşme

---

## 📚 Referanslar

* [LangChain](https://www.langchain.com/)
* [Sentence Transformers](https://www.sbert.net/)
* [FAISS](https://github.com/facebookresearch/faiss)
* [Ollama Llama 3](https://ollama.com/)

---
