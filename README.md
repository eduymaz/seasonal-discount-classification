# 📌 Mevsimsel İndirim Etkisi Sınıflandırıcı API 

## 1. 📝 Tanım  

Bu proje, şirketimizde satış verileri üzerinden daha etkili stratejiler geliştirmek ve veri odaklı karar alma süreçlerini güçlendirmek amacıyla yürütülen çok disiplinli bir modelleme sürecinin bir parçası olarak geliştirilmiştir.

Geliştirilen modeller; müşteri alışveriş davranışlarını analiz etmek, bu davranışları stratejik olarak yönlendirmek ve satış performansını öngörebilmek için tasarlanmıştır. Nihai hedef, satış performansını artırmak, müşteri memnuniyetini yükseltmek ve operasyonel süreçleri optimize etmektir.

Bu kapsamda, satış verileri üzerinde eğitilmiş bir Karar Ağacı (Decision Tree) sınıflandırma modeli ile, belirli bir dönemde uygulanan indirimlerin mevsimsel olarak etkili olup olmadığını tahmin eden bir çözüm geliştirilmiştir. Model, FastAPI tabanlı bir servis olarak yapılandırılmıştır ve bu sayede dış sistemler ile kolaylıkla entegre edilebilen, gerçek zamanlı tahmin sunabilen bir makine öğrenimi hizmeti haline getirilmiştir.

* Sunum için [tıklayınız.](https://drive.google.com/drive/folders/1Wvmwc_rJMGqqqV1RH_OTmaPqiEq2CtjG)

## 2. 📂 Proje Yapısı

Proje, satış verileri üzerinden mevsimsel indirimlerin etkinliğini tahmin etmek üzere oluşturulmuş, veri üretiminden modelleme ve API servisleştirme süreçlerine kadar tüm adımları kapsayan uçtan uca bir makine öğrenimi uygulamasıdır. Aşağıda proje dosyalarının açıklamaları yer almaktadır:

### 2.1. 🔍 Açıklayıcı Dosyalar

`projectQuestion.ipynb`
 → Proje kapsamında ele alınan sorunun detaylı tanımı ve çözüm yaklaşımına dair genel bakış.

`project_tutorial.ipynb`
 → Veri işleme, model oluşturma ve değerlendirme süreçlerinin adım adım açıklandığı, açıklamalı Jupyter defteri.

`Tutorial.pdf`
 → Proje genelinde izlenen yaklaşımı, kullanılan teknikleri ve hedefleri özetleyen dökümantasyon dosyası.

### 2.2. ⚙️ Uygulama Adımları

`step1_makedata.py`
 → Simülasyon veya örneklem yoluyla veri seti oluşturma sürecini yürütür.

`step2_preprocess.py`
 → Verinin temizlenmesi, dönüştürülmesi ve modele hazır hale getirilmesini sağlar.

`step3_bestmodelprediction.py`
 → Farklı modellerin denenmesi, performanslarının karşılaştırılması ve en uygun modelin seçilip eğitilmesini içerir.

`step4_api_main.py`
 → Seçilen modelin FastAPI ile servis haline getirilmesini sağlayan API kodlarını içerir.

## 3.  📂 Klasörler
 
- `data/`
 → Ham veriler ve işlem görmüş veri setleri bu klasörde yer almaktadır.

- `results/`
 → Model çıktıları, görseller (grafikler, matrisler), eğitim sonuçları ve .pkl formatındaki model dosyaları bu klasörde saklanmaktadır.

 ## 4. 🧪 Kurulum
 
 ```markdown
 # 1. Repository'yi klonlayınız:

git clone https://github.com/eduymaz/seasonal-discount-classification.git
cd seasonal-discount-classification

# 2. Sanal ortam oluşturma (opsiyonel ama önerilir)
python -m venv my_environment
source my_environment/bin/activate  
# Windows için: my_environment\Scripts\activate

# 3. Gereken paketleri yükleyin
pip install -r requirements.txt
```

## 5. 🚀 Projenin Çalıştırılması (Usage)

1. Veri Oluşturma
```markdown
python step1_makedata.py
```
2. Veri Ön İşleme
```markdown
python step2_preprocess.py
```
3. Model Eğitimi ve Tahmin
```markdown
python step3_bestmodelprediction.py
```
4. API'yi Başlatma
```markdown
uvicorn step4_api_main:app --reload
```
