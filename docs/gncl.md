## Açılış

PAL-MoE, class-incremental continual learning için geliştirdiğim, dinamik
büyüyebilen bir Mixture-of-Experts mimarisi.

## Problem

Sürekli öğrenmede temel sorun şu: tek bir modelin parametreleri bütün görevler
tarafından paylaşıldığında, yeni bir görev için yapılan güncelleme eski
görevlerin çözümünü bozuyor. Literatürde buna catastrophic forgetting deniyor.
Yeni görev geldiğinde model o görevi öğreniyor ama önceki görevlerdeki
performansı düşüyor.

Bu problemi çözmek için literatürde üç ana yaklaşım var. Birincisi replay:
geçmişten örnek saklayıp yeni görev eğitiminde tekrar göstermek. İkincisi
parametre düzenlileştirmesi: önemli parametreleri değiştirmemeye çalışmak.
Üçüncüsü parametre izolasyonu: her göreve ayrı parametre ayırmak. Benim
çalıştığım nokta üçüncü yaklaşımın bir varyantı, ama iki ek kısıtla birlikte.
Birincisi bellek kısıtı: replay yapacaksam bile ham veri değil, çok daha
küçük bir temsil saklamak istiyorum. İkincisi de şu soru: yeni bilgi mevcut
uzmanlardan birine mi ait, yoksa gerçekten yeni kapasite mi gerekiyor.

Yani araştırma sorusu şu hale geliyor: sınırlı bellek altında, yeni bir görev
geldiğinde mevcut kapasiteyi yeniden mi kullanmalıyım, yoksa yeni kapasite mi
açmalıyım, ve eski görevlerin yönlendirmesi bozulmadan bunu nasıl yaparım.

## Yöntem

### Genel mimari

PAL-MoE'nin genel yapısı şöyle. Önce paylaşılan bir encoder var, ham girdiyi
latent bir vektöre çeviriyor. Bu encoder MNIST'te bir autoencoder MLP, CIFAR
deneylerinde SimCLR ile eğitilmiş bir convolutional ağ, ya da dondurulmuş
ImageNet ResNet-18 ve ViT-B/16 olabiliyor. Encoder'ın çıktısı hem router'a hem
de uzmanlara gidiyor.

Router, latent vektörü alıp her uzman için bir ağırlık üretiyor. Eğitimde ve
değerlendirmede top-1 seçim kullanıyorum, yani her örnek tek bir uzmana
gidiyor. Uzmanlar birer MLP ve her birinin kendi sınıflandırma başlığı var.
Modelin çıktısı, seçilen uzmanın tahmini oluyor. Önemli bir nokta: test
zamanında görev kimliği verilmiyor. Yani model hangi görevden geldiğini
bilmeden, sadece girdiye bakarak hem sınıfı hem de doğru uzmanı bulmak
zorunda. Literatürde buna class-incremental protokol deniyor ve en zor
ayarlardan biri.

### Prototip belleği

Bellek tarafında ham görüntü yerine prototip saklıyorum. Her prototip beş
bileşen taşıyor. Birincisi v_p, latent uzaydaki prototip merkezi; bu temsil
çapası olarak kullanılıyor. İkincisi r_p, o prototip kaydedildiği andaki
router dağılımı; yani model o girdiyi hangi uzmana yönlendiriyordu, bunu
saklıyorum. Üçüncüsü o_p, uzmanın o prototipteki çıktısı; yani geçmiş
davranışın çıktı çapası. Dördüncüsü ve beşincisi x_p ve y_p, yani prototipin
latent örneği ve etiketi. Ayrıca her prototipin bir sahibi var: o görevi
öğrenen uzmanın kimliği.

Prototiplerin sayısı sınırlı, örneğin bin. Yeni prototip geldiğinde yakın olan
prototipler birleştiriliyor, ama yalnızca aynı sahipteki prototipler
birleştirilebiliyor. Bunu özellikle yaptım, çünkü farklı görevlerin
prototipleri birleşirse router distillation yanlış hedef öğreniyor ve ölçtüğüm
kadarıyla iki puana kadar kayıp veriyor.

Bu belleği replay buffer olarak değil, fonksiyon uzayında bir çapa olarak
kullanıyorum. Yani amaç sadece eski örnekleri tekrar göstermek değil, eski
model davranışını yeni modelde korumak.

### Görev döngüsü: genişleme kararı

Yeni bir görev geldiğinde ilk adım, kapasite gerekip gerekmediğine karar
vermek. Bunun için bir trigger kullanıyorum. İki seçenek var: gözetimli bir
bileşik skor, ya da etiketsiz enerji tabanlı yenilik skoru. Deneylerde ayrıca
her göreve bir uzman verilen sabit genişleme protokolünü de karşılaştırma
amacıyla kullandım.

Genişleme kararı verildiğinde yeni uzman rastgele başlatılmıyor. Mevcut en
uygun uzmandan fonksiyon koruyan bir kopya alınıyor. Yani yeni uzman başlangıçta
ebeveyniyle tamamen aynı fonksiyonu hesaplıyor; ardından yeni görev verisiyle
eğitiliyor. Bunun amacı, yeni kapasitenin başlangıçta eski davranışı
bozmaması.

Eğitilen aday uzman bir validation gate'ten geçiyor. Kapı üç şeye bakıyor:
yeni görevdeki doğruluk, prototip kayması, ve eski prototiplerdeki doğruluk
düşüşü. Eğer aday bu kontrolleri geçemezse genişleme reddediliyor ve görev en
yeni uzmanla devam ediyor. Bu mekanizmanın ölçülebilir bir katkısını
bulamadım.

### Eğitim kayıpları

Yeni uzman eğitilirken toplam kayıp birkaç terimden oluşuyor. Birincisi
görev kaybı, yani yeni veride cross-entropy. İkincisi router stabilite kaybı:
saklanan r_p dağılımı ile router'ın şu anki dağılımı arasında KL diverjansı.
Üçüncüsü uzman stabilite kaybı: saklanan o_p çıktıları ile mevcut uzmanların
aynı prototiplerdeki çıktıları arasında MSE. Dördüncüsü negatif sınır terimi:
yeni uzmanın eski prototiplerde kararsız, yani yüksek entropili tahminler
üretmesini istiyorum; böylece "bu benim görevim değil" sinyali öğreniyor.
Beşincisi isteğe bağlı encoder stabilite kaybı; encoder eğitilebilirse EMA
(Exponential Moving Average) kopyasına bağlanıyor.

Burada bir isimlendirme notu düşeyim: negatif sınır terimi klasik anlamda bir
OOD detection değil. Doğru ifade, tarihsel latent bölgeler üzerinde bir
düzensizleştirme.

### Görev sonu

Görev eğitimi bittiğinde sırasıyla şunlar yapılıyor. Önce görev verisinden
prototipler kaydediliyor. Sonra isteğe bağlı olarak kısa bir ortak kalibrasyon
yapılıyor: bütün uzmanlar saklanan latent örnekler üzerinde birkaç epoch
birlikte ince ayar yapıyor, ardından geçmiş yine kilitleniyor. Sonra router
distillation yapılıyor: router, prototiplerin sahiplerini tahmin etmek üzere
eğitiliyor. Bu adım tamamen saklanan latentlerle yapılıyor, ham veri
kullanmıyor, üç yüz adım sürüyor. Son olarak geçmiş uzmanlar ve geçmiş routing
satırları kilitleniyor.

Kilidin tam olması için bir düzeltme yapmam gerekti. Adam optimizer'ının
weight-decay terimi, gradyanı sıfırlanmış kilitli satırları yine de sessizce
sıfıra çekiyordu; çünkü weight-decay güncellemenin içinde ekleniyor. Router
parametrelerini sıfır weight-decay grubuna aldım; artık kilit matematiksel
olarak tam. Bu düzeltme öncesindeki bazı sayılar bu yüzden farklıydı.

### Çıkarım

Çıkarımda router top-1 uzmanı seçiyor. İsteğe bağlı olarak prototip tabanlı
bir çıkarım yönlendirmesi de var: girdi saklanan bir prototipe yeterince
yakınsa, o prototipin geçmiş yönlendirmesini devralıyor. Güven ağırlığı eşik
gerektirmiyor, en yakın iki prototip arasındaki mesafe oranından geliyor.
Deneylerde bu mekanizmanın katkısı nötr çıktı.

### Bellek muhasebesi

Son olarak, bütün sonuç dosyaları bellek miktarını byte cinsinden kaydediyor.
Saklanan veri, saklanan durum ve bunların toplamı ayrı ayrı raporlanıyor.
Bunu özellikle yaptım, çünkü "aynı buffer boyutu" demek "aynı bellek" demek
değil: DER++ logit de saklıyor, iCaRL sınıf başına örnek saklıyor, benim
prototipim ise temsil, yönlendirme ve çıktı çapalarını birlikte saklıyor.

## Deneysel kurulum

Deneylerde dört benchmark kullandım: Split-MNIST beş görev ikişer sınıf,
Split-CIFAR-10 beş görev ikişer sınıf, Split-CIFAR-100 yirmi görev beşer sınıf
ve Tiny-ImageNet yirmi görev onar sınıf. Encoder olarak MNIST'te dondurulmuş
bir autoencoder, CIFAR'da elli epoch SimCLR ile eğitilmiş bir convolutional
ağ, ayrıca dondurulmuş ImageNet ResNet-18 ve ViT-B/16 kullandım.

Burada bir protokol ayrımını baştan söyleyeyim. CIFAR-10 convolutional
deneyini raw pipelinenda koştum; yani replay tabanı gerçekten ham görüntü
saklıyor. CIFAR-100 ve ViT deneylerinde ise encoder çıktılarını önceden
hesaplayan feature cache kullanıyorum; bu durumda bütün yöntemler özellik
vektörü sakladığı için öğe başına bellek farkı ortadan kalkıyor. Bunun
sonuçlara etkisini eşit-byte bölümünde ayrıca gösteriyorum.

Metrik olarak ortalama doğruluk, unutma, geriye transfer, router sahiplik
doğruluğu, yönlendirme korunumu, uzman sayısı, bellek ve gecikme raporluyorum.
Unutma metriği sıfırda kırpılıyor, yani pozitif geriye transferi sıfır olarak
gösteriyor; bunu bilerek muhafazakar seçtim. Seed sayısı CIFAR'da üç, MNIST'te
beş. Rakipler: naive fine-tuning, EWC, Experience Replay (ER), DER++, ER-ACE, AGEM,
iCaRL, MIR, sabit uzmanlı Standard MoE ve single-head Latent Replay.

## Sonuçlar

Aşağıdaki tablolarda bellek sütunu saklanan veri miktarıdır (model
parametreleri hariç); "yer" sütunu ilgili tablonun protokolünü belirtir.

### MNIST (5 seed)

| Method | Avg Acc | Forgetting | Memory |
| :-- | --: | --: | --: |
| DER++ (P=250) | 87.65 ± 0.94 | 6.47 | 777 KB |
| MIR (P=250) | 83.30 ± 1.95 | 15.27 | 768 KB |
| Latent Replay (P=250) | 82.74 ± 0.64 | 16.88 | **127 KB** |
| Experience Replay (ER, P=250) | 82.41 ± 1.30 | 17.42 | 768 KB |
| PAL-MoE hybrid | 80.06 ± 1.11 | **5.56** | 1061 KB |
| PAL-MoE pure | 79.29 ± 1.20 | 8.02 | **285 KB** |
| iCaRL (k=25) | 59.15 ± 2.13 | 10.68 | 771 KB |
| ER-ACE | 55.05 ± 7.97 | 51.20 | 768 KB |
| AGEM | 31.36 ± 5.35 | 83.60 | 768 KB |
| EWC | 19.36 ± 0.02 | 98.70 | - |
| Naive | 19.28 ± 0.07 | 98.66 | - |
| Standard MoE | 16.53 ± 2.65 | 95.56 | - |

PAL-MoE pure 285 kilobaytla yüzde 79.29 doğruluk ve yüzde 8.02 unutma elde
ediyor. Tablodaki en net sonuç Latent Replay satırı: 250 örnekle sadece 127
kilobayt bellekte yüzde 82.74 doğruluk ve yüzde 16.88 unutma, yani deneyim
replayinin altıda biri bellekle aynı doğruluk. Bu, depolama formatının
etkisini gösteren en net tek sonuç.

### CIFAR-10, convolutional encoder (raw pipeline, 3 seed)

| Method | Avg Acc | Forgetting | Memory |
| :-- | --: | --: | --: |
| PAL-MoE hybrid | 39.00 ± 0.09 | 22.85 | 14.47 MB |
| PAL-MoE pure | **37.81 ± 0.57** | **22.03** | **2.18 MB** |
| DER++ (P=250) | 31.72 ± 0.62 | 61.18 | 3.08 MB |
| iCaRL (k=25) | 25.35 ± 1.33 | 15.26 | 3.08 MB |
| Experience Replay (ER, P=250) | 25.14 ± 0.23 | 73.00 | 3.07 MB |

Bu deney raw pipelinenda koştu: replay tabanı gerçekten ham görüntü
saklıyor, benim yöntemim latent. PAL-MoE pure en iyi baseline'ı 6.1 puan geçiyor
ve daha az bellek harcıyor (2.18 MB'a karşı 3.08 MB).

### CIFAR-100, convolutional encoder (feature cache, 3 seed)

| Method | Avg Acc | Forgetting | Memory |
| :-- | --: | --: | --: |
| iCaRL (k=25) | **10.02 ± 0.17** | **11.13** | 2.66 MB |
| PAL-MoE pure | 9.50 ± 0.42 | 23.24 | 4.13 MB |
| PAL-MoE + Latent Replay | 9.47 ± 0.89 | 13.84 | 4.16 MB |
| DER++ (P=250) | 5.95 ± 0.31 | 63.19 | 0.36 MB |

Bu benchmark'ta iCaRL hem daha doğru hem daha az bellek kullanıyor ve bunu
saklamıyorum. Bizim avantajımız unutmanın DER++'a göre çok daha düşük olması.

### CIFAR-100, ResNet-18 omurga (feature cache, 3 seed)

| Method | Avg Acc | Forgetting | Memory |
| :-- | --: | --: | --: |
| PAL-MoE pure | **15.65 ± 0.39** | 31.69 | 4.18 MB |
| iCaRL (k=25) | 13.97 ± 0.87 | **10.96** | 2.66 MB |
| DER++ (P=250) | 13.12 ± 0.08 | 66.91 | 0.36 MB |
| Experience Replay (ER, P=250) | 11.06 ± 0.22 | 66.72 | 0.26 MB |

Omurga değişimi tüm yöntemlerin mutlak seviyesini yaklaşık iki katına
çıkarıyor; yani temsil kalitesi en büyük tek kaldıraç.

### ViT-B/16 (feature cache, 3 seed)

CIFAR-10:

| Method | Avg Acc | Forgetting | Memory |
| :-- | --: | --: | --: |
| PAL-MoE pure | **91.74 ± 0.28** | **5.61** | 6.37 MB |
| PAL-MoE + Latent Replay | 91.74 ± 0.33 | 5.55 | 6.37 MB |
| iCaRL (k=25) | 84.18 | 10.38 | 0.80 MB |
| Experience Replay (ER, P=250) | 82.89 ± 0.10 | 20.29 | 0.77 MB |
| DER++ (P=250) | 73.28 ± 0.31 | 32.10 | 0.78 MB |

CIFAR-100:

| Method | Avg Acc | Forgetting | Memory |
| :-- | --: | --: | --: |
| iCaRL (k=25) | **64.94** | **12.25** | 7.99 MB |
| PAL-MoE pure | 59.34 ± 0.32 | 18.17 | 14.23 MB |
| DER++ (P=250) | 50.97 ± 0.15 | 47.30 | 0.87 MB |
| Experience Replay (ER, P=250) | 34.06 ± 0.90 | 66.56 | 0.77 MB |

CIFAR-10'da en yüksek mutlak sonuçlar burada: yüzde 91.74 doğruluk ve yüzde
5.61 unutma. Ama CIFAR-100 ViT'te iCaRL hem daha doğru hem daha az bellek
kullanıyor. Bu iki sonuç birlikte metodun her koşulda üstün olmadığını
gösteriyor.

### Tiny-ImageNet (20 görev, 10 sınıf; feature cache, 3 seed)

| Method | Avg Acc | Forgetting | Memory |
| :-- | --: | --: | --: |
| iCaRL (k=25) | **14.06** | **10.28** | 10.65 MB |
| PAL-MoE + Latent Replay | 12.70 ± 0.81 | 43.02 | 2.00 MB |
| PAL-MoE pure | 12.13 ± 0.57 | 40.62 | 2.04 MB |
| DER++ (P=250) | 11.66 ± 0.38 | 62.36 | 0.71 MB |
| MIR (P=250) | 7.90 | 65.33 | 0.51 MB |
| Experience Replay (ER, P=250) | 7.67 | 65.72 | 0.51 MB |
| EWC | 3.64 | 72.31 | - |
| Naive | 3.54 | 71.73 | - |

Replay ailesi içinde PAL-MoE açık ara önde: Experience Replay (ER)ne göre 4.5 puan
daha doğru ve unutması 65.7'ye karşı 40.6. iCaRL doğruluk ve unutmada önde
ama 200 sınıfta 5000 ham örnek saklıyor: 10.65 MB'a karşı PAL-MoE 2.04 MB,
yani 5.2 kat bellek. Equal-byte'ta bu tablonun değişmesini bekliyorum; o koşu
sırada.

### Equal-byte karşılaştırması

Şimdiye kadarki tablolar aynı öğe sayısıyla yapılan karşılaştırmalardı. Asıl
adil karşılaştırma aynı byte bütçesiyle yapılanı; bunu iki protokolde ölçtüm.

Raw pipeline, CIFAR-10, 1 MiB, 3 seed. Replay tabanı ham görüntü
saklıyor (örnek başına 12.296 byte), prototip ise 2.184 byte; yani eşit
byte'ta yaklaşık 5.6 kat daha fazla öğe saklayabiliyorum:

| Method | Avg Acc | Forgetting | Realised bytes |
| :-- | --: | --: | --: |
| Latent replay (1016 öğe) | **50.36 ± 0.64** | 41.27 | 1023 KB |
| PAL-MoE pure (480 prototip) | 46.16 ± 1.11 | **21.66** | 1024 KB |
| PAL-MoE hybrid | 39.93 ± 0.56 | 32.68 | 1018 KB |
| DER++ (85 öğe) | 36.81 ± 0.32 | 61.42 | 1024 KB |
| Experience Replay (ER) (85 öğe) | 30.17 ± 1.30 | 72.77 | 1021 KB |
| iCaRL (k=8) | 26.27 ± 2.71 | 22.51 | 970 KB |

PAL-MoE ham replay tabanını 16 puan geçiyor ve unutması 3.4 kat düşük.
4 megabaytta sıralama benzer: Latent Replay 55.39, PAL-MoE 49.19 (unutma
22.06), DER++ 46.60, Experience Replay (ER) 43.40. CIFAR-100'de 1 megabaytta latent
replay 13.74 (unutma 61.18), PAL-MoE 13.70 (unutma 31.79), DER++ 9.00,
Experience Replay (ER) 8.40.

Reservoir sampling ile aynı hücreler: Experience Replay (ER) 33.18 ± 1.11 (unutma
68.04), DER++ 37.79 ± 1.17 (unutma 60.38). Sıralama değişmiyor, yani bu
sonuç recency buffer varsayılanından kaynaklanmıyor.

Feature-cache protokolü, CIFAR-10, 3 seed. Burada bütün yöntemler özellik
vektörü saklıyor, dolayısıyla öğe başına byte farkı kalmıyor:

| Budget | Method | Avg Acc | Forgetting |
| :-- | :-- | --: | --: |
| 1 MiB | Experience Replay (ER) | **49.86 ± 0.66** | 41.43 |
| 1 MiB | DER++ | 48.63 ± 0.66 | 32.53 |
| 1 MiB | PAL-MoE pure | 46.32 ± 0.54 | **21.37** |
| 1 MiB | iCaRL | 35.96 ± 4.39 | 28.52 |
| 4 MiB | Experience Replay (ER) | **54.95 ± 0.26** | 33.60 |
| 4 MiB | PAL-MoE pure | 50.54 ± 1.19 | **19.15** |
| 4 MiB | DER++ | 48.24 ± 0.09 | 30.83 |
| 4 MiB | iCaRL | 41.44 ± 3.45 | 28.74 |

CIFAR-100 feature-cache, 1 MiB, 3 seed: DER++ 16.97 ± 0.63, Experience Replay (ER)
13.64 ± 0.20, PAL-MoE 12.15 ± 0.56 (unutma 34.21), iCaRL 10.72 ± 0.28 (unutma
9.93).

İki tabloyu birlikte sunmamın sebebi şu: iddia "her yerde daha iyi" değil.
İddia, depolama formatına bağlı bir denge. Yayınlanan öğe-bütçesi tabloları
benim yöntemimi kayırıyordu; onları eşit-byte sonuçlarıyla birlikte
raporluyorum.

### Mekanizma ayrıştırması (CIFAR-10 ResNet-18, 3 seed)

| Variant | Avg Acc | Forgetting |
| :-- | --: | --: |
| Naive | 17.71 | 88.95 |
| Experience Replay (ER, P=250) | 38.58 | 60.85 |
| Single-head Latent Replay (P=250) | 38.45 | 60.80 |
| Static MoE (sadece genişleme) | 24.44 | 81.10 |
| + prototype anchoring ve router distillation | **49.87** | **19.84** |
| + gate | 49.45 | 20.63 |
| + OOD terimi (tam reçete) | 48.88 | 20.42 |

Asıl katkı prototype anchoring: Static MoE'ye göre 25 puan doğruluk artışı ve
61 puan unutma azalması. Kapı ve OOD terimi mevcut reçetede nötr; bu, eski
reçetedeki OOD katkısını da revize ediyor. Single-head Latent Replay eşit öğe
sayısında Experience Replay (ER) ile başa baş.

Gerçek hybrid sonucu (raw pipeline, 1000 prototip, 3 seed):

| Benchmark | Method | Avg Acc | Forgetting | Memory |
| :-- | :-- | --: | --: | --: |
| CIFAR-10 | PAL-MoE pure | 48.39 ± 1.55 | 20.82 | **2.18 MB** |
| CIFAR-10 | PAL-MoE hybrid | 49.63 ± 1.19 | 18.03 | 14.47 MB |
| CIFAR-100 | PAL-MoE pure | 15.97 ± 0.31 | 30.65 | **4.18 MB** |
| CIFAR-100 | PAL-MoE hybrid | 16.90 ± 0.19 | 19.05 | 16.47 MB |

Hybrid yaklaşık 1.2 puan kazandırıyor ama 6.6 kat bellek harcıyor. Eşit
byte'ta kaybediyor (1 MiB: 39.93'a karşı 46.16), yani hybrid'i ana sonuç
olarak değil, bellek esnekliği olarak sunmak gerekiyor.

MIR baseline'ı da Experience Replay (ER)ni geçemedi: CIFAR-10'da 37.62 ± 1.04
(unutma 61.79), Experience Replay (ER) 39.92 (unutma 58.87); CIFAR-100'de 10.50
(unutma 66.16), Experience Replay (ER) 10.93 (unutma 66.91).

### Kapasite ve maliyet (CIFAR-100 ResNet-18, 3 seed)

| max_experts | Avg Acc | Forgetting | Params |
| --: | --: | --: | --: |
| 2 | 14.80 | 42.24 | 0.55M |
| 4 | 11.13 | 32.09 | 1.10M |
| 6 | 15.88 | 30.30 | 1.65M |
| 20 | 14.25 | **14.84** | 5.49M |
| Parameter-matched ER (2.46M) | 10.22 | 74.65 | 2.46M |
| Parameter-matched DER++ (2.46M) | 13.49 | 68.41 | 2.46M |

Doğruluk uzman sayısından neredeyse bağımsız; unutma uzman sayısıyla düşüyor.
Yani "daha fazla uzman doğruluğu artırmıyor ama unutmayı azaltıyor". PAL-MoE
1.65 milyon parametreyle 2.46 milyon parametreli single-head rakiplerini geçiyor
ve unutması yarısından az.

### Anchor refresh (CIFAR-100 ResNet-18, 3 seed)

| Cell | Avg Acc | Forgetting |
| :-- | --: | --: |
| Pure, refresh kapalı | 15.65 ± 0.39 | 31.69 |
| Pure, refresh açık | **18.31 ± 1.03** | **18.82** |
| Latent Replay, kapalı | 17.68 ± 0.97 | 17.47 |
| Latent Replay, açık | **21.13 ± 0.34** | **16.09** |

Kalibrasyon sonrası anchor refresh belirgin kazanç sağlıyor (+2.7 puan
doğruluk, -12.9 puan unutma) ve final reçetede varsayılan olmalı. Çıkarım
prototype anchoring (alpha=0.5) yaklaşık nötr.

### Diğer sonuçlar

| Deney | Sonuç |
| :-- | :-- |
| Expert growth, gated karşı forced | 14.25 ± 0.22 vs 14.34 ± 0.41; ikisi de 20/20 uzman; yani kapı yeni uzman gerekip gerekmediğine karar veremiyor. Bu bir sınırlama. |
| Routing retention (RR_t) | Akış boyunca 0.78 ile 0.85 arası |
| Router generalization gap | Prototiplerde sahiplik doğruluğu yüzde 94.5, test girdilerinde en yüksek uzman payı 0.15 ile 0.24 |
| Domain shift (MNIST döndürme, 5 seed) | 85.38 ± 0.47 doğruluk, 6.34 unutma |
| Trainable encoder | Pure varyant 37.81'den 9.97'ye düşüyor; yöntem dondurulmuş temsile bağımlı |
| Latency (batch 1) | ResNet-18: tek kafa 1.09 ms, PAL-MoE 1.71 ms (uzman sayısından bağımsız); ViT: 8.43 ms vs 8.57-8.82 ms |
| Buffer politikası | Feature-cache P=250'de reservoir sampling tabanlara yaklaşık 4 puan doğruluk kazandırıyor: ER 43.23 (recency 39.26), DER++ 48.25 (recency 44.37). Raw equal-byte'ta etki küçük. |

## Sınırlamalar

Birincisi, equal-byte sonucu protokole bağlı. Raw pipelinenda yöntemim önde,
feature-cache hattında doğrulukta replay tabanları önde. Bu yüzden iddiayı
"her yerde daha iyi" diye kurmuyorum; denge olarak kuruyorum.

İkincisi, dinamik allocation hipotezim bu benchmark'ta çürütüldü. Kapı her
genişlemeyi kabul etti, uzman paylaşımı olmadı. Bu yüzden "dinamik expert
allocation" yerine "sabit bütçeli kapasite genişlemesi" ifadesini
kullanıyorum.

Üçüncüsü, negatif sınır terimi mevcut reçetede nötr. Eski reçetedeki katkısı,
router distillation eklenince anlamını yitiriyor.

Dördüncüsü, yöntem dondurulmuş veya güçlü bir temsile bağımlı. Encoder
eğitilebilir olduğunda sonuç çöküyor.

Beşincisi, iCaRL bazı benchmarklarda hem daha doğru hem daha az bellek
kullanıyor. Bunu saklamıyorum.

Altıncısı, router'ın prototiplerden test girdilerine genellemesi zayıf.

Yedincisi, CORe50 gibi daha zorlu bir domain-incremental benchmark henüz yok;
şimdilik MNIST döndürme pilotuyla sınırlı. Prompt tabanlı rehearsal-free
rakipler de kapsam dışı.

Sekizincisi, Tiny-ImageNet equal-byte hücreleri ve reservoir ekinin kalan
seed'leri henüz koşuyor; sonuçlar geldikçe tablolara eklenecek.

## Sonuç ve sonraki adımlar

Özetleyeyim. PAL-MoE, sınırlı bellek altında unutmayı azaltmak için
prototip tabanlı fonksiyon uzayı anchor'larını kullanan, dinamik büyüyebilen bir
Mixture-of-Experts mimarisi. En güçlü kanıtlar şunlar:

- Raw pipelinenda equal-byte'ta replay tabanını 16 puan geçmesi ve unutmanın
  3.4 kat düşük olması.
- CIFAR-10'da en iyi baseline'ı daha az bellekle geçmesi.
- ViT ile CIFAR-10'da yüzde 91.74 doğruluk ve yüzde 5.61 unutma.
- Ablation'da asıl katkının prototype anchoring olduğunun gösterilmesi.
- Tiny-ImageNet'te replay ailesi içinde en iyi doğruluk ve en düşük unutma.

Sonraki adımlar: Tiny-ImageNet equal-byte hücrelerini koşmak, reservoir ile
equal-byte tablosunu tamamlamak, anchor refresh'i varsayılan yapıp başlık
tablolarını yeniden koşmak, router genelleme boşluğunu ayrıştırmak, mümkünse
CORe50 eklemek ve prompt tabanlı rakipler için kapsam kararı vermek.

Kısaca: sonuçlar umut verici, ama iddia "catastrophic forgetting çözüldü"
değil. İddia, belirli bellek bütçelerinde unutma ve bellek dengesinin
iyileştirildiği. Sorularınızı alayım.
