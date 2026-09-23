## Açılış

PAL-MoE, class-incremental continual learning için geliştirdiğim, dinamik
büyüyebilen bir Mixture-of-Experts mimarisi. Konuşmada önce problemi ve
yöntemin nasıl çalıştığını anlatacağım, sonra deneysel kurulumu ve sonuçları
paylaşacağım, en sonda da sonuçların gösterdiği sınırlamaları ve sonraki
adımları açıkça söyleyeceğim.

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
bulamadım; buna sonuçlarda ayrıca değineceğim.

### Eğitim kayıpları

Yeni uzman eğitilirken toplam kayıp birkaç terimden oluşuyor. Birincisi
görev kaybı, yani yeni veride cross-entropy. İkincisi router stabilite kaybı:
saklanan r_p dağılımı ile router'ın şu anki dağılımı arasında KL diverjansı.
Üçüncüsü uzman stabilite kaybı: saklanan o_p çıktıları ile mevcut uzmanların
aynı prototiplerdeki çıktıları arasında MSE. Dördüncüsü negatif sınır terimi:
yeni uzmanın eski prototiplerde kararsız, yani yüksek entropili tahminler
üretmesini istiyorum; böylece "bu benim görevim değil" sinyali öğreniyor.
Beşincisi isteğe bağlı encoder stabilite kaybı; encoder eğitilebilirse EMA
kopyasına bağlanıyor.

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
olarak tam. Bu düzeltme öncesindeki bazı sayılar bu yüzden farklıydı ve
literatüre bu şekilde raporlanmamalı.

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

Deneylerde dört benchmark kullandım. Split-MNIST beş görev, ikişer sınıf.
Split-CIFAR-10 beş görev, ikişer sınıf. Split-CIFAR-100 yirmi görev, beşer
sınıf. Ve Tiny-ImageNet, yirmi görev onar sınıf; bu koşu şu anda devam ediyor.
Encoder olarak MNIST'te dondurulmuş bir autoencoder, CIFAR'da elli epoch
SimCLR ile eğitilmiş bir convolutional ağ, ayrıca dondurulmuş ImageNet
ResNet-18 ve ViT-B/16 kullandım.

Burada bir protokol ayrımını baştan söyleyeyim. CIFAR-10 convolutional
deneyini ham veri hattında koştum; yani replay tabanı gerçekten ham görüntü
saklıyor. CIFAR-100 ve ViT deneylerinde ise encoder çıktılarını önceden
hesaplayan feature cache kullanıyorum; bu durumda bütün yöntemler özellik
vektörü sakladığı için öğe başına bellek farkı ortadan kalkıyor. Bunun
sonuçlara etkisini eşit-byte bölümünde ayrıca anlatacağım.

Metrik olarak ortalama doğruluk, unutma, geriye transfer, router sahiplik
doğruluğu, yönlendirme korunumu, uzman sayısı, bellek ve gecikme raporluyorum.
Unutma metriği sıfırda kırpılıyor, yani pozitif geriye transferi sıfır olarak
gösteriyor; bunu bilerek muhafazakar seçtim. Seed sayısı CIFAR'da üç, MNIST'te
beş. Rakipler: naive fine-tuning, EWC, deneyim replayi, DER++, ER-ACE, AGEM,
iCaRL, MIR, sabit uzmanlı standart MoE ve tek kafalı latent replay.

## Sonuçlar

### MNIST

Önce MNIST. Burada beş seed ortalamasıyla, saf PAL-MoE yüzde 79.29 doğruluk ve
yüzde 8.02 unutma elde ediyor, belleği 285 kilobayt. Hybrid varyant yüzde
80.06 doğruluk ve yüzde 5.56 unutma ile en düşük unutmaya sahip. DER++ yüzde
87.65 ile en yüksek doğruluk, unutması yüzde 6.47, belleği 777 kilobayt.
Deneyim replayi yüzde 82.41 doğruluk ve yüzde 17.42 unutma ile 768 kilobayt
kullanıyor. Bir de tek kafalı latent replay sonucu var: 250 örnekle sadece 127
kilobayt bellekte yüzde 82.74 doğruluk ve yüzde 16.88 unutma elde ediyor.
Yani latent replay, replay tabanının altıda biri bellekle aynı doğruluğu
veriyor. Bu, depolama formatının etkisini gösteren en net tek sonuç.

### CIFAR-10, convolutional encoder

CIFAR-10'da beş görev, üç seed ile şu sonucu aldım. Saf PAL-MoE yüzde 37.81
doğruluk ve yüzde 22.03 unutma, belleği 2.18 megabayt. DER++ yüzde 31.72
doğruluk ve yüzde 61.18 unutma ile 3.08 megabayt kullanıyor. iCaRL yüzde 25.35
doğruluk ve yüzde 15.26 unutma. Yani burada metodum en iyi baseline'ı 6.1 puan
geçiyor ve daha az bellek harcıyor. Bu deney ham veri hattında koştu, yani
replay tabanı gerçekten ham görüntü saklıyor, benim yöntemim ise latent.

### CIFAR-100, convolutional encoder

CIFAR-100'de durum farklı. Yirmi görevde iCaRL yüzde 10.02 doğruluk ve yüzde
11.13 unutma ile 2.66 megabayt kullanıyor. Saf PAL-MoE yüzde 9.50 doğruluk ve
yüzde 23.24 unutma ile 4.13 megabayt. DER++ yüzde 5.95 doğruluk ve yüzde 63.19
unutma. Bu benchmark'ta iCaRL hem daha doğru hem daha az bellek kullanıyor ve
bunu saklamıyorum. Bizim avantajımız unutmanın DER++'a göre çok daha düşük
olması.

### CIFAR-100, ResNet-18 omurga

Aynı deneyi dondurulmuş ImageNet ResNet-18 ile tekrarladım. Saf PAL-MoE yüzde
15.65 doğruluk, yüzde 31.69 unutma, 4.18 megabayt. iCaRL yüzde 13.97 ve yüzde
10.96 unutma. DER++ yüzde 13.12 doğruluk ama yüzde 66.91 unutma. Burada omurga
değişimi tüm yöntemlerin mutlak seviyesini yaklaşık iki katına çıkarıyor; yani
temsil kalitesi en büyük tek kaldıraç.

### ViT-B/16

En yüksek mutlak sonuçlar ViT-B/16 ile geldi. CIFAR-10'da saf PAL-MoE yüzde
91.74 doğruluk ve yüzde 5.61 unutma. iCaRL yüzde 84.18 doğruluk ve yüzde 10.38
unutma, deneyim replayi yüzde 82.89 ve yüzde 20.29. Ama CIFAR-100 ViT'te iCaRL
yüzde 64.94 ile bizim yüzde 59.34'ümüzün önünde, üstelik daha az bellek
kullanıyor. Bu iki sonuç birlikte metodun her koşulda üstün olmadığını
gösteriyor.

### Eşit byte karşılaştırması

Şimdiye kadar gösterdiğim tablolar aynı öğe sayısıyla yapılmış
karşılaştırmalardı. Ama asıl adil karşılaştırma aynı byte bütçesiyle yapılanı.
Bunu iki ayrı protokolde ölçtüm.

Birincisi ham veri hattı. Burada replay tabanı ham görüntü saklıyor: her örnek
12 bin 296 byte. Benim prototipim ise 2 bin 184 byte. Yani eşit byte'ta ben
yaklaşık 5.6 kat daha fazla öğe saklayabiliyorum. CIFAR-10'da 1 megabayt
bütçeyle, üç seed ortalaması: tek kafalı latent replay yüzde 50.36 doğruluk ve
yüzde 41.27 unutma; saf PAL-MoE yüzde 46.16 doğruluk ve yüzde 21.66 unutma;
hybrid yüzde 39.93; DER++ yüzde 36.81 doğruluk ve yüzde 61.42 unutma; deneyim
replayi yüzde 30.17 doğruluk ve yüzde 72.77 unutma; iCaRL yüzde 26.27. Yani bu
protokolde yöntemim ham replay tabanını 16 puan geçiyor ve unutması 3.4 kat
düşük. 4 megabaytta da sıralama benzer: latent replay yüzde 55.39, PAL-MoE
yüzde 49.19 ve unutma yüzde 22.06, DER++ yüzde 46.60, deneyim replayi yüzde
43.40. CIFAR-100'de 1 megabaytta latent replay yüzde 13.74 ve PAL-MoE yüzde
13.70, ama unutma PAL'da yüzde 31.79, latent replay'de yüzde 61.18.

İkincisi feature-cache protokolü. Burada encoder'ın çıktıları önceden
hesaplanıp saklanıyor, yani bütün yöntemler aslında özellik vektörü saklıyor.
Dolayısıyla öğe başına byte farkı ortadan kalkıyor. Bu protokolde CIFAR-10'da
1 megabaytta deneyim replayi yüzde 49.86 ile doğrulukta önde; PAL-MoE yüzde
46.32 ve unutması yüzde 21.37; DER++ yüzde 48.63. 4 megabaytta deneyim replayi
yüzde 54.95, PAL-MoE yüzde 50.54 ve unutma yüzde 19.15. CIFAR-100'de DER++
yüzde 16.97, PAL-MoE yüzde 12.15 ve unutma yüzde 34.21. Yani feature-cache
protokolünde doğruluk lideri replay tabanları, unutma lideri biziz.

Bu iki tabloyu birlikte sunmamın sebebi şu: iddia "her yerde daha iyi"
değil. İddia, depolama formatına bağlı bir denge. Yayınlanan öğe-bütçesi
tabloları benim yöntemimi kayırıyordu; onları eşit-byte sonuçlarıyla birlikte
raporluyorum.

### Mekanizma ayrıştırması

Hangi bileşenin gerçekten çalıştığını görmek için CIFAR-10'da kontrollü bir
ablasyon yaptım. Sonuçlar şöyle. Naive eğitim yüzde 17.71 doğruluk ve yüzde
88.95 unutma. Deneyim replayi yüzde 38.58 ve yüzde 60.85. Tek kafalı latent
replay yüzde 38.45 ve yüzde 60.80; yani eşit öğe sayısında replay ile başa
baş. Sadece kapasite genişleten statik MoE ise yüzde 24.44 doğruluk ve yüzde
81.10 unutma ile deneyim replayinden bile kötü. Prototip çapaları ve router
distillation eklendiğinde sonuç yüzde 49.87 doğruluk ve yüzde 19.84 unutmaya
çıkıyor. Yani asıl katkı burada: 25 puan doğruluk artışı, 61 puan unutma
azalması. Kapı mekanizmasını devreye aldığımda sonuç yüzde 49.45, OOD terimini
eklediğimde yüzde 48.88; yani mevcut reçetede ikisi de nötr. Bu, eski
reçetedeki OOD katkısını da revize ediyor.

Gerçek hybrid sonucunu da ekleyeyim: ham veri hattında, 1000 prototiple,
CIFAR-10'da hybrid yüzde 49.63 doğruluk ve yüzde 18.03 unutma, saf varyant
yüzde 48.39 ve yüzde 20.82. Ama hybrid 14.47 megabayt saklıyor, saf varyant
2.18. Eşit byte'ta hybrid kaybediyor. Yani hybrid'i ana sonuç olarak değil,
bellek esnekliği olarak sunmak gerekiyor.

### Kapasite ve maliyet

Kapasite deneyinde CIFAR-100'de uzman sayısını 2, 4, 6 ve 20 arasında
değiştirdim. Doğruluk neredeyse sabit: yüzde 14.80, 11.13, 15.88 ve 14.25.
Unutma ise uzman sayısıyla düşüyor: yüzde 42.24'ten yüzde 14.84'e. Yani daha
fazla uzman doğruluğu artırmıyor ama unutmayı azaltıyor. Parametre eşleşmeli
tek kafalı rakipler 2.46 milyon parametreyle yüzde 10.22 ve yüzde 13.49
doğruluk alıyor; PAL-MoE 1.65 milyon parametreyle bunları geçiyor ve unutması
yarısından az.

### Anchor refresh

Bir de kalibrasyon sonrası çapaları tazeleme deneyi var. CIFAR-100'de
tazeleme kapalıyken saf varyant yüzde 15.65 doğruluk ve yüzde 31.69 unutma;
açıkken yüzde 18.31 doğruluk ve yüzde 18.82 unutma. Latent replay varyantında
da yüzde 17.68'den yüzde 21.13'e çıkıyor. Yani kalibrasyon sonrası çapaları
tazelemek belirgin kazanç sağlıyor ve final reçetede varsayılan olmalı.

### Diğer sonuçlar

Birkaç sonucu da hızlıca özetleyeyim. Uzman büyüme deneyinde kapı mekanizması
ile zorunlu genişleme istatistiksel olarak aynı sonucu verdi: ikisi de yirmi
göreve yirmi uzman atadı, doğruluk yüzde 14.25 ve 14.34. Yani bu benchmark'ta
kapı yeni uzman gerekip gerekmediğine karar veremiyor; bunu bir sınırlama
olarak söylüyorum. Yönlendirme korunumu akış boyunca 0.78 ile 0.85 arasında
kaldı. Ama prototiplerde sahiplik doğruluğu yüzde 94.5 iken test girdilerinde
yönlendirme dağınık; en yüksek uzman payı 0.15 ile 0.24 arasında. Bu, router
tarafında bir genelleme boşluğu olduğunu gösteriyor.

Sınıf uzayı sabit, girdi dağılımı değişen bir deneyde beş seed ile yüzde 85.38
doğruluk ve yüzde 6.34 unutma aldık. Encoder'ı eğitilebilir yaptığımda saf
varyant yüzde 9.97'ye düşüyor; yani yöntem dondurulmuş temsile bağımlı. Bu
önemli bir tasarım kısıtı. MIR baseline'ı da deneyim replayini geçemedi:
CIFAR-10'da yüzde 37.62'ye karşı yüzde 39.92. Gecikme tarafında ResNet-18'de
tek kafa örnek başına 1.09 milisaniye, PAL-MoE 1.71 milisaniye ve uzman
sayısından bağımsız; ViT'te 8.43'e karşı 8.57 ile 8.82 arası.

Son olarak bir baseline adaleti bulgusu var. Replay tabanlarını literatürdeki
gibi recency buffer politikasıyla koştuğumda zayıf görünüyorlar. Reservoir
sampling ile aynı bütçede deneyim replayi yüzde 25.14'ten yüzde 43.23'e,
DER++ yüzde 31.72'den yüzde 48.25'e çıkıyor. Yani yayınlanan varsayılan,
tabanları olduğundan zayıf gösteriyor. Eşit-byte hücrelerini reservoir ile
tekrar koşuyorum.

## Sınırlamalar

Şimdi sonuçların gösterdiği sınırlamaları açıkça söyleyeyim.

Birincisi, eşit byte sonucu protokole bağlı. Ham veri hattında yöntemim önde,
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

Sekizincisi, bazı tablolar hâlâ tek seed; Tiny-ImageNet dahil birkaç koşu bu
gece tamamlanıyor ve sonuçları ekleyeceğim.

## Sonuç ve sonraki adımlar

Özetleyeyim. PAL-MoE, sınırlı bellek altında unutmayı azaltmak için
prototip tabanlı fonksiyon uzayı çapalarını kullanan, dinamik büyüyebilen bir
Mixture-of-Experts mimarisi. En güçlü kanıtlar şunlar: ham veri hattında eşit
byte'ta replay tabanını 16 puan geçmesi, CIFAR-10'da en iyi baseline'ı daha az
bellekle geçmesi, ViT ile CIFAR-10'da yüzde 91.74 doğruluk ve yüzde 5.61
unutma, ve ablation'da asıl katkının prototype anchoring olduğunun
gösterilmesi.

Sonraki adımlar: reservoir ile eşit-byte tablosunu tamamlamak, Tiny-ImageNet
sonuçlarını almak, çapa tazelemeyi varsayılan yapıp başlık tablolarını yeniden
koşmak, router genelleme boşluğunu ayrıştırmak, mümkünse CORe50 eklemek ve
prompt tabanlı rakipler için kapsam kararı vermek.

Kısaca: sonuçlar umut verici, ama iddia "catastrophic forgetting çözüldü"
değil. İddia, belirli bellek bütçelerinde unutma ve bellek dengesinin
iyileştirildiği. Sorularınızı alayım.
