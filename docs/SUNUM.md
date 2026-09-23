# PAL-MoE sunum notları

Güncel sayılar, sınırlamalar ve muhtemel sorular burada. Ayrıntılı metodoloji `BENCHMARK.md`, deney planı
`EXPERIMENT_PLAN.md`, literatür eşlemesi `RESEARCH_MAP.md`, sonuç dizinleri
`RESULTS_INVENTORY.md` dosyalarında.

## 1. PAL-MoE nedir?

PAL-MoE, class-incremental continual learning için dinamik büyüyebilen bir
Mixture-of-Experts mimarisi. Yeni görev geldiğinde yeni bir expert açılıyor ve
eski expert'ler donduruluyor. Bellekte ham görüntü yerine latent prototipler
tutuluyor: `v_p` temsil, `r_p` geçmiş routing dağılımı, `o_p` geçmiş expert
çıktısı. Router ve expert'ler bu prototiplere demirleniyor, böylece unutma
parametre düzeyinde değil, temsil, routing ve expert davranışının birlikte
stabilize edilmesiyle kontrol ediliyor.

Toplantıda altı çizilecek nokta şu: bileşenlerin hiçbiri tek başına yeni
değil; katkı bu üç mekanizmanın sınırlı bellek altında birlikte tasarımı ve
eşit-byte protokolüyle ölçülmesi. Ölçümler bazı iddiaları da çürüttü; onları
da açıkça söylüyorum.

## 2. Sunum akışı ve slayt planı

Önerilen akış (20-25 dakika):

1. Problem ve araştırma sorusu (2 dk)
2. Mimari ve üç mekanizma (4 dk)
3. Literatürde konum (3 dk)
4. Sonuçlar (8 dk)
5. Mekanizma ayrıştırması (3 dk)
6. Sınırlamalar ve sonraki adımlar (3 dk)
7. Sorular

Slayt planı:

| Slayt | İçerik                                                              | Kaynak                                                      |
| ----: | :-------------------------------------------------------------------- | :---------------------------------------------------------- |
|     1 | Başlık                                                              | -                                                           |
|     2 | Problem: unutma ve bellek kısıtı                                   | bölüm 3                                                   |
|     3 | Fikir: uzmanlaştırma ve "yeniden kullan mı, yeni expert mi" sorusu | bölüm 3                                                   |
|     4 | Mimari ve prototip belleği                                           | bölüm 4                                                   |
|     5 | Görev döngüsü: kalibrasyon, distillation, routing kilidi          | bölüm 4                                                   |
|     6 | Literatür konumu                                                     | bölüm 5                                                   |
|     7 | Sonuçlar: MNIST, CIFAR-10, CIFAR-100, ViT                            | bölüm 6.1-6.3                                             |
|     8 | Eşit byte Pareto figürü                                            | bölüm 6.4,`results/figures/pareto_equalbyte_c10r18.png` |
|     9 | Ablation: asıl mekanizma prototype anchoring                         | bölüm 6.5                                                 |
|    10 | Growth/reuse ve kapasite                                              | bölüm 6.6, 6.8                                            |
|    11 | Sınırlamalar ve sonraki adımlar                                    | bölüm 7-8                                                 |

Sunumda açılabilecek dosyalar: `results/paper_report.md` (otomatik tablolar ve
figürler), `results/growth/` (routing matrisleri), `tests/test_pal_moe.py`
(106 test).

Kullanılabilecek kısa ifadeler:

- "Ana problem expert oluşturmak değil, yeni görev geldiğinde geçmiş
  routing'in bozulmasını engellemek."
- "Prototip belleği bir replay buffer değil; geçmiş model davranışını
  demirleyen bir fonksiyon-uzayı belleği."
- "Ablation, kapasite genişlemesinin tek başına yetmediğini gösteriyor;
  asıl katkı prototype anchor'lar ve router distillation."
- "Eşit byte altında sonuç depolama formatına bağlı; raw pipeline'da PAL önde,
  feature-cache'te replay tabanları doğrulukta önde, PAL unutmada önde."

## 3. Problem ve araştırma sorusu

Standart continual learning'de tek modelin parametreleri bütün görevler
tarafından paylaşılır. Yeni görev, eski görevlerin çözümünü bozar. PAL-MoE
parametre uzayını görevler arasında paylaşmak yerine uzmanlaştırır:

```
f(x) = sum_i g_i(x) E_i(x)
```

Burada `E_i` expert, `g_i` router, `N` zaman içinde büyüyebilen expert sayısı.
Bu durumda soru şu hale gelir:

> Yeni bilgi mevcut expert'lerden hangisine ait, yoksa yeni kapasite mi
> açılmalı? Bu karar sınırlı bellek altında nasıl verilir?

Üç mekanizma birlikte kullanılıyor:

1. Dinamik expert allocation: trigger, validation gate ve
   function-preserving expansion.
2. Latent replay ve prototip belleği: `v_p`, `r_p`, `o_p`, latent örnekler.
3. Prototype-anchored routing: owner distillation, inference anchoring,
   negatif sınır düzenlileştirmesi.

Hipotezlerin güncel durumu:

| Hipotez                                               | Durum                                                                              |
| :---------------------------------------------------- | :--------------------------------------------------------------------------------- |
| H1: Dinamik uzmanlaşma girişimi azaltır            | Kısmen. Ablation'da asıl katkı anchor'larda: static MoE 24.4, anchor'larla 49.9 |
| H2: Eşit byte'ta latent replay rekabetçi            | Destekli. Raw pipeline'da latent replay ER'den çok daha verimli                   |
| H3: Prototip anchor'lar routing'i stabilize eder      | Destekli. Ablation'ın baskın mekanizması                                        |
| H4: Negatif sınır terimi yanlış atamayı azaltır | Mevcut reçetede nötr; distillation varken gereksizleşiyor                       |
| H5: Politika expert'leri yeniden kullanır            | Çürütüldü. CIFAR-100'de gated = forced, 20/20 expert                          |
| H6: Eski routing korunur, yenisi esnek kalır         | Kısmen. RR_t 0.78-0.85, ama test routing'i dağınık                             |

## 4. Mimari

```
x
 |
 v
Shared Encoder --> latent h
 |                   |
 |                   +--> Prototype Memory
 |                   |      v_p: latent prototip
 |                   |      r_p: geçmiş routing dağılımı
 |                   |      o_p: geçmiş expert çıktısı
 |                   |      x_p, y_p: latent örnek ve etiket
 |                   |      owner_expert: görevin expert'i
 |                   v
 |                 Router g(h)
 |                   |
 |                   v
 |              top-1 expert
 |                   |
 |                   v
 |        toplam g_i E_i(h)     (isteğe bağlı generalist expert)
 |                   |
 |                   v
 |              tahmin
 |
 +-- (eğitilebilir encoder ise) EMA / encoder stabilizasyonu
```

Görev sonunda sırasıyla: prototipler kaydediliyor, isteğe bağlı joint latent
kalibrasyon yapılıyor, router prototype owner'larına distill ediliyor ve eski
expert'lerle routing satırları kilitleniyor.

Önemli bir teknik düzeltme: Adam'ın weight-decay terimi kilitli routing
satırlarını sessizce sıfıra çekiyordu. Router sıfır-decay grubuna alındı; kilit
artık tam. Bu düzeltmeden önceki sayılar bu yüzden farklıydı (BENCHMARK.md,
design fact 15).

## 5. Literatürde konum

| Çizgi                | Temsilci çalışmalar                                   | Bu repodaki karşılığı                             |
| :-------------------- | :------------------------------------------------------- | :----------------------------------------------------- |
| Replay                | ER, DER++, ER-ACE, MIR                                   | `pal_moe/baselines/`                                 |
| Regularizasyon        | EWC                                                      | `baselines/ewc.py`                                   |
| Prototip ve temsil    | iCaRL, latent replay                                     | `baselines/icarl.py`, `memory/prototype_memory.py` |
| Parametre izolasyonu  | PackNet, HAT                                             | expert dondurma ve routing kilidi                      |
| Kapasite genişlemesi | PNN, Expert Gate                                         | `models/expert.py`, `builder/` ve `trigger/`     |
| MoE ve CL             | 2024 MoE-CL teorisi, 2025-26 adaptif MoE çalışmaları | dinamik allocation sorusu (H5, E7)                     |
| Değerlendirme        | Mammoth, 2026 eşit-byte protokolü                      | `memory_bytes` muhasebesi, E4 sweep                  |

Yenilik iddiası: bileşenlerin hiçbiri yeni değil. İddia, üç mekanizmanın
class-incremental ve sınırlı bellek koşullarında birlikte tasarımı ile
eşit-byte protokolü altında unutma-doğruluk dengesinin gösterilmesi. H5'in
çürütülmesi bu iddiayı daraltıyor: "dinamik allocation" değil, sabit bütçeli
kapasite genişlemesi ve prototype-anchored routing.

## 6. Sonuçlar

### 6.1 Kısa vadeli benchmarklar

Split-MNIST, 5 seed (2026-09-22 yenilemesi):

| Yöntem               | Doğruluk     | Unutma |    Veri |
| :-------------------- | :------------ | :----- | ------: |
| DER++ (P=250)         | 87.65 ± 0.94 | 6.47   |  777 KB |
| MIR (P=250)           | 83.30 ± 1.95 | 15.27  |  768 KB |
| Latent replay (P=250) | 82.74 ± 0.64 | 16.88  |  127 KB |
| ER (P=250)            | 82.41 ± 1.30 | 17.42  |  768 KB |
| PAL-MoE + hybrid      | 80.06 ± 1.11 | 5.56   | 1061 KB |
| PAL-MoE pure          | 79.29 ± 1.20 | 8.02   |  285 KB |
| iCaRL (k=25)          | 59.15 ± 2.13 | 10.68  |  771 KB |

Latent replay 127 KB ile ER'in 768 KB'da aldığı doğruluğa yaklaşıyor; bu,
depolama formatının etkisini gösteren en net tek sonuç.

Split-CIFAR-10, conv, 3 seed (2026-09-23 yenilemesi):

| Yöntem          | Doğruluk     | Unutma |     Veri |
| :--------------- | :------------ | :----- | -------: |
| PAL-MoE + hybrid | 39.00 ± 0.09 | 22.85  | 14.47 MB |
| PAL-MoE pure     | 37.81 ± 0.57 | 22.03  |  2.18 MB |
| DER++ (P=250)    | 31.72 ± 0.62 | 61.18  |  3.08 MB |
| iCaRL (k=25)     | 25.35 ± 1.33 | 15.26  |  3.08 MB |
| ER (P=250)       | 25.14 ± 0.23 | 73.00  |  3.07 MB |

PAL pure, DER++'ı 6.1 puan geçiyor ve daha az bellek kullanıyor.

### 6.2 Uzun vadeli benchmarklar

Split-CIFAR-100, conv, 3 seed:

| Yöntem                 | Doğruluk     | Unutma |    Veri |
| :---------------------- | :------------ | :----- | ------: |
| iCaRL (k=25)            | 10.02 ± 0.17 | 11.13  | 2.66 MB |
| PAL-MoE pure            | 9.50 ± 0.42  | 23.24  | 4.13 MB |
| PAL-MoE + latent replay | 9.47 ± 0.89  | 13.84  | 4.16 MB |
| DER++ (P=250)           | 5.95 ± 0.31  | 63.19  | 0.36 MB |

Split-CIFAR-100, ResNet-18, 3 seed:

| Yöntem       | Doğruluk     | Unutma |    Veri |
| :------------ | :------------ | :----- | ------: |
| PAL-MoE pure  | 15.65 ± 0.39 | 31.69  | 4.18 MB |
| iCaRL (k=25)  | 13.97 ± 0.87 | 10.96  | 2.66 MB |
| DER++ (P=250) | 13.12 ± 0.08 | 66.91  | 0.36 MB |
| ER (P=250)    | 11.06 ± 0.22 | 66.72  | 0.26 MB |

### 6.3 Güçlü omurga

ViT-B/16, CIFAR-10, 3 seed:

| Yöntem       | Doğruluk     | Unutma |    Veri |
| :------------ | :------------ | :----- | ------: |
| PAL-MoE pure  | 91.74 ± 0.28 | 5.61   | 6.37 MB |
| iCaRL (k=25)  | 84.18 ± 0.00 | 10.38  | 0.80 MB |
| ER (P=250)    | 82.89 ± 0.10 | 20.29  | 0.77 MB |
| DER++ (P=250) | 73.28 ± 0.31 | 32.10  | 0.78 MB |

ViT-B/16, CIFAR-100, 3 seed:

| Yöntem       | Doğruluk     | Unutma |     Veri |
| :------------ | :------------ | :----- | -------: |
| iCaRL (k=25)  | 64.94 ± 0.00 | 12.25  |  7.99 MB |
| PAL-MoE pure  | 59.34 ± 0.32 | 18.17  | 14.23 MB |
| DER++ (P=250) | 50.97 ± 0.15 | 47.30  |  0.87 MB |
| ER (P=250)    | 34.06 ± 0.90 | 66.56  |  0.77 MB |

CIFAR-100 ViT'te iCaRL hem daha doğru hem daha az bellek kullanıyor. 

### 6.4 Eşit byte karşılaştırması

Raw pipeline (asıl test; ER ham görüntü saklar, PAL latent). CIFAR-10
ResNet-18, 1 MiB, 3 seed:

| Yöntem                     | Doğruluk     | Unutma | Gerçekleşen |
| :-------------------------- | :------------ | :----- | ------------: |
| Latent replay (1016 öğe)  | 50.36 ± 0.64 | 41.27  |      1023 KiB |
| PAL-MoE pure (480 prototip) | 46.16 ± 1.11 | 21.66  |      1024 KiB |
| PAL + replay (hybrid)       | 39.93 ± 0.56 | 32.68  |      1018 KiB |
| DER++ (85 öğe)            | 36.81 ± 0.32 | 61.42  |      1024 KiB |
| ER (85 öğe)               | 30.17 ± 1.30 | 72.77  |      1021 KiB |
| iCaRL (k=8)                 | 26.27 ± 2.71 | 22.51  |       970 KiB |

PAL, raw ER'i 16 puan geçiyor ve unutması 3.4 kat düşük; DER++'ı 9.4 puan ve
2.8 kat daha az unutmayla geçiyor. Sebep, prototip başına 2.184 B ile ham
görüntü başına 12.296 B arasındaki fark: eşit byte'ta PAL yaklaşık 5.6 kat
daha fazla öğe tutuyor. 4 MiB'de (seed 42) sıralama: latent replay 55.39,
PAL 49.19 (unutma 22.06), DER++ 46.60, ER 43.40. CIFAR-100 1 MiB'de: latent
replay 13.74, PAL 13.70 (unutma 31.79), DER++ 9.00, ER 8.40.

Feature-cache protokolü (herkes özellik saklar). CIFAR-10 ResNet-18, 3 seed:

| Bütçe | ER                    | DER++         | PAL pure              | iCaRL         |
| :------ | :-------------------- | :------------ | :-------------------- | :------------ |
| 1 MiB   | 49.86 ± 0.66 / 41.43 | 48.63 / 32.53 | 46.32 ± 0.54 / 21.37 | 35.96 / 28.52 |
| 4 MiB   | 54.95 ± 0.26 / 33.60 | 48.24 / 30.83 | 50.54 ± 1.19 / 19.15 | 41.44 / 28.74 |

CIFAR-100 1 MiB, 3 seed: DER++ 16.97 ± 0.63, ER 13.64 ± 0.20, PAL
12.15 ± 0.56 (unutma 34.21), iCaRL 10.72 / 9.93.

Bu protokolde replay tabanları da özellik sakladığı için byte avantajı
kalmıyor; doğrulukta ER ve DER++ önde, PAL unutmada önde. İki tabloyu birlikte
sunmak gerekiyor; iddia "her yerde daha iyi" değil, depolama formatına bağlı
bir denge.

Raw hücrelerin reservoir sampling ile tekrarı sıralamayı değiştirmiyor
(1 MiB, 3 seed: ER 33.18 ± 1.11 / 68.04, DER++ 37.79 ± 1.17 / 60.38; PAL pure
46.16 / 21.66).

### 6.5 Mekanizma ayrıştırması

CIFAR-10 ResNet-18, 3 seed:

| Varyant                                   | Doğruluk | Unutma |
| :---------------------------------------- | :-------- | :----- |
| Naive                                     | 17.71     | 88.95  |
| ER (P=250)                                | 38.58     | 60.85  |
| Latent replay (P=250)                     | 38.45     | 60.80  |
| Static MoE (sadece genişleme)            | 24.44     | 81.10  |
| + prototype anchor ve router distillation | 49.87     | 19.84  |
| + gate (forced yerine)                    | 49.45     | 20.63  |
| + OOD 0.1 (tam reçete)                   | 48.88     | 20.42  |

Kapasite genişlemesi tek başına ER'den kötü. Asıl katkı prototype
anchor'larda: 25.4 puan doğruluk artışı, 61 puan unutma azalması. Gate ve OOD
mevcut reçetede nötr.

Gerçek hybrid (raw pipeline, proto_size=1000, 3 seed): CIFAR-10'da
49.63 ± 1.19 / 18.03, pure 48.39 ± 1.55 / 20.82; ancak hybrid 14.47 MB
saklarken pure 2.18 MB. CIFAR-100'de 16.90 ± 0.19 / 19.05, pure
15.97 ± 0.31 / 30.65 (16.47 MB vs 4.18 MB). Eşit byte'ta hybrid kaybediyor
(1 MiB: 39.93 vs 46.16), yani hybrid'i ana sonuç olarak değil, bellek
esnekliği olarak sunmak gerekiyor.

MIR (feature-cache, P=250, 3 seed): CIFAR-10 37.62 ± 1.04 / 61.79
(ER 39.92 / 58.87); CIFAR-100 10.50 / 66.16 (ER 10.93 / 66.91). Seçim tabanlı
replay bu benchmarklarda rastgele replay'ı geçemiyor.

### 6.6 Kapasite

CIFAR-100 ResNet-18, 3 seed:

|                     max_experts | Doğruluk | Unutma | Parametre |
| ------------------------------: | --------: | -----: | --------: |
|                               2 |     14.80 |  42.24 |     0.55M |
|                               4 |     11.13 |  32.09 |     1.10M |
|                               6 |     15.88 |  30.30 |     1.65M |
|                              20 |     14.25 |  14.84 |     5.49M |
|    Param-eşleşmeli ER (2.46M) |     10.22 |  74.65 |     2.46M |
| Param-eşleşmeli DER++ (2.46M) |     13.49 |  68.41 |     2.46M |

Doğruluk expert sayısından neredeyse bağımsız; unutma expert sayısıyla
düşüyor. PAL 1.65M parametreyle 2.46M'lik ER ve DER++'ı geçiyor, unutması
yarısından az.

### 6.7 Anchor refresh

CIFAR-100 ResNet-18, 3 seed:

| Hücre          | Pure                  | Latent replay varyantı |
| :-------------- | :-------------------- | :---------------------- |
| refresh kapalı | 15.65 ± 0.39 / 31.69 | 17.68 ± 0.97 / 17.47   |
| refresh açık  | 18.31 ± 1.03 / 18.82 | 21.13 ± 0.34 / 16.09   |

Kalibrasyon sonrası anchor'ları tazelemek belirgin kazanç sağlıyor
(+2.7 doğruluk, -12.9 unutma). Inference anchoring (alpha=0.5) yaklaşık nötr.

### 6.8 Diğer sonuçlar

- Growth/reuse (E7): gated ve forced aynı sonucu veriyor (14.25 vs 14.34),
  20/20 expert, owner routing %94.5. Routing retention RR_t 0.78-0.85.
- Domain-shift MNIST, 5 seed: 85.38 ± 0.47 / 6.34 (paylaşılan expert ile).
- Eğitilebilir encoder (CIFAR-10, seed 42): pure 9.97 / 32.81'e düşüyor
  (frozen: 37.81). Yöntem dondurulmuş temsile bağımlı.
- Gecikme (batch 1): ResNet-18'de tek kafa 1.09 ms, PAL 1.71 ms ve expert
  sayısından bağımsız; ViT'te 8.43 ms ve 8.57-8.82 ms.
- Buffer politikası: feature-cache protokolünde P=250'de reservoir sampling
  tabanlara yaklaşık 4 puan doğruluk ve 5-8 puan daha az unutma kazandırıyor
  (seed 42 eki: ER 43.23 / 51.75, recency 39.26 / 59.69; DER++ 48.25 / 40.65,
  recency 44.37 / 45.16). Raw eşit-byte koşusunda etki küçük (ER
  33.18 ± 1.11 / 68.04, recency 30.17 / 72.77; DER++ 37.79 ± 1.17 / 60.38,
  recency 36.81 / 61.42), yani PAL'ın raw üstünlüğü recency varsayılanından
  kaynaklanmıyor. Ekin kalan seed'leri koşuyor.
- Router genelleme boşluğu: prototip owner doğruluğu %94.5, test routing'i
  dağınık (top-expert payı 0.15-0.24).

## 7. Sınırlamalar

1. Eşit byte sonucu protokole bağlı: raw pipeline'da PAL önde,
   feature-cache'te doğrulukta replay tabanları önde. İddia denge üzerine
   kurulmalı, üstünlük üzerine değil.
2. H5 çürütüldü: kapı her genişlemeyi kabul etti, expert reuse yok. "Dinamik
   allocation" yerine sabit bütçeli kapasite genişlemesi denmeli.
3. OOD terimi mevcut reçetede nötr; eski reçetedeki katkısı distillation
   gelince gereksizleşiyor.
4. Yöntem dondurulmuş temsile bağımlı; eğitilebilir encoder ile çöküyor.
5. Router genelleme boşluğu var: prototiplerde %94.5, test girdilerinde
   dağınık.
6. iCaRL bazı benchmarklarda (CIFAR-100 ViT, CIFAR-100 conv) daha doğru ve
   daha az bellek kullanıyor.
7. CORe50 veri seti yok; domain-incremental pilot MNIST-rotate ile sınırlı.
   Prompt tabanlı rehearsal-free baselines kapsam dışı.
8. Yayınlanan recency buffer politikası replay tabanlarını olduğundan zayıf
   gösteriyor; reservoir sonuçları bunu düzeltiyor.

Kullanılabilecek cümle: "Preliminary experiments suggest PAL-MoE improves the
forgetting-memory trade-off under limited memory, especially at long task
horizons."

Kullanılmaması gereken cümle: "PAL-MoE catastrophic forgetting'i çözüyor."

## 8. Sonraki adımlar

1. Reservoir ile eşit-byte tablosunu tamamla.
2. Tiny-ImageNet 20x10 sonuçlarını al (şu an koşuyor).
3. `refresh_anchors_after_calib` varsayılan yapıp başlık tablolarını yeniden
   koş.
4. Router genelleme boşluğunu ayrıştır: hata router'da mı, expert'te mi?
5. Mümkünse CORe50 ile domain-incremental sonuç ekle.
6. Prompt tabanlı baselines için kapsam kararı ver.
7. 5 seed final tablolar ve artefakt yayını.

## 9. Olası sorular

Yenilik ne?
: Bileşenlerin hiçbiri yeni değil: iCaRL prototipi, DER++ replay'i, PNN
  genişlemeyi, Expert Gate seçimi zaten yapıyor. Katkı, bu mekanizmaların
  class-incremental ve sınırlı bellek koşullarında birlikte tasarımı ve
  eşit-byte protokolü altında ölçülmesi. Ablation asıl katkının prototype
  anchoring olduğunu gösteriyor.

MoE kullanmanın kanıtı ne?
: Static MoE tek başına ER'den kötü (24.4 vs 38.6). Kanıt, kapasite
  genişlemesinin anchor'larla birleşince 25 puan kazandırması. Yani "MoE"
  değil, "anchored MoE" çalışıyor.

Gate ne işe yarıyor?
: CIFAR-100'de ölçülebilir doğruluk katkısı yok; gated ve forced aynı, 20/20
  expert. Gate şu an maliyet kontrolü; allocation kararını (H5) bu
  benchmark'ta veremiyoruz.

ER'den daha mı iyi?
: Eşit byte'ta raw pipeline'da evet; feature-cache'te doğrulukta hayır,
  unutmada evet. İfade "belirli bellek bütçelerinde unutma avantajı" olmalı.

CIFAR-100 ViT'te iCaRL sizi geçiyor, neden?
: Evet, 64.9 vs 59.3 ve daha az byte ile. iCaRL güçlü bir baseline; iddia her
  koşulda üstünlük değil, unutma-bellek dengesi.

Task-free mi?
: Ana protokol task-boundary supervised. Energy trigger ve streaming
  evaluator var, ama ana sonuç değil.

Bellek iddiası ne kadar dürüst?
: Her sonuç `memory_bytes` raporluyor. Feature-cache'te herkes özellik
  saklıyor; raw pipeline'da PAL öğe başına 5.6 kat daha az byte harcıyor.
  İki tablo birlikte sunuluyor.

Maildeki yüzde 43 nerede?
: Mail preliminary sonuçlardandı. Sonrasında encoder, freezing, routing lock
  ve baseline tarafı revize edildi; tablolar yeniden üretildi. Güncel sayılar
  bu dosyada.

## 10. Repo haritası

```
pal_moe/          kütüphane: models, memory, adaptation, builder, trigger,
                  baselines, data, evaluation
experiments/      koşucular ve araçlar (README.md'de indeks)
configs/          JSON config'ler (README.md'de indeks)
docs/             bu dosya, BENCHMARK, EXPERIMENT_PLAN, RESEARCH_MAP,
                  RESULTS_INVENTORY, CODE_REVIEW
results/          kanıt dizinleri (RESULTS_INVENTORY.md'de açıklanır)
tests/            106 test
```

## 11. Komutlar

```bash
cd /home/kael/pal-moe

# testler
.venv/bin/python -m pytest tests/test_pal_moe.py -q

# hızlı MNIST demosu (~1 dk, GPU)
.venv/bin/python experiments/run_benchmark.py --config configs/mnist_default.json \
  --methods palmoe,hybrid,derpp --device cuda

# checkpoint teşhisi (hata router'da mı, expert'te mi)
.venv/bin/python experiments/run_benchmark.py --config configs/mnist_default.json \
  --save_checkpoints --device cuda
.venv/bin/python experiments/diagnose_checkpoint.py \
  --checkpoint results/checkpoints_palmoe/task_4.pt --dataset mnist

# tablolar ve figürler
.venv/bin/python experiments/paper_report.py

# arka plan koşularının durumu
bash experiments/recipes/paper_status.sh
```
