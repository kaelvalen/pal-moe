# PAL-MoE — Sunum Rehberi

> **Yarınki toplantı için tek dosya.** Akış, konuşma metni, sayılar, dürüst
> bulgular ve olası hoca soruları burada. İngilizce metodoloji
> [`BENCHMARK.md`](BENCHMARK.md), deney planı [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md),
> sonuç haritası [`RESULTS_INVENTORY.md`](RESULTS_INVENTORY.md), literatür
> eşlemesi [`RESEARCH_MAP.md`](RESEARCH_MAP.md).

---

## 0. Otuz saniyelik cevap

> "PAL-MoE, class-incremental continual learning için tasarladığım, dinamik
> büyüyebilen bir Mixture-of-Experts mimarisi. Yeni görev geldiğinde yeni bir
> expert açılıyor ve eskiler donduruluyor; kritik nokta, geçmiş model
> davranışının bellekte ham görüntü yerine latent prototiplerle (`v_p`
> temsil, `r_p` routing dağılımı, `o_p` expert çıktısı) saklanıp router ve
> expert'lerin bu prototiplere demirlenmesi. Amacım unutmayı parametre değil
> **temsil → routing → expert** üçlüsünün fonksiyon-uzayı stabilizasyonu
> üzerinden kontrol etmek."

Ardından mutlaka:

> "İlk sonuçlar güçlü ama bugün size **dürüst tabloyu** getirdim: eşit byte
> protokolünde doğruluk lideri replay tabanları, bizim avantajımız unutma
> tarafında; ve kapı (gate) politikasının CIFAR-100'de ölçülebilir bir katkısı
> yok. Bu yüzden iddiayı 'her yerde daha iyi' değil, **eşit bellekte
> unutma-doğruluk dengesi** olarak kuruyorum."

---

## 1. Önerilen sunum akışı (20–25 dk)

| # | Bölüm | Süre | Not |
| :-- | :-- | :-- | :-- |
| 1 | Problem ve araştırma sorusu | 2 dk | §2 |
| 2 | Mimari ve üç mekanizma | 4 dk | §3 |
| 3 | Literatürde konum: ne yeni, ne değil | 3 dk | §4 |
| 4 | Sonuçlar | 8 dk | §5: kısa vade → uzun vade → güçlü omurga → equal-byte |
| 5 | Mekanizma ayrıştırması (ablation) | 3 dk | §5.5 |
| 6 | Dürüst sınırlar ve sonraki adımlar | 3 dk | §6, §8 |
| 7 | Sorular | kalan | §9 |

Sunumda göstermek için hazır dosyalar: `results/paper_report.md` (otomatik
tablolar), `results/figures/` (Pareto/growth figürleri), `results/growth/`
(routing matrisleri), `tests/test_pal_moe.py` (105 test).

### 1b. Slayt iskeleti (11 slayt önerisi)

| Slayt | İçerik | Kaynak |
| --: | :-- | :-- |
| 1 | Başlık: PAL-MoE — Prototype-Anchored Lifelong Mixture of Experts | — |
| 2 | Problem: catastrophic forgetting + sabit bellek; formül | §2 |
| 3 | Fikir: `f(x)=Σ g_i E_i`; "reuse mu, yeni expert mi?" | §2 |
| 4 | Mimari diyagram + prototip belleği (`v_p/r_p/o_p/x_p`) | §3 |
| 5 | Görev döngüsü: kalibrasyon → distillation → exact routing lock | §3 |
| 6 | Literatür konumu: ne yeni, ne değil | §4 |
| 7 | Sonuçlar (item-budget): MNIST, CIFAR-10/100, ViT tabloları | §5.1–5.3 |
| 8 | Equal-byte Pareto figürü (`results/figures/pareto_equalbyte_c10r18.png`) | §5.4 |
| 9 | Ablation: asıl mekanizma prototype anchoring | §5.5 |
| 10 | Growth/reuse + capacity: gate nötr, expert sayısı unutmayı düşürüyor | §5.6, §5.8 |
| 11 | Dürüst sınırlar + sonraki adımlar + sorular | §6, §8, §9 |

**Konuşurken kullanılacak 5 cümle** (ezberle):
1. "PAL-MoE'nin ana problemi expert oluşturmak değil, yeni görev geldiğinde
   tarihsel routing'in bozulmasını engellemek."
2. "Prototype memory'yi replay buffer olarak değil, geçmiş model davranışını
   demirleyen function-space memory olarak kullanıyorum."
3. "Ablation şunu gösterdi: kapasite genişlemesi tek başına zarar veriyor;
   asıl katkı prototype anchor'lar ve router distillation."
4. "Eşit byte altında doğruluk lideri replay tabanları; bizim avantajımız
   unutmanın ~1.5–2× düşük olması. İddiayı böyle kuruyorum."
5. "Sonraki adım: raw-pipeline equal-byte, Tiny-ImageNet, refresh-on başlık
   tabloları ve router genelleme boşluğunun teşhisi."

---

## 2. Problem ve araştırma sorusu

Klasik continual learning'de tek model parametreleri bütün görevler tarafından
paylaşılır; yeni görev eski optimumu bozar. PAL-MoE parametre uzayını
uzmanlaştırır:

```
f(x) = Σ_i g_i(x) · E_i(x)        E_i: expert, g_i: router, N: büyüyebilir
```

Bu durumda soru "eski parametreleri nasıl korurum?" değil:

> **"Yeni bilgi mevcut uzmanlardan hangisine ait, yoksa yeni kapasite mi
> açmalıyım; ve bunu sabit bellek altında nasıl yaparım?"**

**Üç mekanizma:**
1. Dinamik expert allocation (trigger + validation gate + function-preserving
   expansion),
2. Latent replay / prototip belleği (`v_p`, `r_p`, `o_p`, latent örnekler),
3. Prototype-anchored routing (owner distillation, inference anchoring,
   negative-boundary regularization).

**Hipotezlerin güncel durumu:**

| ID | Hipotez | Durum |
| :-- | :-- | :-- |
| H1 | Dinamik uzmanlaşma girişimi azaltır | **Kısmen**: ablation'da asıl katkı anchor'larda (static MoE 24.4 → anchors 49.9) |
| H2 | Eşit byte'ta latent replay rekabetçi | **Feature-cache'te ER ≈ latent replay**; asıl raw testi bugünkü koşuda |
| H3 | Prototype anchor routing'i stabilize eder | **Destekli**: ablation'ın baskın mekanizması (+25 puan) |
| H4 | OOD negatif sınırı yanlış atamayı azaltır | **Mevcut reçetede nötr** (distillation varken gereksizleşiyor) |
| H5 | Politika expert'leri yeniden kullanır | **Çürütüldü** (CIFAR-100): gated = forced, 20/20 expert |
| H6 | Eski routing korunur, yenisi esnek kalır | **Ölçüldü**: RR_t 0.78–0.85; ama test routing'i dağınık (§6) |

---

## 3. Mimari

```
x
 │
 ▼
Shared Encoder ──► latent h
 │                   │
 │                   ├──► Prototype Memory
 │                   │       v_p : latent prototype
 │                   │       r_p : tarihsel routing dağılımı
 │                   │       o_p : tarihsel expert çıktısı
 │                   │       x_p/y_p : latent örnek + etiket
 │                   │       owner_expert : görevin expert'i
 │                   ▼
 │                 Router g(h)
 │                   │
 │                   ▼
 │              top-1 expert
 │                   │
 │                   ▼
 │        Σ g_i · E_i(h)          (+ opsiyonel generalist expert)
 │                   │
 │                   ▼
 │              prediction
 │
 └── (eğitilebilir encoder ise) EMA / encoder stabilizasyonu
```

Görev sonunda: prototipleri kaydet → (opsiyonel) joint latent calibration →
router distillation (prototype owner'larına) → eski expert + routing satırlarını
kilitle. **Kritik düzeltme (design fact 15):** Adam'ın weight-decay terimi
kilitli routing satırlarını sessizce sıfıra çekiyordu; router sıfır-decay
grubuna alındı, kilit artık matematiksel olarak tam.

---

## 4. Literatürde konum (ne yeni, ne değil)

| Çizgi | Temsilci | Bizdeki karşılığı |
| :-- | :-- | :-- |
| Replay | ER, DER++ (Buzzega 2020), ER-ACE, MIR | `pal_moe/baselines/{replay,der,mir}.py` |
| Regularizasyon | EWC (Kirkpatrick 2017) | `baselines/ewc.py` |
| Prototip/temsil | iCaRL (Rebuffi 2017), latent replay (Pellegrini 2020) | `baselines/icarl.py`, `memory/prototype_memory.py` |
| Parametre izolasyonu | PackNet, HAT | expert dondurma + routing kilidi |
| Kapasite genişlemesi | PNN (Rusu 2016), Expert Gate (Aljundi 2017) | `models/expert.py` function-preserving clone, `builder/` + trigger/gate |
| MoE-CL | MoE-in-CL teorisi (2024), adaptive/Incremental MoE (2025–26) | dinamik allocation sorusu (H5/E7) |
| Değerlendirme | Mammoth, eşit-byte non-inferiority (2026) | `memory_bytes` muhasebesi, E4 sweep |

**Yenilik iddiası:** tek tek bileşenler yeni değil. İddia, üç mekanizmanın
**class-incremental + sınırlı bellek** koşulunda birlikte tasarımı ve
**eşit-byte protokolü** altında unutma-doğruluk dengesinin gösterilmesi.
H5'in çürütülmesi bu iddiayı daraltıyor: "dinamik allocation" değil,
**sabit bütçeli kapasite genişlemesi + prototype-anchored routing**.

---

## 5. Sonuçlar

### 5.1 Kısa vade (item-bütçesi eşleşmeli, yayınlanmış tablolar)

**Split-MNIST (5 seed, 2026-09-22 yenilemesi, byte muhasebeli):** pure
**79.29 ± 1.20 / 8.02 unutma** (285 KB); hybrid 80.06 ± 1.11 / 5.56 (1061 KB);
DER++ 87.65 ± 0.94 / 6.47 (777 KB); ER(250) 82.41 / 17.42 (768 KB); MIR
83.30 / 15.27; iCaRL 59.15 / 10.68.

> **Öne çıkan H2 kanıtı:** tek kafalı **latent replay** 250 öğeyle **127 KB**
> bellekte 82.74 / 16.88 alıyor — ER'in 768 KB'da aldığı 82.41 / 17.42'ye
> neredeyse eşit, **6× az bellekle**. Raw-vs-latent depolamanın en net
> göstergesi (MNIST feature-cache'siz koşuyor).

**Split-CIFAR-10, conv, 3 seed:** pure **37.30 ± 0.11 / 21.98**, hybrid
38.66 ± 0.33 / 22.62; DER++ 31.83 ± 0.32 / 60.98; ER 25.70 / 72.10; iCaRL
24.90 / 15.75.

### 5.2 Uzun vade (20 görev)

**Split-CIFAR-100, conv, 3 seed:** pure 9.48 ± 0.25 / **23.19**; hybrid
9.98 ± 0.47 / 11.93; DER++ 5.92 ± 0.28 / 63.42; iCaRL 10.13 / 10.54 (2500 ham
örnekle).

**Split-CIFAR-100, ResNet-18, 3 seed (yeni):** pure **15.65 ± 0.39 / 31.69**;
iCaRL 13.97 ± 0.87 / 10.96; DER++ 13.12 / 66.91; ER 11.06 / 66.72. Prototip
belleği 4.18 MB; iCaRL 2.66 MB (ama ham görüntü saklıyor).

### 5.3 Güçlü omurga (ViT-B/16, 3 seed)

**CIFAR-10:** pure **91.74 ± 0.28 / 5.61** (6.37 MB); iCaRL 84.18 / 10.38
(0.80 MB); ER 82.89 ± 0.10 / 20.29; DER++ 73.28 / 32.10.

**CIFAR-100:** iCaRL **64.94 / 12.25** (8.0 MB) — pure 59.34 ± 0.32 / 18.17
(14.2 MB), DER++ 50.97 / 47.30. Bu benchmark'ta iCaRL bizden önde; dürüstçe
söylenmeli.

### 5.4 Equal-byte (adil bellek) — asıl hikâye

**Raw pipeline (asıl H2 testi; ER ham görüntü saklar, PAL latent).**
CIFAR-10 ResNet-18, 1 MiB, 3 seed:

| Yöntem | Doğruluk | Unutma | Gerçekleşen |
| :-- | :-- | :-- | --: |
| Latent replay (1016 öğe) | **50.36 ± 0.64** | 41.27 | 1023 KiB |
| **PAL-MoE pure (480 prototip)** | 46.16 ± 1.11 | **21.66** | 1024 KiB |
| PAL + Replay (hybrid) | 39.93 ± 0.56 | 32.68 | 1018 KiB |
| DER++ (85 öğe) | 36.81 ± 0.32 | 61.42 | 1024 KiB |
| ER (85 öğe) | 30.17 ± 1.30 | 72.77 | 1021 KiB |
| iCaRL (k=8) | 26.27 ± 2.71 | 22.51 | 970 KiB |

4 MiB (seed 42): latent replay 55.39, PAL 49.19 / **22.06**, DER++ 46.60,
ER 43.40, iCaRL 32.48. CIFAR-100 1 MiB (seed 42): latent replay 13.74 /
61.18, PAL 13.70 / **31.79**, hybrid 10.19, DER++ 9.00, ER 8.40.

> **Okuma:** Raw pipeline'da PAL'ın kompakt deposu (2.184 B/prototip vs
> 12.296 B/ham görüntü) byte başına ~5.6× daha fazla öğe alıyor: 1 MiB'de raw
> ER'i **+16 doğruluk puanı** ve **3.4× az unutmayla**, DER++'ı +9.4 puan ve
> 2.8× az unutmayla geçiyor; latent replay'in doğruluğuna ~2× az unutmayla
> ulaşıyor. Bellek iddiası bu protokolde geçerli.

**Feature-cache protokolü (herkes özellik saklar — dürüst karşı-örnek).**
CIFAR-10 ResNet-18, 3 seed:

| Bütçe | ER | DER++ | PAL pure | iCaRL |
| :-- | :-- | :-- | :-- | :-- |
| 1 MiB | **49.86 ± 0.66** / 41.43 | 48.63 / 32.53 | 46.32 ± 0.54 / **21.37** | 35.96 / 28.52 |
| 4 MiB | **54.95 ± 0.26** / 33.60 | 48.24 / 30.83 | 50.54 ± 1.19 / **19.15** | 41.44 / 28.74 |

CIFAR-100, 1 MiB, 3 seed: DER++ 16.97 ± 0.63, ER 13.64 ± 0.20, PAL
12.15 ± 0.56 / **34.21 unutma**, iCaRL 10.72 / 9.93.

> **Okuma:** Bu protokolde replay tabanları da özellik sakladığı için öğe
> başına byte farkı kalmıyor ve doğruluk lideri ER/DER++ oluyor; PAL yine
> ~1.5–2× daha az unutuyor. İki tabloyu birlikte sunmak, iddiayı "her yerde
> daha iyi" olmaktan çıkarıp **depolama formatına bağlı bir denge** haline
> getiriyor. Yayınlanmış item-bütçesi tabloları PAL'ı kayırıyordu (1000
> prototip vs 250 öğe).

### 5.5 Mekanizma ayrıştırması (CIFAR-10 ResNet-18, 3 seed)

| Varyant | Doğruluk | Unutma |
| :-- | :-- | :-- |
| Naive | 17.71 | 88.95 |
| ER (P=250) | 38.58 | 60.85 |
| Latent replay (P=250) | 38.45 | 60.80 |
| Static MoE (sadece genişleme) | 24.44 | 81.10 |
| **+ prototype anchor + router distillation** | **49.87** | **19.84** |
| + gate (forced yerine) | 49.45 | 20.63 |
| + OOD 0.1 (tam reçete) | 48.88 | 20.42 |

**Mesaj:** Kapasite genişlemesi tek başına zarar veriyor; asıl mekanizma
prototype anchoring (+25.4 puan). Gate ve OOD mevcut reçetede nötr.

**Gerçek hybrid (raw pipeline, varsayılan proto_size=1000, 3 seed):**
CIFAR-10 hybrid 49.63 ± 1.19 / 18.03 vs pure 48.39 ± 1.55 / 20.82 — ama
hybrid **14.47 MB** saklarken pure **2.18 MB** (6.6×). CIFAR-100: 16.90 ± 0.19
/ 19.05 vs 15.97 ± 0.31 / 30.65 (16.47 MB vs 4.18 MB). **Eşit byte'ta hybrid
kaybediyor** (1 MiB raw sweep: hybrid 39.93 vs pure 46.16); yani hybrid'in ham
deposu pahalı ve varsayılan ayardaki kazancı bellekten geliyor.

**MIR (feature-cache, P=250, 3 seed):** CIFAR-10 37.62 ± 1.04 / 61.79 (ER
39.92 / 58.87); CIFAR-100 10.50 / 66.16 (ER 10.93 / 66.91) — seçim tabanlı
replay, rastgele replay'ı bu benchmarklarda geçemiyor.

### 5.6 Kapasite ve maliyet (CIFAR-100, 3 seed)

| max_experts | Doğruluk | Unutma | Toplam parametre |
| --: | --: | --: | --: |
| 2 | 14.80 | 42.24 | 0.55M |
| 4 | 11.13 | 32.09 | 1.10M |
| 6 | 15.88 | 30.30 | 1.65M |
| 20 | 14.25 | **14.84** | 5.49M |
| Param-eşleşmeli ER (2.46M) | 10.22 | 74.65 | 2.46M |
| Param-eşleşmeli DER++ (2.46M) | 13.49 | 68.41 | 2.46M |

**Mesaj:** Doğruluk expert sayısından neredeyse bağımsız; unutma expert
sayısıyla düşüyor. "Daha çok expert = daha iyi" değil; ama PAL 1.65M
parametreyle param-eşleşmeli 2.46M'lik ER/DER++'ı geçiyor ve unutması
yarısından az.

### 5.7 Anchor refresh (CIFAR-100, 3 seed)

| Hücre | Pure | Latent replay varyantı |
| :-- | :-- | :-- |
| refresh kapalı | 15.65 ± 0.39 / 31.69 | 17.68 ± 0.97 / 17.47 |
| refresh açık | **18.31 ± 1.03 / 18.82** | **21.13 ± 0.34 / 16.09** |

**Mesaj:** Kalibrasyon sonrası anchor'ları tazelemek zorunlu; stale anchor
riski doğrulandı.

### 5.8 Diğer

- **Growth/reuse (E7):** gated = forced (14.25 vs 14.34), 20/20 expert, owner
  routing %94.5.
- **Domain-shift MNIST (tek seed):** pure 86.07 / 5.64; task-free akış 86.16
  online, surprise 0.740.
- **Routing retention RR_t:** 0.78–0.85 (20 görev boyunca top-1 rota kimliği).

---

## 6. Dürüst bulgular (hocaya kendin söyle)

1. **Eşit byte'ta sonuç protokole bağlı.** Raw pipeline'da (ER ham görüntü,
   PAL latent saklar) PAL byte başına ~5.6× daha fazla öğe alıyor ve 1 MiB'de
   raw ER'i **+16 puan / 3.4× az unutmayla**, DER++'ı +9.4 puan / 2.8× az
   unutmayla geçiyor. Feature-cache protokolünde herkes özellik sakladığı için
   bu avantaj kayboluyor ve doğruluk lideri ER/DER++ oluyor; PAL yine ~1.5–2×
   daha az unutuyor. **İddia: depolama formatına bağlı denge — her yerde
   üstünlük değil.** Yayınlanmış item-bütçesi tabloları PAL'ı kayırıyordu
   (1000 prototip vs 250 öğe).
2. **H5 çürütüldü.** Kapı her genişlemeyi kabul etti; gated = forced; expert
   reuse yok. "Dinamik allocation" iddiası daraltılmalı.
3. **Hybrid eşit byte'ta kazandırmıyor.** Varsayılan proto_size=1000'de hybrid
   pure'a göre ~1.2 puan kazandırıyor ama **6.6× bellek** harcıyor (14.47 MB vs
   2.18 MB); 1 MiB eşit-byte koşusunda hybrid kaybediyor (39.93 vs 46.16).
   Hybrid'i ana sonuç olarak değil, bellek esnekliği olarak sun.
4. **MIR ER'ı geçemiyor** (CIFAR-10 37.62 vs 39.92; CIFAR-100 10.50 vs 10.93):
   seçim tabanlı replay bu benchmarklarda rastgele replay'a üstünlük
   sağlamıyor.
5. **OOD terimi mevcut reçetede nötr.** Eski reçetedeki katkısı (fact 1)
   distillation gelince gereksizleşiyor.
6. **Router genelleme boşluğu.** Prototip owner doğruluğu %94.5 ama test
   routing'i dağınık (top-expert payı 0.15–0.24); 20 expert'te belirgin.
7. **Feature-cache semantiği.** Feature-cache koşularında replay tabanları ve
   iCaRL özellik saklıyor; hybrid'in ham deposu kapalı → "latent replay"
   varyantı. (design fact 19)
8. **iCaRL bazı benchmarklarda önde** (CIFAR-100 ViT) ve daha az byte
   kullanıyor; ham örnek saklamasına rağmen.
9. **CORe50 hâlâ yok**; domain-incremental pilot MNIST-rotate. Prompt tabanlı
   rehearsal-free baselines kapsam dışı (karar bekliyor).

**Kullanılacak cümle:** "Preliminary experiments suggest PAL-MoE improves the
forgetting–memory trade-off under limited memory, especially at long task
horizons."
**Kullanılmayacak:** "PAL-MoE catastrophic forgetting'i çözüyor."

---

## 7. Nerede ne var

```
pal_moe/                  # kütüphane
├── models/               # SharedEncoder (MLP/conv/ResNet/ViT), MLPExpert,
│                         # DynamicRouter/DistanceRouter/AttentionRouter, DynamicMoE
├── memory/               # PrototypeMemory (v_p, r_p, o_p, x_p, y_p, owner)
├── adaptation/ttt.py     # ContinualTrainer: OOD, calibration, distillation,
│                         # freeze/lock, expansion, drift ölçümleri
├── builder/              # ExpertBuilder: candidate train + validation gate
├── trigger/              # composite / energy / always tetikleyiciler
├── baselines/            # naive, ewc, replay, der (DER++/ER-ACE), agem, icarl,
│                         # latent_replay, mir
├── data/                 # split_mnist/cifar/cifar100, folder, domain_shift,
│                         # feature_cache, task_free metrikleri
└── evaluation/           # metrics, diagnostics, geometry, heads, calibration

experiments/              # koşucular (kütüphaneyi kullanır)
├── run_benchmark.py      # 12+ yöntem, tüm bayraklar
├── run_benchmark_multi.py# çok-seed (mean±std) + --aggregate_only
├── run_ablation.py       # kontrollü ablation (paylaşılan encoder)
├── measure_latency.py    # gecikme/maliyet
├── paper_report.py       # results/paper_report.md + figürler
├── prepare_tiny_imagenet.py, repair_missing_rows.py, diagnose_checkpoint.py,
├── debug_routing_asymmetry.py, plot_results.py, run_pure_explore.py
└── recipes/              # paper_wave1b/1c/1d, paper_all*.sh, paper_status.sh,
                          # multiseed_cifar.sh, cifar100_resnet18_multiseed.sh, ...

configs/                  # mnist_default, cifar10_resnet18_frozen[_raw|_trainable],
                          # cifar100_resnet18_frozen[_raw], cifar10_vit, cifar100_vit,
                          # cifar10_big_final, cifar100_big_frozen, ...
docs/                     # bu rehber + BENCHMARK + EXPERIMENT_PLAN + RESEARCH_MAP
                          # + RESULTS_INVENTORY
results/                  # kanıt; her dizin RESULTS_INVENTORY.md'de açıklanır
tests/test_pal_moe.py     # 105 test
```

**Kanıt eşlemesi:** `docs/RESULTS_INVENTORY.md`. Eski/keşif koşuları
`results/archive/` altında (1.6 GB, .gitignore'da).

---

## 8. Sınırlamalar ve sonraki adımlar

**Bu gece koşan kuyruk** (`paper_all_resume.sh`): E9 seed 1/2, E3 yenilemeleri,
E1a/E1b latent-replay satırları, **raw equal-byte sweep**, gerçek hybrid
(raw pipeline), MIR, **Tiny-ImageNet 20×10**, CIFAR conv yenilemeleri, latency,
domain-shift 5-seed, trainable-encoder drift, buffer-policy appendix.

**Sonraki adımlar:**
1. Raw sweep sonucuna göre equal-byte çerçevesini kesinleştir.
2. `refresh_anchors_after_calib` varsayılan yapıp başlık tablolarını yeniden
   koş (E8 kazancı).
3. Tiny-ImageNet ve (mümkünse) CORe50 ile ölçeği büyüt.
4. Router genelleme boşluğunu teşhis et (E6 ayrıştırması: router mı, expert mi?).
5. Prompt tabanlı baselines kapsam kararı (L2P/DualPrompt).
6. E13: 5-seed final tablolar + artefakt yayını.

---

## 9. Olası sorular ve cevaplar

**"Yenilik tam olarak ne?"**
> Tek tek bileşenlerin hiçbirini yeni iddia etmiyorum: iCaRL prototipi, DER++
> replay'i, PNN genişlemeyi, Expert Gate seçimi zaten yapıyor. Yenilik iddiam
> bu üç mekanizmanın class-incremental + sabit bellek koşulunda birlikte
> tasarımı ve eşit-byte protokolü altında unutma-doğruluk dengesinin
> gösterilmesi. Ablation bunu destekliyor: asıl katkı prototype anchoring'de.

**"MoE kullanmanın kanıtı ne?"**
> Static MoE tek başına ER'den kötü (24.4 vs 38.6). Kanıt, kapasite
> genişlemesinin anchor'larla birleşince +25 puan getirmesi. Yani "MoE" değil,
> "anchored MoE" çalışıyor.

**"Gate ne işe yarıyor?"**
> CIFAR-100'de ölçülebilir doğruluk katkısı yok: gated = forced, 20/20 expert.
> Gate şu an bir hesaplama/maliyet kontrolü; allocation politikası kararını
> (H5) bu benchmark'ta veremiyoruz.

**"ER'den daha mı iyi?"**
> Eşit byte'ta doğrulukta hayır; unutmada evet (~1.5–2×). Cümle: "belirli
> bellek bütçelerinde unutma avantajı".

**"CIFAR-100 ViT'te iCaRL sizi geçiyor, neden?"**
> Evet, 64.9 vs 59.3 ve daha az byte. iCaRL güçlü bir baseline; bizim iddiamız
> her koşulda üstünlük değil, unutma-bellek dengesi. Bunu saklamıyoruz.

**"Task-free mi?"**
> Ana protokol task-boundary supervised; energy trigger + streaming evaluator
> var ama ana sonuç değil. Domain-shift pilotu MNIST-rotate.

**"Bellek iddiası ne kadar dürüst?"**
> Artık her sonuç `memory_bytes` raporluyor. Feature-cache'te herkes özellik
> saklıyor; raw pipeline'da PAL 5.6× daha az byte/öğe. İki tabloyu birlikte
> sunuyoruz.

**"43% nerede?" (maildeki sayı)**
> Mail preliminary'di; sonrasında protokolü revize ettim (encoder, freezing,
> exact routing lock, baseline BatchNorm artefaktı) ve tabloları yeniden
> ürettim. Güncel sayılar bu rehberde.

---

## 10. Komutlar (demo için)

```bash
cd /home/kael/pal-moe

# Testler
.venv/bin/python -m pytest tests/test_pal_moe.py -q        # 105 test

# Hızlı MNIST demosu (~1 dk, GPU)
.venv/bin/python experiments/run_benchmark.py --config configs/mnist_default.json \
  --methods palmoe,hybrid,derpp --device cuda

# Checkpoint teşhisi (router mı expert mi?)
.venv/bin/python experiments/run_benchmark.py --config configs/mnist_default.json \
  --save_checkpoints --device cuda
.venv/bin/python experiments/diagnose_checkpoint.py \
  --checkpoint results/checkpoints_palmoe/task_4.pt --dataset mnist

# Rapor ve figürler
.venv/bin/python experiments/paper_report.py
#   -> results/paper_report.md, results/figures/*.png

# Gece kuyruğu (şu an çalışıyor)
bash experiments/recipes/paper_status.sh
```
