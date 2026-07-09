# TrackWood

TrackWood, açık kanal deneylerinde su yüzeyinde taşınan odunsu malzemelerin video üzerinden tespit edilmesi, takip edilmesi ve sayısal olarak değerlendirilmesi için hazırlanmış Python tabanlı bir görüntü işleme aracıdır.

Bu çalışma, TÜBİTAK 1001-Çaba 124M033 numaralı "Akımla Taşınan Odunların Sebep Olduğu Taşkınların Önlenmesinde Geçirgen Mahmuzların Etkilerinin İncelenmesi" projesi kapsamında geliştirilen görüntü işleme modülünü içerir.

![1az214 deney videosundan anotasyonlu kare](1az214%20annoted.jpg)

## Amaç

Deney videolarında çok sayıda odun parçası aynı anda hareket eder, birbirini örter, geçici kümeler oluşturur ve su yüzeyi parlamaları nedeniyle manuel sayım zorlaşır. Bu repo, bu görsel kayıtları tekrar analiz edilebilir verilere dönüştürmek için kullanılır.

Kodun temel amacı şunları üretmektir:

- Her odun parçası için takip kimliği.
- Kare bazlı merkez koordinatı ve sınırlayıcı kutu bilgisi.
- Piksel/kare cinsinden tahmini hız.
- L1 ve L2 gibi sanal çizgilerden geçiş sayıları.
- Anotasyonlu çıktı videosu.
- Yörünge ve geçiş kayıtlarını içeren CSV dosyaları.
- CSV dosyalarından üretilen yörünge, hız, alan ve geçiş grafikleri.

## Yöntem

İşlem hattı iki ana parçadan oluşur.

`main.py`, deney videosunu kare kare işler. Eğitimli YOLO modeli su yüzeyindeki odun parçalarını tespit eder. Ardından SORT mantığına dayalı takip yapısı, Kalman filtresi ve Macar algoritması ile tespitleri zaman içinde aynı izlere bağlar. Böylece her odun parçası için bir `track_id` oluşur.

Kodda tanımlanan sanal dikey çizgiler, örneğin `v:20%,v:80%`, video genişliğinin belirli noktalarına yerleştirilir. Bir odun parçası bu çizgilerden geçtiğinde geçiş olayı CSV dosyasına kaydedilir. Bu sayede L1 ve L2 sayımları kullanılarak tuzaklama verimi hesaplanabilir.

`fortrajectory.py`, `main.py` tarafından üretilen CSV dosyalarını okur ve analiz grafikleri oluşturur. Bu grafikler deneyler arasında yörünge yoğunluğu, hız dağılımı, odun parçası alanı ve çizgi geçişlerinin zamansal değişimini karşılaştırmak için kullanılır.

## Dosyalar

| Dosya | İçerik |
| --- | --- |
| `main.py` | Video işleme, nesne tespiti, takip, çizgi geçişi sayımı ve CSV/video çıktıları. |
| `fortrajectory.py` | Takip CSV dosyalarından analiz grafikleri üretir. |
| `best400.pt` | Odunsu malzeme tespiti için eğitilmiş YOLO model ağırlığı. |
| `1az214.MP4` | Örnek ham deney videosu. |
| `1az214_annotated.mp4` | Örnek anotasyonlu çıktı videosu. |
| `1az214 annoted.jpg` | Anotasyonlu videodan örnek kare. |

## Kurulum

Python 3.10 veya daha güncel bir sürüm önerilir.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install ultralytics opencv-python numpy scipy filterpy pandas matplotlib seaborn
```

Repo büyük video ve model dosyaları içerdiği için klonlama sırasında Git LFS kullanılması gerekir:

```powershell
git lfs install
git clone https://github.com/enesgul23/trackwood.git
cd trackwood
```

## Video Analizi

`main.py` içindeki `CFG` alanında işlenecek video, model yolu, çıktı klasörü ve çizgi konumları ayarlanır.

```python
CFG = {
    "video": "1az214.MP4",
    "model": "best400.pt",
    "out_dir": "results",
    "conf": 0.25,
    "nms_iou": 0.6,
    "lines": "v:20%,v:80%",
    "vis": False,
}
```

Çalıştırmak için:

```powershell
python main.py
```

İşlem sonunda çıktı klasöründe şu dosyalar oluşur:

- `*_annotated.mp4`
- `*_trajectory_detailed.csv`
- `*_line_crossings.csv`

## Grafik Üretimi

`fortrajectory.py` içinde `trajectory_file` ve `crossings_file` yolları, analiz edilecek CSV dosyalarına göre düzenlenir.

```python
trajectory_file = output_dir / "1az214_trajectory_detailed.csv"
crossings_file = output_dir / "1az214_line_crossings.csv"
```

Ardından:

```powershell
python fortrajectory.py
```

Üretilen grafikler; yörünge desenleri, hız dağılımı, alan dağılımı ve çizgi geçişlerinin zamana göre birikimini incelemek için kullanılır.

## Tuzaklama Verimi

L1 giriş çizgisinden geçen odun sayısını, L2 ise çıkış çizgisinden geçen odun sayısını temsil eder. Tuzaklama verimi şu şekilde hesaplanır:

```text
R = 1 - NL2 / NL1
```

Bu değer, mahmuz düzeninin odunsu malzemeyi ne ölçüde tuttuğunu karşılaştırmak için kullanılır. Farklı debi, eğim, mahmuz açısı, mahmuz sayısı ve odun besleme rejimleri bu metrik üzerinden birlikte değerlendirilebilir.

## Notlar

- Hız değerleri varsayılan olarak piksel/kare birimindedir. Fiziksel hız hesabı için kamera kalibrasyonu gerekir.
- Yoğun yığın girişlerinde ve uzun süreli örtüşmelerde takip kimliği değişebilir.
- Farklı kamera açısı, ışık koşulu veya odun tipi için modelin yeniden eğitilmesi gerekebilir.
- Hidrolik yorum yapılırken CSV çıktıları, anotasyonlu video ve deney gözlemi birlikte kontrol edilmelidir.

## Lisans

Bu repo MIT lisansı ile yayımlanmıştır.
