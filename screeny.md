# Co dostarczyc (zrzuty ekranu) (1/2)

> **Uwaga:** dwa slajdy z listą screenshotów!

## Screenshot 1: Konfiguracja datasetu

- Widoczna struktura katalogow, `data.yaml`, przykladowe anotacje

### Struktura katalogów datasetu

```
data/
├── data.yaml
├── train/
│   ├── images/      # 3 839 plików .jpg
│   │   ├── 00001.jpg
│   │   ├── 00002.jpg
│   │   └── ...
│   └── labels/      # 3 839 plików .txt (anotacje YOLO)
│       ├── 00001.txt
│       ├── 00002.txt
│       └── ...
└── test/
    ├── images/      # 259 plików .jpg
    │   └── ...
    └── labels/      # 259 plików .txt
        └── ...
```

### Plik `data/data.yaml`

```yaml
train: train/images
val: test/images
test: test/images

nc: 2
names: [ 'guns', 'knife' ]

roboflow:
  workspace: joao-assalim-xmovq
  project: weapon-2
  version: 2
  license: CC BY 4.0
  url: https://universe.roboflow.com/joao-assalim-xmovq/weapon-2/dataset/2
```

### Przykładowy plik anotacji (format YOLO)

Każda linia w pliku `.txt` opisuje jeden obiekt:
`class_id x_center y_center width height` (wszystkie wartości znormalizowane 0–1).

```text
0 0.512 0.487 0.234 0.156
1 0.732 0.624 0.118 0.092
```

### Przykładowe anotacje wizualnie (z notebooka)

Próbka obrazów treningowych z naniesionymi bounding boxami i etykietami klas (`guns`, `knife`).

![Screenshot 1 — Konfiguracja datasetu](images/01_dataset_config.png)

---

## Screenshot 2: Logi treningowe

- Widoczne: krzywe loss, metryki (precision, recall, mAP)
- Lub `results.png` wygenerowany przez YOLO

`results.png` wygenerowany automatycznie przez YOLO — krzywe `train/box_loss`, `train/cls_loss`, `train/dfl_loss`, `val/*_loss` oraz metryki `precision`, `recall`, `mAP50`, `mAP50-95` w funkcji epoki.

![Screenshot 2 — Logi treningowe](images/02_training_results.png)

---

# Co dostarczyc (zrzuty ekranu) (2/2)

## Screenshot 3: Wyniki ewaluacji

- Confusion matrix lub metryki `model.val()`
- mAP50 i mAP50-95

Wyniki `model.val()` dla najlepszego modelu (`best.pt`) na zbiorze walidacyjnym (259 obrazów, 481 obiektów):

| Klasa   | Precision | Recall | **mAP50**  | **mAP50-95** |
|---------|-----------|--------|------------|--------------|
| `guns`  | 0.868     | 0.650  | 0.776      | 0.588        |
| `knife` | 0.905     | 0.994  | 0.974      | 0.793        |
| **all** | 0.887     | 0.822  | **0.875**  | **0.690**    |

Confusion matrix oraz wartości mAP50 / mAP50-95 zwrócone przez `model.val()`:

![Screenshot 3 — Wyniki ewaluacji](images/03_evaluation.png)

---

## Screenshot 4: Inferencja na nowych zdjeciach

- Minimum 4 zdjecia: bron - nie bron
- Widoczne bounding boxy (lub maski segmentacji) z confidence score

6 zdjęć testowych (4 z bronią + 2 nie-broń) z naniesionymi bounding boxami, etykietami klas oraz wartościami confidence score (próg `conf = 0.25`):

![Screenshot 4 — Inferencja na nowych zdjęciach](images/04_inference.png)
