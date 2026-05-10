# Co dostarczyc (zrzuty ekranu) (1/2)

> **Uwaga:** dwa slajdy z listą screenshotów!

## Screenshot 1: Konfiguracja datasetu

- Widoczna struktura katalogow, `data.yaml`, przykladowe anotacje

![Screenshot 1 — Konfiguracja datasetu](images/01_dataset_config.png)

## Screenshot 2: Logi treningowe

- Widoczne: krzywe loss, metryki (precision, recall, mAP)
- Lub `results.png` wygenerowany przez YOLO

![Screenshot 2 — Logi treningowe](images/02_training_results.png)

---

# Co dostarczyc (zrzuty ekranu) (2/2)

## Screenshot 3: Wyniki ewaluacji

- Confusion matrix lub metryki `model.val()`
- mAP50 i mAP50-95

![Screenshot 3 — Wyniki ewaluacji](images/03_evaluation.png)

## Screenshot 4: Inferencja na nowych zdjeciach

- Minimum 4 zdjecia: bron - nie bron
- Widoczne bounding boxy (lub maski segmentacji) z confidence score

![Screenshot 4 — Inferencja na nowych zdjęciach](images/04_inference.png)
