# TGDB Firearms Dataset

**Źródło:** [TheGunDatabase.com](https://www.thegundatabase.com)  
**Wygenerowano:** tgdb_scraper.py

---

## Statystyki

| ID | Klasa      | Liczba |
|----|------------|--------|
| 0 | handgun    |     98 |
| 1 | rifle      |    112 |
| 2 | shotgun    |     50 |
| 3 | sniper     |      6 |
| 4 | smg        |      5 |
| 5 | other      |      0 |
| — | **ŁĄCZNIE** | **271** |

**Podział datasetu:**
| Split | Obrazów |
|-------|---------|
| train | 111 |
| val   | 30 |
| test  | 19 |

---

## Struktura katalogów

```
tgdb_dataset/
├── images/
│   ├── train/          ← obrazy treningowe
│   ├── val/            ← walidacja
│   └── test/           ← test
├── labels/
│   ├── train/          ← etykiety YOLO (.txt)
│   ├── val/
│   └── test/
├── raw/                ← oryginalne pobrane obrazy
│   ├── handguns/
│   ├── rifles/
│   ├── shotguns/
│   ├── sniper_precision/
│   ├── submachine_guns/
│   └── other_weapons/
├── metadata/
│   ├── weapons.json    ← pełne metadane (JSON)
│   └── weapons.csv     ← tabela (CSV)
├── dataset.yaml        ← konfiguracja YOLOv8
└── README.md
```

---

## Klasy YOLO

| ID | Klasa   | Opis                                    |
|----|---------|-----------------------------------------|
| 0  | handgun | Pistolety i rewolwery                   |
| 1  | rifle   | Karabiny (szturmowe, bojowe, półauto)   |
| 2  | shotgun | Strzelby wszelkich typów                |
| 3  | sniper  | Karabiny snajperskie i precyzyjne       |
| 4  | smg     | Pistolety maszynowe / karabiny maszynowe|
| 5  | other   | Pozostałe kategorie                     |

---

## Trening YOLOv8

```bash
pip install ultralytics

# Klasyfikacja (działa od razu):
yolo classify train \
    data=tgdb_dataset/dataset.yaml \
    model=yolov8n-cls.pt \
    epochs=100 imgsz=224 batch=32

# Detekcja obiektów (po ręcznej anotacji bbox):
yolo detect train \
    data=tgdb_dataset/dataset.yaml \
    model=yolov8n.pt \
    epochs=100 imgsz=640 batch=16

# Walidacja:
yolo detect val \
    data=tgdb_dataset/dataset.yaml \
    model=runs/detect/train/weights/best.pt
```

---

## Metadane (weapons.json)

Każdy rekord zawiera:
- `slug` / `url` — identyfikator i link do strony
- `name` — pełna nazwa broni
- `yolo_class` / `yolo_id` — klasa i ID dla YOLO
- `weapon_type` — podtyp (np. "Pistols", "Revolvers")
- `status` — produkcja aktywna / wycofana
- `manufacturer` / `origin` — producent i kraj
- `production_years` — lata produkcji
- `calibers` — lista kalibrów
- `length_mm` / `barrel_mm` / `weight_g` — wymiary
- `action_type` / `trigger_type` / `capacity` — specs mechaniczne
- `image_url` — oryginalny URL CDN
- `img_width` / `img_height` — rozdzielczość
- `split` — przynależność train/val/test

---

## Uwagi

1. **Anotacje bbox:** Pliki `.txt` w `labels/` zawierają placeholder  
   `class_id 0.5 0.5 1.0 1.0` (bbox = cały obraz).  
   Do trenowania detekcji obiektów wymagana precyzyjna anotacja —  
   zalecane narzędzia: [Label Studio](https://labelstud.io/), [Roboflow](https://roboflow.com/), CVAT.

2. **Klasyfikacja:** Do trenowania modelu klasyfikującego typ broni  
   (nie lokalizującego) obecne etykiety wystarczą — użyj `yolo classify`.

3. **Licencja:** Obrazy i dane należą do TheGunDatabase.com.  
   Sprawdź [Terms of Service](https://www.thegundatabase.com/terms) przed użyciem komercyjnym.
