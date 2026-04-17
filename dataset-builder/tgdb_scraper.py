"""
TheGunDatabase.com Scraper → YOLOv8 Dataset
=============================================
Pobiera wszystkie bronie z https://www.thegundatabase.com
wraz ze zdjęciami i pełnymi metadanymi, tworząc dataset
gotowy do trenowania modelu YOLOv8.

Struktura wyjściowa:
    tgdb_dataset/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── raw/
    │   ├── handguns/
    │   ├── rifles/
    │   ├── shotguns/
    │   ├── sniper_precision/
    │   ├── submachine_guns/
    │   └── other_weapons/
    ├── metadata/
    │   ├── weapons.json        ← pełne metadane
    │   └── weapons.csv         ← tabela do analizy
    ├── dataset.yaml            ← konfiguracja YOLOv8
    └── README.md

Instalacja:
    pip install requests beautifulsoup4 tqdm Pillow

Użycie:
    python tgdb_scraper.py
    python tgdb_scraper.py --output-dir ./moj_dataset --delay 1.5
    python tgdb_scraper.py --resume          # kontynuuj po przerwaniu
    python tgdb_scraper.py --no-download     # tylko metadane, bez obrazów
"""
from __future__ import annotations

import os
import re
import sys
import csv
import json
import time
import random
import shutil
import hashlib
import argparse
import logging
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional
from urllib.parse import urljoin, urlparse, parse_qs, unquote

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from PIL import Image

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("tgdb_scraper.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Stałe
# ─────────────────────────────────────────────────────────────
BASE_URL  = "https://www.thegundatabase.com"
CDN_BASE  = "https://cdn.qrateful.com/tgdb/weapons/main/"

# Kategorie strony → slug URL + klasa YOLO
CATEGORIES = {
    "handguns":         {"slug": "handguns",         "yolo_class": "handgun",  "yolo_id": 0},
    "rifles":           {"slug": "rifles",            "yolo_class": "rifle",    "yolo_id": 1},
    "shotguns":         {"slug": "shotguns",          "yolo_class": "shotgun",  "yolo_id": 2},
    "sniper_precision": {"slug": "sniper-precision",  "yolo_class": "sniper",   "yolo_id": 3},
    "submachine_guns":  {"slug": "submachine-guns",   "yolo_class": "smg",      "yolo_id": 4},
    "other_weapons":    {"slug": "other-weapons",     "yolo_class": "other",    "yolo_id": 5},
}

CLASS_NAMES = ["handgun", "rifle", "shotgun", "sniper", "smg", "other"]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": BASE_URL,
}


# ─────────────────────────────────────────────────────────────
# Model danych
# ─────────────────────────────────────────────────────────────
@dataclass
class WeaponRecord:
    # Identyfikatory
    slug: str = ""
    url: str = ""

    # Podstawowe info
    name: str = ""
    category: str = ""           # np. "handguns"
    yolo_class: str = ""         # np. "handgun"
    yolo_id: int = -1
    weapon_type: str = ""        # np. "Pistols", "Revolvers"
    status: str = ""             # "IN PRODUCTION" / "DISCONTINUED"
    description: str = ""

    # Producent / pochodzenie
    manufacturer: str = ""
    origin: str = ""
    production_years: str = ""

    # Specyfikacje techniczne
    calibers: list = field(default_factory=list)
    length_mm: str = ""
    barrel_mm: str = ""
    weight_g: str = ""
    action_type: str = ""
    trigger_type: str = ""
    sights_type: str = ""
    capacity: str = ""

    # Obraz
    image_url: str = ""          # oryginalny URL CDN
    local_path: str = ""         # ścieżka raw/
    yolo_image_path: str = ""    # ścieżka images/split/
    yolo_label_path: str = ""    # ścieżka labels/split/
    image_hash: str = ""
    img_width: int = 0
    img_height: int = 0
    split: str = ""              # train / val / test


# ─────────────────────────────────────────────────────────────
# HTTP Session
# ─────────────────────────────────────────────────────────────
class Session:
    def __init__(self, delay: float = 1.2, retries: int = 4):
        self.sess = requests.Session()
        self.sess.headers.update(HEADERS)
        self.delay = delay
        self.retries = retries
        self._last = 0.0

    def _wait(self):
        wait = self.delay + random.uniform(0, self.delay * 0.25) - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

    def get(self, url: str, **kw) -> Optional[requests.Response]:
        self._wait()
        for attempt in range(self.retries):
            try:
                r = self.sess.get(url, timeout=20, **kw)
                r.raise_for_status()
                return r
            except requests.HTTPError as e:
                code = e.response.status_code
                if code == 429:
                    wait = 60 * (attempt + 1)
                    log.warning(f"429 Rate limit — czekam {wait}s...")
                    time.sleep(wait)
                elif code == 404:
                    return None
                else:
                    log.warning(f"HTTP {code}: {url}")
                    return None
            except requests.RequestException as e:
                log.warning(f"Błąd (próba {attempt+1}): {e}")
                time.sleep(8 * (attempt + 1))
        return None

    def download(self, url: str, dest: Path) -> bool:
        """Pobierz plik binarny na dysk."""
        self._wait()
        try:
            r = self.sess.get(url, timeout=30, stream=True)
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=16384):
                    f.write(chunk)
            return True
        except Exception as e:
            log.debug(f"Błąd pobierania {url}: {e}")
            if dest.exists():
                dest.unlink()
            return False


# ─────────────────────────────────────────────────────────────
# Parsery HTML
# ─────────────────────────────────────────────────────────────
def _text(el) -> str:
    return el.get_text(strip=True) if el else ""


def parse_category_page(html: str, category_key: str) -> list[dict]:
    """
    Wyciągnij listę broni z strony kategorii.
    Zwraca listę słowników z: slug, name, weapon_type, calibers,
    manufacturer, origin, production_years, status, image_url, description
    """
    soup = BeautifulSoup(html, "html.parser")
    weapons = []

    # Każda broń to link <a href="/weapons/SLUG"> zawierający kartę
    for card in soup.select("a[href^='/weapons/']"):
        href = card.get("href", "")
        slug = href.split("/weapons/")[-1].rstrip("/")
        if not slug:
            continue

        rec = {
            "slug": slug,
            "url": f"{BASE_URL}/weapons/{slug}",
            "category": category_key,
            "yolo_class": CATEGORIES[category_key]["yolo_class"],
            "yolo_id": CATEGORIES[category_key]["yolo_id"],
        }

        # Nazwa
        h3 = card.find("h3")
        rec["name"] = _text(h3)

        # Status (IN PRODUCTION / DISCONTINUED)
        status_el = card.find(string=re.compile(r"IN PRODUCTION|DISCONTINUED", re.I))
        rec["status"] = status_el.strip() if status_el else ""

        # Typ broni (np. "Pistols", "Revolvers") — tekst po h3
        all_texts = [t.strip() for t in card.stripped_strings]
        # Zbieramy kaliber (zawiera zazwyczaj "mm" lub "ACP" itp.)
        calibers = []
        weapon_type = ""
        for t in all_texts:
            if t == rec["name"] or t in (rec["status"], ""):
                continue
            # Producent i kraj to ostatnie wpisy — crude heuristic
            if re.search(r'\d+x\d+|ACP|NATO|Magnum|\.\d{2,3}|Gauge|Colt|Rimfire', t, re.I):
                calibers.append(t)
            elif not weapon_type and len(t) < 40:
                weapon_type = t
        rec["weapon_type"] = weapon_type
        rec["calibers"] = calibers

        # Obraz — Next.js image URL zawiera zakodowany CDN url
        img = card.find("img")
        if img:
            src = img.get("src", "") or img.get("data-src", "")
            # Odkoduj Next.js /_next/image?url=...
            cdn_url = _extract_cdn_url(src)
            rec["image_url"] = cdn_url

        # Krótki opis (p lub ostatni długi tekst)
        p = card.find("p")
        rec["description"] = _text(p)

        weapons.append(rec)

    return weapons


def _extract_cdn_url(next_img_src: str) -> str:
    """
    Wyciągnij oryginalny URL z Next.js image proxy:
    /_next/image?url=https%3A%2F%2Fcdn...&w=3840&q=75
    """
    if "/_next/image" in next_img_src:
        parsed = urlparse(next_img_src)
        qs = parse_qs(parsed.query)
        if "url" in qs:
            return unquote(qs["url"][0])
    # Fallback: bezpośredni URL
    return next_img_src


def parse_weapon_detail(html: str, rec: WeaponRecord) -> WeaponRecord:
    """
    Wypełnij szczegółowe specs z podstrony /weapons/SLUG.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Opis (najdłuższy paragraf przed tabelą specs)
    for p in soup.find_all("p"):
        text = _text(p)
        if len(text) > 60:
            rec.description = text
            break

    # Status (IN PRODUCTION etc.)
    status_el = soup.find(string=re.compile(r"IN PRODUCTION|DISCONTINUED", re.I))
    if status_el:
        rec.status = status_el.strip()

    # Obraz — duży z właściwości
    hero_img = soup.select_one("img[alt]")
    if hero_img:
        src = hero_img.get("src", "")
        if src:
            rec.image_url = _extract_cdn_url(src)

    # Quick Overview — szukamy par label:wartość
    # Strona renderuje je jako divs / dts / dds
    spec_map = {}

    # Strategia 1: szukaj wszystkich par tekst w sąsiadujących elementach
    for el in soup.find_all(["dt", "th"]):
        label = _text(el).lower().strip().rstrip(":")
        value_el = el.find_next_sibling(["dd", "td"])
        if value_el:
            spec_map[label] = _text(value_el)

    # Strategia 2: divs z klasami zawierającymi "spec" / "stat"
    for row in soup.select("div, li"):
        children = [c for c in row.children if hasattr(c, "get_text")]
        if len(children) == 2:
            k = children[0].get_text(strip=True).lower()
            v = children[1].get_text(strip=True)
            if k and v and len(k) < 40:
                spec_map[k] = v

    # Strategia 3: tekst z sekcji "Quick Overview" przez kolejne elementy
    for heading in soup.find_all(string=re.compile("Quick Overview", re.I)):
        parent = heading.parent
        section = parent.find_next_sibling() or parent.parent
        if section:
            items = section.find_all(["p", "span", "div", "li"])
            for i in range(0, len(items) - 1, 2):
                k = _text(items[i]).lower().rstrip(":")
                v = _text(items[i+1])
                if k and v and len(k) < 40:
                    spec_map[k] = v

    # Mapowanie znanych pól
    FIELD_MAP = {
        "manufacturer":      "manufacturer",
        "origin":            "origin",
        "production":        "production_years",
        "caliber":           None,       # specjalne
        "overall length":    "length_mm",
        "barrel length":     "barrel_mm",
        "weight":            "weight_g",
        "action type":       "action_type",
        "action":            "action_type",
        "trigger type":      "trigger_type",
        "trigger":           "trigger_type",
        "sights type":       "sights_type",
        "sights":            "sights_type",
        "magazine capacity": "capacity",
        "capacity":          "capacity",
    }

    for label, value in spec_map.items():
        for key, attr in FIELD_MAP.items():
            if key in label:
                if attr is None:  # kaliber — lista
                    if value not in rec.calibers:
                        rec.calibers.append(value)
                elif attr and not getattr(rec, attr):
                    setattr(rec, attr, value)

    # Producent i kraj z linków (najdokładniejsza metoda)
    for a in soup.select("a[href*='/manufacturer/']"):
        rec.manufacturer = _text(a)
    for a in soup.select("a[href*='/country/']"):
        rec.origin = _text(a)
    for a in soup.select("a[href*='/caliber/']"):
        cal = _text(a)
        if cal and cal not in rec.calibers:
            rec.calibers.append(cal)

    return rec


# ─────────────────────────────────────────────────────────────
# Pomocnicze
# ─────────────────────────────────────────────────────────────
def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def valid_image(path: Path, min_px: int = 64) -> tuple[bool, int, int]:
    try:
        with Image.open(path) as img:
            w, h = img.size
            return (w >= min_px and h >= min_px), w, h
    except Exception:
        return False, 0, 0


def assign_split(records: list[WeaponRecord], ratios=(0.7, 0.2, 0.1)) -> None:
    """Przypisz split stratyfikowany wg klasy."""
    by_class: dict[str, list] = {}
    for r in records:
        by_class.setdefault(r.yolo_class, []).append(r)

    for cls_recs in by_class.values():
        random.shuffle(cls_recs)
        n = len(cls_recs)
        n_train = int(n * ratios[0])
        n_val   = int(n * ratios[1])
        for i, r in enumerate(cls_recs):
            if i < n_train:
                r.split = "train"
            elif i < n_train + n_val:
                r.split = "val"
            else:
                r.split = "test"


# ─────────────────────────────────────────────────────────────
# Główny scraper
# ─────────────────────────────────────────────────────────────
class TGDBScraper:
    def __init__(
        self,
        output_dir: Path,
        delay: float = 1.2,
        split_ratios: tuple = (0.7, 0.2, 0.1),
        no_download: bool = False,
        resume: bool = False,
        detail_pages: bool = True,
    ):
        self.out = output_dir
        self.no_download = no_download
        self.detail_pages = detail_pages
        self.split_ratios = split_ratios
        self.session = Session(delay=delay)
        self.records: list[WeaponRecord] = []
        self.seen_slugs: set[str] = set()
        self.seen_hashes: set[str] = set()

        # Katalogi
        for sub in ["train", "val", "test"]:
            (self.out / "images" / sub).mkdir(parents=True, exist_ok=True)
            (self.out / "labels" / sub).mkdir(parents=True, exist_ok=True)
        for cat in CATEGORIES:
            (self.out / "raw" / cat).mkdir(parents=True, exist_ok=True)
        (self.out / "metadata").mkdir(parents=True, exist_ok=True)

        self.meta_path = self.out / "metadata" / "weapons.json"

        if resume and self.meta_path.exists():
            self._load_state()

    def _load_state(self):
        with open(self.meta_path, encoding="utf-8") as f:
            data = json.load(f)
        for d in data.get("records", []):
            r = WeaponRecord(**d)
            self.records.append(r)
            self.seen_slugs.add(r.slug)
            if r.image_hash:
                self.seen_hashes.add(r.image_hash)
        log.info(f"Wczytano {len(self.records)} poprzednich rekordów (resume).")

    def _save_state(self):
        counts = {}
        for r in self.records:
            counts[r.yolo_class] = counts.get(r.yolo_class, 0) + 1
        data = {
            "total": len(self.records),
            "class_counts": counts,
            "records": [asdict(r) for r in self.records],
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # ── Krok 1: zbierz listę broni z kategorii ──────────────────────
    def scrape_category_list(self, category_key: str) -> list[dict]:
        slug = CATEGORIES[category_key]["slug"]
        url  = f"{BASE_URL}/category/{slug}"
        log.info(f"Pobieram listę kategorii: {url}")

        r = self.session.get(url)
        if not r:
            log.error(f"Nie można pobrać kategorii: {url}")
            return []

        weapons = parse_category_page(r.text, category_key)
        log.info(f"  Znaleziono {len(weapons)} broni w '{category_key}'")
        return weapons

    # ── Krok 2: pobierz szczegóły każdej broni ──────────────────────
    def scrape_weapon_detail(self, base: dict) -> WeaponRecord:
        rec = WeaponRecord(
            slug=base["slug"],
            url=base["url"],
            name=base.get("name", ""),
            category=base.get("category", ""),
            yolo_class=base.get("yolo_class", ""),
            yolo_id=base.get("yolo_id", -1),
            weapon_type=base.get("weapon_type", ""),
            calibers=base.get("calibers", []),
            status=base.get("status", ""),
            description=base.get("description", ""),
            image_url=base.get("image_url", ""),
        )

        if not self.detail_pages:
            return rec

        r = self.session.get(rec.url)
        if r:
            rec = parse_weapon_detail(r.text, rec)

        return rec

    # ── Krok 3: pobierz obraz ────────────────────────────────────────
    def download_image(self, rec: WeaponRecord) -> bool:
        if self.no_download or not rec.image_url:
            return False

        # Ustal rozszerzenie z URL
        path_part = urlparse(rec.image_url).path
        ext = os.path.splitext(path_part)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
            ext = ".jpg"

        fname = f"{rec.slug}{ext}"
        dest  = self.out / "raw" / rec.category / fname

        # Pomiń jeśli już pobrany
        if dest.exists():
            ok, w, h = valid_image(dest)
            if ok:
                rec.local_path = str(dest)
                rec.image_hash = md5(dest)
                rec.img_width  = w
                rec.img_height = h
                return True

        # Pobierz
        ok = self.session.download(rec.image_url, dest)
        if not ok:
            # Fallback: spróbuj zbudować URL z CDN slug
            fallback = f"{CDN_BASE}{rec.slug}.jpg"
            if fallback != rec.image_url:
                ok = self.session.download(fallback, dest)

        if not ok or not dest.exists():
            return False

        valid, w, h = valid_image(dest)
        if not valid:
            dest.unlink(missing_ok=True)
            return False

        img_hash = md5(dest)
        if img_hash in self.seen_hashes:
            dest.unlink(missing_ok=True)
            return False

        self.seen_hashes.add(img_hash)
        rec.local_path = str(dest)
        rec.image_hash = img_hash
        rec.img_width  = w
        rec.img_height = h
        return True

    # ── Główna pętla ─────────────────────────────────────────────────
    def run(self):
        log.info("=== TheGunDatabase.com Scraper START ===")

        for cat_key in CATEGORIES:
            # Pobierz listę broni z kategorii
            weapons_raw = self.scrape_category_list(cat_key)

            for w in tqdm(weapons_raw, desc=f"Scraping {cat_key}"):
                if w["slug"] in self.seen_slugs:
                    continue

                self.seen_slugs.add(w["slug"])

                # Pobierz szczegóły
                rec = self.scrape_weapon_detail(w)

                # Pobierz obraz
                img_ok = self.download_image(rec)
                if not img_ok and not self.no_download:
                    log.debug(f"Brak obrazu dla: {rec.slug}")

                self.records.append(rec)

            # Zapisz po każdej kategorii
            self._save_state()
            log.info(f"Kategoria '{cat_key}' gotowa. Łącznie: {len(self.records)} broni.")

        log.info(f"\nScraping zakończony. Łącznie {len(self.records)} broni.")
        self._print_stats()

    def _print_stats(self):
        counts: dict[str, int] = {}
        for r in self.records:
            counts[r.yolo_class] = counts.get(r.yolo_class, 0) + 1
        log.info("Podsumowanie klas:")
        for cls in CLASS_NAMES:
            log.info(f"  {cls:12s}: {counts.get(cls, 0)}")

    # ── Budowanie struktury YOLOv8 ────────────────────────────────────
    def build_yolo_dataset(self):
        log.info("Buduję strukturę YOLOv8...")

        # Przypisz split
        records_with_img = [r for r in self.records if r.local_path and Path(r.local_path).exists()]
        log.info(f"Obrazów z plikiem na dysku: {len(records_with_img)}")
        assign_split(records_with_img, self.split_ratios)

        for rec in tqdm(records_with_img, desc="Kopiowanie do YOLO"):
            src = Path(rec.local_path)
            ext = src.suffix.lower()
            fname_stem = rec.slug

            dst_img   = self.out / "images" / rec.split / f"{fname_stem}{ext}"
            dst_label = self.out / "labels" / rec.split / f"{fname_stem}.txt"

            shutil.copy2(src, dst_img)
            rec.yolo_image_path = str(dst_img)
            rec.yolo_label_path = str(dst_label)

            # Plik etykiety YOLO: class_id cx cy w h (bbox = cały obraz = placeholder)
            # Do precyzyjnego trenowania detekcji obiektów wymagana ręczna anotacja.
            # Do klasyfikacji (classify) można trenować wprost na tych labelkach.
            with open(dst_label, "w") as lf:
                lf.write(f"{rec.yolo_id} 0.500 0.500 1.000 1.000\n")

        self._save_state()
        log.info("Struktura YOLOv8 gotowa.")

    # ── Pliki konfiguracyjne ──────────────────────────────────────────
    def write_yaml(self):
        counts: dict[str, int] = {}
        for r in self.records:
            counts[r.yolo_class] = counts.get(r.yolo_class, 0) + 1

        yaml_path = self.out / "dataset.yaml"
        abs_out = str(self.out.resolve())

        lines = [
            "# YOLOv8 Dataset — TheGunDatabase.com",
            f"# Wygenerowano automatycznie przez tgdb_scraper.py",
            f"",
            f"path: {abs_out}",
            f"",
            f"train: images/train",
            f"val:   images/val",
            f"test:  images/test",
            f"",
            f"nc: {len(CLASS_NAMES)}  # liczba klas",
            f"",
            "names:",
        ]
        for i, name in enumerate(CLASS_NAMES):
            lines.append(f"  {i}: {name}  # {counts.get(name, 0)} obrazów")

        lines += [
            "",
            "# ─────────────────────────────────────────────",
            "# Trening YOLOv8 — przykładowe komendy:",
            "# ─────────────────────────────────────────────",
            "#",
            "# pip install ultralytics",
            "#",
            "# Detekcja obiektów (wymaga anotacji bbox):",
            "# yolo detect train data=dataset.yaml model=yolov8n.pt epochs=100 imgsz=640",
            "#",
            "# Klasyfikacja (działa z obecnymi labelkami):",
            "# yolo classify train data=dataset.yaml model=yolov8n-cls.pt epochs=100 imgsz=224",
            "#",
            "# Walidacja:",
            "# yolo detect val data=dataset.yaml model=runs/detect/train/weights/best.pt",
            "#",
            "# Predykcja na nowym zdjęciu:",
            "# yolo detect predict model=best.pt source=foto.jpg",
        ]

        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        log.info(f"Zapisano: {yaml_path}")

    def write_csv(self):
        csv_path = self.out / "metadata" / "weapons.csv"
        fields = [
            "slug", "name", "yolo_class", "yolo_id", "category",
            "weapon_type", "status", "manufacturer", "origin",
            "production_years", "calibers", "length_mm", "barrel_mm",
            "weight_g", "action_type", "capacity", "split",
            "img_width", "img_height", "image_url", "local_path",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for rec in self.records:
                row = asdict(rec)
                row["calibers"] = "; ".join(rec.calibers) if rec.calibers else ""
                w.writerow(row)
        log.info(f"Zapisano: {csv_path}")

    def write_readme(self):
        counts: dict[str, int] = {}
        split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
        for r in self.records:
            counts[r.yolo_class] = counts.get(r.yolo_class, 0) + 1
            if r.split:
                split_counts[r.split] = split_counts.get(r.split, 0) + 1

        rows = "\n".join(
            f"| {i} | {name:10s} | {counts.get(name, 0):6d} |"
            for i, name in enumerate(CLASS_NAMES)
        )
        total = len(self.records)

        readme = f"""# TGDB Firearms Dataset

**Źródło:** [TheGunDatabase.com](https://www.thegundatabase.com)  
**Wygenerowano:** tgdb_scraper.py

---

## Statystyki

| ID | Klasa      | Liczba |
|----|------------|--------|
{rows}
| — | **ŁĄCZNIE** | **{total}** |

**Podział datasetu:**
| Split | Obrazów |
|-------|---------|
| train | {split_counts.get('train', 0)} |
| val   | {split_counts.get('val', 0)} |
| test  | {split_counts.get('test', 0)} |

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
yolo classify train \\
    data=tgdb_dataset/dataset.yaml \\
    model=yolov8n-cls.pt \\
    epochs=100 imgsz=224 batch=32

# Detekcja obiektów (po ręcznej anotacji bbox):
yolo detect train \\
    data=tgdb_dataset/dataset.yaml \\
    model=yolov8n.pt \\
    epochs=100 imgsz=640 batch=16

# Walidacja:
yolo detect val \\
    data=tgdb_dataset/dataset.yaml \\
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
"""
        readme_path = self.out / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme)
        log.info(f"Zapisano: {readme_path}")

    # ── Uruchomienie wszystkiego ──────────────────────────────────────
    def build_all(self):
        self.run()
        self.build_yolo_dataset()
        self.write_yaml()
        self.write_csv()
        self.write_readme()
        log.info("=== WSZYSTKO GOTOWE ===")
        log.info(f"Dataset zapisany w: {self.out.resolve()}")


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Scraper TheGunDatabase.com → YOLOv8 dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output-dir", default="./tgdb_dataset",
                   help="Katalog wyjściowy (domyślnie: ./tgdb_dataset)")
    p.add_argument("--delay", type=float, default=1.2,
                   help="Opóźnienie między requestami w sekundach (domyślnie: 1.2)")
    p.add_argument("--split", nargs=3, type=float, default=[0.7, 0.2, 0.1],
                   metavar=("TRAIN", "VAL", "TEST"),
                   help="Podział train/val/test (domyślnie: 0.7 0.2 0.1)")
    p.add_argument("--no-download", action="store_true",
                   help="Zbierz tylko metadane, nie pobieraj obrazów")
    p.add_argument("--no-detail", action="store_true",
                   help="Pomiń odwiedzanie stron szczegółowych (szybsze, mniej metadanych)")
    p.add_argument("--resume", action="store_true",
                   help="Kontynuuj poprzednie pobieranie (wczytaj weapons.json)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    split = tuple(args.split)
    assert abs(sum(split) - 1.0) < 0.01, "Suma split musi wynosić 1.0"

    scraper = TGDBScraper(
        output_dir=Path(args.output_dir),
        delay=args.delay,
        split_ratios=split,
        no_download=args.no_download,
        resume=args.resume,
        detail_pages=not args.no_detail,
    )

    scraper.build_all()
