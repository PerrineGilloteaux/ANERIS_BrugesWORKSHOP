"""
AEIS_MAC_libforminka
--------------------
Outils pour entraîner un classifieur sur les photos MINKA annotées (niveau
famille) : ResNet18 (ImageNet) ou BioCLIP / BioCLIP 2 (OpenCLIP, pré-entraîné
sur l'arbre du vivant), générer des CAM, raffiner avec SAM, et filtrer les
observations situées dans des zones géographiques (polygones KML, p. ex. sites eDNA).
"""
import os
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import glob
import math
import urllib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


# =============================================================================
# 1. Exclusion géographique à partir d'un KML (polygones Google Earth / Maps)
# =============================================================================
_KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}


def _parse_kml_coordinates(text):
    """Convertit le texte d'une balise <coordinates> en array (N, 2) [lon, lat]."""
    pts = []
    for token in text.strip().split():
        parts = token.split(",")
        if len(parts) >= 2:
            pts.append((float(parts[0]), float(parts[1])))
    return np.array(pts, dtype=float)


def load_kml_polygons(kml_path):
    """
    Lit un fichier KML et renvoie la liste des polygones qu'il contient.

    Chaque polygone est un dict :
        {"name": str, "outer": array (N,2) lon/lat, "holes": [array (M,2), ...]}
    Gère les Placemark contenant un Polygon ou un MultiGeometry de Polygons.
    """
    tree = ET.parse(kml_path)
    root = tree.getroot()
    polygons = []
    for placemark in root.iter(f"{{{_KML_NS['kml']}}}Placemark"):
        name_el = placemark.find("kml:name", _KML_NS)
        name = name_el.text.strip() if name_el is not None and name_el.text else "sans_nom"
        for poly in placemark.iter(f"{{{_KML_NS['kml']}}}Polygon"):
            outer_el = poly.find("kml:outerBoundaryIs/kml:LinearRing/kml:coordinates", _KML_NS)
            if outer_el is None or not outer_el.text:
                continue
            outer = _parse_kml_coordinates(outer_el.text)
            holes = [
                _parse_kml_coordinates(h.text)
                for h in poly.findall("kml:innerBoundaryIs/kml:LinearRing/kml:coordinates", _KML_NS)
                if h.text
            ]
            polygons.append({"name": name, "outer": outer, "holes": holes})
    if not polygons:
        raise ValueError(f"Aucun polygone trouvé dans {kml_path}")
    return polygons


def _points_in_ring(lon, lat, ring):
    """Test point-dans-polygone vectorisé (ray casting). lon/lat : arrays 1D."""
    inside = np.zeros(lon.shape, dtype=bool)
    xs, ys = ring[:, 0], ring[:, 1]
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi, xj, yj = xs[i], ys[i], xs[j], ys[j]
        crosses = (yi > lat) != (yj > lat)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_inter = (xj - xi) * (lat - yi) / (yj - yi) + xi
        inside ^= crosses & (lon < x_inter)
        j = i
    return inside


def _distance_to_ring_m(lon, lat, ring):
    """Distance minimale (m) de chaque point au contour (projection équirectangulaire locale)."""
    lat0 = np.deg2rad(np.mean(ring[:, 1]))
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * np.cos(lat0)
    px, py = lon * m_per_deg_lon, lat * m_per_deg_lat
    rx, ry = ring[:, 0] * m_per_deg_lon, ring[:, 1] * m_per_deg_lat
    dmin = np.full(lon.shape, np.inf)
    for k in range(len(ring) - 1):
        ax, ay, bx, by = rx[k], ry[k], rx[k + 1], ry[k + 1]
        dx, dy = bx - ax, by - ay
        seg_len2 = dx * dx + dy * dy
        if seg_len2 == 0:
            t = np.zeros_like(px)
        else:
            t = np.clip(((px - ax) * dx + (py - ay) * dy) / seg_len2, 0, 1)
        d = np.hypot(px - (ax + t * dx), py - (ay + t * dy))
        dmin = np.minimum(dmin, d)
    return dmin


def points_in_kml_zones(lat, lon, polygons, buffer_m=0.0):
    """
    Pour chaque point (lat, lon), renvoie le nom de la zone qui le contient
    (ou None). `buffer_m` élargit chaque polygone de cette distance en mètres.
    Les points sans coordonnées (NaN) renvoient None.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    zone = np.full(lat.shape, None, dtype=object)
    valid = ~(np.isnan(lat) | np.isnan(lon))
    for poly in polygons:
        outer = poly["outer"]
        hit = _points_in_ring(lon, lat, outer)
        for hole in poly["holes"]:
            hit &= ~_points_in_ring(lon, lat, hole)
        if buffer_m > 0:
            hit |= _distance_to_ring_m(lon, lat, outer) <= buffer_m
        hit &= valid
        zone[hit & (zone == None)] = poly["name"]  # noqa: E711
    return zone


def exclude_kml_zones(df, kml_path, lat_col="latitude", lon_col="longitude",
                      buffer_m=0.0, drop_missing_coords=False,
                      exclude_obscured=False, obscured_col="obscured",
                      verbose=True, return_excluded=False):
    """
    Exclut d'un DataFrame (df_obs ou df_photos) les lignes situées dans les
    polygones du fichier KML (par exemple les sites d'échantillonnage eDNA),
    afin de les garder comme jeu de test indépendant.

    Paramètres
    ----------
    df : DataFrame avec colonnes latitude / longitude
    kml_path : chemin du fichier .kml
    buffer_m : marge (mètres) ajoutée autour de chaque polygone
    drop_missing_coords : si True, retire aussi les lignes sans coordonnées
    exclude_obscured : si True, retire les observations dont les coordonnées
        sont floutées (colonne `obscured`), car leur position réelle peut
        tomber dans une zone sans que l'on puisse le détecter
    return_excluded : si True, renvoie (df_kept, df_excluded) ; df_excluded
        contient une colonne `kml_zone` avec le nom de la zone.
    """
    polygons = load_kml_polygons(kml_path)
    zones = points_in_kml_zones(df[lat_col].values, df[lon_col].values, polygons, buffer_m)
    in_zone = pd.Series(zones != None, index=df.index)  # noqa: E711

    missing = df[lat_col].isna() | df[lon_col].isna()
    to_drop = in_zone.copy()
    if drop_missing_coords:
        to_drop |= missing
    if exclude_obscured and obscured_col in df.columns:
        to_drop |= df[obscured_col].fillna(False).astype(bool)

    df_kept = df.loc[~to_drop].copy()
    df_excluded = df.loc[to_drop].copy()
    df_excluded["kml_zone"] = zones[to_drop.values]

    if verbose:
        print(f"Polygones chargés : {[p['name'] for p in polygons]}")
        print(f"Lignes dans les zones KML : {int(in_zone.sum())} / {len(df)}")
        for name, count in pd.Series(zones[in_zone.values]).value_counts().items():
            print(f"   - {name}: {count}")
        if drop_missing_coords:
            print(f"Lignes sans coordonnées retirées : {int(missing.sum())}")
        if exclude_obscured and obscured_col in df.columns:
            print(f"Lignes à coordonnées floutées retirées : "
                  f"{int(df[obscured_col].fillna(False).astype(bool).sum())}")
        print(f"Lignes conservées : {len(df_kept)}")

    if return_excluded:
        return df_kept, df_excluded
    return df_kept


# =============================================================================
# 2. Préparation des données MINKA
# =============================================================================
def is_valid_photo_url(url):
    """Rejette NaN et l'image par défaut de MINKA (attachment_defaults)."""
    return (isinstance(url, str) and url.startswith("http")
            and "attachment_defaults" not in url)


def read_image(path_or_url, timeout=20, max_retries=3, backoff=2.0, verbose=True):
    """
    Lit une image (fichier local ou URL) et la renvoie en BGR (numpy) ou None.
    Les URL sont retentées `max_retries` fois.
    """
    if not isinstance(path_or_url, str):
        return None
    if not path_or_url.startswith("http"):
        img = cv2.imread(path_or_url, cv2.IMREAD_COLOR)
        if img is None and verbose:
            print(f"Impossible de lire {path_or_url}")
        return img
    for attempt in range(max_retries):
        try:
            response = requests.get(path_or_url, timeout=timeout)
            response.raise_for_status()
            np_array = np.frombuffer(response.content, dtype=np.uint8)
            image_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError("Échec du décodage de l'image.")
            return image_bgr
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"Tentative {attempt + 1}/{max_retries} échouée pour {path_or_url} : {e}")
            if attempt < max_retries - 1:
                time.sleep(backoff * (attempt + 1))
    return None


# Compatibilité avec le code existant
def read_image_from_url(url, timeout=20, max_retries=3):
    return read_image(url, timeout=timeout, max_retries=max_retries)


def download_images(urls, filenames, cache_dir="minka_images", n_workers=8,
                    timeout=20, max_retries=3):
    """
    Télécharge les images une seule fois dans `cache_dir` (déjà présentes :
    ignorées). Renvoie la liste des chemins locaux, None si échec.
    Évite de re-télécharger 11 000 images à chaque epoch.
    """
    os.makedirs(cache_dir, exist_ok=True)
    local_paths = [os.path.join(cache_dir, f) for f in filenames]

    def _fetch(i):
        path = local_paths[i]
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return i, path
        url = urls[i]
        if not is_valid_photo_url(url):
            return i, None
        for attempt in range(max_retries):
            try:
                r = requests.get(url, timeout=timeout)
                r.raise_for_status()
                if cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR) is None:
                    raise ValueError("image illisible")
                with open(path, "wb") as f:
                    f.write(r.content)
                return i, path
            except Exception:  # noqa: BLE001
                time.sleep(2.0 * (attempt + 1))
        return i, None

    results = [None] * len(urls)
    done = 0
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for fut in as_completed([ex.submit(_fetch, i) for i in range(len(urls))]):
            i, path = fut.result()
            results[i] = path
            done += 1
            if done % 500 == 0 or done == len(urls):
                print(f"  {done}/{len(urls)} images traitées")
    n_fail = sum(p is None for p in results)
    print(f"Téléchargement terminé : {len(urls) - n_fail} OK, {n_fail} échecs")
    return results


# =============================================================================
# 3. Dataset / entraînement
# =============================================================================
def get_train_transform(size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform(size=224):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class CustomDataset(Dataset):
    """Images (chemins locaux ou URL) + labels numériques.
    Une image illisible renvoie None ; elle est écartée par `collate_skip_none`
    (au lieu d'être étiquetée arbitrairement avec une classe)."""

    def __init__(self, image_paths, labels, transform=None):
        assert len(image_paths) == len(labels), "image_paths et labels n'ont pas la même longueur"
        self.image_paths = list(image_paths)
        self.labels = [int(l) for l in labels]
        self.transform = transform or get_eval_transform()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_bgr = read_image(self.image_paths[idx])
        if image_bgr is None:
            print(f"Warning: image ignorée (index {idx}) : {self.image_paths[idx]}")
            return None
        image_pil = transforms.ToPILImage()(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        return self.transform(image_pil), torch.tensor(self.labels[idx], dtype=torch.long)


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def build_resnet18(num_classes, pretrained=True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for batch in loader:
        if batch is None:
            continue
        images, labels = batch[0].to(device), batch[1].to(device)
        outputs = model(images)
        if criterion is not None:
            loss_sum += criterion(outputs, labels).item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return (loss_sum / max(total, 1)), correct / max(total, 1)


def train_classifier(image_paths, labels, num_classes, epochs=10, suffix="",
                     val_image_paths=None, val_labels=None, batch_size=16,
                     lr=1e-3, output_dir="models", num_workers=4,
                     class_names=None, class_weights=None,
                     resume_from=None, resume_lr=None, **kwargs):
    """
    Entraîne un ResNet18 (pré-entraîné ImageNet).
    - sauvegarde un checkpoint à chaque epoch et le meilleur modèle (validation)
      dans `output_dir` avec le suffixe `suffix`
    - `class_weights` : tenseur optionnel pour compenser le déséquilibre
    - `resume_from` : chemin d'un checkpoint pour reprendre l'entraînement.
      `epochs` est alors le nombre TOTAL d'epochs (ex. 20 après 10 déjà faites).
      Le learning rate repart de `resume_lr` (défaut : lr / 3) avec une nouvelle
      décroissance cosinus sur les epochs restantes (le cosinus précédent
      avait amené le lr à ~0).
    Les kwargs inconnus (ex. ancien `label_none`) sont ignorés.
    """
    if kwargs:
        print(f"Arguments ignorés : {list(kwargs)}")
    os.makedirs(output_dir, exist_ok=True)
    tag = f"_{suffix}" if suffix else ""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    train_loader = DataLoader(
        CustomDataset(image_paths, labels, get_train_transform()),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=collate_skip_none, pin_memory=device.type == "cuda")
    val_loader = None
    if val_image_paths is not None and len(val_image_paths) > 0:
        val_loader = DataLoader(
            CustomDataset(val_image_paths, val_labels, get_eval_transform()),
            batch_size=batch_size, shuffle=False, num_workers=num_workers,
            collate_fn=collate_skip_none)

    model = build_resnet18(num_classes, pretrained=resume_from is None).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    start_epoch, best_acc, history = 0, -1.0, []
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        if ckpt.get("num_classes") not in (None, num_classes):
            raise ValueError(f"Le checkpoint a {ckpt['num_classes']} classes, "
                             f"mais num_classes={num_classes}")
        if class_names is not None and ckpt.get("class_names") not in (None, list(class_names)):
            raise ValueError("La liste des familles diffère de celle du checkpoint : "
                             "les labels ne correspondraient plus.")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        history = list(ckpt.get("history") or [])
        scores = [h.get("val_acc", h.get("train_acc", -1.0)) for h in history]
        best_acc = max(scores) if scores else -1.0
        if start_epoch >= epochs:
            print(f"Le checkpoint est déjà à l'epoch {start_epoch} >= epochs={epochs} : rien à faire.")
            return model, history
        lr = resume_lr if resume_lr is not None else lr / 3
        for g in optimizer.param_groups:
            g["lr"] = lr
            g["initial_lr"] = lr
        print(f"Reprise depuis {resume_from} : epochs {start_epoch + 1} à {epochs}, "
              f"lr={lr:g}, meilleure accuracy actuelle {best_acc:.3f}")
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs - start_epoch, 1))

    for epoch in range(start_epoch, epochs):
        model.train()
        running_loss, correct, total, n_batches = 0.0, 0, 0, 0
        print(f"Epoch {epoch + 1}/{epochs}")
        for i, batch in enumerate(train_loader):
            if batch is None:
                continue
            images, targets = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)
            if i % 50 == 49:
                print(f"  Batch {i + 1}/{len(train_loader)}, "
                      f"Loss: {running_loss / n_batches:.4f}, Acc: {correct / max(total, 1):.3f}")
        scheduler.step()
        train_loss, train_acc = running_loss / max(n_batches, 1), correct / max(total, 1)
        record = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc}
        msg = f"  -> train loss {train_loss:.4f}, train acc {train_acc:.3f}"
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, device, criterion)
            record.update(val_loss=val_loss, val_acc=val_acc)
            msg += f" | val loss {val_loss:.4f}, val acc {val_acc:.3f}"
        print(msg)
        history.append(record)

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": train_loss,
            "num_classes": num_classes,
            "class_names": list(class_names) if class_names is not None else None,
            "history": history,
            "arch": "resnet18",
        }
        torch.save(ckpt, os.path.join(output_dir, f"checkpoint{tag}_epoch_{epoch}.pth"))
        score = record.get("val_acc", train_acc)
        if score > best_acc:
            best_acc = score
            torch.save(ckpt, os.path.join(output_dir, f"best{tag}.pth"))

    torch.save(ckpt, os.path.join(output_dir, f"resnet18_final{tag}.pth"))
    print(f"Meilleure accuracy ({'val' if val_loader else 'train'}) : {best_acc:.3f}")
    return model, history


def load_model_from_checkpoint(checkpoint_path, num_classes=None, device="cpu"):
    """Recharge un checkpoint produit par train_classifier (ResNet18) ou
    train_bioclip_classifier (BioCLIP). Renvoie (model, optimizer, epoch, loss,
    class_names). Pour BioCLIP, optimizer vaut None (non nécessaire en inférence)
    et le modèle porte son propre prétraitement dans `model.eval_transform`."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("arch", "resnet18") != "resnet18":
        model = load_bioclip_classifier(checkpoint_path, device=device, checkpoint=checkpoint)
        return (model, None, checkpoint["epoch"], checkpoint.get("loss"),
                checkpoint.get("class_names"))
    num_classes = num_classes or checkpoint.get("num_classes")
    model = build_resnet18(num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval_transform = get_eval_transform()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return (model, optimizer, checkpoint["epoch"], checkpoint["loss"],
            checkpoint.get("class_names"))


# =============================================================================
# 4. Prédiction / évaluation
# =============================================================================
def _model_transform(model, transform=None):
    """Prétraitement à utiliser pour un modèle : celui passé en argument, sinon
    celui attaché au modèle (BioCLIP), sinon celui du ResNet (ImageNet)."""
    if transform is not None:
        return transform
    return getattr(model, "eval_transform", None) or get_eval_transform()


def _preprocess(image_bgr, size=224, transform=None):
    """Prétraitement IDENTIQUE à la validation pendant l'entraînement
    (PIL + Resize avec anti-aliasing). L'ancienne version utilisait cv2.resize
    sans anti-aliasing, ce qui faisait perdre ~5 points d'accuracy en évaluation
    (0.58 au lieu de 0.63 sur la validation)."""
    transform = transform or get_eval_transform(size)
    image_pil = transforms.ToPILImage()(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    return transform(image_pil).unsqueeze(0)


def predict_class(model, image_path, device, transform=None):
    img = read_image(image_path)
    if img is None:
        return None
    model.eval()
    with torch.no_grad():
        output = model(_preprocess(img, transform=_model_transform(model, transform)).to(device))
    return output.argmax(1).item()


class _PathDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = list(paths), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = read_image(self.paths[idx], verbose=False)
        if img is None:
            return None
        pil = transforms.ToPILImage()(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return self.transform(pil), idx


@torch.no_grad()
def predict_proba(model, image_paths, device, transform=None, batch_size=64,
                  num_workers=4, verbose=True):
    """Probabilités (N, num_classes) pour une liste d'images, par batch.
    Les images illisibles ont une ligne de NaN."""
    model.to(device).eval()
    transform = _model_transform(model, transform)
    loader = DataLoader(_PathDataset(image_paths, transform), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, collate_fn=collate_skip_none)
    probs = None
    done = 0
    for batch in loader:
        if batch is None:
            continue
        images, idx = batch
        p = torch.softmax(model(images.to(device)).float(), dim=1).cpu().numpy()
        if probs is None:
            probs = np.full((len(image_paths), p.shape[1]), np.nan, dtype=np.float32)
        probs[idx.numpy()] = p
        done += len(idx)
        if verbose and (done // batch_size) % 50 == 0:
            print(f"  {done}/{len(image_paths)} images", end="\r")
    if verbose:
        print(f"  {done}/{len(image_paths)} images prédites")
    return probs


def generate_confusion_matrix(model, image_paths, device, true_labels, num_classes=None,
                              transform=None, batch_size=64, num_workers=4):
    """Prédit toutes les images (par batch) et renvoie (cm, y_pred, y_true).
    Utilise le même prétraitement qu'à la validation pendant l'entraînement."""
    probs = predict_proba(model, image_paths, device, transform, batch_size, num_workers)
    ok = ~np.isnan(probs).any(axis=1)
    y_true = list(np.asarray(true_labels)[ok])
    y_pred = list(probs[ok].argmax(axis=1))
    labels = list(range(num_classes)) if num_classes else None
    return confusion_matrix(y_true, y_pred, labels=labels), y_pred, y_true


def topk_accuracy(probs, y_true, k=3):
    """Accuracy top-k (la vraie famille est parmi les k plus probables)."""
    ok = ~np.isnan(probs).any(axis=1)
    topk = np.argsort(-probs[ok], axis=1)[:, :k]
    return float((topk == np.asarray(y_true)[ok][:, None]).any(axis=1).mean())


def aggregate_by_observation(obs_ids, probs, y_true):
    """Moyenne les probabilités des photos d'une même observation MINKA.
    Renvoie (obs_ids, probs_obs, y_true_obs) : une ligne par observation."""
    df = pd.DataFrame(probs)
    df["obs"] = list(obs_ids)
    df["y"] = list(y_true)
    df = df.dropna()
    g = df.groupby("obs")
    probs_obs = g[[c for c in df.columns if c not in ("obs", "y")]].mean().values
    return g.size().index.values, probs_obs, g["y"].first().values


def calculate_label_proportions(true_labels, predicted_labels, class_names):
    n = len(class_names)
    true_p = np.bincount(true_labels, minlength=n) / len(true_labels)
    pred_p = np.bincount(predicted_labels, minlength=n) / len(predicted_labels)
    return dict(zip(class_names, true_p)), dict(zip(class_names, pred_p))


def plot_confusion_matrix(cm, class_names, annot=None):
    size = max(10, len(class_names) * 0.25)
    plt.figure(figsize=(size, size * 0.8))
    if annot is None:
        annot = len(class_names) <= 30
    sns.heatmap(cm, annot=annot, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion matrix")
    plt.xlabel("Predicted labels")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    plt.show()


def plot_label_proportions(true_proportions, pred_proportions, class_names):
    x = np.arange(len(class_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(12, len(class_names) * 0.2), 6))
    ax.bar(x - width / 2, list(true_proportions.values()), width, label="Vrais labels")
    ax.bar(x + width / 2, list(pred_proportions.values()), width, label="Labels prédits")
    ax.set_ylabel("Proportion")
    ax.set_title("Proportions of label per class")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=90)
    ax.legend()
    fig.tight_layout()
    plt.show()


# =============================================================================
# 5. CAM + SAM
# =============================================================================
def generate_cam(model, image_path, target_layer, label):
    """CAM pour ResNet uniquement (utilise model.fc et une couche convolutive)."""
    img_ori = read_image(image_path)
    device = next(model.parameters()).device
    img_tensor = _preprocess(img_ori).to(device)

    features_blobs = []
    handle = model._modules.get(target_layer).register_forward_hook(
        lambda m, i, o: features_blobs.append(o.detach().cpu().numpy()))
    model.eval()
    with torch.no_grad():
        model(img_tensor)
    handle.remove()  # sinon les hooks s'accumulent à chaque appel

    weight_softmax = model.fc.weight.detach().cpu().numpy()
    activations = features_blobs[0].squeeze(0)  # (C, H, W)
    cam = np.tensordot(weight_softmax[label], activations, axes=1)
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-10)
    return cam


def load_predictor(sam_checkpoint="sam_vit_h_4b8939.pth"):
    # import local : segment_anything n'est requis que pour cette fonction
    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    return SamPredictor(sam)


def sample_points_from_mask(cam_mask, num_points=10):
    y_coords, x_coords = np.where(cam_mask)
    point_coords = np.column_stack((x_coords, y_coords))
    if len(point_coords) == 0:
        return np.empty((0, 2), dtype=int)
    if len(point_coords) <= num_points:
        return point_coords
    center_x, center_y = np.mean(x_coords), np.mean(y_coords)
    central_idx = np.argmin(np.hypot(point_coords[:, 0] - center_x, point_coords[:, 1] - center_y))
    other = np.delete(point_coords, central_idx, axis=0)
    idx = np.linspace(0, len(other) - 1, num_points - 1, dtype=int)
    return np.vstack((point_coords[central_idx][None, :], other[idx]))


def calculate_mask_to_image_area_ratio(mask, image, verbose=True):
    h, w = image.shape[:2]
    ratio = np.sum(mask) / (h * w)
    if verbose:
        print(f"mask Area : {np.sum(mask)} pixels | Image Area : {h * w} pixels | "
              f"Ratio : {ratio:.4f} ({ratio * 100:.2f}%)")
    return ratio


def refine_mask_with_sam(predictor, image_path, cam, threshold=0.8):
    image = cv2.cvtColor(read_image(image_path), cv2.COLOR_BGR2RGB)
    predictor.set_image(image)
    cam_resized = cv2.resize(cam.astype(np.float32), (image.shape[1], image.shape[0]))
    cam_norm = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-10)
    point_coords = sample_points_from_mask(cam_norm > threshold, 10)
    if len(point_coords) == 0:
        point_coords = np.array([[image.shape[1] // 2, image.shape[0] // 2]])
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=np.ones(len(point_coords), dtype=int),
        multimask_output=True)
    return masks[np.argmax(scores)]


def display_results(original_image_path, cam, refined_mask, threshold=0.5):
    original_image = cv2.cvtColor(read_image(original_image_path), cv2.COLOR_BGR2RGB)
    h, w = original_image.shape[:2]
    cam_seed = cv2.resize((cam > threshold).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    cam = cv2.resize(cam.astype(np.float32), (w, h))
    cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-10)
    heatmap = cv2.cvtColor(cv2.applyColorMap((cam_norm * 255).astype(np.uint8), cv2.COLORMAP_JET),
                           cv2.COLOR_BGR2RGB)
    cam_overlay = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)  # déjà en RGB

    mask_bool = refined_mask.astype(bool)
    refined_mask_colored = original_image.copy()
    refined_mask_colored[~mask_bool] = 255
    ratio = calculate_mask_to_image_area_ratio(mask_bool, original_image)

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    for ax, img, title in zip(axes,
                              [original_image, cam_overlay, cam_seed, refined_mask_colored],
                              ["Original Image", "Class Activation Map (CAM)",
                               "Thresholded CAM", "Masque raffiné"]):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    return ratio, refined_mask_colored


# =============================================================================
# 6. BioCLIP / BioCLIP 2 (OpenCLIP) : zéro-shot, linear probe, fine-tuning
# =============================================================================
# Dépendance : pip install open_clip_torch   (Python 3.8 -> open_clip_torch 2.32)
# Modèles :
#   "hf-hub:imageomics/bioclip"     ViT-B/16, embeddings 512  (léger, conseillé pour commencer)
#   "hf-hub:imageomics/bioclip-2"   ViT-L/14, embeddings 768  (meilleur, ~4x plus lourd)
#   "hf-hub:ReefNet/finetuned-bioclip"  BioCLIP affiné sur 39 genres de coraux durs
#       tropicaux : domaine très étroit, a priori moins adapté à 365 familles
#       (oiseaux, plantes, insectes, estran…). Supporté pour comparaison.
#   chemin local vers un checkpoint (.pt/.bin/.safetensors) chargé sur l'architecture
#   `fallback_arch`.
DEFAULT_BIOCLIP = "hf-hub:imageomics/bioclip"


def _import_open_clip():
    try:
        import open_clip  # noqa: F401
    except ImportError as e:
        raise ImportError("open_clip n'est pas installé : pip install open_clip_torch") from e
    return open_clip


def _load_weights_into(model, path):
    """Charge un fichier de poids (formats variés) dans un modèle open_clip.
    Lève une erreur si la majorité des poids ne correspondent pas."""
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(sd, dict) and isinstance(sd.get(key), dict):
            sd = sd[key]
    sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in sd.items()}
    n_total = len(model.state_dict())
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(missing) > 0.5 * n_total:  # poids de l'encodeur visuel seul ?
        sd_v = {}
        for k, v in sd.items():
            for prefix in ("visual.", "model.visual.", "image_encoder.", "vision_model."):
                if k.startswith(prefix):
                    sd_v[k[len(prefix):]] = v
        sd_v = sd_v or sd
        missing_v, unexpected_v = model.visual.load_state_dict(sd_v, strict=False)
        if len(missing_v) > 0.5 * len(model.visual.state_dict()):
            raise RuntimeError(f"Les poids de {path} ne correspondent pas à l'architecture "
                               f"({len(missing_v)} tenseurs manquants). Vérifiez `fallback_arch`.")
        print(f"Poids visuels chargés depuis {path} "
              f"({len(missing_v)} manquants, {len(unexpected_v)} ignorés)")
    else:
        print(f"Poids chargés depuis {path} ({len(missing)} manquants, {len(unexpected)} ignorés ; "
              f"des manquants sont normaux si seul l'encodeur visuel est fourni)")


def load_open_clip_model(base_model=DEFAULT_BIOCLIP, fallback_arch=DEFAULT_BIOCLIP, pretrained=None):
    """
    Charge un modèle OpenCLIP. Renvoie (clip_model, preprocess_val, tokenizer, arch_name)
    où `arch_name` est le nom à utiliser pour reconstruire l'architecture.
    - "hf-hub:..." au format open_clip : chargé directement
    - "hf-hub:..." qui n'est pas au format open_clip (simple fichier de poids) :
      téléchargé et chargé sur l'architecture `fallback_arch`
    - chemin local d'un fichier de poids : chargé sur `fallback_arch`
    - nom d'architecture open_clip (ex. "ViT-B-16") + `pretrained`
    """
    oc = _import_open_clip()
    if os.path.isfile(base_model):
        model, _, preprocess_val = oc.create_model_and_transforms(fallback_arch)
        _load_weights_into(model, base_model)
        return model, preprocess_val, oc.get_tokenizer(fallback_arch), fallback_arch
    try:
        model, _, preprocess_val = oc.create_model_and_transforms(base_model, pretrained=pretrained)
        return model, preprocess_val, oc.get_tokenizer(base_model), base_model
    except Exception as e:  # noqa: BLE001
        if not base_model.startswith("hf-hub:"):
            raise
        from huggingface_hub import snapshot_download
        repo_dir = snapshot_download(base_model[len("hf-hub:"):])
        files = [f for ext in ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt")
                 for f in glob.glob(os.path.join(repo_dir, "**", ext), recursive=True)]
        if not files:
            raise RuntimeError(f"Aucun fichier de poids trouvé dans {base_model}") from e
        print(f"{base_model} n'est pas au format open_clip ({type(e).__name__}) : "
              f"chargement de {os.path.basename(files[0])} sur l'architecture {fallback_arch}")
        model, _, preprocess_val = oc.create_model_and_transforms(fallback_arch)
        _load_weights_into(model, files[0])
        return model, preprocess_val, oc.get_tokenizer(fallback_arch), fallback_arch


def _normalize_stats(preprocess):
    """Moyenne / écart-type de normalisation du prétraitement open_clip."""
    for t in getattr(preprocess, "transforms", []):
        if isinstance(t, transforms.Normalize):
            return list(t.mean), list(t.std)
    return CLIP_MEAN, CLIP_STD


def get_clip_train_transform(preprocess_val, size=224):
    """Augmentation pour le fine-tuning, avec la normalisation du modèle CLIP."""
    mean, std = _normalize_stats(preprocess_val)
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.5, 1.0),
                                     interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def build_family_prompts(class_names, taxon_tree=None, template="a photo of {}."):
    """
    Textes décrivant chaque famille pour initialiser la tête (zéro-shot).
    BioCLIP est entraîné avec des chaînes taxonomiques : avec `taxon_tree`
    (df_taxon de mecoda_minka), on produit par ex.
    "a photo of Animalia Mollusca Gastropoda Trochida Trochidae."
    """
    if taxon_tree is None:
        return [template.format(c) for c in class_names]
    tt = taxon_tree.drop_duplicates("taxon_id").set_index("taxon_id")
    fam = (taxon_tree[taxon_tree["rank"] == "family"]
           .drop_duplicates("taxon_name").set_index("taxon_name"))
    prompts = []
    for c in class_names:
        names = []
        if c in fam.index:
            for a in str(fam.loc[c, "ancestry"]).split("/"):
                try:
                    a = int(a)
                except ValueError:
                    continue
                if a in tt.index and tt.loc[a, "rank"] in ("kingdom", "phylum", "class", "order"):
                    names.append(str(tt.loc[a, "taxon_name"]))
        prompts.append(template.format(" ".join(names + [c])))
    return prompts


@torch.no_grad()
def text_embeddings(clip_model, tokenizer, prompts, device, batch_size=128):
    """Embeddings texte normalisés (num_prompts, embed_dim)."""
    clip_model.to(device).eval()
    out = []
    for i in range(0, len(prompts), batch_size):
        tokens = tokenizer(prompts[i:i + batch_size]).to(device)
        out.append(F.normalize(clip_model.encode_text(tokens).float(), dim=-1).cpu())
    return torch.cat(out)


class BioCLIPClassifier(nn.Module):
    """Encodeur visuel CLIP + tête linéaire sur les features normalisées.
    Avec la tête initialisée par les embeddings texte x logit_scale, la sortie
    initiale est exactement la prédiction zéro-shot de BioCLIP."""

    def __init__(self, visual, embed_dim, num_classes, logit_scale=100.0):
        super().__init__()
        self.visual = visual
        self.head = nn.Linear(embed_dim, num_classes)
        self.logit_scale = float(logit_scale)

    def features(self, x):
        return F.normalize(self.visual(x), dim=-1)

    def forward(self, x):
        return self.head(self.features(x))

    @torch.no_grad()
    def init_head(self, weight, bias=None):
        self.head.weight.copy_(weight.to(self.head.weight))
        if bias is None:
            self.head.bias.zero_()
        else:
            self.head.bias.copy_(bias.to(self.head.bias))


def build_bioclip_classifier(num_classes, class_names=None, base_model=DEFAULT_BIOCLIP,
                             fallback_arch=DEFAULT_BIOCLIP, prompts=None, taxon_tree=None,
                             head_init="text", device="cpu"):
    """
    Construit un BioCLIPClassifier.
    head_init : "text" (zéro-shot à partir des noms de familles), None (aléatoire),
                nn.Linear ou (weight, bias) issus d'un linear probe.
    Le modèle porte `eval_transform`, `train_transform` et `bioclip_config`.
    """
    clip_model, preprocess_val, tokenizer, arch_name = load_open_clip_model(base_model, fallback_arch)
    logit_scale = float(clip_model.logit_scale.exp().item()) if hasattr(clip_model, "logit_scale") else 100.0
    visual = clip_model.visual
    embed_dim = getattr(visual, "output_dim", None)
    size = getattr(visual, "image_size", 224)
    size = size[0] if isinstance(size, (tuple, list)) else size
    if embed_dim is None:
        with torch.no_grad():
            embed_dim = visual(torch.zeros(1, 3, size, size)).shape[-1]
    model = BioCLIPClassifier(visual, embed_dim, num_classes, logit_scale)

    if isinstance(head_init, str) and head_init == "text":
        if class_names is None:
            raise ValueError("class_names est requis pour head_init='text'")
        prompts = prompts or build_family_prompts(class_names, taxon_tree)
        emb = text_embeddings(clip_model, tokenizer, prompts, device)
        model.init_head(emb * logit_scale)
        print(f"Tête initialisée en zéro-shot ({len(prompts)} familles), ex. : {prompts[0]!r}")
    elif isinstance(head_init, nn.Linear):
        model.init_head(head_init.weight.detach().cpu(), head_init.bias.detach().cpu())
    elif isinstance(head_init, (tuple, list)):
        model.init_head(*[torch.as_tensor(t) for t in head_init])
    del clip_model  # l'encodeur texte n'est plus nécessaire

    model.eval_transform = preprocess_val
    model.train_transform = get_clip_train_transform(preprocess_val, size)
    model.bioclip_config = {"base_model": base_model, "arch_name": arch_name,
                            "embed_dim": int(embed_dim), "logit_scale": logit_scale,
                            "image_size": int(size)}
    return model.to(device)


def load_bioclip_classifier(checkpoint_path, device="cpu", checkpoint=None):
    """Recharge un BioCLIPClassifier sauvegardé par train_bioclip_classifier
    ou save_bioclip_classifier."""
    ckpt = checkpoint or torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["bioclip_config"]
    oc = _import_open_clip()
    clip_model, _, preprocess_val = oc.create_model_and_transforms(cfg["arch_name"])
    model = BioCLIPClassifier(clip_model.visual, cfg["embed_dim"], ckpt["num_classes"],
                              cfg["logit_scale"])
    del clip_model
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval_transform = preprocess_val
    model.train_transform = get_clip_train_transform(preprocess_val, cfg["image_size"])
    model.bioclip_config = cfg
    return model.to(device).eval()


def save_bioclip_classifier(model, path, class_names, epoch=0, history=None, loss=None, **extra):
    """Sauvegarde un BioCLIPClassifier dans un format lisible par
    load_model_from_checkpoint (ex. après le linear probe)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ckpt = {"arch": "bioclip", "bioclip_config": model.bioclip_config,
            "model_state_dict": model.state_dict(), "num_classes": model.head.out_features,
            "class_names": list(class_names), "epoch": epoch, "history": history or [],
            "loss": loss}
    ckpt.update(extra)
    torch.save(ckpt, path)


# ---------- Linear probe (encodeur figé, features calculées une seule fois) --
@torch.no_grad()
def extract_features(model, image_paths, device, batch_size=128, num_workers=8,
                     cache_file=None, amp=True):
    """
    Features visuelles normalisées (N, D) en float16 pour une liste d'images
    (lignes NaN pour les images illisibles). Résultat mis en cache dans
    `cache_file` (.npz) : relancer ne recalcule pas si la liste d'images est identique.
    """
    image_paths = [str(p) for p in image_paths]
    if cache_file and os.path.exists(cache_file):
        d = np.load(cache_file, allow_pickle=False)
        if len(d["paths"]) == len(image_paths) and (d["paths"] == np.array(image_paths)).all():
            print(f"Features chargées depuis {cache_file}")
            return d["features"]
        print(f"{cache_file} ne correspond pas à cette liste d'images : recalcul")
    device = torch.device(device)
    model.to(device).eval()
    loader = DataLoader(_PathDataset(image_paths, model.eval_transform), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, collate_fn=collate_skip_none)
    feats = None
    done, t0 = 0, time.time()
    use_amp = amp and device.type == "cuda"
    for batch in loader:
        if batch is None:
            continue
        images, idx = batch
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            f = model.features(images.to(device, non_blocking=True))
        f = f.float().cpu().numpy().astype(np.float16)
        if feats is None:
            feats = np.full((len(image_paths), f.shape[1]), np.nan, dtype=np.float16)
        feats[idx.numpy()] = f
        done += len(idx)
        if (done // batch_size) % 100 == 0:
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{len(image_paths)} images ({rate:.0f} img/s)")
    print(f"  {done}/{len(image_paths)} images en {time.time() - t0:.0f} s")
    if cache_file:
        np.savez(cache_file, features=feats, paths=np.array(image_paths))
    return feats


def train_linear_probe(feat_train, y_train, num_classes, feat_val=None, y_val=None,
                       init_head=None, epochs=30, lr=1e-3, weight_decay=1e-4,
                       batch_size=1024, label_smoothing=0.1, device=None, seed=0):
    """
    Entraîne uniquement la tête linéaire sur des features pré-calculées
    (quelques minutes, même sur CPU). Renvoie (head nn.Linear, history) ;
    la tête renvoyée est celle de la meilleure epoch en validation.
    `init_head` : nn.Linear de départ (ex. model.head initialisé en zéro-shot).
    """
    torch.manual_seed(seed)
    device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    ok = ~np.isnan(feat_train).any(axis=1)
    Xtr = torch.as_tensor(feat_train[ok], dtype=torch.float32, device=device)
    ytr = torch.as_tensor(np.asarray(y_train)[ok], dtype=torch.long, device=device)
    Xva = yva = None
    if feat_val is not None:
        okv = ~np.isnan(feat_val).any(axis=1)
        Xva = torch.as_tensor(feat_val[okv], dtype=torch.float32, device=device)
        yva = torch.as_tensor(np.asarray(y_val)[okv], dtype=torch.long, device=device)

    head = nn.Linear(Xtr.shape[1], num_classes).to(device)
    if init_head is not None:
        with torch.no_grad():
            head.weight.copy_(init_head.weight.detach().to(device))
            head.bias.copy_(init_head.bias.detach().to(device))
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * math.ceil(len(Xtr) / batch_size)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps,
                                              pct_start=0.1)

    def _acc(X, y):
        with torch.no_grad():
            logits = torch.cat([head(X[i:i + 8192]) for i in range(0, len(X), 8192)])
            return (logits.argmax(1) == y).float().mean().item(), criterion(logits, y).item()

    history, best_acc, best_state = [], -1.0, None
    if Xva is not None:
        acc0, loss0 = _acc(Xva, yva)
        print(f"Avant entraînement (zéro-shot si tête texte) : val acc {acc0:.3f}")
    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(len(Xtr), device=device)
        tot = 0.0
        for i in range(0, len(Xtr), batch_size):
            b = perm[i:i + batch_size]
            loss = criterion(head(Xtr[b]), ytr[b])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            tot += loss.item() * len(b)
        head.eval()
        train_acc, _ = _acc(Xtr, ytr)
        rec = {"epoch": epoch, "train_loss": tot / len(Xtr), "train_acc": train_acc}
        msg = f"Epoch {epoch + 1}/{epochs} : train loss {rec['train_loss']:.4f}, acc {train_acc:.3f}"
        if Xva is not None:
            rec["val_acc"], rec["val_loss"] = _acc(Xva, yva)
            msg += f" | val loss {rec['val_loss']:.4f}, acc {rec['val_acc']:.3f}"
        print(msg)
        history.append(rec)
        score = rec.get("val_acc", train_acc)
        if score > best_acc:
            best_acc = score
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    print(f"Meilleure accuracy : {best_acc:.3f}")
    return head.cpu(), history


# ---------- Fine-tuning complet (encodeur + tête) ----------------------------
def _param_groups(model, lr_backbone, lr_head, weight_decay):
    """Pas de weight decay sur biais / normalisations / embeddings 1D."""
    groups = {"bb_decay": [], "bb_no_decay": [], "head": []}
    for name, p in model.named_parameters():
        if name.startswith("head."):
            groups["head"].append(p)
        elif p.ndim < 2 or "bias" in name or "ln" in name or "norm" in name or "embedding" in name:
            groups["bb_no_decay"].append(p)
        else:
            groups["bb_decay"].append(p)
    return [
        {"params": groups["bb_decay"], "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": groups["bb_no_decay"], "lr": lr_backbone, "weight_decay": 0.0},
        {"params": groups["head"], "lr": lr_head, "weight_decay": 1e-4},
    ]


def train_bioclip_classifier(image_paths, labels, num_classes, class_names, epochs=10,
                             suffix="bioclip", val_image_paths=None, val_labels=None,
                             base_model=DEFAULT_BIOCLIP, fallback_arch=DEFAULT_BIOCLIP,
                             model=None, head_init="text", prompts=None, taxon_tree=None,
                             batch_size=64, lr_backbone=1e-5, lr_head=1e-3, weight_decay=0.05,
                             warmup_epochs=1, freeze_backbone_epochs=0, label_smoothing=0.1,
                             class_weights=None, amp=True, grad_checkpointing=False,
                             output_dir="models", num_workers=8, resume_from=None):
    """
    Fine-tuning de BioCLIP (ou BioCLIP 2 / ReefNet / checkpoint local) sur les familles.

    - `model` : BioCLIPClassifier déjà construit (ex. tête issue du linear probe =
      stratégie LP-FT, la plus robuste aux changements de domaine). Sinon il est
      construit à partir de `base_model` avec `head_init` ("text" = zéro-shot).
    - lr faible pour l'encodeur (`lr_backbone`), plus fort pour la tête (`lr_head`),
      AdamW, warmup puis décroissance cosinus, précision mixte sur GPU.
    - `freeze_backbone_epochs` : nb d'epochs où seule la tête est entraînée.
    - Sauvegarde `last_<suffix>.pth` (reprise) et `best_<suffix>.pth` (meilleure val acc),
      lisibles par load_model_from_checkpoint.
    - `resume_from` : checkpoint `last_<suffix>.pth` ; `epochs` = nombre TOTAL.
    Renvoie (model, history).
    """
    os.makedirs(output_dir, exist_ok=True)
    tag = f"_{suffix}" if suffix else ""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cpu":
        print("⚠ Pas de GPU : le fine-tuning d'un ViT sera très lent. "
              "Préférez le linear probe (extract_features + train_linear_probe).")

    ckpt = None
    if resume_from is not None:
        ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)
        if ckpt.get("class_names") not in (None, list(class_names)):
            raise ValueError("La liste des familles diffère de celle du checkpoint.")
        model = load_bioclip_classifier(resume_from, device="cpu", checkpoint=ckpt)
    elif model is None:
        model = build_bioclip_classifier(num_classes, class_names, base_model, fallback_arch,
                                         prompts, taxon_tree, head_init, device)
    model = model.to(device)
    if grad_checkpointing and hasattr(model.visual, "set_grad_checkpointing"):
        model.visual.set_grad_checkpointing(True)

    loader_kw = dict(num_workers=num_workers, collate_fn=collate_skip_none,
                     pin_memory=device.type == "cuda", persistent_workers=num_workers > 0)
    train_loader = DataLoader(CustomDataset(image_paths, labels, model.train_transform),
                              batch_size=batch_size, shuffle=True, drop_last=True, **loader_kw)
    val_loader = None
    if val_image_paths is not None and len(val_image_paths) > 0:
        val_loader = DataLoader(CustomDataset(val_image_paths, val_labels, model.eval_transform),
                                batch_size=batch_size * 2, shuffle=False, **loader_kw)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None,
        label_smoothing=label_smoothing)
    optimizer = optim.AdamW(_param_groups(model, lr_backbone, lr_head, weight_decay),
                            betas=(0.9, 0.999))
    steps_per_epoch = len(train_loader)
    total_steps = max(epochs * steps_per_epoch, 1)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    use_amp = amp and device.type == "cuda"
    amp_dtype = (torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    start_epoch, best_acc, history = 0, -1.0, []
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        history = list(ckpt.get("history") or [])
        best_acc = max([h.get("val_acc", h.get("train_acc", -1.0)) for h in history] or [-1.0])
        print(f"Reprise à l'epoch {start_epoch + 1}/{epochs} (meilleure accuracy {best_acc:.3f})")
        if scheduler.last_epoch >= total_steps:
            print("⚠ Le planning du learning rate est terminé : augmentez `epochs` "
                  "ou relancez un fine-tuning à partir de best (model=...).")

    @torch.no_grad()
    def _evaluate():
        model.eval()
        tot, correct, loss_sum = 0, 0, 0.0
        for batch in val_loader:
            if batch is None:
                continue
            x, y = batch[0].to(device, non_blocking=True), batch[1].to(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(x)
            loss_sum += criterion(out.float(), y).item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            tot += y.size(0)
        return loss_sum / max(tot, 1), correct / max(tot, 1)

    for epoch in range(start_epoch, epochs):
        frozen = epoch < freeze_backbone_epochs
        for name, p in model.named_parameters():
            p.requires_grad = (not frozen) or name.startswith("head.")
        model.train()
        running_loss, correct, total, n_batches = 0.0, 0, 0, 0
        t_epoch, t_data = time.time(), 0.0
        t_wait = time.time()
        print(f"Epoch {epoch + 1}/{epochs}" + (" (encodeur figé)" if frozen else ""))
        for i, batch in enumerate(train_loader):
            t_data += time.time() - t_wait
            if batch is None:
                t_wait = time.time()
                continue
            x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(x)
                loss = criterion(out.float(), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()
            n_batches += 1
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
            if i % 100 == 99:
                elapsed = time.time() - t_epoch
                print(f"  Batch {i + 1}/{steps_per_epoch}, Loss: {running_loss / n_batches:.4f}, "
                      f"Acc: {correct / max(total, 1):.3f} | {total / elapsed:.0f} img/s, "
                      f"attente données {100 * t_data / elapsed:.0f}%")
            t_wait = time.time()
        train_loss, train_acc = running_loss / max(n_batches, 1), correct / max(total, 1)
        rec = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc}
        msg = f"  -> train loss {train_loss:.4f}, train acc {train_acc:.3f}"
        if val_loader is not None:
            rec["val_loss"], rec["val_acc"] = _evaluate()
            msg += f" | val loss {rec['val_loss']:.4f}, val acc {rec['val_acc']:.3f}"
        print(msg + f" ({time.time() - t_epoch:.0f} s)")
        history.append(rec)

        meta = {"arch": "bioclip", "bioclip_config": model.bioclip_config,
                "num_classes": num_classes, "class_names": list(class_names),
                "epoch": epoch, "history": history, "loss": train_loss}
        torch.save(dict(meta, model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        scheduler_state_dict=scheduler.state_dict(),
                        scaler_state_dict=scaler.state_dict()),
                   os.path.join(output_dir, f"last{tag}.pth"))
        score = rec.get("val_acc", train_acc)
        if score > best_acc:
            best_acc = score
            torch.save(dict(meta, model_state_dict=model.state_dict()),
                       os.path.join(output_dir, f"best{tag}.pth"))
            print(f"  nouveau meilleur modèle -> {os.path.join(output_dir, f'best{tag}.pth')}")

    print(f"Meilleure accuracy ({'val' if val_loader else 'train'}) : {best_acc:.3f}")
    return model, history

def minka_photo_urls(photo_id, size="medium", exts=("jpg", "jpeg")):
    """URL candidates for a MINKA photo id: .jpg first, then .jpeg."""
    return [f"https://observe.minka-sdg.org/attachments/local_photos/files/{photo_id}/{size}.{e}" for e in exts]

def minka_photo_url(photo_id, size="medium"):
    return minka_photo_urls(photo_id, size)[0]

def candidate_urls(source):
    """Liste des sources à essayer : id MINKA -> .jpg puis .jpeg ; URL en .jpg/.jpeg -> l'URL puis l'autre extension."""
    source = str(source)
    if source.isdigit():
        return minka_photo_urls(source)
    if source.startswith("http"):
        m = re.match(r"^(.*)\.(jpe?g)(\?.*)?$", source, flags=re.I)
        if m:
            other = "jpeg" if m.group(2).lower() == "jpg" else "jpg"
            return [source, f"{m.group(1)}.{other}{m.group(3) or ''}"]
    return [source]

def fetch_first_working(source):
    """Essaie chaque candidat (jpg puis jpeg) et renvoie (source_valide, image_BGR)."""
    tried = []
    for url in candidate_urls(source):
        img = read_image(url, max_retries=1, verbose=False)
        if img is not None:
            print("Image OK :", url)
            return url, img
        print("Échec :", url)
        tried.append(url)
    raise ValueError(f"Image introuvable, essayé : {tried}")

@torch.no_grad()
def predict_topk(model, image_path, device, class_names, k=3):
    """Top-k familles (nom, probabilité) avec le prétraitement du modèle."""
    img = read_image(image_path)
    x = _preprocess(img, transform=_model_transform(model)).to(device)
    p = torch.softmax(model(x).float(), dim=1)[0].cpu()
    top = p.topk(min(k, len(class_names)))
    return [(int(i), class_names[int(i)], float(v)) for v, i in zip(top.values, top.indices)]

def analyse_image(source, model, device, class_names, predictor, th=0.5, top_k=3, block=-1, full_image=True, show=True):
    """
    Classification BioCLIP -> CAM -> masque SAM pour une image.
    th : seuil de la CAM (0-1) pour choisir les points donnés à SAM.
    Renvoie un dict : family, prob, topk, cam, mask, ratio, image (RGB masquée).
    """
    t0 = time.time()
    source, img = fetch_first_working(source)     # jpg puis jpeg si le lien ne marche pas

    # Image enregistrée une seule fois, puis lue en local par toutes les étapes
    local = source
    if source.startswith("http"):
        local = "_tmp_inference.jpg"
        cv2.imwrite(local, img)

    topk = predict_topk(model, local, device, class_names, k=top_k)
    label, family, prob = topk[0]
    print(f"Famille prédite : {family} ({prob:.1%})")
    for _, name, p in topk[1:]:
        print(f"   alternative : {name} ({p:.1%})")
    t1 = time.time()

    cam, _ = generate_cam_bioclip(model, local, label=label, block=block, full_image=full_image)
    t2 = time.time()

    mask = refine_mask_with_sam(predictor, local, cam, th)
    t3 = time.time()
    print(f"Temps : classification {t1 - t0:.1f}s | CAM {t2 - t1:.1f}s | SAM {t3 - t2:.1f}s")

    ratio, masked_image = display_results(local, cam, mask, threshold=th) if show else (None, None)
    return {"family": family, "prob": prob, "topk": topk, "cam": cam,
            "mask": mask, "ratio": ratio, "image": masked_image}

def generate_cam_bioclip(model, image_path, label=None, block=-1, full_image=True):
    """
    Grad-CAM pour un BioCLIPClassifier (ViT OpenCLIP).
    Renvoie (cam, label) : cam = array (g, g) normalisé dans [0, 1] (g = 14 pour ViT-B/16),
    label = famille utilisée (celle passée en argument, sinon la famille prédite).
    block : bloc du Transformer utilisé (-1 = dernier ; essayer -2, -3 si la carte est trop diffuse).
    """
    img_bgr = read_image(image_path)
    if img_bgr is None:
        raise ValueError(f"Image illisible : {image_path}")
    device = next(model.parameters()).device
    size = model.bioclip_config.get("image_size", 224)

    if full_image:
        mean, std = _normalize_stats(model.eval_transform)
        tf = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        tf = model.eval_transform
    pil = transforms.ToPILImage()(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    x = tf(pil).unsqueeze(0).to(device).requires_grad_(True)

    store = {}
    def fwd_hook(module, inputs, output):
        store["act"] = output
        output.register_hook(lambda g: store.__setitem__("grad", g))

    handle = model.visual.transformer.resblocks[block].ln_1.register_forward_hook(fwd_hook)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(x)
            if label is None:
                label = int(logits.argmax(1).item())
            logits[0, label].backward()
    finally:
        handle.remove()      # sinon les hooks s'accumulent à chaque appel

    act, grad = store["act"].detach(), store["grad"].detach()
    if act.shape[0] != 1:                        # disposition (N, B, C) -> (B, N, C)
        act, grad = act.transpose(0, 1), grad.transpose(0, 1)
    act, grad = act[0, 1:], grad[0, 1:]          # on retire le token CLS -> (n_patchs, C)

    g = int(round(act.shape[0] ** 0.5))
    if g * g != act.shape[0]:
        g = getattr(model.visual, "grid_size", (g, g))[0]
    weights = grad.mean(dim=0)                                   # (C,)
    cam = torch.relu((act * weights).sum(dim=-1)).reshape(g, g)  # (g, g)
    cam = cam.float().cpu().numpy()
    if cam.max() <= 0:
        print("⚠ CAM nulle (toutes les contributions négatives) : essayer block=-2 ou un autre seuil")
    cam = cam / (cam.max() + 1e-10)
    return cam, label
    
def download_if_missing(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"Téléchargement de {path} ...")
    urllib.request.urlretrieve(url, path)
