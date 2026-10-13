"""
Scanner récursif pour détecter les images et gérer les chemins des fichiers de description.
"""

import math
import os
from typing import List, Optional, Sequence, Set, Tuple


DEFAULT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
IGNORED_DIRS = {".git", ".venv", "venv", "__pycache__", ".idea", ".vscode", "node_modules"}


def parse_extensions(ext_str: str) -> Set[str]:
    """Parse une chaîne d'extensions séparées par des virgules (ex: '.png,.jpg,.jpeg')."""
    if not ext_str:
        return DEFAULT_EXTENSIONS
    result = set()
    for item in ext_str.split(","):
        cleaned = item.strip().lower()
        if cleaned:
            if not cleaned.startswith("."):
                cleaned = "." + cleaned
            result.add(cleaned)
    return result or DEFAULT_EXTENSIONS


def find_image_files(base_path: str, extensions: Optional[Sequence[str]] = None) -> List[str]:
    """
    Parcourt récursivement le répertoire base_path et retourne la liste triée
    de tous les chemins d'images correspondants aux extensions autorisées.
    Ignore les répertoires cachés/techniques et les fichiers de description.
    """
    valid_exts = set(e.lower() for e in extensions) if extensions else DEFAULT_EXTENSIONS
    image_paths: List[str] = []

    if not os.path.isdir(base_path):
        return image_paths

    for root, dirs, files in os.walk(base_path):
        # Élagage des répertoires ignorés
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in IGNORED_DIRS]

        for file in files:
            # Ne pas traiter les fichiers de description eux-mêmes
            if file.endswith(".description.txt"):
                continue

            _, ext = os.path.splitext(file)
            if ext.lower() in valid_exts:
                image_paths.append(os.path.join(root, file))

    image_paths.sort(key=lambda p: p.lower())
    return image_paths


def get_caption_path(image_path: str, append_extension: bool = False) -> str:
    """
    Retourne le chemin du fichier de description pour une image donnée.
    
    Par défaut (append_extension=False) :
      'dossier/photo.png' -> 'dossier/photo.description.txt'
    Si append_extension=True :
      'dossier/photo.png' -> 'dossier/photo.png.description.txt'
    """
    if append_extension:
        return f"{image_path}.description.txt"
    stem, _ = os.path.splitext(image_path)
    return f"{stem}.description.txt"


def is_already_captioned(image_path: str) -> Tuple[bool, Optional[str]]:
    """
    Vérifie si un fichier .description.txt existe déjà pour cette image
    (vérifie aussi bien 'nom.description.txt' que 'nom.ext.description.txt').
    Retourne un tuple (existe, chemin_existant).
    """
    stem, _ = os.path.splitext(image_path)
    candidate_replace = f"{stem}.description.txt"
    candidate_append = f"{image_path}.description.txt"

    if os.path.isfile(candidate_replace) and os.path.getsize(candidate_replace) > 0:
        return True, candidate_replace
    if os.path.isfile(candidate_append) and os.path.getsize(candidate_append) > 0:
        return True, candidate_append
    return False, None


def format_file_size(bytes_val: int) -> str:
    """Formate une taille en octets en chaîne lisible (Ko, Mo, etc.)."""
    if not bytes_val or bytes_val <= 0:
        return "0 o"
    units = ["o", "Ko", "Mo", "Go"]
    i = int(math.floor(math.log(bytes_val, 1024)))
    i = min(i, len(units) - 1)
    p = math.pow(1024, i)
    s = round(bytes_val / p, 1)
    return f"{s} {units[i]}"
