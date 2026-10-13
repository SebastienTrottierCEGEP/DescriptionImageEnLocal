"""
Module de configuration pour le captioning d'images.
Lit et valide les propriétés depuis config.properties.
"""

import os
from typing import Any, Dict


def load_config(filepath: str = "config.properties") -> Dict[str, Any]:
    """
    Charge les propriétés depuis un fichier clé=valeur (ex: config.properties).
    Résout les chemins relatifs par rapport au dossier racine du projet.
    """
    props: Dict[str, Any] = {
        "model_id": "llama3.2-vision:11b",
        "host": "http://localhost:11434",
        "base_path": "../TeamsBlinder/OC1_Ex1_Anonymise",
        "caption_prompt": (
            "Décris précisément et en détail cette image en français. "
            "Mentionne les éléments clés, le texte visible, les diagrammes ou schémas, "
            "et l'agencement général."
        ),
        "system_prompt": (
            "Tu es un assistant expert en vision par ordinateur et en analyse d'images. "
            "Fournis des descriptions claires, détaillées, structurées et fidèles de l'image fournie."
        ),
        "image_extensions": ".png,.jpg,.jpeg,.gif,.webp,.bmp",
        "temperature": "0.2",
        "max_tokens": "1024",
        "num_ctx": "8192",
        "max_image_dim": "1280",
        "timeout": "120",
        "overwrite": "false",
    }

    # Recherche dans le dossier racine du projet
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_path = os.path.join(script_dir, filepath) if not os.path.isabs(filepath) else filepath

    if os.path.exists(target_path):
        with open(target_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    props[key.strip()] = val.strip()

    # Normalisation du chemin base_path si relatif
    base_val = props.get("base_path", "")
    if base_val and not os.path.isabs(base_val):
        props["base_path"] = os.path.normpath(os.path.join(script_dir, base_val))

    return props
