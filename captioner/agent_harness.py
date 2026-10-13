"""
Agent Harness utilisant le Strands Agent SDK et Ollama pour le captioning d'images.
Fournit une intégration native Strands SDK avec optimisation d'images et mécanismes de repli robustes.
"""

import base64
import io
import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Détection de la disponibilité de Strands SDK
try:
    from strands import Agent
    from strands.models.ollama import OllamaModel
    HAS_STRANDS = True
except ImportError:
    HAS_STRANDS = False

# Détection de la bibliothèque ollama
try:
    import ollama
    HAS_OLLAMA_LIB = True
except ImportError:
    HAS_OLLAMA_LIB = False

# Détection de Pillow pour l'optimisation des images volumineuses
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# Correspondance entre extension de fichier et format attendu par Strands SDK
EXT_TO_STRANDS_FORMAT: Dict[str, str] = {
    ".png": "png",
    ".jpg": "jpeg",
    ".jpeg": "jpeg",
    ".gif": "gif",
    ".webp": "webp",
    ".bmp": "jpeg",
    ".tiff": "jpeg",
    ".tif": "jpeg",
}


def optimize_image_if_needed(image_bytes: bytes, max_dim: int = 1280) -> Tuple[bytes, str]:
    """
    Optimise l'image pour le modèle vision si Pillow est installé :
    - Redimensionne proportionnellement si max(width, height) > max_dim
    - Convertit les modes complexes (RGBA, palettes P, CMYK, etc.) en RGB JPEG
    Cela permet de réduire drastiquement l'empreinte VRAM et le nombre de tuiles/tokens
    requis par les modèles vision d'Ollama (évite les erreurs HTTP 500 Out of Memory).
    """
    if not HAS_PIL or max_dim <= 0:
        return image_bytes, ""

    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            w, h = img.size
            needs_resize = max(w, h) > max_dim
            needs_mode_convert = img.mode not in ("RGB", "L")

            if needs_resize or needs_mode_convert:
                if needs_resize:
                    ratio = min(max_dim / w, max_dim / h)
                    new_w = max(1, int(w * ratio))
                    new_h = max(1, int(h * ratio))
                    resample_filter = getattr(Image, "Resampling", Image).LANCZOS
                    img = img.resize((new_w, new_h), resample_filter)

                if img.mode != "RGB":
                    img = img.convert("RGB")

                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=85, optimize=True)
                return buf.getvalue(), "jpeg"
    except Exception as e:
        logger.debug(f"Optimisation de l'image ignorée : {e}")

    return image_bytes, ""


class ImageCaptionAgentHarness:
    """
    Harness pilotant l'agent Strands pour décrire des images via un modèle Ollama Vision.
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model_id: str = "llama3.2-vision:11b",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        num_ctx: int = 8192,
        max_image_dim: int = 1280,
        system_prompt: Optional[str] = None,
        timeout: int = 120,
    ):
        self.host = host.rstrip("/")
        self.model_id = model_id
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.num_ctx = int(num_ctx)
        self.max_image_dim = int(max_image_dim)
        self.system_prompt = (
            system_prompt
            or "Tu es un assistant expert en vision par ordinateur et en analyse d'images. "
               "Fournis des descriptions claires, détaillées, structurées et fidèles de l'image fournie."
        )
        self.timeout = timeout
        self._strands_model: Optional[Any] = None

        if HAS_STRANDS:
            try:
                self._strands_model = OllamaModel(
                    host=self.host,
                    model_id=self.model_id,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    options={
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                        "num_ctx": self.num_ctx,
                    },
                )
            except Exception as e:
                logger.debug(f"Impossible d'initialiser OllamaModel Strands : {e}")

    def check_connection(self) -> Tuple[bool, str]:
        """Vérifie si le serveur Ollama est joignable."""
        # 1. Tentative sur /api/version
        try:
            req = urllib.request.Request(f"{self.host}/api/version", headers={"User-Agent": "ImageCaptioner"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    self.server_version = data.get("version", "inconnue")
                    return True, f"Ollama v{self.server_version}"
        except Exception:
            pass

        # 2. Repli sur la racine /
        try:
            req = urllib.request.Request(f"{self.host}/", headers={"User-Agent": "ImageCaptioner"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    self.server_version = "actif"
                    return True, "Ollama actif"
        except Exception as e:
            return False, str(e)

        return False, "Non joignable"

    def check_if_ollama_is_latest(self) -> Tuple[bool, str, str]:
        """
        Compare la version d'Ollama locale avec la dernière version officielle publiée sur GitHub.
        Retourne (is_latest, current_version, latest_version).
        """
        curr = (getattr(self, "server_version", "") or "").lstrip("v").strip()
        if not curr or curr in ("actif", "inconnue"):
            return True, curr, ""

        try:
            req = urllib.request.Request(
                "https://api.github.com/repos/ollama/ollama/releases/latest",
                headers={"User-Agent": "ImageCaptioner"}
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                latest = data.get("tag_name", "").lstrip("v").strip()
                if latest:
                    return (curr == latest), curr, latest
        except Exception:
            pass

        return True, curr, ""

    def check_model_availability(self) -> Tuple[bool, List[str]]:
        """Vérifie si le modèle sélectionné est présent localement dans Ollama."""
        try:
            req = urllib.request.Request(f"{self.host}/api/tags", headers={"User-Agent": "ImageCaptioner"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]

                target = self.model_id.lower()
                for m in models:
                    m_lower = m.lower()
                    if m_lower == target or m_lower.split(":")[0] == target.split(":")[0]:
                        return True, models
                return False, models
        except Exception:
            return False, []

    def _determine_image_format(self, image_path: str) -> str:
        """Détermine le format image pour Strands SDK."""
        _, ext = os.path.splitext(image_path)
        return EXT_TO_STRANDS_FORMAT.get(ext.lower(), "jpeg")

    def describe_image(self, image_path: str, prompt: str) -> Tuple[str, str]:
        """
        Génère une description détaillée pour une image donnée avec gestion de retry.
        
        Retourne un tuple: (description_text, backend_utilise)
        Backend peut être: 'Strands Agent SDK', 'Ollama Client' ou 'Ollama API HTTP'.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"L'image n'existe pas : '{image_path}'")

        with open(image_path, "rb") as f:
            raw_bytes = f.read()

        if not raw_bytes:
            raise ValueError(f"Le fichier image est vide : '{image_path}'")

        # Optimisation préliminaire si image volumineuse / haute résolution
        opt_bytes, opt_format = optimize_image_if_needed(raw_bytes, self.max_image_dim)
        if opt_format:
            image_bytes = opt_bytes
            img_format = opt_format
        else:
            image_bytes = raw_bytes
            img_format = self._determine_image_format(image_path)

        last_error: Optional[Exception] = None
        max_attempts = 2

        for attempt in range(1, max_attempts + 1):
            if attempt > 1:
                time.sleep(2.0)  # Pause pour laisser Ollama libérer la VRAM / recharger le runner

            # -------------------------------------------------------------
            # 1. Tentative avec Strands Agent SDK
            # -------------------------------------------------------------
            if HAS_STRANDS and self._strands_model is not None:
                try:
                    agent = Agent(
                        model=self._strands_model,
                        system_prompt=self.system_prompt,
                        callback_handler=None,
                    )
                    content_blocks = [
                        {"text": prompt},
                        {
                            "image": {
                                "format": img_format,
                                "source": {"bytes": image_bytes},
                            }
                        },
                    ]
                    agent_res = agent(content_blocks)
                    desc = str(agent_res).strip()
                    if desc:
                        return desc, "Strands Agent SDK"
                except Exception as e:
                    logger.debug(f"Tentative {attempt} - Échec Strands Agent SDK pour {image_path}: {e}")
                    last_error = e

            # -------------------------------------------------------------
            # 2. Repli via la bibliothèque officielle ollama (si installée)
            # -------------------------------------------------------------
            if HAS_OLLAMA_LIB:
                try:
                    client = ollama.Client(host=self.host)
                    resp = client.generate(
                        model=self.model_id,
                        prompt=f"{self.system_prompt}\n\n{prompt}",
                        images=[image_bytes],
                        options={
                            "temperature": self.temperature,
                            "num_predict": self.max_tokens,
                            "num_ctx": self.num_ctx,
                        },
                    )
                    desc = resp.get("response", "").strip()
                    if desc:
                        return desc, "Ollama Client (Fallback)"
                except Exception as e:
                    logger.debug(f"Tentative {attempt} - Échec Ollama Client pour {image_path}: {e}")
                    last_error = e

            # -------------------------------------------------------------
            # 3. Repli direct via l'API HTTP REST d'Ollama (/api/generate)
            # -------------------------------------------------------------
            try:
                b64_img = base64.b64encode(image_bytes).decode("utf-8")
                payload = {
                    "model": self.model_id,
                    "prompt": f"{self.system_prompt}\n\n{prompt}",
                    "images": [b64_img],
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                        "num_ctx": self.num_ctx,
                    },
                }
                req = urllib.request.Request(
                    f"{self.host}/api/generate",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json", "User-Agent": "ImageCaptionerAgent"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    desc = data.get("response", "").strip()
                    if desc:
                        return desc, "Ollama API HTTP (Fallback)"
            except urllib.error.HTTPError as e:
                err_detail = ""
                try:
                    raw_body = e.read().decode("utf-8", errors="replace")
                    err_json = json.loads(raw_body)
                    err_detail = err_json.get("error", raw_body.strip())
                except Exception:
                    pass
                last_error = RuntimeError(f"Ollama HTTP {e.code}: {err_detail or e.reason}")
            except Exception as e:
                last_error = e

        err_lower = str(last_error).lower()
        if "mllama" in err_lower or "unknown model architecture" in err_lower:
            error_msg = (
                f"Erreur Ollama ({last_error})\n"
                f"   Le moteur Ollama ne parvient pas à charger les tenseurs 'mllama' du modèle.\n"
                f"   Causes courantes :\n"
                f"   - Le modèle a été téléchargé avec une ancienne version d'Ollama (cache de couches corrompu/obsolète).\n"
                f"   - Le démon Ollama n'a pas été redémarré après la mise à jour du binaire.\n\n"
                f"   Étapes de résolution :\n"
                f"   1. Redémarrer le service Ollama :\n"
                f"      sudo systemctl daemon-reload && sudo systemctl restart ollama\n"
                f"   2. Supprimer puis re-télécharger proprement le modèle :\n"
                f"      ollama rm {self.model_id}\n"
                f"      ollama pull {self.model_id}\n"
                f"   3. Tester son chargement dans le terminal :\n"
                f"      ollama run {self.model_id} \"bonjour\"\n"
                f"   (Alternative immédiate : tester avec llava via 'python3 caption_images.py -m llava:7b')"
            )
            raise RuntimeError(error_msg) from last_error

        error_msg = f"Impossible de légender '{os.path.basename(image_path)}'. Cause : {last_error}"
        raise RuntimeError(error_msg) from last_error
