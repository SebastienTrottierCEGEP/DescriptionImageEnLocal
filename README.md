# DescriptionImageEnLocal

Description automatique et locale de fichiers images par vision artificielle avec **Ollama** et le **Strands Agent SDK**.

Ce projet permet de parcourir récursivement une arborescence de dossiers (`base_path`), de détecter tous les fichiers images (.png, .jpg, .jpeg, .webp, etc.), et de générer une description détaillée pour chacun d'eux à l'aide d'un modèle multimodal local (ex: `llama3.2-vision:11b`).

Chaque description est écrite dans un fichier texte situé au même endroit que l'image avec l'extension `.description.txt` (ex: `schema.png` → `schema.description.txt`).

---

## 🚀 Démarrage rapide

### 1. Configuration initiale (WSL / Linux)
Pour installer Ollama, configurer l'environnement virtuel et télécharger le modèle :
```bash
./setup_wsl.sh
```

### 2. Gestion des modèles Ollama (TUI interactive)
Pour visualiser, basculer ou supprimer les modèles disponibles :
```bash
./manage_models.sh
```

### 3. Lancer le captioning des images
```bash
./run_captioner.sh
```
Ou directement avec Python :
```bash
python3 caption_images.py
```

---

## ⚙️ Configuration (`config.properties`)

Le fichier `config.properties` permet de définir les paramètres par défaut :

```properties
# Modèle Ollama multimodal
model_id=llama3.2-vision:11b

# Hôte Ollama local
host=http://localhost:11434

# Répertoire source contenant les images à parcourir récursivement
base_path=../TeamsBlinder/OC1_Ex1_Anonymise

# Prompt de description d'image
caption_prompt=Décris précisément et en détail cette image en français. Mentionne les éléments clés, le texte visible, les diagrammes ou schémas, et l'agencement général.

# Extensions ciblées
image_extensions=.png,.jpg,.jpeg,.gif,.webp,.bmp

# Ignorer ou écraser les descriptions existantes
overwrite=false
```

---

## 🛠️ Options de la ligne de commande

Le script `caption_images.py` accepte plusieurs arguments pour surcharger la configuration :

| Option | Description | Exemple |
| :--- | :--- | :--- |
| `-b, --base-path` | Répertoire racine à analyser | `python3 caption_images.py -b ./mes_images` |
| `-m, --model` | Modèle Ollama Vision à utiliser | `python3 caption_images.py -m llama3.2-vision:11b` |
| `--host` | Adresse du serveur Ollama | `python3 caption_images.py --host http://localhost:11434` |
| `-p, --prompt` | Prompt personnalisé pour la description | `python3 caption_images.py -p "Décris ce schéma électrique"` |
| `-f, --overwrite` | Forcer la regénération des descriptions existantes | `python3 caption_images.py -f` |
| `--append-ext` | Utiliser `photo.png.description.txt` au lieu de `photo.description.txt` | `python3 caption_images.py --append-ext` |
| `-n, --limit` | Traiter au maximum N images | `python3 caption_images.py -n 5` |
| `--dry-run` | Lister les images sans appeler le modèle | `python3 caption_images.py --dry-run` |
| `--delay` | Pause en secondes entre chaque image | `python3 caption_images.py --delay 1.0` |
| `--extensions` | Filtrer par extensions spécifiques | `python3 caption_images.py --extensions .png,.jpg` |
| `--num-ctx` | Taille de la fenêtre de contexte (ex: 8192 pour les modèles vision) | `python3 caption_images.py --num-ctx 8192` |
| `--max-image-dim` | Redimensionner les images dépassant cette taille en px | `python3 caption_images.py --max-image-dim 1280` |

---

## 🧠 Architecture

1. **`caption_images.py`** : Script CLI principal avec affichage en console, barre de progression, gestion du signal `Ctrl+C` et rapport d'exécution final.
2. **`run_captioner.sh`** : Script d'exécution bash avec activation automatique de l'environnement virtuel `.venv` et vérification d'Ollama.
3. **`captioner/agent_harness.py`** : Harness pilotant le modèle Vision :
   - Priorité 1 : **Strands Agent SDK** (`Agent` + `OllamaModel`)
   - Priorité 2 : Bibliothèque officielle `ollama` (Fallback)
   - Priorité 3 : API REST HTTP directe `/api/generate` (Fallback)
4. **`captioner/scanner.py`** : Exploration récursive des répertoires, détection des extensions et calcul des chemins de sortie `.description.txt`.
5. **`captioner/config.py`** : Chargement et normalisation des propriétés depuis `config.properties`.
