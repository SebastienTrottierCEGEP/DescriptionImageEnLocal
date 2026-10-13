#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "    Captioning d'images avec Ollama       "
echo "        Strands Agent SDK Vision          "
echo "=========================================="

# 1. Activation de l'environnement virtuel Python s'il existe
if [ -d ".venv" ] && [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# 2. Vérification rapide de l'état d'Ollama (avec timeout)
if ! curl -s --connect-timeout 2 --max-time 3 "http://localhost:11434" &> /dev/null; then
    echo "❌ Erreur : Le serveur Ollama ne répond pas sur http://localhost:11434."
    echo "   Le captioning ne peut pas continuer."
    echo "   Pour démarrer Ollama au préalable : ./setup_wsl.sh (ou 'ollama serve')"
    exit 1
fi

echo "✔ Serveur Ollama actif sur http://localhost:11434"

echo ""

# 3. Lancement du script principal de captioning
python3 caption_images.py "$@"
