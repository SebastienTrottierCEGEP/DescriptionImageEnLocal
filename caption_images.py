#!/usr/bin/env python3
"""
CaptionImages - Script de description d'images par vision locale.

Lit la configuration depuis config.properties, parcourt récursivement le dossier base_path,
génère la description de chaque fichier image trouvé à l'aide d'un modèle Ollama Vision
piloté par le Strands Agent SDK, et enregistre chaque description dans un fichier
au même endroit avec l'extension .description.txt.
"""

import argparse
import os
import sys
import time
from typing import List

# Configuration sécurisée de l'encodage de la console (Windows cp1252 / Linux UTF-8)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from captioner.config import load_config
from captioner.scanner import (
    find_image_files,
    format_file_size,
    get_caption_path,
    is_already_captioned,
    parse_extensions,
)
from captioner.agent_harness import ImageCaptionAgentHarness


def parse_arguments() -> argparse.Namespace:
    """Parse les arguments passés en ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Génération automatique de descriptions d'images locales via Strands Agent SDK et Ollama."
    )
    parser.add_argument(
        "-b", "--base-path",
        help="Chemin du dossier racine à parcourir récursivement (par défaut: valeur de config.properties)."
    )
    parser.add_argument(
        "-m", "--model",
        help="Nom du modèle Ollama Vision (ex: llama3.2-vision:11b)."
    )
    parser.add_argument(
        "--host",
        help="URL du serveur Ollama (ex: http://localhost:11434)."
    )
    parser.add_argument(
        "-p", "--prompt",
        help="Prompt utilisateur pour la description des images."
    )
    parser.add_argument(
        "--system-prompt",
        help="System prompt pour l'agent vision."
    )
    parser.add_argument(
        "-f", "--overwrite",
        action="store_true",
        help="Forcer la regénération même si un fichier .description.txt existe déjà."
    )
    parser.add_argument(
        "--append-ext",
        action="store_true",
        help="Utiliser 'nom.ext.description.txt' au lieu de 'nom.description.txt'."
    )
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=None,
        help="Limiter le traitement aux N premières images trouvées."
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Pause (en secondes) entre chaque image pour ménager les ressources (défaut: 0)."
    )
    parser.add_argument(
        "--extensions",
        help="Liste d'extensions d'images séparées par des virgules (ex: .png,.jpg,.jpeg)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Afficher la liste des images sans effectuer les appels de modèle."
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help="Taille du contexte pour Ollama (défaut: valeur de config.properties, ex: 8192)."
    )
    parser.add_argument(
        "--max-image-dim",
        type=int,
        default=None,
        help="Dimension max en pixels pour optimiser les images (défaut: valeur de config.properties, ex: 1280)."
    )
    parser.add_argument(
        "-c", "--config",
        default="config.properties",
        help="Fichier de configuration alternatif (défaut: config.properties)."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    print("=" * 65)
    print("      CAPTIONING D'IMAGES EN LOCAL - STRANDS AGENT & OLLAMA     ")
    print("=" * 65)

    # 1. Chargement de la configuration
    config = load_config(args.config)
    base_path = args.base_path or config.get("base_path", "./SampleDir")
    model_id = args.model or config.get("model_id", "llama3.2-vision:11b")
    host = args.host or config.get("host", "http://localhost:11434")
    caption_prompt = args.prompt or config.get("caption_prompt", (
        "Décris précisément et en détail cette image en français. "
        "Mentionne les éléments clés, le texte visible, les diagrammes ou schémas, "
        "et l'agencement général."
    ))
    system_prompt = args.system_prompt or config.get("system_prompt")
    extensions_str = args.extensions or config.get("image_extensions", ".png,.jpg,.jpeg,.gif,.webp,.bmp")
    extensions = parse_extensions(extensions_str)
    overwrite = args.overwrite or (str(config.get("overwrite", "false")).strip().lower() in ("true", "1", "yes"))
    append_ext = args.append_ext
    temperature = float(config.get("temperature", 0.2))
    max_tokens = int(config.get("max_tokens", 1024))
    num_ctx = args.num_ctx or int(config.get("num_ctx", 8192))
    max_image_dim = args.max_image_dim if args.max_image_dim is not None else int(config.get("max_image_dim", 1280))
    timeout = int(config.get("timeout", 120))

    naming_mode = "nom.ext.description.txt" if append_ext else "nom.description.txt"

    print(f"[*] Dossier cible (base_path) : {base_path}")
    print(f"[*] Modèle Ollama (Vision)     : {model_id} ({host})")
    print(f"[*] Fenêtre de contexte (num_ctx): {num_ctx}")
    if max_image_dim > 0:
        print(f"[*] Optimisation image max-dim : {max_image_dim}px (si Pillow présent)")
    print(f"[*] Extensions recherchées     : {', '.join(sorted(extensions))}")
    print(f"[*] Mode de nommage fichier    : {naming_mode}")
    print(f"[*] Écrasement (overwrite)     : {'Oui' if overwrite else 'Non (skip si déjà présent)'}")
    if args.limit:
        print(f"[*] Limite maximale            : {args.limit} image(s)")
    if args.dry_run:
        print("[!] Mode simulation activé     : Aucune description ne sera générée")
    print("-" * 65)

    # 2. Validation du dossier racine
    if not os.path.isdir(base_path):
        print(f"[!] Erreur : Le dossier cible '{base_path}' n'existe pas.")
        sys.exit(1)

    # 3. Initialisation du harness et vérification d'Ollama
    agent_harness = ImageCaptionAgentHarness(
        host=host,
        model_id=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        num_ctx=num_ctx,
        max_image_dim=max_image_dim,
        system_prompt=system_prompt,
        timeout=timeout,
    )

    if not args.dry_run:
        online, version_info = agent_harness.check_connection()
        if not online:
            print(f"[!] Erreur critique : Le serveur Ollama ne répond pas sur {host}.")
            print(f"    Raison : {version_info}")
            print("    Le captioning ne peut pas être effectué sans Ollama.")
            print("    Démarrez Ollama avec './setup_wsl.sh' ou 'ollama serve' avant de relancer.")
            sys.exit(1)

        print(f"[✔] Serveur Ollama actif et joignable : {version_info}")

        is_latest, curr_v, latest_v = agent_harness.check_if_ollama_is_latest()
        if latest_v:
            if is_latest:
                print(f"[✔] Ollama est à jour (v{curr_v}).")
            else:
                print(f"[ℹ] Mise à jour Ollama disponible : v{curr_v} -> v{latest_v}")
                print("    Pour mettre à jour : curl -fsSL https://ollama.com/install.sh | sh")

        server_ver = getattr(agent_harness, "server_version", "")
        if "llama3.2-vision" in model_id.lower() and server_ver and server_ver != "actif":
            ver_clean = server_ver.lstrip("v").strip()
            parts = ver_clean.split(".")
            try:
                major = int(parts[0])
                minor = int(parts[1]) if len(parts) > 1 else 0
                if major == 0 and minor < 4:
                    print(f"[⚠] ATTENTION : Le modèle '{model_id}' utilise l'architecture 'mllama'")
                    print(f"    qui requiert Ollama >= 0.4.0 (version actuelle : v{server_ver}).")
                    print("    Pour mettre à jour Ollama :")
                    print("      curl -fsSL https://ollama.com/install.sh | sh")
                    print("    puis redémarrez Ollama (ex: sudo systemctl restart ollama ou ./setup_wsl.sh).")
            except Exception:
                pass

        model_ok, available_models = agent_harness.check_model_availability()
        if not model_ok:
            print(f"[!] Erreur critique : Modèle '{model_id}' introuvable dans Ollama.")
            installed_list = ", ".join(available_models) if available_models else "aucun modèle trouvé"
            print(f"    Modèles installés localement : {installed_list}")
            print(f"    Veuillez installer le modèle avec : ollama pull {model_id}")
            sys.exit(1)

        print(f"[✔] Modèle '{model_id}' disponible localement.")
        print("-" * 65)

    # 4. Découverte récursive des fichiers images
    print("[*] Recherche récursive des images en cours...")
    images = find_image_files(base_path, extensions)

    if not images:
        print(f"[!] Aucune image trouvée dans '{base_path}' avec les extensions {extensions_str}.")
        sys.exit(0)

    print(f"[+] {len(images)} image(s) détectée(s) au total.\n")

    if args.limit and args.limit < len(images):
        images = images[:args.limit]
        print(f"[*] Traitement limité aux {len(images)} première(s) image(s).\n")

    # 5. Traitement image par image
    total_processed = 0
    total_skipped = 0
    total_failed = 0
    start_time = time.time()

    try:
        for idx, img_path in enumerate(images, start=1):
            rel_path = os.path.relpath(img_path, base_path)
            target_caption_path = get_caption_path(img_path, append_extension=append_ext)
            caption_filename = os.path.basename(target_caption_path)

            print(f"[{idx}/{len(images)}] {rel_path}")

            # Vérification de présence préalable
            already_done, existing_file = is_already_captioned(img_path)
            if already_done and not overwrite:
                print(f"   [↷] Déjà légendée ({os.path.basename(existing_file or target_caption_path)}). Ignorée.")
                total_skipped += 1
                continue

            try:
                img_size = os.path.getsize(img_path)
                print(f"   * Fichier source : {format_file_size(img_size)}")
            except Exception:
                pass

            if args.dry_run:
                print(f"   [DRY-RUN] Fichier cible prévu : {caption_filename}")
                total_processed += 1
                continue

            print("   * Analyse visuelle et génération de la description...")
            img_start = time.time()
            try:
                caption_text, backend = agent_harness.describe_image(img_path, caption_prompt)
                img_duration = time.time() - img_start

                # Enregistrement du fichier de description
                with open(target_caption_path, "w", encoding="utf-8") as f:
                    f.write(caption_text.strip() + "\n")

                total_processed += 1
                preview = caption_text.splitlines()[0] if caption_text else ""
                if len(preview) > 90:
                    preview = preview[:87] + "..."

                print(f"   [+] Terminé en {img_duration:.2f}s via [{backend}]")
                print(f"   [+] Description écrite : {caption_filename}")
                if preview:
                    print(f"   > \"{preview}\"")
                print()

            except Exception as e:
                total_failed += 1
                print(f"   [!] Échec du traitement : {e}\n")

            if args.delay > 0:
                time.sleep(args.delay)

    except KeyboardInterrupt:
        print("\n[!] Interruption manuelle reçue (Ctrl+C). Arrêt du traitement en cours...")

    elapsed = time.time() - start_time

    # 6. Rapport final
    print("=" * 65)
    print("                    RAPPORT DE CAPTIONING                      ")
    print("=" * 65)
    print(f"[+] Images ciblées            : {len(images)}")
    print(f"[+] Images décrites avec succès: {total_processed}")
    print(f"[↷] Images déjà décrites (skip): {total_skipped}")
    if total_failed > 0:
        print(f"[!] Échecs / Erreurs          : {total_failed}")
    print(f"[*] Modele utilise            : {model_id}")
    print(f"[*] Dossier racine            : {base_path}")
    print(f"[*] Temps total d'execution   : {elapsed:.2f}s")
    print("=" * 65)
    print("[+] Opération de captioning terminée !\n")


if __name__ == "__main__":
    main()
