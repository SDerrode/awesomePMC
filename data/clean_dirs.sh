#!/usr/bin/env bash
set -euo pipefail   # toute erreur de find/rm interrompt le script

# ── Usage ──────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<EOF
Usage: bash clean_dirs.sh [--dry-run|--help]

Vide les répertoires de sortie (datafile/, historyTracker/, Plots/) tout en
préservant les fichiers .gitkeep qui marquent l'arborescence.

Options:
  --dry-run   Affiche ce qui serait supprimé sans rien toucher.
  --help, -h  Affiche cette aide et quitte.
EOF
    exit 0
fi

# Récupère le dossier où se trouve le script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Répertoires à nettoyer (relatifs au dossier du script).
# NOTE : doit rester synchronisé avec les patterns "data/<dir>/*" du .gitignore
#        à la racine du dépôt.
DIRS=("datafile" "historyTracker" "Plots")

# option --dry-run pour inspecter sans supprimer
DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[dry-run] Aucune suppression effectuée."
fi

# Logging horodaté (cohérence avec les autres scripts du projet)
log() { echo "[$(date +%H:%M:%S)] $*"; }

for dir in "${DIRS[@]}"; do
    TARGET_DIR="$SCRIPT_DIR/$dir"

    if [ ! -d "$TARGET_DIR" ]; then
        log "Répertoire inexistant : $TARGET_DIR"
        continue
    fi

    log "Nettoyage : $dir"

    # -type f/-type d séparés pour éviter de suivre les symlinks avec rm -rf
    # les dossiers contenant un .gitkeep sont également préservés

    # Supprimer les fichiers (hors .gitkeep)
    if [ "$DRY_RUN" = true ]; then
        find "$TARGET_DIR" -mindepth 1 -type f ! -name ".gitkeep" -print
    else
        find "$TARGET_DIR" -mindepth 1 -type f ! -name ".gitkeep" -delete
    fi

    # Supprimer les dossiers vides (les dossiers avec .gitkeep ne seront pas vides)
    if [ "$DRY_RUN" = true ]; then
        find "$TARGET_DIR" -mindepth 1 -type d -empty -print
    else
        find "$TARGET_DIR" -mindepth 1 -type d -empty -delete
    fi

done

log "Nettoyage terminé."