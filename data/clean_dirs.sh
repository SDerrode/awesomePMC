#!/usr/bin/env bash
set -euo pipefail   # FIX : ajouté — toute erreur de find/rm interrompt le script

# Récupère le dossier où se trouve le script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Répertoires à nettoyer (relatifs au dossier du script)
DIRS=("datafile" "historyTracker" "plot")

# FIX : --dry-run pour inspecter sans supprimer
#       Usage : bash clean_dirs.sh --dry-run
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

    # FIX 1 : -type f/-type d séparés pour éviter de suivre les symlinks avec rm -rf
    # FIX 2 : les dossiers contenant un .gitkeep sont également préservés
    #          (l'original supprimait le dossier parent même si .gitkeep était dedans)

    # Supprimer les fichiers (hors .gitkeep et hors realdata/)
    if [ "$DRY_RUN" = true ]; then
        find "$TARGET_DIR" -mindepth 1 -not -path "*/realdata*" -type f ! -name ".gitkeep" -print
    else
        find "$TARGET_DIR" -mindepth 1 -not -path "*/realdata*" -type f ! -name ".gitkeep" -delete
    fi

    # Supprimer les dossiers vides (hors realdata/, les dossiers avec .gitkeep ne seront pas vides)
    if [ "$DRY_RUN" = true ]; then
        find "$TARGET_DIR" -mindepth 1 -not -path "*/realdata*" -type d -empty -print
    else
        find "$TARGET_DIR" -mindepth 1 -not -path "*/realdata*" -type d -empty -delete
    fi

done

log "Nettoyage terminé."