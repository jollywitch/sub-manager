#!/usr/bin/env bash
set -euo pipefail

TARGETS=(
  "./tools"
  "$HOME/.config/sub-manager/sub-manager.conf"
  "$HOME/.cache/huggingface/hub/models--zai-org--GLM-OCR"
)

DRY_RUN=false
ASSUME_YES=false

for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=true
      ;;
    --yes)
      ASSUME_YES=true
      ;;
    *)
      echo "Unknown option: $arg" >&2
      echo "Usage: $0 [--dry-run] [--yes]" >&2
      exit 2
      ;;
  esac
done

echo "Targets:"
for target in "${TARGETS[@]}"; do
  echo "  - $target"
done

if [ "$DRY_RUN" = true ]; then
  echo "Dry run only. No files were removed."
  exit 0
fi

if [ "$ASSUME_YES" = false ]; then
  read -r -p "Delete these paths? [y/N] " reply
  case "$reply" in
    [yY]|[yY][eE][sS]) ;;
    *)
      echo "Cancelled."
      exit 0
      ;;
  esac
fi

for target in "${TARGETS[@]}"; do
  if [ -e "$target" ] || [ -L "$target" ]; then
    rm -rf "$target"
    echo "Removed: $target"
  else
    echo "Skipped (not found): $target"
  fi
done

echo "Done."
