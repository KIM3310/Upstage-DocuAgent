#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="$ROOT/tools/cloudflare_pages.env"

if [[ -f "$CONFIG_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$CONFIG_FILE"
fi

BUILD_COMMAND="${BUILD_COMMAND:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
ROOT_DIR="${ROOT_DIR:-.}"
NOTES="${NOTES:-}"

IGNORE_GLOBS=(
  --glob '!.git'
  --glob '!node_modules'
  --glob '!dist'
  --glob '!.next'
  --glob '!build'
  --glob '!coverage'
  --glob '!*.png'
  --glob '!*.jpg'
  --glob '!*.jpeg'
  --glob '!*.gif'
  --glob '!*.pdf'
)

log() { printf '%s\n' "$*"; }
err() { printf 'ERROR: %s\n' "$*" >&2; }

usage() {
  cat <<USAGE
Usage:
  tools/release_ops.sh cloudflare
  tools/release_ops.sh apply-adsense <ca-pub-xxxxxxxxxxxxxxxx> <slot-id>
  tools/release_ops.sh check
  tools/release_ops.sh report <ca-pub-xxxxxxxxxxxxxxxx> <slot-id>
USAGE
}

detect_web_root() {
  local candidates=(
    "site"
    "docs"
    "app/frontend/public"
    "frontend/public"
    "public"
    "."
  )

  local c
  for c in "${candidates[@]}"; do
    if [[ -f "$ROOT/$c/ads.txt" && -f "$ROOT/$c/robots.txt" ]]; then
      printf '%s\n' "$c"
      return 0
    fi
  done

  printf '%s\n' "."
}

show_cloudflare() {
  log "[Cloudflare Pages Mapping]"
  log "repo: $(basename "$ROOT")"
  log "root_directory: ${ROOT_DIR}"
  log "build_command: ${BUILD_COMMAND:-<none>}"
  log "output_directory: ${OUTPUT_DIR:-<none>}"
  if [[ -n "$NOTES" ]]; then
    log "notes: $NOTES"
  fi
}

apply_adsense() {
  local client="${1:-}"
  local slot="${2:-}"

  if [[ -z "$client" || -z "$slot" ]]; then
    err "apply-adsense requires <client> <slot>."
    usage
    exit 1
  fi

  if [[ ! "$client" =~ ^ca-pub-[0-9]{16}$ ]]; then
    err "client must match ca-pub-<16digits>."
    exit 1
  fi

  if [[ ! "$slot" =~ ^[0-9]{8,20}$ ]]; then
    err "slot must be numeric (8-20 digits)."
    exit 1
  fi

  local pub="${client#ca-pub-}"
  mapfile -t files < <(
    rg -l "ca-pub-0000000000000000|ca-pub-xxxxxxxxxxxxxxxx|pub-0000000000000000|1234567890" \
      "$ROOT" "${IGNORE_GLOBS[@]}" \
      --glob '!*.md' \
      --glob '!README*'
  )

  if [[ ${#files[@]} -eq 0 ]]; then
    log "No placeholder targets found."
    return 0
  fi

  local f
  for f in "${files[@]}"; do
    perl -i -pe "s/ca-pub-0000000000000000/${client}/g; s/ca-pub-xxxxxxxxxxxxxxxx/${client}/g; s/pub-0000000000000000/${pub}/g; s/1234567890/${slot}/g" "$f"
  done

  log "Updated ${#files[@]} files with AdSense values."
  git -C "$ROOT" diff --name-only
}

check_one_file() {
  local path="$1"
  local label="$2"
  if [[ -f "$path" ]]; then
    log "OK   $label"
  else
    log "FAIL $label"
    return 1
  fi
}

check_policy() {
  local name="$1"
  local root_path="$2"

  if [[ -f "$ROOT/$root_path/${name}.html" || -f "$ROOT/${name}.html" || -f "$ROOT/src/app/${name}/page.tsx" ]]; then
    log "OK   policy:${name}"
    return 0
  fi

  log "FAIL policy:${name}"
  return 1
}

check_review() {
  local fail=0
  local web_root
  web_root="$(detect_web_root)"

  log "[AdSense/Cloudflare Review Check]"
  log "repo: $(basename "$ROOT")"
  log "web_root: $web_root"

  check_one_file "$ROOT/$web_root/ads.txt" "ads.txt" || fail=1
  check_one_file "$ROOT/$web_root/robots.txt" "robots.txt" || fail=1
  check_one_file "$ROOT/$web_root/sitemap.xml" "sitemap.xml" || fail=1

  if [[ -f "$ROOT/$web_root/_headers" || -f "$ROOT/_headers" ]]; then
    log "OK   _headers"
  else
    log "WARN _headers (recommended)"
  fi

  check_policy "privacy" "$web_root" || fail=1
  check_policy "terms" "$web_root" || fail=1
  check_policy "contact" "$web_root" || fail=1
  check_policy "compliance" "$web_root" || fail=1

  if rg -n "google-adsense-account" "$ROOT" "${IGNORE_GLOBS[@]}" --glob '!*.md' >/dev/null; then
    log "OK   adsense account meta"
  else
    log "FAIL adsense account meta"
    fail=1
  fi

  if rg -n "ca-pub-0000000000000000|ca-pub-xxxxxxxxxxxxxxxx|pub-0000000000000000|data-ad-slot=\"1234567890\"|VITE_ADSENSE_SLOT=1234567890|NEXT_PUBLIC_ADSENSE_SLOT=1234567890" \
    "$ROOT" "${IGNORE_GLOBS[@]}" --glob '!*.md' --glob '!README*' >/dev/null; then
    log "FAIL placeholder AdSense values remain"
    fail=1
  else
    log "OK   no placeholder AdSense values"
  fi

  show_cloudflare

  if [[ $fail -eq 0 ]]; then
    log "PASS review gate"
    return 0
  fi

  log "FAIL review gate"
  return 1
}

cmd="${1:-help}"
case "$cmd" in
  cloudflare)
    show_cloudflare
    ;;
  apply-adsense)
    apply_adsense "${2:-}" "${3:-}"
    ;;
  check)
    check_review
    ;;
  report)
    apply_adsense "${2:-}" "${3:-}"
    check_review
    ;;
  help|--help|-h)
    usage
    ;;
  *)
    err "Unknown command: $cmd"
    usage
    exit 1
    ;;
esac
