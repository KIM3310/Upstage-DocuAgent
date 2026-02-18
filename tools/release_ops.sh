#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="$ROOT/tools/cloudflare_pages.env"
STRICT_ADSENSE_VALUES="${STRICT_ADSENSE_VALUES:-0}"

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

Environment:
  STRICT_ADSENSE_VALUES=1  # make placeholder AdSense values fail the check gate
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

contains_any() {
  local target="$1"
  shift
  local pattern
  for pattern in "$@"; do
    if rg -n -i "$pattern" "$target" >/dev/null 2>&1; then
      return 0
    fi
  done
  return 1
}

check_robots_quality() {
  local path="$1"
  local fail=0

  if ! rg -n "^User-agent:" "$path" >/dev/null; then
    log "FAIL robots: missing User-agent"
    fail=1
  fi
  if ! rg -n "^Allow:" "$path" >/dev/null; then
    log "FAIL robots: missing Allow"
    fail=1
  fi
  if ! rg -n "^Sitemap:" "$path" >/dev/null; then
    log "FAIL robots: missing Sitemap"
    fail=1
  fi

  if [[ $fail -eq 0 ]]; then
    log "OK   robots quality"
    return 0
  fi

  return 1
}

check_ads_txt_quality() {
  local path="$1"
  local line
  line="$(head -n 1 "$path" | tr -d '\r')"

  if [[ "$line" =~ ^google\.com,\ pub-[0-9]{16},\ DIRECT,\ f08c47fec0942fa0$ ]]; then
    log "OK   ads.txt quality"
    return 0
  fi

  if [[ "$STRICT_ADSENSE_VALUES" == "1" ]]; then
    log "FAIL ads.txt quality (invalid publisher id format)"
    return 1
  fi

  if [[ "$line" =~ ^google\.com,\ pub-0000000000000000,\ DIRECT,\ f08c47fec0942fa0$ ]]; then
    log "WARN ads.txt uses placeholder publisher id (expected before production onboarding)"
    return 0
  fi

  log "FAIL ads.txt quality"
  return 1
}

check_index_discoverability() {
  local fail=0
  local scan_targets=()

  if [[ -d "$ROOT/$1" ]]; then
    scan_targets+=("$ROOT/$1")
  fi
  if [[ -d "$ROOT/src/app" ]]; then
    scan_targets+=("$ROOT/src/app")
  fi
  if [[ -d "$ROOT/src/components" ]]; then
    scan_targets+=("$ROOT/src/components")
  fi
  if [[ -d "$ROOT/components" ]]; then
    scan_targets+=("$ROOT/components")
  fi
  if [[ -d "$ROOT/app/frontend/src" ]]; then
    scan_targets+=("$ROOT/app/frontend/src")
  fi
  if [[ -d "$ROOT/frontend/src" ]]; then
    scan_targets+=("$ROOT/frontend/src")
  fi

  if [[ ${#scan_targets[@]} -eq 0 ]]; then
    log "WARN discoverability: no scan targets"
    return 0
  fi

  if rg -n -i "privacy" "${scan_targets[@]}" --glob '*.{html,js,jsx,ts,tsx}' >/dev/null; then
    log "OK   homepage links privacy"
  else
    log "FAIL homepage links privacy"
    fail=1
  fi
  if rg -n -i "terms" "${scan_targets[@]}" --glob '*.{html,js,jsx,ts,tsx}' >/dev/null; then
    log "OK   homepage links terms"
  else
    log "FAIL homepage links terms"
    fail=1
  fi
  if rg -n -i "contact" "${scan_targets[@]}" --glob '*.{html,js,jsx,ts,tsx}' >/dev/null; then
    log "OK   homepage links contact"
  else
    log "FAIL homepage links contact"
    fail=1
  fi
  if rg -n -i "compliance" "${scan_targets[@]}" --glob '*.{html,js,jsx,ts,tsx}' >/dev/null; then
    log "OK   homepage links compliance"
  else
    log "FAIL homepage links compliance"
    fail=1
  fi
  if rg -n -i "about" "${scan_targets[@]}" --glob '*.{html,js,jsx,ts,tsx}' >/dev/null; then
    log "OK   homepage links about"
  else
    log "WARN homepage links about (recommended)"
  fi

  return $fail
}

check_contact_quality() {
  local fail=0
  local scan_targets=()

  if [[ -d "$ROOT/$1" ]]; then
    scan_targets+=("$ROOT/$1")
  fi
  if [[ -d "$ROOT/src/app" ]]; then
    scan_targets+=("$ROOT/src/app")
  fi
  if [[ -d "$ROOT/components" ]]; then
    scan_targets+=("$ROOT/components")
  fi
  if [[ -d "$ROOT/app/frontend/src" ]]; then
    scan_targets+=("$ROOT/app/frontend/src")
  fi
  if [[ -d "$ROOT/frontend/src" ]]; then
    scan_targets+=("$ROOT/frontend/src")
  fi

  if [[ ${#scan_targets[@]} -eq 0 ]]; then
    log "WARN contact quality: no scan targets"
    return 0
  fi

  if rg -n -i "@[^[:space:]]+\\.(local|test)\\b|ops\\.local|team\\.local" "${scan_targets[@]}" \
    --glob '*.{html,js,jsx,ts,tsx}' >/dev/null; then
    log "FAIL contact quality: local/test address found"
    fail=1
  else
    log "OK   contact quality: no local/test address"
  fi

  return $fail
}

check_ad_separation_signal() {
  local fail=0
  local scan_targets=()

  if [[ -d "$ROOT/$1" ]]; then
    scan_targets+=("$ROOT/$1")
  fi
  if [[ -d "$ROOT/src" ]]; then
    scan_targets+=("$ROOT/src")
  fi
  if [[ -d "$ROOT/components" ]]; then
    scan_targets+=("$ROOT/components")
  fi
  if [[ -d "$ROOT/app/frontend/src" ]]; then
    scan_targets+=("$ROOT/app/frontend/src")
  fi
  if [[ -d "$ROOT/frontend/src" ]]; then
    scan_targets+=("$ROOT/frontend/src")
  fi

  if [[ ${#scan_targets[@]} -eq 0 ]]; then
    log "WARN ad separation: no scan targets"
    return 0
  fi

  if rg -n -i "sponsored|ad slot|광고" "${scan_targets[@]}" --glob '*.{html,js,jsx,ts,tsx}' >/dev/null; then
    log "OK   ad separation label signal"
  else
    log "FAIL ad separation label signal"
    fail=1
  fi

  return $fail
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
  check_robots_quality "$ROOT/$web_root/robots.txt" || fail=1
  check_ads_txt_quality "$ROOT/$web_root/ads.txt" || fail=1

  if [[ -f "$ROOT/$web_root/_headers" || -f "$ROOT/_headers" ]]; then
    log "OK   _headers"
  else
    log "WARN _headers (recommended)"
  fi

  check_policy "privacy" "$web_root" || fail=1
  check_policy "terms" "$web_root" || fail=1
  check_policy "contact" "$web_root" || fail=1
  check_policy "compliance" "$web_root" || fail=1
  check_policy "about" "$web_root" || fail=1
  check_index_discoverability "$web_root" || fail=1
  check_contact_quality "$web_root" || fail=1
  check_ad_separation_signal "$web_root" || fail=1

  if rg -n "google-adsense-account" "$ROOT" "${IGNORE_GLOBS[@]}" --glob '!*.md' >/dev/null; then
    log "OK   adsense account meta"
  else
    log "FAIL adsense account meta"
    fail=1
  fi

  if rg -n "ca-pub-0000000000000000|ca-pub-xxxxxxxxxxxxxxxx|pub-0000000000000000|data-ad-slot=\"1234567890\"|VITE_ADSENSE_SLOT=1234567890|NEXT_PUBLIC_ADSENSE_SLOT=1234567890" \
    "$ROOT" "${IGNORE_GLOBS[@]}" --glob '!*.md' --glob '!README*' >/dev/null; then
    if [[ "$STRICT_ADSENSE_VALUES" == "1" ]]; then
      log "FAIL placeholder AdSense values remain (STRICT_ADSENSE_VALUES=1)"
      fail=1
    else
      log "WARN placeholder AdSense values remain (expected before production AdSense onboarding)"
    fi
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
