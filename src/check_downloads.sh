#!/usr/bin/env bash
# Health check for the EC-EARTH3 downloads. Prints no URLs or access tokens.
#
#   bash src/check_downloads.sh        # measures current speed over 60 s
#   bash src/check_downloads.sh 20     # quicker, 20 s measurement

cd ~/IKT464-CMIP-project || { echo "project folder not found"; exit 1; }
M=${1:-60}
declare -A SIZE=([historical]=59982783406 [ssp126]=60656201821 [ssp585]=60874380344)
SCEN="historical ssp126 ssp585"
hr()  { numfmt --to=iec --suffix=B --format=%.1f "$1"; }
pct() { awk -v n="$1" -v t="$2" 'BEGIN{printf "%.1f", 100*n/t}'; }
hide_tokens() { sed -E 's/access_token=[^&" ]*/access_token=***/g'; }

echo "== Workspace =="
echo "  container: $(hostname | awk -F- '{print $(NF-1)"-"$NF}')"
echo "  up since:  $(ps -o lstart= -p 1)   <- if this is after you last left, the workspace restarted"
echo "  disk free: $(df -h --output=avail . | tail -1 | tr -d ' ')"
mem=$(cat /sys/fs/cgroup/memory.current 2>/dev/null) && \
  echo "  memory:    $(( mem / 1073741824 )) GiB used of 96"

echo; echo "== Running downloads =="
running=$(pgrep -af download_data | grep -o 'raw/[a-z0-9]*/' | cut -d/ -f2 | sort -u)
if [ -n "$running" ]; then echo "$running" | sed 's/^/  /'; else echo "  none running"; fi

echo; echo "== Progress =="
declare -A START
for s in $SCEN; do
  f=data/raw/$s/EC-EARTH3.mat
  if [ -f "$f" ]; then
    n=$(stat -c%s "$f")
    if [ "$n" -eq "${SIZE[$s]}" ]; then
      printf "  %-10s DONE     %s (exact expected size)\n" "$s" "$(hr "$n")"
    else
      printf "  %-10s !! WRONG SIZE: %s instead of %s - not a real download, delete %s\n" \
        "$s" "$(hr "$n")" "$(hr "${SIZE[$s]}")" "$f"
    fi
  elif [ -f "$f.part" ]; then
    n=$(stat -c%s "$f.part"); START[$s]=$n
    printf "  %-10s %5s%%   %s of %s   last write %s\n" "$s" "$(pct "$n" "${SIZE[$s]}")" \
      "$(hr "$n")" "$(hr "${SIZE[$s]}")" "$(date -r "$f.part" +'%a %H:%M')"
    grep -qx "$s" <<<"$running" || echo "  !! $s is incomplete and NOT running - restart it"
  else
    printf "  %-10s not started\n" "$s"
  fi
done

if [ ${#START[@]} -gt 0 ]; then
  echo; echo "== Current speed (measuring ${M} s) =="
  sleep "$M"
  for s in $SCEN; do
    [ -n "${START[$s]}" ] || continue
    n=$(stat -c%s "data/raw/$s/EC-EARTH3.mat.part" 2>/dev/null) || n=${SIZE[$s]}  # finished meanwhile
    rate=$(( (n - START[$s]) / M / 1024 ))
    left=$(( SIZE[$s] - n ))
    if [ "$rate" -gt 0 ]; then
      eta=$(awk -v l="$left" -v r="$rate" 'BEGIN{printf "%.1f h", l/(r*1024)/3600}')
    else
      eta="no progress"
    fi
    printf "  %-10s %5d KB/s   time left at this speed: %s\n" "$s" "$rate" "$eta"
  done
fi

echo; echo "== Link expiry =="
if [ -f data/raw/urls.env ]; then
  # shellcheck disable=SC1091
  source data/raw/urls.env
  now=$(date +%s)
  for v in URL_HIST URL_126 URL_585; do
    ts=$(grep -o 'access_token=[^&]*' <<<"${!v}" | grep -o '[0-9]*$')
    if [ -n "$ts" ]; then
      printf "  %-9s %s   (%d h left)\n" "$v" "$(date -u -d @"$ts" +'%a %d %b %H:%M UTC')" $(( (ts - now) / 3600 ))
    else
      printf "  %-9s missing or malformed in urls.env\n" "$v"
    fi
  done
else
  echo "  data/raw/urls.env not found"
fi

echo; echo "== Recent warnings in the logs (last 200 lines each) =="
found=0
for s in $SCEN; do
  l=data/raw/$s/dl.log
  [ -f "$l" ] || continue
  w=$(tail -n 200 "$l" | grep -E 'retry|Giving up|HTTP [0-9]|Bad URL|mismatch|Not enough|Fatal' | tail -n 3 | hide_tokens)
  if [ -n "$w" ]; then found=1; echo "  $s:"; echo "$w" | cut -c1-140 | sed 's/^/    /'; fi
done
[ "$found" -eq 0 ] && echo "  none"