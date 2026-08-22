#!/usr/bin/env bash
# Kiem toan bo goi: cu phap, --help chay duoc, duong dan tuyet doi, file thua.
#
#   bash tools/selfcheck.sh
#
# Muc dich: bat cac loi chi lo ra khi ai do CHAY THAT tren may khac — import
# thieu, duong dan hardcode, tool khong khoi dong noi. Chay truoc khi ban giao.
set -u
P=${PYTHON:-/home/nqhia/miniconda3/envs/cv-base/bin/python}
cd "$(dirname "$0")/.." || exit 1
bad=0

echo "=== 1 · cu phap ==="
for f in core/*.py tools/*.py tests/*.py training/core/*.py training/tools/*.py; do
  if ! "$P" -c "import ast,sys; ast.parse(open('$f',encoding='utf-8').read())" 2>/dev/null; then
    echo "  LOI CU PHAP: $f"; bad=$((bad+1))
  fi
done
echo "  $(ls core/*.py tools/*.py tests/*.py training/core/*.py training/tools/*.py | wc -l) file, $bad loi"

echo
echo "=== 2 · tool co khoi dong duoc khong (--help) ==="
for f in tools/*.py; do
  case "$(basename "$f")" in __init__.py) continue;; esac
  out=$("$P" "$f" --help 2>&1)
  if [ $? -ne 0 ] && ! echo "$out" | grep -q "usage:"; then
    echo "  HONG: $(basename "$f") -> $(echo "$out" | tail -1 | cut -c1-80)"
    bad=$((bad+1))
  fi
done
echo "  xong"

echo
echo "=== 3 · duong dan tuyet doi ngoai training/ ==="
# tu loai chinh file nay: no CHUA chuoi mau de tim, khong phai dung no
n=$(grep -rn "/mnt/c/Users\|C:/Users" --include="*.py" --include="*.yaml" \
      --include="*.sh" core tools tests config 2>/dev/null | grep -v selfcheck.sh | wc -l)
if [ "$n" -gt 0 ]; then
  grep -rn "/mnt/c/Users\|C:/Users" --include="*.py" --include="*.yaml" \
    --include="*.sh" core tools tests config 2>/dev/null | grep -v selfcheck.sh | sed 's|^|  |'
  echo "  -> $n cho, se hong tren may khac"
  bad=$((bad+1))
else
  echo "  khong co"
fi

echo
echo "=== 4 · file config duoc tham chieu co ton tai ==="
for k in weights; do
  v=$(grep -E "^\s+$k:" config/day_cfg.yaml | head -1 | awk '{print $2}')
  if [ -n "$v" ] && [ ! -f "$v" ]; then
    echo "  THIEU: config tro toi '$v' nhung khong co file"; bad=$((bad+1))
  else
    echo "  $k: $v  OK"
  fi
done

echo
echo "=== 5 · file la o thu muc goc ==="
extra=$(ls *.jpg *.png *.log 2>/dev/null)
if [ -n "$extra" ]; then
  echo "$extra" | sed 's|^|  |'
  echo "  -> anh tam, nen xoa hoac chuyen vao test_cases/"
else
  echo "  sach"
fi

echo
echo "=== 6 · test suite ==="
"$P" tests/selftest.py >/dev/null 2>&1 && echo "  selftest: PASS" \
  || { echo "  selftest: HONG"; bad=$((bad+1)); }
"$P" tests/integration_test.py >/dev/null 2>&1 && echo "  integration: PASS" \
  || { echo "  integration: HONG"; bad=$((bad+1)); }

echo
if [ "$bad" -gt 0 ]; then echo "==> $bad VAN DE"; exit 1; fi
echo "==> khong co van de chan"
