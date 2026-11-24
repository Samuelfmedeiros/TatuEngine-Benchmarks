#!/usr/bin/env bash
# Gate de não-exposição de engenharia (ordem do Samuel, 15/11/2026).
# O repo publica amostra e benchmark — nunca kernel, layout de struct ou roadmap.
# Falha (rc=1) se qualquer marcador reaparecer em arquivos publicados.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

PADRAO='TriangleLens|traverse_rays|refraction_index|0,2812|0\.2812|__shfl_xor_sync|288 B|static_assert\(|sizeof\(\s*[A-Z]|lenToPosState|kTreeBits|kRelaxLen'

ARQUIVOS=$(find . -path ./.git -prune -o \( -name '*.md' -o -name '*.html' -o -name '*.txt' \) -type f -print)
ACHADOS=$(grep -nE "$PADRAO" $ARQUIVOS 2>/dev/null || true)

if [ -n "$ACHADOS" ]; then
    echo "FALHA: marcadores de engenharia expostos no repo publico:"
    echo "$ACHADOS"
    exit 1
fi
echo "OK: nenhum marcador de engenharia encontrado"
exit 0
