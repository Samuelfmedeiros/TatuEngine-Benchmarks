#!/usr/bin/env bash
# Gate de não-exposição de engenharia (ordem do Samuel, 15/11/2026).
# O repo publica amostra e benchmark — nunca kernel, layout de struct ou roadmap.
# Falha (rc=1) se marcador aparecer no working tree OU no historico git.
# Sem historico completo (clone raso) sai rc=2 (NAO VERIFICAVEL), nunca verde falso.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2

PADRAO='TriangleLens|traverse_rays|refraction_index|0,2812|0\.2812|__shfl_xor_sync|288 B|static_assert\(|sizeof\(\s*[A-Z]|lenToPosState|kTreeBits|kRelaxLen'

ARQUIVOS=$(find . -path ./.git -prune -o \( -name '*.md' -o -name '*.html' -o -name '*.txt' \) -type f -print)
ACHADOS=$(grep -nE "$PADRAO" $ARQUIVOS 2>/dev/null || true)

if [ -n "$ACHADOS" ]; then
    echo "FALHA: marcadores de engenharia expostos no WORKING TREE:"
    echo "$ACHADOS"
    exit 1
fi

# O vazamento de 15/11 veio de COMMITS: force-push nao apaga objeto do GitHub.
# Varrer todo o historico alcancavel, nao so os arquivos atuais.
COMMITS=0
HISTORICO=""
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    COMMITS=$(git rev-list --all --count 2>/dev/null || echo 0)
    if [ "$COMMITS" -lt 2 ]; then
        echo "NAO VERIFICAVEL: clone raso ($COMMITS commit visivel)."
        echo "O gate precisa do historico completo para provar nao-exposicao."
        echo "Use 'fetch-depth: 0' no checkout (CI) ou clone sem --depth."
        exit 2
    fi
    # So linhas ADICIONADAS ('+') sao exposicao; linhas removidas ('-') sao a
    # propria limpeza e nao devem falhar o gate.
    HISTORICO=$(git log --all -p -- '*.md' '*.html' '*.txt' 2>/dev/null \
        | grep -E "^\+[^+].*($PADRAO)" | head -40 || true)
fi

if [ -n "$HISTORICO" ]; then
    echo "FALHA: marcadores no HISTORICO git do repo publico."
    echo "Working tree limpo, mas force-push NAO apaga objeto do GitHub:"
    echo "a solucao de 16/11 foi renomear o repo antigo (private) e criar outro limpo."
    echo "Exige decisao do dono — nao resolver com push normal."
    echo "$HISTORICO" | head -25
    exit 1
fi
echo "OK: nenhum marcador de engenharia (working tree + historico, $COMMITS commits varridos)"
exit 0
