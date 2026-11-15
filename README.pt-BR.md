# 🧠 TatuEngine-Benchmarks — Samuel Medeiros

Benchmarks e provas de coerência do motor de inferência ternário TatuEngine.

**Live:** https://github.com/Samuelfmedeiros/TatuEngine-Benchmarks

> 🌐 [🇺🇸 English](README.md) · **Português**

---

## 🎯 O que este repositório é de fato

Isto **não** é um recorde de velocidade nem de compressão. Não supera o
`bitmamba.c`, o `bitnet.cpp` nem o `xz`, e os números abaixo não devem ser lidos
como se superasse — 69,4× é o caminho GPU deste motor contra o próprio caminho CPU,
e 7,1× é contra o próprio caminho BF16. Os dois medem um ponto de partida, não um
concorrente.

O que vale ler aqui, e o que o resto do repo trata:

**1. Uma forma diferente de dispor os pesos.** O TatuEngine não guarda pesos como
escalares e não calcula um matmul simples sobre eles. A unidade nativa carrega um
pouco de estado por peso junto do valor, e a forma empacotada é um bloco ternário com
cabeçalho de metadados de tamanho fixo. A questão de projeto que ele explora é se um
*layout* — não a aritmética — consegue carregar informação útil e comprimir ao mesmo
tempo. O arranjo de campos, os tamanhos de struct e os kernels são proprietários; o
que está publicado aqui é o comportamento medido do resultado.

**2. `.tlz`, um codec companheiro.** Um compressor de verdade, com modelagem própria,
ajustado no último ciclo de 6,32% de gap contra o `xz` para 1,74%. Não é melhor que o
`xz`, e não precisa ser: o alvo natural dele é blob binário estruturado, não prosa. O
interior do codec é proprietário.

**3. Uma estrutura de documentação em três camadas.** Toda afirmação neste repo é
arquivada sob *reproduzível por terceiros* / *verificável em princípio* / *não
estabelecido*. Essa estrutura é a parte reutilizável — deveria ser obrigatória em
qualquer repositório de benchmark.

**4. Um detector de contaminação barato.** Assinatura de 32 tokens dos primeiros
tokens de cada documento, varredura por mmap sobre o tensor de treino de 2,4B tokens,
e o truque de avaliar o complemento (o cache do held-out vira o conjunto
in-distribution, então os dois lados saem de um split único e pareado). Reutilizável
para qualquer treino próprio.

Os números do motor continuam documentados por inteiro abaixo, com o que eles
estabelecem e o que não estabelecem.

### Do que o artefato é feito

`bitmamba_1b.bin` tem **614,45 MiB** em disco. A maior parte não são os pesos
comprimidos: é o embedding e o `lm_head` em fp16, 205,85M parâmetros a 2 bytes cada =
**392,6 MiB, 63,9% do arquivo**. O corpo do transformer codificado em ternário é os
36,1% restantes.

Essa divisão importa mais que qualquer razão de empacotamento. Todo argumento de banda
sobre formato ternário é um argumento sobre o terço menor deste arquivo, e a tabela de
embedding é justamente a parte que nenhum truque ternário toca. Quantizar
`embed`+`lm_head` para int8 sozinho levaria o artefato a **354,1 MiB (−42,4%)** —
cerca de quatro vezes o que um reempacotamento perfeito do payload ternário renderia.

O arranjo interno de blocos que produz os 36,1% é proprietário e não está documentado
aqui. O que é público é o tamanho do artefato, a contagem de parâmetros e a aritmética
acima.

## 🏗️ Arquitetura

```
tatuengine-benchmarks/
├── README.md                    # Paper (EN)
├── README.pt-BR.md              # Paper (PT-BR)
├── LICENSE                      # MIT
├── .gitignore
├── CHANGELOG.md
├── CONTRIBUTING.md
├── SECURITY.md
├── logs/
│   ├── coherence-test.txt       # 16/16 testes de coerência (CPU vs GPU)
│   └── prediction-120m.txt      # predição de token + compressão
└── architecture/
    └── diagram.md               # Diagrama de arquitetura de alto nível
```

**Stack:** C++ · CUDA · Python

---

## 📊 Resultados (RTX 3060, 12GB VRAM)

**O modelo desta tabela — medido, não assumido:** o artefato que produziu estes
números é um **`state-spaces/mamba2-1.3b` podado em profundidade** (48 → 32
camadas, vocab 50288 → 50257), gerado por `scripts/prune_mamba.py`
(`REPO_ID = "state-spaces/mamba2-1.3b"`). Tem **64 heads** (`ssm.A_log` com shape
`(64,)`) e 4.978 valores distintos em `in_proj` — **não é ternário em
armazenamento**, e **não é** o Zhayr1/BitMamba-2, que tem 32 heads e exatamente 3
valores distintos. O `logs/coherence-test.txt` rotula o modelo como
"BitMamba-2-1B"; esse rótulo não corresponde ao artefato medido e está em correção.

Também **não** é o modelo de 120M cujos números de predição aparecem mais abaixo.
São dois modelos distintos e os números não são intercambiáveis. O modelo de 120M
a 2 bits teria ~29 MiB, não 245 MiB.

`state-spaces/mamba2-1.3b` é Apache-2.0 (Microsoft/CMU). Atribuição pendente.

**Baseline externo, medido na mesma máquina** (i5-12400, mesmo arquivo
`bitmamba_1b.bin`, mesmo prompt, 200 tokens, seed 42).

`bitmamba_1b.bin` tem **614,45 MiB (644.297.164 B)** em disco — o BitMamba-2 real
(32 heads, vocab 50257). Revisões anteriores deste README diziam "993.664 blocos =
272,92 MB"; esse número veio de um log de debug
(`build_py/Testing/Temporary/LastTest.log`) e está errado para o artefato publicado
em 2,25×. Desses, 221,9 MiB são o corpo do transformer codificado em ternário e 392,6
MiB são embedding e `lm_head` em fp16.

| Motor | Hardware | tok/s |
|---|---|---|
| `jserv/bitmamba.c` (C11, 8 threads) | i5-12400 CPU | **73,7** |
| TatuEngine | RTX 3060 GPU | **24,3** |
| TatuEngine | i5-12400 CPU | **0,09** |

O `bitmamba.c` na CPU é **3,0× mais rápido que o TatuEngine na GPU** e **819× mais
rápido que o TatuEngine na mesma CPU**, no mesmo arquivo.

A 614,45 MiB lidos por token, isso dá **47,5 GB/s** para o `bitmamba.c` — 93% do
teto teórico de 51,2 GB/s do dual-channel DDR4-3200. Ele está na parede mesmo. O
TatuEngine a 24,3 tok/s move 15,7 GB/s, que é **4,3%** dos 360 GB/s da RTX 3060. O README dele declara
52,86 tok/s num i3-12100F; 73,7 num i5-12400 é consistente com os clocks maiores na
mesma geração Alder Lake.

Duas ressalvas que decidem se a tabela significa alguma coisa:

- **Armazenamento pesa no TatuEngine, não no `bitmamba.c`.** O segundo faz mmap
  zero-copy e lê do page cache; o TatuEngine faz `cudaMemcpy` por camada. Com 400
  tokens, o mesmo run do TatuEngine dá 5,04 tok/s com o modelo num HDD NTFS externo
  e 16,09 tok/s com o modelo num SSD local. Os números acima são o regime de 60
  tokens com pesos quentes.
- **Carga do host.** Estes runs rodaram com load average 43–59 e 28–30 GB de swap em
  uso. Remeça antes de tratar os valores absolutos como reprodutíveis.

É um confronto limpo: um arquivo, uma máquina, os dois motores. É também o primeiro
baseline externo que este repositório reporta.

| Métrica | Valor |
|---------|-------|
| Compressão | 113,8× vs checkpoint de treino · 14,2× vs FP32 denso · 7,1× vs BF16 |
| MSE (in_proj) | 3.7e-9 (duas ordens abaixo do epsilon do FP32) |
| Phase diff | 0,00e+00 (fase bit-exact) · amp 1,19e-07 = 1 épsilon FP32 |
| Coerência | 16/16 testes passados (semântica, não bit-exact) |
| Speedup | 69.4× (4.092s/tok CPU → 0.059s/tok GPU) |
| Blocos | 893K (in_proj 530K, out_proj 262K, lm_head 100K) |

---

## 🧠 Predição de próximo token (modelo 120M)

Corpora: **jurídico PT-BR** e **wiki PT geral** — 30 documentos cada, janelas de
512 tokens. Cross-entropy medida sobre 15.025 e 15.330 tokens preditos
respectivamente. Avaliação determinística (argmax, sem amostragem).

| Checkpoint | Corpus | Loss (nats) | Perplexidade | Top-1 | Top-5 | Bits/token |
|---|---|---|---|---|---|---|
| 884K | jurídico PT-BR | 1.6361 | 5.14 | 68.7% | 82.7% | 2.36 |
| 886K | jurídico PT-BR | 1.6252 | 5.08 | 69.2% | 82.8% | 2.34 |
| **888K** | **jurídico PT-BR** | **1.6141** | **5.02** | **69.3%** | **82.9%** | **2.33** |
| 884K | wiki PT | 2.9879 | 19.84 | 48.7% | 67.0% | 4.31 |
| 886K | wiki PT | 2.9891 | 19.87 | 48.6% | 66.5% | 4.31 |
| **888K** | **wiki PT** | **2.9815** | **19.72** | **48.6%** | **67.1%** | **4.30** |

**Leitura da trajetória:** entre os steps 884K e 888K o corpus jurídico melhorou
(1.6361 → 1.6141, top-1 68.7% → 69.3%) enquanto o corpus geral ficou parado
(2.9879 → 2.9815, top-1 48.7% → 48.6%).

**Correção (ver a seção held-out abaixo):** isto foi originalmente lido como "o
modelo está se especializando no domínio dominante do treino". Essa leitura está
errada. O ganho no jurídico é memorização recuperada de documentos que já estão no
tensor de treino, não habilidade adquirida. Em held-out pareado (mesmo checkpoint,
mesmo corpus), o jurídico perde 30,5% de loss ao sair do treino (1,9547 → 2,5505) e
o wiki perde 41,2% (2,9257 → 4,1302). Não há especialização; há memorização. O
corpus de treino é 68% jurídico PT-BR, que é exatamente por que a memorização cai ali.

O texto gerado é coerente tanto em decodificação greedy quanto amostrada
(temperatura 0.8, top-k 50, top-p 0.95).

> **⚠️ Estes são números in-distribution — e a contaminação foi medida, não
> suposta.** Os corpora de avaliação vêm dos mesmos pools usados no pretraining.
> Medi o tamanho do problema com amostragem aleatória de row-groups em **todos** os
> parquets (uma medição anterior de 12 documentos pegava só o início da ordem dos
> arquivos e superestimava a contaminação). Assinatura dos primeiros 32 tokens
> buscada no tensor de treino de 2.396.160.000 tokens:

| Corpus de eval | Documentos varridos | Achados no treino | Taxa |
|---|---|---|---|
| jurídico PT-BR | 200 | 129 | **64%** |
| wiki PT | 120 | 53 | **44%** |

A taxa real de contaminação é 64% e 44% — não os 83%/92% de uma medição anterior
feita em 12 documentos do início dos parquets. A maioria dos documentos jurídicos
que estão no treino aparece 21 vezes. Trate 1,6141 e 2,9815 como "o quanto o
modelo se ajusta a texto que já viu", não como generalização. O que os números
sustentam é a comparação *relativa* entre checkpoints na mesma distribuição — e o
gap pareado contra held-out está medido na seção abaixo. Ver
`logs/prediction-120m.txt` para a ressalva completa.

---

## ⚙️ Split híbrido CPU/GPU — onde o tempo está

O pipeline não é "GPU para tudo". Por token, por camada:

| Etapa | Dispositivo | Medido |
|---|---|---|
| lookup de embedding | CPU | — |
| RMSNorm | CPU | — |
| matmul `in_proj` | GPU (bridge) | maior parte do tempo de GPU |
| conv1d + SiLU | CPU | — |
| passo do scan SSM | CPU | — |
| matmul `out_proj` | GPU (bridge) | — |
| soma residual | CPU | — |
| RMSNorm final + `lm_head` | CPU → GPU | `lm_head` tem 100K blocos |

**O que está medido:** ponta a ponta 0,059 s/token em GPU contra 4,092 s/token em
CPU (69,4×), e 245 MiB de VRAM para os pesos codificados (mamba2-1.3b podado, 893K
blocos).

Para escala, usando os números medidos do bench pareado abaixo: 614,45 MiB lidos por
token dão **15,7 GB/s** no caminho GPU a 24,3 tok/s, contra ~360 GB/s de banda da RTX
3060 — **4,3%** da banda disponível. O `jserv/bitmamba.c` a 73,7 tok/s no mesmo
arquivo dá **47,5 GB/s**, ou seja, **93%** do teto teórico de dual-channel DDR4-3200,
e é **3,0× mais rápido que o caminho GPU daqui**. O 69,4× é
GPU-contra-o-próprio-CPU, não uma medida competitiva. (Uma revisão anterior deste
README usava 245,3 MiB e derivava 4,36 GB/s / 1,21% de uma contagem de blocos que não
correspondia ao arquivo publicado; os números de banda acima substituem aqueles.)

**O que NÃO está medido — e essa é a lacuna honesta:** latência por operação,
volume de transferência H2D/D2H e custo de sincronização por token. Sem isso, o
69,4× diz "o caminho GPU é muito mais rápido ponta a ponta", não "o split híbrido
é ótimo". Um trace etapa por etapa mostraria se os handoffs CPU↔GPU comem o
ganho, e esse trace não está neste repositório.

---

## 🗜️ Compressão lossless (`.tlz`)

Medido em corpora reais, contra `xz -9e` e `zstd -19`:

| Corpus | Original | xz -9e | zstd -19 | .tlz | vs xz |
|---|---|---|---|---|---|
| Wiki PT (texto) | 645.527 B | 188.080 B (3,432×) | 191.508 B (3,371×) | **191.350 B (3,374×)** | +1,74% |
| Código-fonte | 484.599 B | 90.564 B (5,351×) | 92.774 B (5,223×) | **93.933 B (5,159×)** | +3,72% |

**Gap contra o xz ao longo do dia:** texto 6,32% → **1,74%**; código 7,28% → **3,72%**.
Duas mudanças de modelagem trouxeram isso, ambas internas ao codec e não descritas
aqui.

### Como reproduzir estes números

Entradas, ambiente e comandos — tudo necessário para refazer a comparação:

```
# Corpora de entrada (SHA-256 dos arquivos exatos usados)
real.txt  17e1f54020d44670aa75389d0a778d669951d72adc2ea1a37a113259485137fd  (645.527 B)
code.txt  4505cd42b71a022e4ac10cb80799fba9c5ce729d9420d3272b5afc1c92885c8b  (484.599 B)

# Versões dos compressores
xz (XZ Utils) 5.8.3
zstd CLI v1.5.7 (Yann Collet)
```

```bash
# Compressores de referência
xz -9e -c real.txt > real.txt.xz
zstd -19 -c real.txt > real.txt.zst

# .tlz (a partir da árvore de build do TatuEngine)
./build/tools/tlz_compacto compact real.txt real.tlz -v
./build/tools/tlz_compacto extract real.tlz back.txt

# Integridade: comparação byte a byte, não só hash
cmp real.txt back.txt && echo "byte-identical"
sha256sum real.txt back.txt    # 17e1f54020d44670aa75389d0a778d669951d72adc2ea1a37a113259485137fd
```

O roundtrip do `.tlz` foi verificado com `cmp` (byte a byte) nos dois corpora, não
só com hash — hash bate identificação, `cmp` prova igualdade.

**O que é o .tlz, em um parágrafo:** um compressor de propósito específico com
modelagem própria, ajustado neste ciclo para reduzir o gap contra o `xz` em texto de
6,32% para 1,74%. O desenho interno (estrutura de matches, modelo de distância,
codificação de literais) é proprietário. O que se pode verificar aqui são os tamanhos,
os SHA-256 dos arquivos de entrada e o roundtrip por `cmp`.

**Nota sobre parâmetros:** os defaults do encoder acima são os do CLI; não há
dicionário tunado nem arquivo de metadados externo. `.tlz` é um stream único
autocontido, que é por que a comparação de tamanho contra `xz`/`zstd` é justa.

**Nota de metodologia:** uma medição anterior reportou gap de 14,2% no texto.
Era artefato de um corpus sintético montado com documentos repetidos — o
buscador de matches encontrava repetições enormes que não existem em texto real.
Todo número acima vem de corpora reais.

---

## 🔬 O que está aqui

- **`logs/coherence-test.txt`** — 16/16 testes de coerência (fase bit-exact, amplitude até 1 épsilon FP32)
- **`logs/prediction-120m.txt`** — saída bruta das medições de predição e compressão
- **`architecture/diagram.md`** — diagrama de alto nível do pipeline
- **`README.md`** — paper técnico com números, arquitetura e links

## Avaliação held-out (o número honesto)

Os números in-distribution acima superestimam o modelo de forma significativa.
A medição honesta exige comparar o **mesmo corpus** nos dois lados: quando o
documento está no treino e quando não está. Comparar loss absoluto entre corpora
diferentes é inválido — cada corpus tem entropia própria.

Amostragem aleatória de row-groups em **todos** os parquets (não só no início da
ordem, que era o viés da primeira medição), assinatura de 32 tokens contra o
tensor de treino, checkpoint `step_902000`:

| Corpus | Docs varridos | Limpos | Contaminação | Docs avaliados (cada lado) |
|---|---|---|---|---|
| Jurídico PT-BR | 200 | 40 | 64% | 40 |
| Wiki PT | 120 | 40 | 44% | 40 |

Resultado pareado — mesmo checkpoint, mesmo tokenizador, mesma sequência de 1024
tokens por documento:

| Corpus | Métrica | In-distribution | Held-out | Delta |
|---|---|---|---|---|
| Jurídico | loss | 1,9547 | **2,5505** | +0,596 (+30,5%) |
| Jurídico | ppl | 7,06 | **12,81** | +5,75 |
| Jurídico | top-1 | 64,85% | **57,92%** | −6,93 pp |
| Jurídico | bits/token | 2,820 | **3,680** | +0,860 |
| Wiki | loss | 2,9257 | **4,1302** | +1,205 (+41,2%) |
| Wiki | ppl | 18,65 | **62,19** | +43,54 |
| Wiki | top-1 | 49,75% | **40,65%** | −9,10 pp |
| Wiki | bits/token | 4,221 | **5,959** | +1,738 |

Três leituras que esses números sustentam:

**1. Não há especialização jurídica.** O held-out jurídico (2,5505) fica *abaixo*
do held-out wiki (4,1302) — mas o in-distribution jurídico também fica muito
abaixo do in-distribution wiki. O que separa os dois corpora é a dificuldade
intrínseca do texto, não habilidade do modelo no domínio.

**2. O gap de generalização é maior no wiki.** Jurídico perde 30,5% de loss do
in-distribution para o held-out; wiki perde 41,2%. Se o jurídico tivesse
aprendido estrutura de domínio, o gap dele seria o menor — e é. Isso é
consistente com o corpus jurídico ser mais templático e repetitivo, o que também
explica a contaminação maior (64% vs 44%).

**3. O número in-distribution não é propriedade do modelo.** É memória. A queda
de 30-41% de loss ao sair do treino é a medida direta disso.

Reproduzir (os dois lados, mesmo checkpoint):

```bash
# held-out: filtra contaminados e avalia os limpos
python scripts/eval_heldout.py --ckpt <dir> --corpus legalpt \
    --candidatos 200 --quero 40 --seed 42 --device cuda --saida /tmp/eval_heldout.json

# in-distribution: le o cache acima e avalia o complemento (os contaminados)
python scripts/eval_indist.py --ckpt <dir> --corpus legalpt \
    --saida-heldout /tmp/eval_heldout.json --device cuda --saida /tmp/eval_indist_legal.json
```

Os dois scripts imprimem os índices usados e gravam um cache ao lado da saída,
então o mesmo split pode ser reconferido sem repetir a varredura do tensor de
treino (que custa ~20 min por corpus).

**Ressalva que permanece:** 40 documentos por lado é amostra defensável mas não
grande, e a varredura de 200/120 candidatos dá uma estimativa de contaminação com
margem de alguns pontos percentuais. O que estes números estabelecem com firmeza
é a *direção e a ordem de grandeza* do gap — não o valor exato de loss em
produção. Uma varredura de 1000+ documentos reduziria a incerteza; não mudaria a
conclusão de que o número in-distribution era memória.

---

## O que dá e o que não dá para verificar

Ser explícito sobre a fronteira é mais útil do que sugerir que tudo é
reproduzível. Três camadas:

**Reprodutível por terceiros (artefatos publicados aqui)**
- Todo número de loss / perplexidade / top-k: tamanhos de corpus, contagem de
  tokens e steps dos checkpoints estão em `logs/prediction-120m.txt`, e a
  aritmética é conferível (`ppl = exp(loss)`, `bits = loss / ln 2`).
- Identidades aritméticas do layout do bloco (2 bits × 1024 pesos = 256 bytes)
  e a tabela de razões em `architecture/diagram.md`.
- Roundtrip byte a byte — terceiros podem rodar `xz -9e` / `zstd -19` nos mesmos
  corpora públicos e comparar tamanhos.

**Verificável em princípio, não publicado aqui**
- Coerência CPU-vs-GPU: o log do teste é publicado, o motor que o produziu não.
  Terceiros leem o resultado, mas não conseguem reexecutar.
- Speedup (69,4×, tok/s): depende de hardware e foi medido só na máquina do
  autor. Sem reprodução independente.

**Não estabelecido**
- Capacidade geral do modelo de 120M além dos dois corpora avaliados. 30
  documentos por corpus é demonstração, não afirmação de generalização.
- Comparação com outros modelos de tamanho semelhante — não feita. A única
  referência no log é o acaso uniforme (`ln 50257` = 10,83 nats).
- **Nenhum baseline externo de motor de inferência.** Não há comparação contra
  bitnet.cpp, os tipos ternários do llama.cpp/ggml, ou qualquer outra
  implementação. O 69,4× compara o caminho GPU contra o caminho CPU deste próprio
  motor — não diz nada sobre competitividade. Lê-lo como afirmação competitiva é
  uma leitura equivocada que os números convidam, e por isso está dito aqui.
- **Nenhum trace GPU por etapa.** O split CPU/GPU não é demonstrado ótimo; é
  demonstrado apenas mais rápido ponta a ponta que o caminho CPU escalar.
- Os 2,33 bits/token dependem da tarefa (domínio jurídico); não é afirmação
  sobre compressibilidade geral do modelo.

## ⚠️ O que NÃO está aqui

O código-fonte do motor, o kernel CUDA real e a lógica de alinhamento em C++ são proprietários. Este repositório serve como portfólio de provas técnicas. Nenhum checkpoint de pesos, nenhum dado de treino tokenizado e nenhum conteúdo de corpus é publicado — só métricas agregadas.

## 🔗 Links

- **Blog:** [lifelog.seu.pet](https://lifelog.seu.pet)
- **E-book Técnico:** [R$ 37 → skills.seu.pet/p/o-agente-que-nao-mente](https://skills.seu.pet/p/o-agente-que-nao-mente)
- **Setup Guiado:** [R$ 1.500 → skills.seu.pet/servico](https://skills.seu.pet/servico)

---

## 📋 Changelog

Ver [CHANGELOG.md](CHANGELOG.md).
