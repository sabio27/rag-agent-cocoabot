# CacaoBot — RAG hybride et agent IA sur la filière cacao

CacaoBot répond à des questions sur la filière cacao à partir de 14 rapports en français et en anglais (Banque mondiale, ICCO, NORC, Fairtrade, règlement européen sur la déforestation…). Chaque réponse cite l'organisme, l'année et la page de sa source. Quand l'information n'est pas dans les documents, l'assistant le signale au lieu d'inventer une réponse.

Le projet combine un RAG hybride, un agent à outils et une évaluation automatique. Il fonctionne entièrement en local, avec des modèles open source.

## Fonctionnalités

**Recherche hybride bilingue.** La recherche sémantique (embeddings `bge-m3`, ChromaDB) comprend les reformulations mais manque les sigles et les noms exacts. Elle est donc combinée à une recherche par mots-clés (BM25), lancée sur la question en français et sur sa traduction anglaise afin de retrouver aussi les rapports en anglais. Les classements sont fusionnés par Reciprocal Rank Fusion.

**Agent à outils.** Au lieu de chercher une seule fois, le modèle choisit ses outils : plusieurs recherches pour une question qui compare deux sources, ou une calculatrice plutôt qu'un calcul de tête. La boucle est implémentée sans framework, avec une limite d'itérations et une détection des appels répétés. La calculatrice analyse l'expression sans l'exécuter comme du code.

**Évaluation automatique.** Un script pose 21 questions dont les réponses ont été vérifiées dans les rapports. Il mesure séparément la recherche (le bon passage figure-t-il parmi les 5 récupérés ?) et la réponse (est-elle juste ?), ce qui permet d'identifier l'origine de chaque erreur.

## Résultats

| Version | Recherche | Recall@5 | MRR | Exactitude* |
|---|---|---|---|---|
| Base | Sémantique seule | 82 % | 0,70 | 76 % |
| v1 | Sémantique + BM25 | 82 % | 0,68 | 76 % |
| **v1b** | Sémantique + BM25 français et anglais | **94 %** | **0,80** | **82 %** |

\* Évaluée par un LLM correcteur, qui surestime d'environ 8 points d'après une vérification manuelle. Les questions hors sujet sont refusées dans toutes les versions.

Deux autres pistes, un prompt plus strict et un modèle deux fois plus gros, n'ont pas amélioré les résultats et ont été abandonnées. Le détail de ces essais figure dans le [journal de bord](JOURNAL.md).

## Installation

Prérequis : Python 3.12, [uv](https://docs.astral.sh/uv/) et [Ollama](https://ollama.com).

```bash
ollama pull qwen2.5:7b
ollama pull bge-m3
uv sync
uv run python download_corpus.py   # les PDF ne sont pas versionnés ; la liste figure dans data/sources.csv
uv run python chunking.py          # découpage des rapports
uv run python indexation.py        # calcul des embeddings (environ 30 minutes sur processeur)
uv run python agent.py             # agent (ou rag.py pour la version sans agent)
uv run python evaluation.py        # évaluation
```

