# CacaoBot — RAG hybride et agent IA sur la filière cacao

Assistant qui répond en français à des questions sur la filière cacao, à partir de 14 rapports en français et en anglais (Banque mondiale, ICCO, NORC, Fairtrade, règlement européen RDUE…), en citant ses sources (organisme, année, page).

## Fonctionnalités

- **RAG hybride bilingue** : recherche sémantique (`bge-m3`, ChromaDB) + BM25 en français et en anglais, fusion par Reciprocal Rank Fusion.
- **Agent à outils** (boucle *tool calling* codée sans framework) : décide s'il faut chercher et combien de fois, découpe les questions composées, délègue les calculs à une calculatrice sécurisée (`ast`, sans exécution de code). Garde-fous : itérations limitées, détection des appels répétés.
- **Évaluation automatique** : recall@5, MRR, exactitude par un juge LLM, refus des questions hors corpus.

## Résultats

21 questions annotées et vérifiées dans les sources (12 factuelles, 5 multilingues FR→EN, 4 hors corpus).

| Version | Recherche | Recall@5 | MRR | Exactitude* |
|---|---|---|---|---|
| v0 | Sémantique seule | 82 % | 0,70 | 76 % |
| v1 | + BM25 | 82 % | 0,68 | 76 % |
| **v1b** | + BM25 bilingue | **94 %** | **0,80** | **82 %** |

\* Juge LLM local, qui surestime d'environ 8 points selon une vérification manuelle. Les 4 questions hors corpus sont refusées dans toutes les versions.

Deux pistes ont été testées puis rejetées après mesure : un prompt plus strict (−8 points d'exactitude) et un modèle deux fois plus gros (gain non significatif, temps doublé). Détails dans [`JOURNAL.md`](JOURNAL.md).

## Fichiers

| Fichier | Rôle |
|---|---|
| `chunking.py` | Découpage des PDF par page, retrait des en-têtes et de la bibliographie |
| `indexation.py` | Embeddings et indexation dans ChromaDB |
| `rag.py` | Pipeline RAG |
| `agent.py` | Agent à outils |
| `evaluation.py` | Évaluation sur `data/eval.csv` |

## Installation

Prérequis : Python 3.12, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com).

```bash
ollama pull qwen2.5:7b
ollama pull bge-m3
uv sync
uv run python download_corpus.py   # les PDF ne sont pas versionnés ; liste et URL dans data/sources.csv
uv run python chunking.py
uv run python indexation.py
uv run python agent.py             # ou rag.py, evaluation.py
```

## Limites et suite

- Tableaux et graphiques des PDF extraits « à plat » : erreurs sur certaines questions chiffrées.
- Le modèle local (7B) suit imparfaitement les consignes ; jeu d'évaluation encore petit.
- Prochaines étapes : évaluation de l'agent, migration vers LangGraph, modèle via API.