import chromadb # pour la base vectorielle
import ollama # pour les embeddings et pour le llm
import json # pour lire chunks.jsonl (BM25 a besoin des textes)
import re # pour découper les textes en mots
from rank_bm25 import BM25Okapi  # l'implémentation de BM25

DOSSIER_CHROMA = "data/chroma"  # dossier ou chroma enregistre la base sur le disque
NOM_COLLECTION = "cacao"
MODELE_EMBEDDING = "bge-m3" # Doit etre le meme que dans indexation.py
MODELE_LLM = "qwen2.5:7b" # le modele qui redige la reponse
NB_PASSAGES = 5 # le "k" du top-k : nombre de chunks envoyés au LLM
AFFICHER_PASSAGES = True # si True, on affiche les passages envoyés au LLM

FICHIER_CHUNKS = "data/chunks.jsonl"  # les mêmes chunks que ceux indexés dans Chroma
MODE_RECHERCHE = "hybride"  # "semantique" (v0) ou "hybride" (v1) : on peut revenir en arrière en changeant un mot
NB_CANDIDATS = 20   # nombre de candidats demandés à CHAQUE méthode avant la fusion
K_RRF = 60   # constante standard de la fusion RRF

PROMPT_SYSTEME = """
Tu es CacaoBot, un assistant spécialisé sur la filière cacao.
Règles strictes :
1. Réponds UNIQUEMENT à partir des passages fournis. N'utilise jamais tes connaissances personnelles.
2. Après chaque information, cite sa source avec son numéro entre crochets, par exemple [2].
3. Si les passages ne contiennent pas la réponse, réponds exactement : "Je ne trouve pas cette information dans les documents."
4. Réponds dans la langue de la question, de façon claire et concise.
"""


client = chromadb.PersistentClient(path=DOSSIER_CHROMA) # ouvre la base créée par indexation.py
collection = client.get_collection(NOM_COLLECTION) # get_collection : on réutilise la collection existante

with open(FICHIER_CHUNKS, encoding="utf-8") as f:
    CHUNKS = [json.loads(ligne) for ligne in f]   # la liste de tous les chunks
CHUNKS_PAR_ID = {c["id"]: c for c in CHUNKS}    # dictionnaire id -> chunk, pour retrouver un chunk à partir de son id

def tokeniser(texte):
    """Découpe un texte en mots en minuscules : "Côte d'Ivoire" -> ["côte", "d", "ivoire"]."""
    return re.findall(r"\w+", texte.lower())   # \w+ = une suite de lettres ou de chiffres (accents compris)

BM25 = BM25Okapi([tokeniser(c["texte"]) for c in CHUNKS])  # construit l'index BM25 sur tous les chunks (quelques secondes)


def recherche_semantique(question, n):
    """Renvoie les ids des n chunks les plus proches par le sens (embeddings)."""
    vecteur = ollama.embed(model=MODELE_EMBEDDING, input=question)["embeddings"][0]
    res = collection.query(query_embeddings=[vecteur], n_results=n)
    return res["ids"][0]   # liste d'ids, du plus proche au moins proche
    
def recherche_bm25(question, n):
    """Renvoie les ids des n chunks avec le meilleur score BM25 (mots-clés)."""
    scores = BM25.get_scores(tokeniser(question))      # un score par chunk, dans l'ordre de CHUNKS
    meilleurs = sorted(range(len(CHUNKS)), key=lambda i: scores[i], reverse=True)[:n]  # les positions des n meilleurs scores
    return [CHUNKS[i]["id"] for i in meilleurs]

def fusion_rrf(*classements, k=K_RRF):
    """Fusionne plusieurs classements d'ids. Renvoie une liste de couples (id, score), du meilleur au moins bon."""
    scores = {}
    for classement in classements:                     # *classements accepte un ou plusieurs classements
        for rang, id_chunk in enumerate(classement, start=1):
            scores[id_chunk] = scores.get(id_chunk, 0) + 1 / (k + rang)  # .get(..., 0) : 0 si le chunk n'a pas encore de score
    return sorted(scores.items(), key=lambda paire: paire[1], reverse=True)

def rechercher(question, k=NB_PASSAGES):
    """Renvoie les k meilleurs passages, selon le mode de recherche choisi."""
    if MODE_RECHERCHE == "semantique":
        classement = fusion_rrf(recherche_semantique(question, k))  # un seul classement : l'ordre ne change pas
    else:
        classement = fusion_rrf(recherche_semantique(question, NB_CANDIDATS), recherche_bm25(question, NB_CANDIDATS))
    return [
        {"texte": CHUNKS_PAR_ID[id_chunk]["texte"], "meta": CHUNKS_PAR_ID[id_chunk], "score": score}
        for id_chunk, score in classement[:k]          # on garde les k premiers
    ]

def construire_prompt(question, passages):
    """Assemble les passages numérotés et la question dans un seul message"""
    blocs = []
    for n, p in enumerate(passages, start=1):     # numérotation à partir de 1 : [1], [2]...
        m = p["meta"]
        blocs.append(f"[{n}] ({m['organisme']}, {m['annee']}, p. {m['page']})\n{p['texte']}")  # chaque passage annoncé par son numéro et sa source
    contexte = "\n\n".join(blocs) # une ligne vide entre chaque passage
    return f"PASSAGES :\n\n{contexte}\n\nQUESTION : {question}" # le prompt final, avec les passages puis la question


def repondre(question):
    """Le pipeline RAG complet : recherche  -> prompt -> génération."""
    passages = rechercher(question)
    reponse = ollama.chat(
        model=MODELE_LLM,
        messages=[
            {"role": "system", "content": PROMPT_SYSTEME}, # les consignes
            {"role": "user", "content": construire_prompt(question, passages)}, # les passages + la question
        ],
        options={"temperature": 0, "num_ctx": 8192}, # température 0 = réponses factuelles et reproductibles; num_ctx = taille de la fenetre de contexte
    )
    return reponse["message"]["content"], passages

if __name__ == "__main__":   # vrai seulement quand on lance "python rag.py" ; faux quand un autre script fait "from rag import ..."
    # Boucle de conversation : pose des questions jusqu'à ce que tu tapes "q"
    while True: # boucle infinie, interrompue par "break"
        question = input("\nTa question (q pour quitter) : ").strip() # input attend que tu tapes une ligne au clavier
        if question.lower() in ("q", ""): # "q" ou une ligne vide : on arrete
            break
        
        reponse, passages = repondre(question)
        
        if AFFICHER_PASSAGES:
            print("\n--- Passages recuperés ---")
            for n, p in enumerate(passages, start=1):
                m = p["meta"]
                print(f"[{n}] score {p['score']:.4f} | {m['organisme']} ({m['annee']}), p. {m['page']} | {p['texte'][:120]}...")
                
        print(f"\n--- Réponse ---\n{reponse}")