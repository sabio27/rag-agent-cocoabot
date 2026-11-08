import chromadb # pour la base vectorielle
import ollama # pour les embeddings et pour le llm

DOSSIER_CHROMA = "data/chroma"  # dossier ou chroma enregistre la base sur le disque
NOM_COLLECTION = "cacao"
MODELE_EMBEDDING = "bge-m3" # Doit etre le meme que dans indexation.py
MODELE_LLM = "qwen2.5:7b" # le modele qui redige la reponse
NB_PASSAGES = 5 # le "k" du top-k : nombre de chunks envoyés au LLM
AFFICHER_PASSAGES = True # si True, on affiche les passages envoyés au LLM

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

def rechercher(question, k=NB_PASSAGES):
    """Renvoie les k chunks les plus proches de la question."""
    vecteur = ollama.embed(model=MODELE_EMBEDDING, input=question)["embeddings"][0] # [0] : on extrait LE vecteur de la liste. la question devient un vecteur, avec le meme modele qu'à l'indexation.py
    res = collection.query(query_embeddings=[vecteur], n_results=k)
    return [
        {"texte": doc, "meta": meta, "distance": dist} # un dictionnaire par passage, plus simple à manipuler
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
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
                print(f"[{n}] distance {p['distance']:.3f} | {m['organisme']} ({m['annee']}), p. {m['page']} | {p['texte'][:120]}...")
                
        print(f"\n--- Réponse ---\n{reponse}")