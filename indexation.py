import json # pour lire les chunks.jsonl
import time # pour mesurer la durée et estimer le temps restant
from pathlib import Path # pour les chemins
import chromadb # une base vectorielle
import ollama # pour calculer les embeddings


FICHIER_CHUNKS = Path("data/chunks.jsonl")
DOSSIER_CHROMA = "data/chroma"  # dossier ou chroma enregistre la base sur le disqque
NOM_COLLECTION = "cacao" # une "collection" Chroma ressemble à une table : un ensemble de vecteurs + textes + métadonnées
MODELE_EMBEDDING = "bge-m3" # LE modèle d'embedding ; il faudra utiliser exactement le même pour les questions
TAILLE_LOT = 32 # le nombre de chunks envoyés à ollama en une seule fois

# 1. Charger les chunks : une ligne du fichier = un chunk au format JSON
with FICHIER_CHUNKS.open(encoding="utf-8") as f:
    chunks = [json.loads(ligne) for ligne in f]  # json.loads transforme une ligne de texte en JSON de dictionnaire python
print(f"{len(chunks)} chunks chargés")

# 2. Créer (ou recréer) la collection
client = chromadb.PersistentClient(path=DOSSIER_CHROMA)  # "Persistent" : la base est enregistrée sur le disque et survit à l'arret du script
try:
    client.delete_collection(NOM_COLLECTION) # on supprime l'ancienne version pour repartir de zéro à chauqe indexation
except Exception:     # si elle n'existe pas encore (au premier lancement), chroma lève une erreur...
    pass              # qu'on ignore volontairement
collection = client.create_collection(
    name=NOM_COLLECTION,
    configuration={"hnsw": {"space": "cosine"}}, # HNSW est l'alggo de recherche rapide de chroma. on lui demande une distance cosinus
)

# 3. Calculer les embeddings par lots et les ajouter à la collection
debut = time.time() # heure de debut, en secondes
for i in range(0, len(chunks), TAILLE_LOT): # i vaut 0, 32, 64, 96... jusqu'à la fin
    lot = chunks[i:i+TAILLE_LOT] # le paquet de 32 chunks à traiter
    textes = [c["texte"] for c in lot] # on extrait seulement les textes
    vecteurs = ollama.embed(model=MODELE_EMBEDDING, input=textes)["embeddings"] # un appel à Ollama pour tout le lot: renvoie une liste de vecteurs de 1024 nombres
    
    collection.add(
        ids=[c["id"] for c in lot], # identifiants uniques (ex. "norc_child_labor_2020_p98_c0")
        documents=textes, # le texte, stocké pour pouvoir l'afficher et l'envoyer au LLM
        embeddings=vecteurs, # les vecteurs qu'on a calculés nous-mêmes
        metadatas=[{k: v for k, v in c.items() if k not in ("id", "texte")} for c in lot], # toutes les autres infos (titre, organisme, année, page...) ; on retire id et texte, déjà stockés à part
    )
    
    fait = i + len(lot) # nombre de chunks traités jusqu'ici
    ecoule = time.time() - debut # secondes écoulées depuis le depart
    reste = ecoule / fait * (len(chunks) - fait) # estimation: temps moyen par chunk x chunks restants
    print(f"\r{fait}/{len(chunks)} chunks indexés ({ecoule:.0f} s écoulés, ~{reste:.0f} s restantes)", end="")
    # "\r" ramène le curseur en début de ligne et end="" évite le retour à la ligne : la même ligne se met à jour

print(f"\n Indexation terminée: {collection.count()} chunks dans la collection '{NOM_COLLECTION}'")

# 4. premier test de recherche
question = "Quel est le prix bord champ du cacao payé aux producteurs en Côte d'Ivoire ?"
vecteur_question = ollama.embed(model=MODELE_EMBEDDING, input=question)["embeddings"][0] # [0] :  on a envoyé un seul texte, on prend le seul vecteur
resultats = collection.query(query_embeddings=[vecteur_question], n_results=3) # les 3 chunks les plus proches

print(f"\nQuestion: {question}")
for doc, meta, dist in zip(resultats["documents"][0], resultats["metadatas"][0], resultats["distances"][0]):
    #chroma renvoie des listes (une liste par question posée) : [0] prend les résultats de notre unique question
    # zip parcourt les trois listes en parallèle 
    print(f"\n[distance {dist:.3f}] {meta['organisme']} ({meta['annee']}), p. {meta['page']}")
    print(doc[:300] + "...") # les 300 premiers caractères du chunk.
