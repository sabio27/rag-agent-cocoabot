import csv  # pour lire les fichiers csv
import json     # pour écrire les chunks au format json
import random # pour tirer quelques chunks au hasard et les inspecter
import re  # expressions regulieres, pour nettoyer le texte
from collections import Counter #Counter compte combien de fois chaque élément apparait
from pathlib import Path  # pour manipuler les chemins de fichiers
from pypdf import PdfReader  # pour ouvrir les pdf

DOSSIER_PDF = Path("data/raw")  # ou sont les pdf
FICHIERS_SOURCES = Path("data/sources.csv")  # fiche descriptive de chaque document
FICHIER_SORTIE = Path("data/chunks.jsonl")   # ou on enregistre les chunks
TAILLE = 1000 # taille cible du chunk ≈ 250 tokens
CHEVAUCHEMENT = 200 # nombre de caractere repris du chunk precedent au debut du suivant
TAILLE_MIN = 50  # on jette les morceaux plus courts 
SEUIL_ENTETE = 0.3 # ligne presente sur plus de 30% des pages = en-tete ou pied de page


def signature(ligne):
    """Version normalisée d'une ligne, pour reconnaître un en-tête même quand le numéro de page change."""
    return re.sub(r"\d+", "#", ligne.strip().lower())  # chaque nombre devient "#" : "FINAL REPORT | 78" et "| 79" donnent la même signature

def detecter_entetes(pages_lignes):
    """Renvoie l'ensemble des signatures de lignes repetees en haut ou en bas de nombreuses pages"""
    compteur = Counter()  # {signature: nombre de pages où elle apparaît}
    for lignes in pages_lignes:    # pour chaque page (une liste de ligne)
        bords = lignes[:2] + lignes[-2:]  # les 2 premières et 2 dernières lignes de la page
        compteur.update({signature(l) for l in bords if l.strip()})  # le set {...} évite de compter deux fois la même ligne sur une page
    seuil = max(3, SEUIL_ENTETE * len(pages_lignes))  # au moins 3 pages: un document de 2 pages n'a pas d'en-tête "répété"
    return {sig for sig, n in compteur.items() if n >= seuil}  # garde les signatures assez fréquentes
    
def nettoyer(texte):
    """Nettoie le texte brut extrait d'une page pdf"""
    texte = re.sub(r"-\n(\w)", r"\1", texte)  # recolle les mots coupés en fin de ligne : "produc-\ntion" devient "production"
    texte = re.sub(r"\s+", " ", texte) # remplace toute suite d'espace, tabulations et retours à la ligne par un seul espace
    texte = re.sub(r"(\w) ([’'])(\w)", r"\1\2\3", texte)  # supprime l'espace parasite avant une apostrophe : "d ’Ivoire" -> "d’Ivoire"
    return texte.strip()  # enlève les espaces au début et à la fin

def est_bibliographie(morceau):
    """Heuristique : un chunk rempli de liens ou d'années façon "2019." ressemble à une liste de références."""
    nb_liens = len(re.findall(r"https?://|doi\.org|doi:", morceau))       # compte les URL et les DOI
    nb_style_parentheses = len(re.findall(r"\((?:19|20)\d{2}[a-z]?\)\.", morceau))  # style "(2016)." : année entre parenthèses suivie d'un point
    nb_style_initiale = len(re.findall(r"[A-Z]\. (?:19|20)\d{2}[a-z]?\.", morceau))  # style "Ruf F. 2019." : initiale, espace, année, point
    return nb_liens >= 2 or nb_style_parentheses + nb_style_initiale >= 3  # au moins 3 références formatées dans le même chunk

def decouper(texte, taille=TAILLE, chevauchement=CHEVAUCHEMENT):
    """Decoupe un texte en morceaux d'environ 'taille' caracteres qui se chevauchent."""
    morceaux = []  # la liste des chunks qu'on va remplir
    debut = 0 # position de depart du chunk en cours
    while debut < len(texte) : # tant qu'on n'a pas atteint la fin du texte
        fin = min(debut + taille, len(texte))  # fin théorique : début + taille, sans dépasser la fin du texte
        if fin < len(texte) :  # si on n'est pas au dernier morceau...
            coupure = texte.rfind(". ", debut, fin) #...on cherche le DERNIER ". " (fin de phrase) avant la limite
            if coupure > debut + taille //2:  # on n'accepte cette coupure que si le chunk garde au moins la moitié de la taille visée
                fin = coupure + 1 # on coupe juste apres le point
        morceau = texte[debut:fin].strip()  # extrait le morceau entre debut et fin
        if len(morceau) >= TAILLE_MIN: # on ne garde que les morceaux assez longs
            morceaux.append(morceau) 
        if fin == len(texte):  # si on vient de traiter la fin du texte
            break   #  on sort de la boucle
        debut = fin - chevauchement  # le chunk suivant recommence 200 caractere avant la fin: c'est le chevauchement
        espace = texte.find(" ", debut, fin) # cherche le premier à partir du nouveau point de départ
        if espace != -1: #find renvoie -1 s'il ne trouve rien
            debut = espace + 1 # on demarre juste apres l'espace, jamais au milieu du mot
    return morceaux

# 1. Charger les méthadonnées: un dictionnaire {nom_de_fichier: ligne_du_csv}
with FICHIERS_SOURCES.open(encoding="utf-8") as f:
    metadonnees = {ligne["fichier"]: ligne for ligne in csv.DictReader(f)}  # compréhension de dictionnaire" : construit le dict en une ligne
    
# 2. Lire chaque pdf page par page et decouper
tous_les_chunks = []  # liste de tous les chunks du corpus
exemples_biblio = [] # on garde les chunks écartés, pour verifier que le filtre ne se trompe pas
for pdf in sorted(DOSSIER_PDF.glob("*.pdf")): # parcourt les pdf par ordre alphabetique
    meta = metadonnees.get(pdf.name)  # recupere la fiche du document; .get renvoie None si le fichier n'est pas dans le CSV
    if meta is None:    # un pdf sans fiche ne pourra pas être cité correctement...
        print(f"Ignoré (absent de sources.csv): {pdf.name}")
        continue     # ... donc on le saute
    
    pages_lignes = [(p.extract_text() or "").splitlines() for p in PdfReader(pdf).pages]  # chaque page devient une liste de lignes
    entetes = detecter_entetes(pages_lignes)  # les en-têtes et pieds de page propres à CE document
    nb_avant = len(tous_les_chunks)  # memorise le nombre avant ce docuement (pour compter les siens)
    nb_biblio = 0      #compteur de chunks écartés pour ce document
    for num_page, lignes in enumerate(pages_lignes, start=1): # on parcourt maintenant les pages déjà lues
        lignes_utiles = [l for l in lignes if signature(l) not in entetes]  # retire les en-têtes et pieds de page
        texte = nettoyer("\n".join(lignes_utiles)) # recolle les lignes restantes, puis nettoie
        for i, morceau in enumerate(decouper(texte)): #decoupe la page et numerote les morceaux
            if est_bibliographie(morceau):    # chunk de références ? on l'écarte...
                nb_biblio += 1
                exemples_biblio.append(morceau)  # ...mais on le garde de côté pour inspection
                continue # passe au morceau suivant sans l'ajouter
            tous_les_chunks.append({  # chaque chunk est un dictionnaire: le texte + ses metadonnees
                "id": f"{pdf.stem}_p{num_page}_c{i}", # identifiant unique; pdf.stem = nom du fichier sans ".pdf"
                "texte": morceau,
                "fichier": pdf.name,
                "titre": meta["titre"],
                "organisme": meta["organisme"],
                "annee": int(meta["annee"]),  # convertit l'année en nombre entier (utile pour filtrer : annee >= 2023)
                "langue": meta["langue"],
                "page": num_page,
            })
    print(f"{pdf.name:45} {len(tous_les_chunks) - nb_avant:5} chunks | {len(entetes):2} en-têtes retirés | {nb_biblio:3} chunks biblio écartés")
    
# 3. Enregistrer au format JSONL : un chunk par ligne
with FICHIER_SORTIE.open("w", encoding="utf-8") as f:  # "w" = écriture (écrase le fichier s'il existe)
    for chunk in tous_les_chunks:
        f.write(json.dumps(chunk, ensure_ascii=False) + "\n")  # ensure_ascii=False garde les accents lisibles au lieu de \u00e9

# 4. Statistiques et inspection
longueurs = [len(c["texte"]) for c in tous_les_chunks]  # liste des longueurs de tous les chunks
print(f"\nTOTAL : {len(tous_les_chunks)} chunks gardés, {len(exemples_biblio)} écartés comme bibliographie")
print(f"Longueur moyenne : {sum(longueurs) / len(longueurs):.0f} caractères (min {min(longueurs)}, max {max(longueurs)})")

random.seed(42)                               # graine fixe : le même tirage à chaque exécution, pour pouvoir comparer
for chunk in random.sample(tous_les_chunks, 3):  # tire 3 chunks au hasard, sans remise
    print(f"\n--- {chunk['id']} ({chunk['organisme']}, {chunk['annee']}, p. {chunk['page']}) ---")
    print(chunk["texte"])
    
print("\n=== 2 exemples de chunks ÉCARTÉS comme bibliographie (à vérifier !) ===")
for morceau in random.sample(exemples_biblio, min(2, len(exemples_biblio))):  # min(...) évite une erreur s'il y en a moins de 2
    print(f"\n{morceau[:400]}...")           # n'affiche que les 400 premiers caractères
        