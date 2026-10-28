from pathlib import Path   # pour les chemins
from pypdf import PdfReader  # pour ouvrir les pdf
#import fonttools 

DOSSIER = Path("data/raw")  # dossier contenenant les pdf
SEUIL_SCAN = 100  # en dessous de 100 caracteres par page en moyenne, le pdf est probablement scanné

total_pages = 0  # compteur global de pages, initialisé à zéro
total_caracteres = 0  # compteur global de caracteres

for pdf in sorted(DOSSIER.glob("*.pdf")): # glob("*.pdf") liste tous les fichiers.pdf du dossier
    try:
        reader = PdfReader(pdf) # ouvre le pdf
        textes = [page.extract_text() or "" for page in reader.pages]  #liste du texte de chaque page, "or ''" remplace un resultat vide par une chaine vide
    except Exception as erreur:  # si le pdf est illisible ...
        print(f"{pdf.name:45} ILLISIBLE: {erreur}")  #  ... on le signale
        continue   # et on passe au pdf suivant
    
    nb_pages = len(textes)  # nombres de pages = nombres d'éléments de la liste
    nb_caracteres = sum(len(t) for t in textes)   # addtitione la longueur de chaque page
    moyenne = nb_caracteres / nb_pages if nb_pages else 0  # moyenne de caracteres par page, en évitant la division par zéro
    
    alerte = " SCANNE ?" if moyenne < SEUIL_SCAN else ""  # message d'alerte seulement si la moyenne est trop basse
    print(f"{pdf.name:45} {nb_pages:4} pages {nb_caracteres:>9} car. ({moyenne:.0f}/page){alerte}")
    # Dans les f-strings : ":45" réserve 45 caractères (alignement en colonnes),
    # ":>9" aligne à droite sur 9 caractères, ":.0f" affiche sans décimale.
    
    total_pages += nb_pages  # on ajoute le nombre de pages de ce pdf au total
    total_caracteres += nb_caracteres  # idem pour les caractères

print(f"\nTOTAL : {total_pages} pages, {total_caracteres} caractères")  # \n saute une ligne avant le total
print(f"≈ {total_caracteres // 4} tokens (estimation : 1 token ≈ 4 caractères)")  # règle approximative pour estimer le nombre de tokens
