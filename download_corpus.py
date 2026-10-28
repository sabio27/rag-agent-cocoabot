import csv  # module standard pour lire les fichiers csv
import urllib.request #module standard pour faire des requetes HTTP
from pathlib import Path #Path manipule les chemins de fichiers
import truststore # fait utiliser à python les certificats du système
truststore.inject_into_ssl() #remplace le magasin de certificats par défaut par celui de Windows, avant toute requête

SOURCES = Path("data/sources.csv") #chemin du fichier qui décrit les documents
DOSSIER = Path("data/raw")  #dossier ou seront rangés les pdf
DOSSIER.mkdir(parents=True, exist_ok=True)  #crée le dossier et ses parents s'il n'existe pas

with SOURCES.open(encoding="utf-8") as f:  #ouvre le csv eb utf-8 pour lire correctement les accents
    for ligne in csv.DictReader(f): #DictReader transforme chaque ligne en dictionnaire {"fichier": ..., "url": ...}
        destination = DOSSIER / ligne["fichier"] #l'operateur / assemble des chemins: data/raw/nom_du_fichier.pdf
        
        if destination.exists(): #si le fichier existe 
            print(f"Deja present: {destination.name}")
            continue  # .. on passe directement à la ligne suivante du csv
        
        requete = urllib.request.Request( # prepare (sans l'envoyer) une requete HTTP vers l'url
            ligne["url"],
            headers={"User-Agent": "Mozilla/5.0"}, # on se présente comme un navidateur, pour eviter que certains sites bloquent les scripts
        )
        
        try:   # on essaie le telechargement. s'il échoue, on saute au bloc except
            with urllib.request.urlopen(requete, timeout=60) as reponse: # envoie la requete; abandon après 60 s
                contenu = reponse.read() #recupere tout le contenu recu, sous forme d'octets
                
            if not contenu.startswith(b"%PDF"): # tout vrai PDF commence par les octets "%PDF"; sinon c'est souvent une page html d'erreur
                print(f"Pas un pdf: {destination.name} (telecharger à la main)")
                continue  # on n'enregistre pas ce faux fichier
            
            destination.write_bytes(contenu) # écrit les octets recus dans le fichier sur le disque
            print(f"OK: {destination.name} ({len(contenu) // 2024} ko)")  # // == division entiere
        except Exception as erreur:    # attrape toute erreur (site indisponible, accès refusé, délai dépassé)
            print(f"ECHEC : {destination.name} -> {erreur}")  # On affiche l'erreur et la boucle continue avec le document suivant