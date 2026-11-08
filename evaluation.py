import csv                                   # pour lire eval.csv et écrire les résultats
from datetime import datetime                # pour dater le fichier de résultats
from pathlib import Path
import ollama
from rag import repondre, MODELE_LLM         # on réutilise le pipeline de rag.py : on évalue EXACTEMENT le même bot

FICHIER_EVAL = Path("data/eval.csv")
DOSSIER_RESULTATS = Path("resultats")        # chaque évaluation y sera enregistrée, pour comparer les versions
PHRASE_REFUS = "je ne trouve pas cette information"  # la phrase de refus imposée par le prompt de rag.py (en minuscules)

PROMPT_JUGE = """Tu es un correcteur rigoureux. Compare la réponse d'un assistant à la réponse attendue.

Question : {question}
Réponse attendue : {attendue}
Réponse de l'assistant : {obtenue}

- CORRECTE : la réponse contient l'information essentielle attendue, même formulée autrement ou avec des détails en plus.
- PARTIELLE : elle ne contient qu'une partie de l'information attendue.
- INCORRECTE : l'information est fausse ou absente, ou l'assistant dit ne pas savoir.

Règles strictes :
- Si un chiffre, une date ou un montant diffère de la réponse attendue, la réponse est INCORRECTE.
- Si la réponse donne d'autres éléments que ceux attendus sans les mentionner, elle est INCORRECTE.
- Si elle ne contient qu'une partie des éléments attendus, elle est PARTIELLE.

Réponds par un seul mot : CORRECTE, PARTIELLE ou INCORRECTE."""
# Les {question}, {attendue} et {obtenue} sont des emplacements remplis plus bas par .format(...)


def juger(question, attendue, obtenue):
    """Demande au LLM juge si la réponse obtenue correspond à la réponse attendue."""
    prompt = PROMPT_JUGE.format(question=question, attendue=attendue, obtenue=obtenue)  # remplit les emplacements
    rep = ollama.chat(model=MODELE_LLM, messages=[{"role": "user", "content": prompt}], options={"temperature": 0})
    texte = rep["message"]["content"].upper()  # en majuscules, pour comparer sans se soucier de la casse
    for verdict in ("INCORRECTE", "PARTIELLE", "CORRECTE"):  # INCORRECTE en premier : le mot "CORRECTE" est contenu dans "INCORRECTE" !
        if verdict in texte:
            return verdict
    return "INDETERMINE"                     # si le juge n'a pas respecté le format demandé

def rang_bon_passage(passages, attendus):
    """Renvoie le rang du premier passage dont le couple (fichier, page) fait partie des sources attendues, ou None."""
    for rang, p in enumerate(passages, start=1):
        if (p["meta"]["fichier"], p["meta"]["page"]) in attendus:  # un couple (fichier, page) se compare d'un coup
            return rang
    return None

# 1. Charger les questions
with FICHIER_EVAL.open(encoding="utf-8") as f:
    questions = list(csv.DictReader(f))      # une liste de dictionnaires, un par ligne du CSV
print(f"{len(questions)} questions chargées\n")

# 2. Faire passer l'examen
resultats = []
for q in questions:
    print(f"{q['id']} ({q['type']})...", end=" ", flush=True)  # flush=True affiche le texte tout de suite, sans attendre la fin de la ligne
    reponse, passages = repondre(q["question"])  # le bot répond, exactement comme dans rag.py
    a_refuse = PHRASE_REFUS in reponse.lower()   # le bot a-t-il dit "je ne trouve pas" ?

    if q["type"] == "hors_corpus":
        rang = None                          # pas de "bon passage" à chercher
        verdict = "CORRECTE" if a_refuse else "INCORRECTE"  # ici, la bonne réponse EST le refus
    else:
        attendus = set()   # ensemble des couples (fichier, page) acceptés
        dernier_fichier = None  # mémorise le dernier fichier cité, pour les pages abrégées
        for s in q["sources"].split(";"):
            s = s.strip()  # retire les espaces parasites
            if not s: # élément vide (point-virgule en trop) : on l'ignore
                continue
            if ":" in s: # forme complète : "fichier.pdf:11"
                fichier, page = s.rsplit(":", 1)  # coupe au dernier ":"
                dernier_fichier = fichier.strip()  # ce fichier devient le "dernier cité"
            elif dernier_fichier:  # forme abrégée : "30" seul = même fichier que juste avant 
                page = s
            else:   # une page seule en premier : impossible de savoir de quel fichier il s'agit
                raise ValueError(f"{q['id']} : source mal formée '{s}' (format attendu : fichier.pdf:page)")
            attendus.add((dernier_fichier, int(page)))  # ajoute le couple (fichier, page)
        rang = rang_bon_passage(passages, attendus)
        verdict = "INCORRECTE" if a_refuse else juger(q["question"], q["reponse_attendue"], reponse)  # un refus sur une question qui a une réponse = échec, pas besoin du juge

    resultats.append({
        "id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "reponse_attendue": q["reponse_attendue"],
        "reponse_bot": reponse,
        "rang_bon_passage": rang if rang else "",  # vide si non trouvé
        "verdict": verdict,
        "sources": " | ".join(f"{p['meta']['fichier']} p.{p['meta']['page']}" for p in passages),  # les 5 passages récupérés, pour analyser les échecs
    })
    print(f"retrieval : {'rang ' + str(rang) if rang else ('-' if q['type'] == 'hors_corpus' else 'ÉCHEC')} | réponse : {verdict}")
    
# 3. Calculer les scores
POINTS = {"CORRECTE": 1, "PARTIELLE": 0.5, "INCORRECTE": 0, "INDETERMINE": 0}  # barème de l'exactitude

def scores(lignes):
    """Calcule recall@5, MRR et exactitude sur un groupe de questions qui ont une réponse dans le corpus."""
    n = len(lignes)
    recall = sum(1 for r in lignes if r["rang_bon_passage"] != "") / n
    mrr = sum(1 / r["rang_bon_passage"] for r in lignes if r["rang_bon_passage"] != "") / n  # 1/rang : 1 si premier, 0,5 si deuxième...
    exactitude = sum(POINTS[r["verdict"]] for r in lignes) / n
    return recall, mrr, exactitude

print("\n========== RÉSULTATS ==========")
for groupe in ("factuel", "multilingue"):
    lignes = [r for r in resultats if r["type"] == groupe]
    if lignes:
        rc, mrr, ex = scores(lignes)
        print(f"{groupe:12} ({len(lignes):2} q.) | recall@5 {rc:5.0%} | MRR {mrr:.2f} | exactitude {ex:5.0%}")

avec_reponse = [r for r in resultats if r["type"] != "hors_corpus"]
rc, mrr, ex = scores(avec_reponse)
print(f"{'TOTAL':12} ({len(avec_reponse):2} q.) | recall@5 {rc:5.0%} | MRR {mrr:.2f} | exactitude {ex:5.0%}")

hors = [r for r in resultats if r["type"] == "hors_corpus"]
print(f"Hors corpus correctement refusés : {sum(r['verdict'] == 'CORRECTE' for r in hors)}/{len(hors)}")

# 4. Enregistrer le détail pour l'analyse
DOSSIER_RESULTATS.mkdir(exist_ok=True)
fichier_sortie = DOSSIER_RESULTATS / f"eval_{datetime.now():%Y%m%d_%H%M}.csv"  # ex. resultats/eval_20261107_2245.csv : un fichier par évaluation, jamais écrasé
with fichier_sortie.open("w", encoding="utf-8", newline="") as f:  # newline="" évite des lignes vides parasites sous Windows
    writer = csv.DictWriter(f, fieldnames=resultats[0].keys())
    writer.writeheader()
    writer.writerows(resultats)
print(f"\nDétail enregistré dans {fichier_sortie}")