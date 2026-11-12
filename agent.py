import time # pour mesurer la durée de chaque réponse
import ast  # pour analyser une expression mathématique sans l'exécuter comme du code
import json   # pour comparer des arguments d'appels (détection des répétitions)
import operator    # les opérations +, -, *, / sous forme de fonctions
import ollama
from rag import rechercher, MODELE_LLM  # on réutilise la recherche hybride bilingue (v1b) et le même modèle


MAX_ITERATIONS = 6  # garde-fou : nombre maximal de tours de boucle
AFFICHER_TRACE = True  # affiche chaque appel d'outil : c'est notre "observabilité" maison

PROMPT_AGENT = """Tu es CacaoBot, un assistant de recherche sur la filière cacao.
Tu disposes de deux outils :
- rechercher_documents : cherche des passages dans un corpus de rapports sur le cacao.
- calculer : effectue un calcul arithmétique exact.

Méthode :
1. Si la question porte sur la filière cacao, utilise rechercher_documents avant de répondre. Tu peux faire plusieurs recherches : une par sous-question, ou une reformulation si les résultats sont insuffisants.
2. Pour tout calcul (différence, somme, pourcentage), utilise l'outil calculer. Ne calcule jamais de tête.
3. Réponds uniquement à partir des passages trouvés. Cite chaque information sous la forme (Organisme, année, p. X).
4. Si les documents ne contiennent pas la réponse, réponds : "Je ne trouve pas cette information dans les documents."
5. Réponds dans la langue de la question."""

# Description des outils pour le LLM. C'est TOUT ce qu'il voit pour décider : les descriptions doivent être précises.
OUTILS = [
    {
        "type": "function",
        "function": {
            "name": "rechercher_documents",
            "description": "Cherche dans le corpus (Banque mondiale, ICCO, NORC, Fairtrade, DG Trésor, règlement européen RDUE, etc.) "
                           "les 5 passages les plus pertinents pour une requête. Renvoie chaque passage avec sa source.",
            "parameters": {
                "type": "object",
                "properties": {
                    "requete": {"type": "string", "description": "La requête de recherche, avec des mots-clés précis."},
                },
                "required": ["requete"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculer",
            "description": "Calcule une expression arithmétique avec + - * / et des parenthèses. "
                           "Utilise le point pour les décimales et aucun séparateur de milliers. Exemple : '1250 - 225 - 275'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "L'expression, uniquement des nombres et des opérateurs."},
                },
                "required": ["expression"],
            },
        },
    },
]

# ---------- Les fonctions réellement exécutées par le code ----------

OPERATEURS = {  # les seules opérations autorisées
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.USub: operator.neg,   # le signe moins devant un nombre : -5
}


def evaluer(noeud):
    """Calcule un arbre d'expression en refusant tout ce qui n'est pas un nombre ou une opération autorisée."""
    if isinstance(noeud, ast.Constant) and isinstance(noeud.value, (int, float)):
        return noeud.value                   # un nombre : on le renvoie tel quel
    if isinstance(noeud, ast.BinOp) and type(noeud.op) in OPERATEURS:
        return OPERATEURS[type(noeud.op)](evaluer(noeud.left), evaluer(noeud.right))  # opération à deux termes : on calcule chaque côté
    if isinstance(noeud, ast.UnaryOp) and type(noeud.op) in OPERATEURS:
        return OPERATEURS[type(noeud.op)](evaluer(noeud.operand))
    raise ValueError("élément non autorisé")  # tout le reste (noms, appels de fonction...) est refusé


def calculer(expression):
    """Outil calculatrice : renvoie un texte, jamais une exception, pour que le LLM puisse réagir à une erreur."""
    try:
        arbre = ast.parse(expression, mode="eval")  # transforme le texte en arbre, SANS l'exécuter
        return f"{expression} = {round(evaluer(arbre.body), 4)}"
    except Exception as erreur:
        return f"Erreur : impossible de calculer '{expression}' ({erreur}). Utilise seulement des nombres et + - * / ( )."


def rechercher_documents(requete):
    """Outil de recherche : met en forme les passages avec leur source, pour que le LLM puisse les citer."""
    passages = rechercher(requete)
    if not passages:
        return "Aucun résultat. Essaie d'autres mots-clés."
    blocs = [f"({p['meta']['organisme']}, {p['meta']['annee']}, p. {p['meta']['page']})\n{p['texte']}" for p in passages]
    return "\n\n".join(blocs)


FONCTIONS = {"rechercher_documents": rechercher_documents, "calculer": calculer}  # nom de l'outil -> fonction Python


# ---------- La boucle agentique ----------

def agent(question):
    messages = [
        {"role": "system", "content": PROMPT_AGENT},
        {"role": "user", "content": question},
    ]
    deja_fait = set()   # mémorise les appels déjà exécutés, pour détecter les boucles
    
    stats = {"llm":0, "traductions":0} # compteurs d'appels au LLM
    debut = time.time() # heure de départ

    for iteration in range(1, MAX_ITERATIONS + 1):
        rep = ollama.chat(model=MODELE_LLM, messages=messages, tools=OUTILS,
                          options={"temperature": 0, "num_ctx": 8192})
        stats["llm"] += 1 # un appel de l'agent lui-même
        message = rep.message
        messages.append(message)   # IMPORTANT : la demande du LLM entre dans l'historique

        if not message.tool_calls:  # aucun outil demandé : c'est la réponse finale
            if AFFICHER_TRACE:
                total = stats["llm"] + stats["traductions"]
                print(f"  [bilan] {stats['llm']} appels de l'agent + {stats['traductions']} traductions = {total} appels LLM, {time.time() - debut:.0f} s")
            return message.content

        for appel in message.tool_calls:  # le LLM peut demander plusieurs outils d'un coup
            nom = appel.function.name
            args = appel.function.arguments  # un dictionnaire, par exemple {"requete": "prix bord champ"}
            cle = (nom, json.dumps(args, sort_keys=True))  # signature de l'appel, pour repérer une répétition

            if nom not in FONCTIONS:
                resultat = f"Erreur : l'outil '{nom}' n'existe pas."
            elif cle in deja_fait:
                resultat = "Tu as déjà fait exactement cet appel. Utilise les résultats précédents ou reformule."
            else:
                deja_fait.add(cle)
                if nom == "rechercher_documents":
                    stats["traductions"] += 1 # en mode "hybride_bilingue", chaque recherche appelle aussi le LLM pour traduire
                try:
                    resultat = FONCTIONS[nom](**args)  # **args transforme le dictionnaire en arguments nommés
                except TypeError as erreur:          # mauvais nom d'argument inventé par le LLM
                    resultat = f"Erreur : arguments invalides ({erreur})."

            if AFFICHER_TRACE:
                print(f"  [tour {iteration}] {nom}({args}) -> {resultat[:100]}...")
            messages.append({"role": "tool", "content": resultat, "tool_name": nom})  # IMPORTANT : sans cette ligne, le LLM ne "voit" jamais le résultat

    return "Je n'ai pas pu terminer dans le nombre d'étapes autorisé."  # garde-fou atteint


if __name__ == "__main__":
    while True:
        question = input("\nTa question (q pour quitter) : ").strip()
        if question.lower() in ("q", ""):
            break
        print(f"\n--- Réponse ---\n{agent(question)}")