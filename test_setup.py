import ollama

#Test du LLM
rep = ollama.chat(model="qwen2.5:7b",
                  messages=[{"role": "user", "content": "Explique le RAG en une phrase"}])
print("LLM:", rep["message"]["content"])

#Test de l'embedding
emb = ollama.embed(model="bge-m3", input="Le prix du cacao bord champ")
vecteur = emb["embeddings"][0]
print("Dimension du vecteur : ", len(vecteur))
print("5 premieres valeurs :", vecteur[:5])