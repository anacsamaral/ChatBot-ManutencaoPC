import requests
import numpy as np
import os

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
DIRETORIO_ATUAL = os.path.dirname(__file__)
CAMINHO_BASE = os.path.abspath(os.path.join(DIRETORIO_ATUAL, '..', 'base_conhecimento', 'base_conhecimentos.txt'))
CAMINHO_NPY = os.path.abspath(os.path.join(DIRETORIO_ATUAL, '..', 'base_conhecimento', 'embedding.npy'))

def carregar_base():
    with open(CAMINHO_BASE, "r", encoding="utf-8") as f:
        texto = f.read()
    
    chunks_brutos = texto.split("\n\n")
    return [chunk.strip() for chunk in chunks_brutos if chunk.strip()]

def gerar_embedding(texto):
    response = requests.post(OLLAMA_EMBED_URL, json={
        "model": "nomic-embed-text",
        "prompt": texto
    })
    return response.json()["embedding"]

def inicializar_embeddings():
    base_chunks = carregar_base()
    
    if os.path.exists(CAMINHO_NPY):
        matriz_embeddings = np.load(CAMINHO_NPY)
        
        return [
            {"texto": chunk, "embedding": matriz_embeddings[i]} 
            for i, chunk in enumerate(base_chunks) 
            if i < len(matriz_embeddings)
        ]
    
    base_embeddings = []
    lista_vetores = []
    
    for chunk in base_chunks:
        emb = np.array(gerar_embedding(chunk))
        base_embeddings.append({"texto": chunk, "embedding": emb})
        lista_vetores.append(emb)
        
    np.save(CAMINHO_NPY, np.array(lista_vetores))
    
    return base_embeddings

base_embeddings = inicializar_embeddings()

def buscar_contexto(pergunta, top_k=3):
    emb_pergunta = np.array(gerar_embedding(pergunta))
    similaridades = []
    
    for item in base_embeddings:
        sim = np.dot(emb_pergunta, item["embedding"]) / (
            np.linalg.norm(emb_pergunta) * np.linalg.norm(item["embedding"])
        )
        similaridades.append((sim, item["texto"]))
    
    similaridades.sort(reverse=True, key=lambda x: x[0])
    
    return [texto for _, texto in similaridades[:top_k]]