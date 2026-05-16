import requests
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)

OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
DIRETORIO_ATUAL = os.path.dirname(__file__)

# Ajuste os caminhos conforme sua estrutura de pastas
CAMINHO_BASE = os.path.abspath(os.path.join(DIRETORIO_ATUAL, '..', 'base_conhecimento', 'base_conhecimentos.txt'))
CAMINHO_NPY = os.path.abspath(os.path.join(DIRETORIO_ATUAL, '..', 'base_conhecimento', 'embedding.npy'))

def carregar_base():
    if not os.path.exists(CAMINHO_BASE):
        logging.error(f"Arquivo não encontrado: {CAMINHO_BASE}")
        return []
        
    with open(CAMINHO_BASE, "r", encoding="utf-8") as f:
        texto = f.read()
    
    chunks_brutos = texto.split("\n\n")
    return [chunk.strip() for chunk in chunks_brutos if len(chunk.strip()) > 10]

def gerar_embedding(texto):
    try:
        response = requests.post(OLLAMA_EMBED_URL, json={
            "model": "nomic-embed-text",
            "prompt": texto
        }, timeout=10)
        response.raise_for_status()
        return response.json().get("embedding")
    except Exception as e:
        logging.error(f"Erro ao gerar embedding: {e}")
        return None

def inicializar_embeddings():
    base_chunks = carregar_base()
    if not base_chunks:
        return []
    
    # Lógica de Cache Inteligente: Verifica se o TXT é mais novo que o NPY
    precisa_atualizar = True
    if os.path.exists(CAMINHO_NPY):
        tempo_mod_txt = os.path.getmtime(CAMINHO_BASE)
        tempo_mod_npy = os.path.getmtime(CAMINHO_NPY)
        
        if tempo_mod_npy >= tempo_mod_txt:
            precisa_atualizar = False
            logging.info("Base de conhecimento não foi alterada. Carregando embeddings do disco...")
            try:
                matriz_embeddings = np.load(CAMINHO_NPY)
                return [
                    {"texto": chunk, "embedding": matriz_embeddings[i]} 
                    for i, chunk in enumerate(base_chunks) 
                    if i < len(matriz_embeddings)
                ]
            except Exception as e:
                logging.error("Erro ao ler embedding.npy, recriando...")
                precisa_atualizar = True

    if precisa_atualizar:
        logging.info("Gerando novos embeddings... Isso pode levar alguns segundos.")
        base_embeddings = []
        lista_vetores = []
        
        for chunk in base_chunks:
            emb = gerar_embedding(chunk)
            if emb:
                emb_array = np.array(emb)
                base_embeddings.append({"texto": chunk, "embedding": emb_array})
                lista_vetores.append(emb_array)
            
        if lista_vetores:
            np.save(CAMINHO_NPY, np.array(lista_vetores))
            logging.info("Novos embeddings salvos com sucesso.")
        
        return base_embeddings

# Inicializa globalmente na primeira importação
base_embeddings = inicializar_embeddings()

def buscar_contexto(pergunta, top_k=3):
    if not base_embeddings:
        return []

    emb_pergunta = gerar_embedding(pergunta)
    if not emb_pergunta:
        return []
        
    emb_pergunta = np.array(emb_pergunta)
    similaridades = []
    
    for item in base_embeddings:
        # Cálculo de Similaridade de Cosseno
        sim = np.dot(emb_pergunta, item["embedding"]) / (
            np.linalg.norm(emb_pergunta) * np.linalg.norm(item["embedding"])
        )
        similaridades.append((sim, item["texto"]))
    
    # Ordena do maior para o menor e filtra resultados irrelevantes (Threshold > 0.5)
    similaridades.sort(reverse=True, key=lambda x: x[0])
    return [texto for sim, texto in similaridades[:top_k] if sim > 0.5]