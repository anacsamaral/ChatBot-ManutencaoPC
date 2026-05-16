from flask import Flask, request, jsonify, render_template
from groq import Groq
import uuid
import logging
import os
from dotenv import load_dotenv

# Carrega as variáveis contidas no arquivo .env para a memória do sistema
load_dotenv()

# Recupera a chave de forma segura da memória
API_KEY_GROQ = os.getenv("GROQ_API_KEY")

if not API_KEY_GROQ:
    logging.error("AVISO: A variável de ambiente GROQ_API_KEY não foi encontrada no arquivo .env!")

# Inicializa o cliente usando a variável protegida
client = Groq(api_key=API_KEY_GROQ)

# Import do seu módulo local de RAG
from gerar_embedding.gerar_embedding import buscar_contexto

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Isolamento de memória por sessão
historico_sessoes = {}

def processar_mensagem(mensagem, session_id):
    if session_id not in historico_sessoes:
        historico_sessoes[session_id] = []
        
    historico = historico_sessoes[session_id]
    historico.append(f"Usuário: {mensagem}")
    
    contextos = buscar_contexto(mensagem)
    contexto_texto = "\n\n".join(contextos) if contextos else "Nenhuma informação específica encontrada na base oficial."
    
    historico_formatado = "\n".join(historico[-6:])
    
    prompt = f"""
    Você é um técnico especialista em suporte de TI e manutenção de computadores.
    Sua missão é ajudar o usuário a diagnosticar e resolver problemas de hardware e software.
    Responda de forma clara, objetiva, em português brasileiro, e limite sua resposta a no máximo 2 parágrafos curtos.
    
    BASE DE CONHECIMENTO TÉCNICO (Use para embasar sua resposta):
    {contexto_texto}
    
    HISTÓRICO DA CONVERSA:
    {historico_formatado}
    
    Pergunta do Usuário: {mensagem}
    Sua Resposta:
    """

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Você é um técnico de TI focado em manutenção de computadores. Quando o usuário pedir um passo a passo de limpeza ou manutenção física, separe claramente as etapas em tópicos organizados e use as informações da base de forma lógica."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.1-8b-instant", 
            temperature=0.2,
            max_tokens=500
        )
        resposta = response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Erro na API Groq: {e}")
        resposta = "Desculpe, meu sistema de processamento central está indisponível."
    
    historico.append(f"Assistente: {resposta}")
    historico_sessoes[session_id] = historico
    
    return resposta

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat_unificado():
    data = request.json
    mensagem = data.get("message")
    session_id = data.get("session_id", "sessao_web_anonima")

    # Correção aqui: mudou de 'message' para 'mensagem'
    if not mensagem:
        return jsonify({"reply": "Por favor, descreva o problema."}), 400

    resposta = processar_mensagem(mensagem, session_id)
    return jsonify({"reply": resposta})

if __name__ == "__main__":
    app.run(port=5000, debug=True)