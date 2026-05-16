from flask import Flask, request, jsonify, render_template
import requests
from gerar_embedding.gerar_embedding import buscar_contexto

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"

historico = []

def processar_mensagem(mensagem):
    global historico
    
    historico.append(f"Usuário: {mensagem}")
    
    contextos = buscar_contexto(mensagem)
    contexto_texto = "\n\n".join(contextos)
    
    historico_formatado = "\n".join(historico[-10:])
    
    prompt = f"""
    Você é um especialista em suporte técnico de computadores.
    Responda de forma clara, objetiva e profissional.
    
    BASE DE CONHECIMENTO (Use as informações abaixo para embasar sua resposta):
    {contexto_texto}
    
    HISTÓRICO DA CONVERSA:
    {historico_formatado}
    
    Pergunta: {mensagem}
    Resposta:
    """

    response = requests.post(OLLAMA_URL, json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    })

    resposta = response.json()["response"]
    
    historico.append(f"Assistente: {resposta}")
    
    return resposta

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat_unificado():
    data = request.json
    mensagem = data.get("message")

    resposta = processar_mensagem(mensagem)

    return jsonify({"reply": resposta})

@app.route("/webhook", methods=["POST"])
def dialogflow_webhook():
    req = request.get_json(silent=True, force=True)
    
    try:
        mensagem = req.get("queryResult").get("queryText")
    except AttributeError:
        return jsonify({"fulfillmentText": "Erro ao processar a mensagem do Dialogflow."})

    resposta = processar_mensagem(mensagem)

    return jsonify({"fulfillmentText": resposta})

if __name__ == "__main__":
    app.run(debug=True)