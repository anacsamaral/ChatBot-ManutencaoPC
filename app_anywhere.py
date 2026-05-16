from flask import Flask, request, jsonify
import requests
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Aponta para o backend principal
URL_BACKEND_LOCAL = "http://localhost:5000/chat"

@app.route('/webhook', methods=['POST'])
def dialogflow_webhook():
    req = request.get_json(silent=True, force=True)
    
    if not req:
        return jsonify({"fulfillmentText": "Payload inválido recebido."})

    try:
        query_result = req.get("queryResult", {})
        mensagem = query_result.get("queryText", "")
        # O Dialogflow envia um session ID único que identifica a conversa do usuário
        session_id = req.get("session", "sessao_dialogflow_padrao")
        
        if not mensagem:
            return jsonify({"fulfillmentText": "Não consegui compreender a mensagem."})

        # Encaminha a mensagem para o nosso motor inteligente (app.py)
        res = requests.post(URL_BACKEND_LOCAL, json={
            "message": mensagem,
            "session_id": session_id
        }, timeout=45)
        
        res.raise_for_status()
        resposta_ia = res.json().get("reply", "Erro ao extrair resposta da IA.")
        
        return jsonify({
            "fulfillmentText": resposta_ia,
            "source": "suporte-ti-webhook"
        })

    except requests.exceptions.Timeout:
        return jsonify({"fulfillmentText": "O diagnóstico está demorando muito. Pode repetir a pergunta?"})
    except Exception as e:
        logging.error(f"Erro no webhook: {e}")
        return jsonify({"fulfillmentText": "Ocorreu um erro interno de comunicação no servidor."})

if __name__ == '__main__':
    # Roda em porta diferente para não conflitar com o Flask principal
    app.run(port=5001, debug=True)