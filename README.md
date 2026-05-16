# 🛠️ ChatBot de Suporte de TI & Manutenção de PCs (RAG Architecture)

Este projeto consiste em um ecossistema inteligente de chatbot focado em suporte técnico de TI e manutenção de computadores. O sistema utiliza uma arquitetura **RAG (Retrieval-Augmented Generation)** para responder dúvidas de usuários com base em manuais e bases de conhecimento locais proprietárias, integrando o **Google Dialogflow ES** como interface de processamento de intenções e a API de alta velocidade da **Groq Cloud** para inferência da LLM.

## 🚀 Principais Funcionalidades

* **Arquitetura RAG Local:** Mecanismo de busca semântica local que consome uma base de conhecimentos em formato de texto (`base_conhecimentos.txt`), gerando embeddings vetoriais para contextualizar as respostas da inteligência artificial.
* **Inferência de Alta Velocidade (Fim dos Timeouts):** Integração com a API do Groq utilizando o modelo `llama-3.1-8b-instant`. O processamento em nuvem ocorre em milissegundos, solucionando o limite estrito de timeout de 5 segundos exigido pelos webhooks do Dialogflow ES.
* **Isolamento de Memória por Sessão:** Gerenciamento dinâmico de histórico baseado no `session_id`. Múltiplos usuários podem interagir simultaneamente sem vazamento de contexto ou misturas de históricos de conversas.
* **Interface Dupla:** Disponibilização de uma rota de Webhook para integração nativa com o Dialogflow ES e uma rota Web convencional (`/`) que renderiza uma interface de chat fluida em HTML/CSS/JS para testes isolados.
* **Segurança e Boas Práticas (.env):** Isolamento total de credenciais sensíveis através do uso de variáveis de ambiente (`python-dotenv`), evitando o vazamento acidental de chaves de API em repositórios públicos.

## 🛠️ Tecnologias Utilizadas

* **Linguagem:** Python 3.x
* **Framework Web:** Flask (Gerenciamento de rotas HTTP e Webhooks)
* **Orquestração de IA:** Groq SDK (`llama-3.1-8b-instant`)
* **Processamento de Vetores (RAG):** NumPy / Mecanismo de similaridade vetorial próprio
* **NLU / Processador de Linguagem Natural:** Google Dialogflow ES
* **Exposição de Túnel Local:** Ngrok (Para testes em tempo real do Webhook local)
* **Gerenciamento de Ambiente:** Python-dotenv & Ambientes Virtuais (`venv`)

## 📂 Estrutura do Projeto

```text
├── gerar_embedding/
│   ├── gerar_embedding.py   # Script responsável por carregar a base e fazer a busca vetorial
│   ├── base_conhecimentos.txt # Seu arquivo com procedimentos técnicos de suporte
│   └── embedding.npy        # Cache vetorial gerado localmente (ignorado no Git)
├── templates/
│   └── index.html           # Interface Web local para interação com o chatbot
├── .env                     # Arquivo de credenciais privadas (ignorado no Git)
├── .env.example             # Modelo de configuração para novos ambientes
├── .gitignore               # Filtro de arquivos protegidos para o Git
├── app.py                   # Servidor central Flask com as rotas de chat e webhook
└── requirements.txt         # Arquivo de dependências do ecossistema Python
