# 🚘 Chatbot Automotivo

Este repositório contém os arquivos necessários para configurar e executar uma aplicação Streamlit que interage com um banco de dados Milvus para processar e responder a perguntas baseadas em manuais automotivos em PDF apoiado por um modelo de linguagem da OpenAI.

## 📁 Estrutura do Repositório

- `app_chat_automotive.py`: script principal da aplicação Streamlit;
- `docker-compose.yml`: arquivo do Docker Compose para orquestrar os containers necessários;
- `Dockerfile`: procedimento para construir a imagem Docker da aplicação Streamlit;
- `requirements.txt`: dependências Python necessárias para a aplicação;
- `img/`: pasta contendo imagens utilizadas pela aplicação.

## ✅ Pré-requisitos

Antes de começar, você precisará ter o Docker 🐳 e o Docker Compose instalados em sua máquina. As instruções de instalação podem ser encontradas na [documentação oficial do Docker](https://docs.docker.com/get-docker/).

## 🚀 Instruções de Uso

1. **Clonar o Repositório**

   Clone este repositório para sua máquina local.

2. **Construir e Executar com o Docker Compose**

   No diretório raiz do projeto, execute o comando:
   ```bash
   docker-compose up --build
   ```
   Isso irá construir a imagem Docker para o aplicativo Streamlit usando o `Dockerfile` e também irá iniciar os serviços definidos no `docker-compose.yml`, como `etcd`, `minio`, e `milvus`.

4. **Acessar a Aplicação**

   Após os containers estarem em execução, a aplicação Streamlit pode ser acessada através do navegador em `http://localhost:8501`.

5. **Uso da Aplicação**

   - Carregue os arquivos PDF dos manuais automotivos através da interface do usuário 📤;
   - Processar os PDFs para construir a base de conhecimento 📚;
   - Faça perguntas através da interface de chat 💬 e receba respostas baseadas nos dados processados.
