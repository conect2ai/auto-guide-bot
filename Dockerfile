# Use a imagem base do Python
FROM python:3.10-slim

# Atualize os pacotes
RUN apt-get update && apt-get install -y && rm -rf /var/lib/apt/lists/*

# Defina o diretório de trabalho
WORKDIR /app

# Copie o arquivo de requirements e instale as dependências
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copie a pasta img para o contêiner
COPY img/ img/

# Copie o restante dos arquivos da sua aplicação
COPY . .

# Inicie sua aplicação Streamlit
CMD ["streamlit", "run", "app_chat_automotive.py"]