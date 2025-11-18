# Dockerfile pour Railway avec Ollama + Ministral-8B
FROM python:3.11-slim

# Installation des dépendances système
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Installation d'Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Création du répertoire de travail
WORKDIR /app

# Copie des fichiers requirements
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie de tous les fichiers du projet
COPY . .

# Script de démarrage qui lance Ollama ET l'app
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Exposition du port Gradio
EXPOSE 7860

# Commande de démarrage
CMD ["/app/start.sh"]