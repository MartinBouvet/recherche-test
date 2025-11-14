#!/bin/bash

# Script de démarrage du serveur de recherche intelligente

echo "🚀 Démarrage du serveur de recherche intelligente..."
echo ""

# Vérifier que Python est installé
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 n'est pas installé"
    exit 1
fi

# Vérifier que Ollama est disponible
if ! command -v ollama &> /dev/null; then
    echo "⚠️  Ollama n'est pas installé ou n'est pas dans le PATH"
    echo "   Assurez-vous qu'Ollama est installé et que le modèle est disponible"
fi

# Vérifier que les dépendances sont installées
if [ ! -d "venv" ]; then
    echo "📦 Création de l'environnement virtuel..."
    python3.11 -m venv venv 2>/dev/null || python3 -m venv venv
fi

echo "📦 Activation de l'environnement virtuel..."
source venv/bin/activate

# Vérifier si les dépendances sont installées
if ! python -c "import fastapi" 2>/dev/null; then
    echo "📦 Installation des dépendances..."
    pip install -q -r requirements.txt
else
    echo "✅ Dépendances déjà installées"
fi

echo ""
echo "✅ Démarrage du serveur sur http://localhost:8000"
echo "   Appuyez sur Ctrl+C pour arrêter"
echo ""

python app.py

