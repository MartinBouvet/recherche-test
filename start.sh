#!/bin/bash
set -e

echo "🚀 Démarrage du système..."

# Démarrer Ollama en arrière-plan
echo "📦 Lancement d'Ollama..."
ollama serve &
OLLAMA_PID=$!

# Attendre qu'Ollama soit prêt
echo "⏳ Attente du démarrage d'Ollama..."
sleep 5

# Vérifier si Ollama est accessible
until curl -s http://localhost:11434/api/tags > /dev/null; do
    echo "⏳ Ollama n'est pas encore prêt, nouvelle tentative..."
    sleep 2
done

echo "✅ Ollama est prêt!"

# Télécharger le modèle Ministral-8B s'il n'existe pas
echo "📥 Vérification du modèle Ministral-8B..."
if ! ollama list | grep -q "ministral-8b"; then
    echo "📥 Téléchargement de Ministral-8B (peut prendre quelques minutes)..."
    ollama pull nchapman/ministral-8b-instruct-2410:8b
    echo "✅ Ministral-8B téléchargé!"
else
    echo "✅ Ministral-8B déjà présent"
fi

# Lancer l'application Gradio
echo "🎨 Lancement de l'application Gradio..."
python app_gradio.py

# Si l'app crash, garder Ollama en vie
wait $OLLAMA_PID