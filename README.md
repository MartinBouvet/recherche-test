# Moteur de Recherche Intelligent pour Matériaux de Construction

Ce projet propose une solution de recherche intelligente pour un catalogue de matériaux de construction. Contrairement aux moteurs de recherche classiques qui nécessitent des requêtes exactes, celui-ci comprend le langage naturel et s'adapte aux besoins réels des utilisateurs.

## Contexte et Objectifs

L'objectif était de créer un moteur de recherche capable de rivaliser avec des solutions professionnelles comme Algolia ou ElasticSearch, mais sans dépendre de services externes. Le défi principal n'était pas seulement d'intégrer une interface IA, mais de rendre les données exploitables, cohérentes et rapidement consultables.

## Fonctionnalités Principales

### Compréhension du Langage Naturel
Le système utilise un modèle LLM (Large Language Model) pour comprendre les intentions derrière les requêtes. Vous pouvez taper "je cherche du sable pour faire du béton" et le système comprendra que vous recherchez des agrégats fins pour béton, pas du sable de plage.

### Recherche Hybride
La recherche combine deux approches complémentaires :
- **Recherche sémantique** : Utilise des embeddings pour trouver des produits similaires même si les mots exacts ne sont pas présents
- **Recherche textuelle** : Recherche classique par mots-clés avec un système de scoring intelligent

### Gestion des Fautes de Frappe
Le système corrige automatiquement les erreurs de saisie. Tapez "sabble" au lieu de "sable", et vous obtiendrez quand même des résultats pertinents.

### Synonymes Métier
Un dictionnaire de synonymes spécialisé permet de comprendre que "béton" peut aussi signifier "ciment" ou "mortier" selon le contexte, ou que "pavé" et "dalle" sont souvent interchangeables.

### Filtrage par Prix
Le système comprend les contraintes de budget exprimées naturellement. Par exemple, "carrelage gris extérieur à moins de 40 euros" filtrera automatiquement les résultats selon le prix.

### Recherche Vocale
Une fonctionnalité de recherche vocale permet de dicter vos requêtes au lieu de les taper. Cliquez sur le bouton micro, parlez votre demande (ex: "je cherche des bétonnières de 25 L"), et le système transcrit automatiquement votre demande avant d'effectuer la recherche. Une phrase de confirmation personnalisée s'affiche ensuite pour confirmer la compréhension de votre demande.

### Performance Optimisée
- Cache des embeddings pour éviter de les régénérer à chaque démarrage
- Cache des résultats de recherche pour les requêtes fréquentes
- Recherche rapide (généralement moins de 200ms)

## Installation

### Prérequis

- Python 3.11 ou supérieur
- Ollama installé et configuré
- Le modèle `nchapman/ministral-8b-instruct-2410:8b` disponible
- Un navigateur moderne avec support de l'API MediaRecorder (pour la recherche vocale)

### Étapes d'Installation

1. **Cloner ou télécharger le projet**

2. **Créer un environnement virtuel** (recommandé) :
```bash
python -m venv venv
source venv/bin/activate  # Sur macOS/Linux
# ou
venv\Scripts\activate  # Sur Windows
```

3. **Installer les dépendances** :
```bash
pip install -r requirements.txt
```

4. **Vérifier qu'Ollama est installé et que le modèle est disponible** :
```bash
ollama list
# Si le modèle n'est pas présent :
ollama pull nchapman/ministral-8b-instruct-2410:8b
```

5. **Lancer Ollama** (dans un terminal séparé) :
```bash
ollama serve
```

## Utilisation

### Démarrer le Serveur

```bash
source venv/bin/activate  # Si vous utilisez un venv
python app.py
```

Le serveur démarre sur `http://localhost:8000` par défaut.

### Interface Web

Ouvrez simplement votre navigateur à l'adresse `http://localhost:8000` et utilisez la barre de recherche. Vous pouvez formuler vos requêtes en langage naturel :

- "sable pour béton"
- "isolation laine de roche"
- "pavé terrasse"
- "carrelage gris extérieur à moins de 40 euros"
- "je veux faire un mur en béton"

**Recherche vocale** : Cliquez sur le bouton micro 🎤 à côté de la barre de recherche, parlez votre demande, puis cliquez à nouveau pour arrêter l'enregistrement. Le système transcrit automatiquement votre demande et effectue la recherche.

### API REST

Le système expose une API REST simple :

**Recherche (GET)** :
```bash
curl "http://localhost:8000/api/search?q=carrelage%20extérieur&limit=10"
```

**Recherche (POST)** :
```bash
curl -X POST "http://localhost:8000/api/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "sable pour béton", "limit": 10}'
```

**Transcription Audio** (POST) :
```bash
curl -X POST "http://localhost:8000/api/transcribe" \
  -F "audio=@recording.wav"
```

**Health Check** :
```bash
curl "http://localhost:8000/api/health"
```

## Architecture Technique

### Structure du Projet

- **`app.py`** : Application FastAPI principale, gère les endpoints et le cycle de vie
- **`data_loader.py`** : Charge et parse le fichier SQL, nettoie les données
- **`search_engine.py`** : Cœur du moteur de recherche avec LLM, embeddings et scoring
- **`synonyms.py`** : Dictionnaire de synonymes métier et fonctions d'expansion
- **`templates/index.html`** : Interface web frontend
- **`analyze_relevance.py`** : Script d'analyse de la pertinence des résultats
- **`check_images.py`** : Script de vérification des références d'images

### Flux de Recherche

1. **Compréhension LLM** : Le modèle analyse la requête et extrait :
   - Les termes techniques principaux
   - Les contraintes (prix, dimensions, etc.)
   - Une reformulation optimisée pour la recherche

2. **Correction des Fautes de Frappe** : Les termes sont comparés avec le vocabulaire du catalogue pour corriger les erreurs

3. **Expansion par Synonymes** : Les termes sont enrichis avec leurs synonymes métier

4. **Recherche Hybride** :
   - **Textuelle (85%)** : Recherche par mots-clés avec scoring intelligent
   - **Sémantique (15%)** : Recherche par similarité d'embeddings

5. **Filtrage** : Application des contraintes (prix, score minimum)

6. **Tri et Retour** : Résultats triés par pertinence

## Configuration

### Modèle LLM

Par défaut, le système utilise `nchapman/ministral-8b-instruct-2410:8b`. Pour changer de modèle, modifiez dans `search_engine.py` :

```python
model_name: str = "votre-modele"
```

### Modèle d'Embeddings

Le modèle par défaut est `paraphrase-multilingual-MiniLM-L12-v2` (multilingue, léger, rapide). Pour changer :

```python
self.embedding_model = SentenceTransformer('votre-modele')
```

### Modèle Whisper (Recherche Vocale)

Le modèle par défaut est `base` (bon compromis vitesse/qualité). Pour améliorer la précision, vous pouvez utiliser un modèle plus grand :

**Modèles disponibles** (du plus rapide au plus précis) :
- `tiny` : Très rapide, moins précis (~39 Mo)
- `base` : Bon compromis, par défaut (~74 Mo)
- `small` : Meilleure qualité, un peu plus lent (~244 Mo)
- `medium` : Très bonne qualité, plus lent (~769 Mo)
- `large` : Meilleure qualité, beaucoup plus lent (~1550 Mo)

**Pour changer le modèle**, définissez la variable d'environnement avant de lancer l'application :

```bash
export WHISPER_MODEL_SIZE=small
python app.py
```

Ou créez un fichier `.env` :
```
WHISPER_MODEL_SIZE=small
```

**Optimisations activées** :
- Prompt initial avec vocabulaire métier pour guider la transcription
- Paramètres optimisés (temperature=0.0, beam_size=5, best_of=5)
- Post-traitement automatique pour corriger les termes techniques

### Désactiver le LLM

Pour des recherches ultra-rapides sans compréhension contextuelle, vous pouvez désactiver le LLM :

```python
search_engine = SearchEngine(articles, use_llm=False)
```

## Gestion des Images

Les images sont référencées dans la base de données par leur nom de fichier. Pour qu'elles s'affichent correctement :

1. Créez un dossier `images/` à la racine du projet :
```bash
mkdir images
```

2. Placez les images dans ce dossier avec les noms correspondants à ceux de la base de données (ex: `49.jpg`, `2060.jpg`)

3. L'API servira automatiquement les images via l'endpoint `/images/{filename}`

Si une image n'existe pas, un placeholder sera affiché automatiquement dans l'interface.

**Note** : Les noms de fichiers dans la base de données sont automatiquement nettoyés (quotes supprimées). Si vous voyez des erreurs 404, vérifiez que les noms correspondent exactement.

## Performance et Optimisations

### Cache des Embeddings

Les embeddings sont générés une première fois et sauvegardés dans `embeddings_cache.npy`. Au démarrage suivant, ils sont chargés depuis le cache, ce qui économise environ 12-15 secondes.

### Cache des Résultats

Les résultats de recherche sont mis en cache en mémoire pour les requêtes identiques, permettant des réponses quasi-instantanées pour les recherches répétées.

### Temps de Réponse

- **Premier lancement** : ~30-60 secondes (génération des embeddings + chargement Whisper)
- **Lancements suivants** : ~2-5 secondes (chargement depuis cache)
- **Recherche** : ~80-200ms (selon la complexité de la requête)
- **Transcription vocale** : ~1-3 secondes (selon la longueur de l'audio)

## Scripts Utiles

### Analyse de Pertinence

Pour analyser la qualité des résultats et des données :

```bash
python analyze_relevance.py
```

Ce script permet de :
- Vérifier la qualité des données chargées
- Tester la pertinence des recherches avec des requêtes prédéfinies
- Analyser en détail une requête spécifique

### Vérification des Images

Pour voir quelles images sont référencées dans la base de données :

```bash
python check_images.py
```

## Limitations et Améliorations Futures

### Limitations Actuelles

- Le catalogue est chargé en mémoire (limité à quelques dizaines de milliers d'articles)
- Pas de facettes avancées (filtrage par catégorie, marque, etc.)
- Le LLM peut parfois être lent (dépend de votre configuration)

### Améliorations Possibles

1. **Base de données vectorielle** : Utiliser FAISS, Qdrant ou Pinecone pour gérer des millions d'articles
2. **Facettes et filtres** : Ajouter des filtres par catégorie, prix, dimensions, etc.
3. **Historique et suggestions** : Sauvegarder les recherches fréquentes et proposer des suggestions
4. **A/B Testing** : Tester différents poids pour la recherche hybride
5. **Analytics** : Suivre les recherches les plus fréquentes et les produits les plus consultés

## Dépannage

### Erreur "Model not found"

Assurez-vous qu'Ollama est lancé et que le modèle est disponible :
```bash
ollama serve
ollama pull nchapman/ministral-8b-instruct-2410:8b
```

### Erreur de chargement des données

Vérifiez que le fichier SQL (`newflat_sograma_produits_seuls.sql`) est présent et lisible.

### Recherche lente

- Réduisez le nombre de résultats (`limit`)
- Augmentez `min_score` pour filtrer les résultats peu pertinents
- Désactivez le LLM avec `use_llm=False` pour des recherches plus rapides

### Images qui ne s'affichent pas

- Vérifiez que le dossier `images/` existe
- Vérifiez que les noms de fichiers correspondent exactement (sans quotes)
- Utilisez `check_images.py` pour voir quelles images sont référencées

### Recherche vocale ne fonctionne pas

- **Whisper non chargé** : Vérifiez les logs au démarrage. Si Whisper ne se charge pas, la recherche vocale sera désactivée mais l'application fonctionnera normalement pour la recherche textuelle.
- **Permissions microphone** : Assurez-vous que votre navigateur a les permissions d'accès au microphone. Dans Chrome/Edge, cliquez sur l'icône de cadenas dans la barre d'adresse.
- **Format audio non supporté** : Whisper accepte plusieurs formats (WAV, MP3, WebM, OGG). Si vous avez des problèmes, essayez un autre navigateur.
- **Premier chargement lent** : Le modèle Whisper "base" fait environ 74 Mo et est téléchargé automatiquement au premier lancement. Cela peut prendre quelques minutes selon votre connexion.
- **Transcription imprécise** : Si la transcription n'est pas assez précise, essayez d'utiliser un modèle plus grand (small, medium, ou large) via la variable d'environnement `WHISPER_MODEL_SIZE`. Notez que cela augmentera le temps de transcription et la consommation mémoire.

## Contribution

Ce projet a été développé dans le cadre d'une démonstration de faisabilité. N'hésitez pas à proposer des améliorations ou signaler des bugs !

## Licence

Ce projet est fourni tel quel, sans garantie particulière.
