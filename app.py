"""
API FastAPI pour la recherche intelligente de produits
Utilise un modèle LLM pour comprendre les requêtes en langage naturel
et effectue une recherche hybride (sémantique + textuelle)
"""

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
from contextlib import asynccontextmanager
import uvicorn
import whisper
import tempfile
import os
import re
from search_engine import SearchEngine
from data_loader import DataLoader

# Initialisation des composants
data_loader = DataLoader()
search_engine = None
whisper_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application"""
    # Startup
    global search_engine
    print("Chargement des données...")
    articles = data_loader.load_articles()
    print(f"{len(articles)} articles chargés")
    
    print("Initialisation du moteur de recherche...")
    search_engine = SearchEngine(articles)
    print("Moteur de recherche prêt!")
    
    print("Chargement du modèle Whisper pour la transcription vocale...")
    global whisper_model
    try:
        # Modèles disponibles (du plus rapide au plus précis) :
        # - tiny : très rapide, moins précis
        # - base : bon compromis (par défaut)
        # - small : meilleure qualité, un peu plus lent
        # - medium : très bonne qualité, plus lent
        # - large : meilleure qualité, beaucoup plus lent
        whisper_model_size = os.getenv("WHISPER_MODEL_SIZE", "base")  # Peut être changé via variable d'environnement
        whisper_model = whisper.load_model(whisper_model_size)
        print(f"Modèle Whisper '{whisper_model_size}' chargé!")
    except Exception as e:
        print(f"Attention: Impossible de charger Whisper ({e}). La recherche vocale sera désactivée.")
        whisper_model = None
    
    yield
    
    # Shutdown (si nécessaire)
    pass

app = FastAPI(
    title="Recherche Intelligente Produits",
    version="1.0.0",
    lifespan=lifespan
)

class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = 10
    min_score: Optional[float] = 0.0

class SearchResult(BaseModel):
    id: int
    libelle: str
    designation: Optional[str]
    reference: str
    prix_vente: float
    prix_vente_particulier: Optional[float]
    photo1: Optional[str]
    score: float
    categorieId: int
    motsClefs: Optional[str]

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query_understood: str
    total_results: int
    search_time: float
    confirmation_message: Optional[str] = None
    knowledge_context: Optional[str] = None

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Page d'accueil avec interface de recherche"""
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/images/{filename}")
async def get_image(filename: str):
    """
    Endpoint pour servir les images
    Note: Les images doivent être dans un dossier 'images/' à la racine
    Si elles n'existent pas, on retourne une image placeholder
    """
    import os
    from fastapi.responses import FileResponse, Response
    from fastapi import status
    
    # Nettoyer le nom de fichier (enlever les quotes si présentes)
    clean_filename = filename.strip("'").strip('"').strip()
    
    # Sécurité : empêcher les accès en dehors du dossier images
    if '..' in clean_filename or '/' in clean_filename or '\\' in clean_filename:
        return Response(status_code=status.HTTP_400_BAD_REQUEST)
    
    # Chemin possible de l'image
    image_path = f"images/{clean_filename}"
    
    if os.path.exists(image_path) and os.path.isfile(image_path):
        return FileResponse(image_path)
    else:
        # Retourner une réponse 404 - le frontend affichera un placeholder
        return Response(status_code=status.HTTP_404_NOT_FOUND)

@app.post("/api/search", response_model=SearchResponse)
async def search_products(request: SearchRequest):
    """
    Endpoint de recherche intelligente
    
    Utilise le modèle LLM pour comprendre la requête et effectue
    une recherche hybride (sémantique + textuelle)
    """
    if not search_engine:
        raise HTTPException(status_code=503, detail="Moteur de recherche non initialisé")
    
    try:
        results = search_engine.search(
            query=request.query,
            limit=request.limit,
            min_score=request.min_score
        )
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la recherche: {str(e)}")

@app.get("/api/search", response_model=SearchResponse)
async def search_products_get(
    q: str = Query(..., description="Requête de recherche"),
    limit: int = Query(10, ge=1, le=100),
    min_score: float = Query(0.0, ge=0.0, le=1.0)
):
    """Version GET de l'endpoint de recherche"""
    if not search_engine:
        raise HTTPException(status_code=503, detail="Moteur de recherche non initialisé")
    
    try:
        results = search_engine.search(
            query=q,
            limit=limit,
            min_score=min_score
        )
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la recherche: {str(e)}")

@app.post("/api/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """
    Endpoint pour transcrire un fichier audio en texte
    Utilise Whisper pour la transcription avec optimisations pour le vocabulaire métier
    """
    if not whisper_model:
        raise HTTPException(status_code=503, detail="Modèle Whisper non disponible")
    
    try:
        # Sauvegarder temporairement le fichier audio
        # Whisper accepte plusieurs formats (wav, mp3, webm, ogg, etc.)
        file_extension = audio.filename.split('.')[-1] if audio.filename else 'wav'
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
            content = await audio.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        try:
            # Prompt initial avec vocabulaire métier pour guider la transcription
            initial_prompt = (
                "Matériaux de construction, béton, ciment, mortier, sable, gravier, "
                "pavé, dalle, carrelage, isolation, laine de roche, laine de verre, "
                "enduit, plâtre, terrasse, extérieur, intérieur, bétonnière, "
                "agrégat, gravillon, faïence, grès cérame, mortier colle, "
                "isolation thermique, isolation phonique, pavé autobloquant, "
                "pavé drainant, sable alluvionnaire, mélange à béton."
            )
            
            # Paramètres optimisés pour une meilleure transcription
            transcribe_options = {
                "language": "fr",
                "initial_prompt": initial_prompt,  # Guide la transcription avec vocabulaire métier
                "temperature": 0.0,  # Plus déterministe pour une meilleure précision
                "beam_size": 5,  # Augmente la précision (par défaut: 5)
                "best_of": 5,  # Teste plusieurs hypothèses et prend la meilleure
                "word_timestamps": False,  # Pas besoin des timestamps pour notre usage
                "fp16": False,  # Désactivé pour éviter les problèmes de compatibilité
            }
            
            # Transcrire avec Whisper
            result = whisper_model.transcribe(tmp_path, **transcribe_options)
            transcribed_text = result["text"].strip()
            
            # Post-traitement : correction des termes techniques courants
            transcribed_text = _correct_transcription(transcribed_text)
            
            return {
                "text": transcribed_text,
                "language": result.get("language", "fr")
            }
        finally:
            # Nettoyer le fichier temporaire
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la transcription: {str(e)}")

def _correct_transcription(text: str) -> str:
    """
    Post-traitement pour corriger les erreurs de transcription courantes
    dans le domaine des matériaux de construction
    """
    corrections = {
        # Corrections de mots composés souvent mal transcrits (espaces incorrects)
        "béton nière": "bétonnière",
        "béton nières": "bétonnières",
        "béton nière de": "bétonnière de",
        "mélange a béton": "mélange à béton",
        "mélange a": "mélange à",
        # Corrections de ponctuation et espaces
        " ,": ",",
        " .": ".",
        " ?": "?",
        " !": "!",
        "  ": " ",
        # Corrections de nombres et unités
        " 25 l": " 25 L",
        " 25l": " 25 L",
        "25 litres": "25 L",
        "25 litre": "25 L",
        " 50 l": " 50 L",
        " 50l": " 50 L",
        "50 litres": "50 L",
        "50 litre": "50 L",
    }
    
    # Appliquer les corrections (dans l'ordre, les plus spécifiques en premier)
    corrected_text = text
    for wrong, correct in corrections.items():
        corrected_text = corrected_text.replace(wrong, correct)
    
    # Nettoyer les espaces multiples
    corrected_text = re.sub(r'\s+', ' ', corrected_text).strip()
    
    # Capitaliser la première lettre
    if corrected_text:
        corrected_text = corrected_text[0].upper() + corrected_text[1:] if len(corrected_text) > 1 else corrected_text.upper()
    
    return corrected_text

@app.get("/api/health")
async def health_check():
    """Vérification de l'état de l'API"""
    return {
        "status": "healthy",
        "search_engine_ready": search_engine is not None,
        "whisper_ready": whisper_model is not None,
        "articles_loaded": len(data_loader.articles) if data_loader else 0
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

