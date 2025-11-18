"""
Application Gradio SÉCURISÉE pour Railway
Moteur de recherche intelligent avec Ministral-8B + Authentification
"""

import gradio as gr
import os
import time
from typing import List, Dict, Any

# Import des modules du projet
from data_loader import DataLoader
from search_engine import SearchEngine

# ============================================
# CONFIGURATION SÉCURITÉ
# ============================================
DEMO_PASSWORD = os.getenv("SEARCH_PASSWORD", "demo2024")
print(f"🔐 Authentification activée - Mot de passe: {DEMO_PASSWORD}")

# Variables globales
search_engine = None
data_loader = None

def initialize_app():
    """Initialise le moteur de recherche au démarrage"""
    global search_engine, data_loader
    
    print("=" * 70)
    print("🚀 DÉMARRAGE DU MOTEUR DE RECHERCHE INTELLIGENT")
    print("=" * 70)
    
    try:
        print("\n📊 Étape 1/3 : Chargement des données...")
        data_loader = DataLoader()
        articles = data_loader.load_articles()
        print(f"   ✅ {len(articles)} articles chargés")
        
        print("\n🤖 Étape 2/3 : Initialisation Ministral-8B...")
        search_engine = SearchEngine(
            articles, 
            model_name="nchapman/ministral-8b-instruct-2410:8b",
            use_llm=True
        )
        print("   ✅ Moteur IA opérationnel")
        
        print("\n🎉 Étape 3/3 : Système prêt!")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n❌ ERREUR: {str(e)}")
        return False

def format_results_html(results: Dict[str, Any]) -> str:
    """Formate les résultats en HTML"""
    
    if not results.get('products'):
        return """
        <div style='padding: 20px; text-align: center;'>
            <h3>😕 Aucun résultat trouvé</h3>
            <p>Essayez de reformuler votre recherche.</p>
        </div>
        """
    
    html = "<div style='padding: 10px;'>"
    
    for i, product in enumerate(results['products'], 1):
        score_color = "green" if product['score'] > 0.7 else "orange" if product['score'] > 0.5 else "gray"
        
        html += f"""
        <div style='border: 1px solid #ddd; border-radius: 8px; padding: 15px; margin-bottom: 15px; background: white;'>
            <div style='display: flex; justify-content: space-between;'>
                <div style='flex: 1;'>
                    <h3 style='margin: 0 0 10px 0; color: #2c3e50;'>
                        {i}. {product.get('DESIGNATION', 'Sans nom')}
                    </h3>
                    <div style='color: #666; font-size: 0.9em; margin-bottom: 8px;'>
                        <strong>Réf:</strong> {product.get('REFERENCE', 'N/A')} | 
                        <strong>Calibre:</strong> {product.get('CALIBRE', 'N/A')}
                    </div>
                    <div style='color: #555; margin-bottom: 8px;'>
                        {product.get('TEXTDESCRIPTIF', 'Pas de description')[:200]}...
                    </div>
                    <div style='display: flex; gap: 20px;'>
                        <span><strong>Prix:</strong> {product.get('PVTTC', 'N/A')} €</span>
                        <span><strong>Cond:</strong> {product.get('CONDITIONNEMENT', 'N/A')}</span>
                        <span><strong>Poids:</strong> {product.get('POIDS', 'N/A')} kg</span>
                    </div>
                </div>
                <div style='margin-left: 15px; text-align: right;'>
                    <div style='background: {score_color}; color: white; padding: 8px 12px; border-radius: 5px; font-weight: bold;'>
                        {product['score']:.0%}
                    </div>
                </div>
            </div>
        </div>
        """
    
    html += "</div>"
    return html

def search_products(query: str, limit: int = 10, min_score: float = 0.3, use_ai: bool = True):
    """Fonction de recherche"""
    if not search_engine:
        return "❌ Moteur non initialisé", ""
    
    if not query or len(query.strip()) < 2:
        return "⚠️ Veuillez entrer une requête", ""
    
    try:
        original_use_llm = search_engine.use_llm
        search_engine.use_llm = use_ai
        
        start_time = time.time()
        results = search_engine.search(
            query=query.strip(),
            limit=limit,
            min_score=min_score
        )
        search_time = time.time() - start_time
        
        search_engine.use_llm = original_use_llm
        
        html_output = format_results_html(results)
        
        stats = f"""
        📊 **Statistiques:**
        - Résultats: {results.get('total_results', 0)}
        - Temps: {search_time:.2f}s
        - Requête comprise: {results.get('query_understood', query)}
        """
        
        return html_output, stats
        
    except Exception as e:
        return f"❌ Erreur: {str(e)}", ""

def create_interface():
    """Crée l'interface Gradio avec authentification"""
    
    with gr.Blocks(
        title="Recherche Thomas-Sograma",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown("""
        # 🔍 Recherche Intelligente Thomas-Sograma
        
        Propulsé par **Ministral-8B** - Compréhension du langage naturel
        
        **Exemples:**
        - "Je cherche du gravier pour une allée"
        - "Isolation thermique pour combles"
        - "Carrelage extérieur gris"
        """)
        
        with gr.Row():
            query_input = gr.Textbox(
                label="🔎 Que recherchez-vous ?",
                placeholder="Ex: sable pour béton, isolation laine de roche...",
                lines=2
            )
            
        with gr.Row():
            with gr.Column(scale=1):
                limit_slider = gr.Slider(5, 50, value=10, step=5, label="📊 Résultats")
            with gr.Column(scale=1):
                min_score_slider = gr.Slider(0.0, 1.0, value=0.3, step=0.1, label="🎯 Pertinence min")
            with gr.Column(scale=1):
                use_ai_checkbox = gr.Checkbox(value=True, label="🤖 IA activée")
        
        search_btn = gr.Button("🚀 Rechercher", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column(scale=3):
                results_html = gr.HTML(label="Résultats")
            with gr.Column(scale=1):
                stats_output = gr.Markdown(label="Stats")
        
        gr.Examples(
            examples=[
                ["sable pour béton", 10, 0.3, True],
                ["Je cherche du gravier pour une allée", 10, 0.3, True],
                ["isolation thermique combles", 15, 0.3, True],
                ["carrelage extérieur gris", 10, 0.2, True],
            ],
            inputs=[query_input, limit_slider, min_score_slider, use_ai_checkbox],
            label="💡 Exemples"
        )
        
        search_btn.click(
            fn=search_products,
            inputs=[query_input, limit_slider, min_score_slider, use_ai_checkbox],
            outputs=[results_html, stats_output]
        )
        
        query_input.submit(
            fn=search_products,
            inputs=[query_input, limit_slider, min_score_slider, use_ai_checkbox],
            outputs=[results_html, stats_output]
        )
        
        gr.Markdown("""
        ---
        ### ℹ️ À propos
        - **Ministral-8B** pour langage naturel
        - **Recherche hybride** textuelle + sémantique
        - **~200 produits** Thomas-Sograma
        """)
    
    return demo

if __name__ == "__main__":
    print("🚀 Initialisation...")
    initialize_app()
    
    demo = create_interface()
    
    # Lancement avec authentification
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        auth=("demo", DEMO_PASSWORD)  # 🔐 AUTHENTIFICATION
    )