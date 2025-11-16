"""
Application Gradio optimisée pour Hugging Face Spaces
Moteur de recherche intelligent avec Ministral-8B
"""

import gradio as gr
import os
import time
from typing import List, Dict, Any
from data_loader import DataLoader
from search_engine import SearchEngine

# Variables globales
search_engine = None
data_loader = None

def initialize_app():
    """Initialise le moteur de recherche au démarrage"""
    global search_engine, data_loader
    
    print("🚀 Initialisation de l'application...")
    print("📊 Chargement des données...")
    
    data_loader = DataLoader()
    articles = data_loader.load_articles()
    print(f"✅ {len(articles)} articles chargés")
    
    print("🤖 Initialisation du moteur de recherche avec Ministral-8B...")
    search_engine = SearchEngine(
        articles, 
        model_name="nchapman/ministral-8b-instruct-2410:8b",
        use_llm=True  # Activé car on a 16GB RAM sur HF Spaces
    )
    print("✅ Moteur de recherche prêt!")
    
    return "Application initialisée avec succès!"

def format_price(price):
    """Formate le prix pour l'affichage"""
    if price and price > 0:
        return f"{price:.2f} €"
    return "Prix sur demande"

def format_results_html(data: Dict[str, Any]) -> str:
    """Formate les résultats en HTML pour l'affichage Gradio"""
    
    if not data or data.get('total_results', 0) == 0:
        return """
        <div style='padding: 40px; text-align: center; background: #f8f9fa; border-radius: 10px;'>
            <h2 style='color: #6c757d;'>😕 Aucun résultat trouvé</h2>
            <p>Essayez avec d'autres mots-clés ou reformulez votre recherche.</p>
        </div>
        """
    
    html = f"""
    <style>
        .search-info {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .confirmation {{
            background: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-weight: bold;
        }}
        .product-card {{
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .product-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .product-header {{
            display: flex;
            justify-content: space-between;
            align-items: start;
            margin-bottom: 15px;
        }}
        .product-title {{
            font-size: 1.3em;
            font-weight: bold;
            color: #2c3e50;
            margin: 0;
        }}
        .product-score {{
            background: #28a745;
            color: white;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }}
        .product-ref {{
            color: #6c757d;
            font-size: 0.9em;
            margin-bottom: 10px;
        }}
        .product-desc {{
            color: #495057;
            margin: 10px 0;
            line-height: 1.6;
        }}
        .product-price {{
            font-size: 1.5em;
            color: #e74c3c;
            font-weight: bold;
            margin-top: 15px;
        }}
        .product-price-normal {{
            font-size: 0.9em;
            color: #6c757d;
            text-decoration: line-through;
            margin-left: 10px;
        }}
        .material-section {{
            margin-bottom: 30px;
        }}
        .material-title {{
            background: #f8f9fa;
            padding: 12px 20px;
            border-left: 4px solid #667eea;
            font-size: 1.2em;
            font-weight: bold;
            color: #2c3e50;
            margin: 20px 0 15px 0;
        }}
    </style>
    """
    
    # Info de recherche
    query_understood = data.get('query_understood', '')
    total_results = data.get('total_results', 0)
    search_time = data.get('search_time', 0)
    
    html += f"""
    <div class='search-info'>
        <h3 style='margin: 0 0 10px 0;'>🔍 Recherche : {query_understood}</h3>
        <p style='margin: 0;'>{total_results} résultat(s) trouvé(s) en {search_time:.2f}s</p>
    </div>
    """
    
    # Message de confirmation si projet
    if data.get('confirmation_message'):
        html += f"""
        <div class='confirmation'>
            {data['confirmation_message']}
        </div>
        """
    
    # Résultats par matériau (si projet)
    if data.get('is_project') and data.get('materiaux'):
        for materiau_type, materiau_data in data['materiaux'].items():
            if materiau_data.get('results'):
                html += f"<div class='material-section'>"
                html += f"<div class='material-title'>🔧 {materiau_type.title()}</div>"
                
                for result in materiau_data['results']:
                    html += format_product_card(result)
                
                html += "</div>"
    
    # Résultats classiques
    elif data.get('results'):
        for result in data['results']:
            html += format_product_card(result)
    
    return html

def format_product_card(result: Dict[str, Any]) -> str:
    """Formate une carte produit"""
    libelle = result.get('libelle', 'Sans nom')
    designation = result.get('designation', '')
    reference = result.get('reference', 'N/A')
    prix_vente = result.get('prix_vente', 0)
    prix_particulier = result.get('prix_vente_particulier', 0)
    score = result.get('score', 0)
    
    # Score en pourcentage et couleur
    score_percent = int(score * 100)
    if score_percent >= 80:
        score_color = "#28a745"
    elif score_percent >= 60:
        score_color = "#ffc107"
    else:
        score_color = "#6c757d"
    
    card = f"""
    <div class='product-card'>
        <div class='product-header'>
            <div>
                <h3 class='product-title'>{libelle}</h3>
                <div class='product-ref'>Réf: {reference}</div>
            </div>
            <div class='product-score' style='background: {score_color};'>
                {score_percent}% pertinent
            </div>
        </div>
    """
    
    if designation:
        card += f"<p class='product-desc'>{designation}</p>"
    
    # Prix
    if prix_vente > 0:
        card += f"<div class='product-price'>{format_price(prix_vente)}"
        if prix_particulier and prix_particulier != prix_vente:
            card += f"<span class='product-price-normal'>{format_price(prix_particulier)}</span>"
        card += "</div>"
    
    card += "</div>"
    
    return card

def search_products(query: str, limit: int = 10, min_score: float = 0.3, use_ai: bool = True) -> tuple:
    """
    Effectue une recherche de produits
    
    Args:
        query: Requête de recherche
        limit: Nombre maximum de résultats
        min_score: Score minimum de pertinence
        use_ai: Utiliser l'IA pour comprendre la requête
    
    Returns:
        tuple: (html_results, stats_text)
    """
    if not search_engine:
        return "❌ Moteur de recherche non initialisé", ""
    
    if not query or len(query.strip()) < 2:
        return "⚠️ Veuillez entrer une requête de recherche", ""
    
    try:
        # Temporairement changer use_llm si nécessaire
        original_use_llm = search_engine.use_llm
        search_engine.use_llm = use_ai
        
        start_time = time.time()
        results = search_engine.search(
            query=query.strip(),
            limit=limit,
            min_score=min_score
        )
        search_time = time.time() - start_time
        
        # Restaurer la valeur originale
        search_engine.use_llm = original_use_llm
        
        # Ajouter le temps de recherche
        results['search_time'] = search_time
        
        # Formater en HTML
        html_output = format_results_html(results)
        
        # Statistiques
        stats = f"""
        📊 **Statistiques de recherche:**
        - Résultats trouvés: {results.get('total_results', 0)}
        - Temps de recherche: {search_time:.2f}s
        - Requête comprise: {results.get('query_understood', query)}
        - Type: {'Projet' if results.get('is_project') else 'Recherche simple'}
        """
        
        return html_output, stats
        
    except Exception as e:
        error_msg = f"❌ Erreur lors de la recherche: {str(e)}"
        return error_msg, ""

def create_interface():
    """Crée l'interface Gradio"""
    
    with gr.Blocks(
        title="Recherche Intelligente Thomas-Sograma",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown("""
        # 🔍 Recherche Intelligente de Matériaux de Construction
        
        Moteur de recherche propulsé par **Ministral-8B** - Comprend le langage naturel et trouve les produits parfaits pour votre projet.
        
        **Exemples de recherches:**
        - "Je cherche du gravier pour faire une allée"
        - "Isolation thermique pour combles"
        - "Carrelage extérieur gris à moins de 50 euros"
        - "Je veux faire un mur en parpaings"
        """)
        
        with gr.Row():
            with gr.Column(scale=4):
                query_input = gr.Textbox(
                    label="🔎 Que recherchez-vous ?",
                    placeholder="Ex: sable pour béton, carrelage extérieur, isolation laine de roche...",
                    lines=2
                )
            
        with gr.Row():
            with gr.Column(scale=1):
                limit_slider = gr.Slider(
                    minimum=5,
                    maximum=50,
                    value=10,
                    step=5,
                    label="📊 Nombre de résultats"
                )
            with gr.Column(scale=1):
                min_score_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.3,
                    step=0.1,
                    label="🎯 Pertinence minimale"
                )
            with gr.Column(scale=1):
                use_ai_checkbox = gr.Checkbox(
                    value=True,
                    label="🤖 Utiliser l'IA (Ministral-8B)"
                )
        
        search_btn = gr.Button("🚀 Rechercher", variant="primary", size="lg")
        
        with gr.Row():
            with gr.Column(scale=3):
                results_html = gr.HTML(label="Résultats")
            with gr.Column(scale=1):
                stats_output = gr.Markdown(label="Statistiques")
        
        # Exemples
        gr.Examples(
            examples=[
                ["sable pour béton", 10, 0.3, True],
                ["Je cherche du gravier pour faire une allée", 10, 0.3, True],
                ["isolation thermique combles", 15, 0.3, True],
                ["carrelage extérieur gris pas cher", 10, 0.2, True],
                ["Je veux faire un mur en parpaings", 10, 0.3, True],
                ["pavé autobloquant pour terrasse", 10, 0.3, True],
            ],
            inputs=[query_input, limit_slider, min_score_slider, use_ai_checkbox],
            label="💡 Exemples de recherches"
        )
        
        # Actions
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
        
        Ce moteur de recherche intelligent utilise:
        - **Ministral-8B** pour comprendre vos demandes en langage naturel
        - **Recherche hybride** combinant recherche textuelle et sémantique
        - **Base de connaissances** Thomas-Sograma avec ~200 produits
        
        **Développé avec:** Python, Ollama, Sentence Transformers, Gradio
        """)
    
    return demo

if __name__ == "__main__":
    # Initialiser l'application
    print("=" * 60)
    initialize_app()
    print("=" * 60)
    
    # Créer et lancer l'interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )