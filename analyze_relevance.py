"""
Script pour analyser la pertinence des résultats de recherche
et vérifier la qualité des données
"""

from data_loader import DataLoader
from search_engine import SearchEngine
import json

def analyze_data_quality():
    """Analyse la qualité des données chargées"""
    print("=" * 60)
    print("ANALYSE DE LA QUALITÉ DES DONNÉES")
    print("=" * 60)
    
    loader = DataLoader()
    articles = loader.load_articles()
    
    print(f"\n📊 Statistiques générales:")
    print(f"   Total d'articles: {len(articles)}")
    
    # Statistiques sur les champs importants
    with_libelle = sum(1 for a in articles if a.get('libelle'))
    with_designation = sum(1 for a in articles if a.get('designation'))
    with_mots_clefs = sum(1 for a in articles if a.get('motsClefs'))
    with_photo = sum(1 for a in articles if a.get('photo1'))
    with_prix = sum(1 for a in articles if a.get('prix_vente', 0) > 0)
    
    print(f"\n📝 Champs remplis:")
    print(f"   - Libellé: {with_libelle} ({with_libelle/len(articles)*100:.1f}%)")
    print(f"   - Désignation: {with_designation} ({with_designation/len(articles)*100:.1f}%)")
    print(f"   - Mots-clés: {with_mots_clefs} ({with_mots_clefs/len(articles)*100:.1f}%)")
    print(f"   - Photo: {with_photo} ({with_photo/len(articles)*100:.1f}%)")
    print(f"   - Prix: {with_prix} ({with_prix/len(articles)*100:.1f}%)")
    
    # Vérifier les problèmes de quotes dans les photos
    print(f"\n🖼️  Analyse des photos:")
    photos_with_quotes = []
    photos_clean = []
    for article in articles[:100]:  # Échantillon
        photo = article.get('photo1', '')
        if photo:
            if "'" in photo or '"' in photo:
                photos_with_quotes.append((article.get('id'), photo))
            else:
                photos_clean.append(photo)
    
    if photos_with_quotes:
        print(f"   ⚠️  {len(photos_with_quotes)} photos avec quotes détectées (échantillon):")
        for art_id, photo in photos_with_quotes[:5]:
            print(f"      ID {art_id}: '{photo}'")
    else:
        print(f"   ✅ Toutes les photos sont propres (échantillon)")
    
    # Analyser les catégories
    categories = {}
    for article in articles:
        cat_id = article.get('categorieId', 0)
        categories[cat_id] = categories.get(cat_id, 0) + 1
    
    print(f"\n📂 Catégories:")
    print(f"   Nombre de catégories: {len(categories)}")
    top_cats = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]
    for cat_id, count in top_cats:
        print(f"   - Catégorie {cat_id}: {count} articles")
    
    return articles

def test_search_relevance(articles):
    """Teste la pertinence des recherches"""
    print("\n" + "=" * 60)
    print("TEST DE PERTINENCE DES RECHERCHES")
    print("=" * 60)
    
    engine = SearchEngine(articles, use_llm=False)  # Sans LLM pour plus de rapidité
    
    test_queries = [
        ("sable pour béton", ["sable", "béton"]),
        ("isolation laine de roche", ["isolation", "laine", "roche"]),
        ("pavé terrasse", ["pavé", "terrasse"]),
        ("carrelage extérieur", ["carrelage", "extérieur"]),
    ]
    
    for query, expected_terms in test_queries:
        print(f"\n🔍 Requête: '{query}'")
        print(f"   Termes attendus: {expected_terms}")
        
        results = engine.search(query, limit=5, min_score=0.2)
        
        print(f"   ⏱️  Temps: {results['search_time']}s")
        print(f"   📊 Résultats: {results['total_results']}")
        
        if results['results']:
            print(f"   🏆 Top 3 résultats:")
            for i, result in enumerate(results['results'][:3], 1):
                libelle = result['libelle']
                score = result['score']
                
                # Vérifier si les termes attendus sont présents
                libelle_lower = libelle.lower()
                terms_found = [term for term in expected_terms if term.lower() in libelle_lower]
                
                match_indicator = "✅" if terms_found else "⚠️"
                print(f"      {i}. {match_indicator} Score: {score:.3f} | {libelle[:60]}")
                if terms_found:
                    print(f"         → Termes trouvés: {terms_found}")
                else:
                    print(f"         → Aucun terme attendu trouvé dans le libellé")
        else:
            print(f"   ❌ Aucun résultat")

def analyze_specific_query(articles, query: str):
    """Analyse détaillée d'une requête spécifique"""
    print("\n" + "=" * 60)
    print(f"ANALYSE DÉTAILLÉE: '{query}'")
    print("=" * 60)
    
    engine = SearchEngine(articles, use_llm=False)
    results = engine.search(query, limit=10, min_score=0.1)
    
    print(f"\n📊 Statistiques:")
    print(f"   Temps de recherche: {results['search_time']}s")
    print(f"   Nombre de résultats: {results['total_results']}")
    print(f"   Requête comprise: '{results['query_understood']}'")
    
    if results['results']:
        print(f"\n📋 Détail des résultats:")
        for i, result in enumerate(results['results'], 1):
            print(f"\n   {i}. {result['libelle']}")
            print(f"      ID: {result['id']}")
            print(f"      Référence: {result['reference']}")
            print(f"      Score: {result['score']:.4f} ({result['score']*100:.1f}%)")
            print(f"      Prix: {result['prix_vente']} €")
            print(f"      Photo: {result['photo1'] or 'Aucune'}")
            if result.get('designation'):
                print(f"      Description: {result['designation'][:80]}...")
            if result.get('motsClefs'):
                print(f"      Mots-clés: {result['motsClefs'][:60]}...")

def main():
    """Fonction principale"""
    print("\n" + "=" * 60)
    print("🔍 ANALYSE DE PERTINENCE DU MOTEUR DE RECHERCHE")
    print("=" * 60 + "\n")
    
    # 1. Analyser la qualité des données
    articles = analyze_data_quality()
    
    # 2. Tester la pertinence
    test_search_relevance(articles)
    
    # 3. Analyse détaillée d'une requête spécifique
    print("\n" + "=" * 60)
    query = input("\n💡 Entrez une requête à analyser en détail (ou appuyez sur Entrée pour passer): ").strip()
    if query:
        analyze_specific_query(articles, query)
    
    print("\n" + "=" * 60)
    print("✅ ANALYSE TERMINÉE")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()

