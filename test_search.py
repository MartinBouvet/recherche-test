"""
Script de test pour vérifier le fonctionnement du moteur de recherche
"""

from data_loader import DataLoader
from search_engine import SearchEngine
import time

def test_data_loading():
    """Test du chargement des données"""
    print("=" * 60)
    print("TEST 1: Chargement des données")
    print("=" * 60)
    
    loader = DataLoader()
    articles = loader.load_articles()
    
    print(f"✅ {len(articles)} articles chargés")
    
    if articles:
        print(f"\nExemple d'article:")
        article = articles[0]
        print(f"  - ID: {article.get('id')}")
        print(f"  - Libellé: {article.get('libelle')}")
        print(f"  - Référence: {article.get('reference')}")
        print(f"  - Prix: {article.get('prix_vente')} €")
    
    return articles

def test_search_engine(articles):
    """Test du moteur de recherche"""
    print("\n" + "=" * 60)
    print("TEST 2: Initialisation du moteur de recherche")
    print("=" * 60)
    
    print("⏳ Initialisation en cours... (cela peut prendre 30-60 secondes)")
    start = time.time()
    engine = SearchEngine(articles)
    elapsed = time.time() - start
    print(f"✅ Moteur initialisé en {elapsed:.2f} secondes")
    
    return engine

def test_search_queries(engine):
    """Test de différentes requêtes de recherche"""
    print("\n" + "=" * 60)
    print("TEST 3: Tests de recherche")
    print("=" * 60)
    
    test_queries = [
        "sable pour béton",
        "carrelage extérieur",
        "isolation laine de roche",
        "pavé terrasse",
        "enduit plâtre"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Recherche: '{query}'")
        start = time.time()
        results = engine.search(query, limit=5)
        elapsed = time.time() - start
        
        print(f"   ⏱️  Temps: {elapsed:.2f}s")
        print(f"   📊 Résultats: {results['total_results']}")
        print(f"   💡 Requête comprise: '{results['query_understood']}'")
        
        if results['results']:
            print(f"   🏆 Top résultat:")
            top = results['results'][0]
            print(f"      - {top['libelle']}")
            print(f"      - Score: {top['score']:.3f}")
            print(f"      - Prix: {top['prix_vente']} €")

def main():
    """Fonction principale de test"""
    print("\n" + "=" * 60)
    print("🧪 TESTS DU MOTEUR DE RECHERCHE INTELLIGENT")
    print("=" * 60 + "\n")
    
    try:
        # Test 1: Chargement des données
        articles = test_data_loading()
        
        if not articles:
            print("❌ Aucun article chargé. Vérifiez le fichier SQL.")
            return
        
        # Test 2: Initialisation du moteur
        engine = test_search_engine(articles)
        
        # Test 3: Recherches
        test_search_queries(engine)
        
        print("\n" + "=" * 60)
        print("✅ TOUS LES TESTS SONT TERMINÉS")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

