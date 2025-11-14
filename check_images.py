"""
Script pour vérifier quelles images existent dans la base de données
et où elles devraient être stockées
"""

from data_loader import DataLoader
import os
from collections import Counter

def check_images():
    """Vérifie les images dans la base de données"""
    print("=" * 60)
    print("VÉRIFICATION DES IMAGES")
    print("=" * 60)
    
    loader = DataLoader()
    articles = loader.load_articles()
    
    # Collecter toutes les photos
    all_photos = []
    photos_by_extension = Counter()
    
    for article in articles:
        for photo_field in ['photo1', 'photo2', 'photo3', 'photo4']:
            photo = article.get(photo_field)
            if photo:
                # Nettoyer
                clean_photo = photo.strip("'").strip('"').strip()
                if clean_photo:
                    all_photos.append(clean_photo)
                    ext = os.path.splitext(clean_photo)[1].lower()
                    photos_by_extension[ext] += 1
    
    print(f"\n📊 Statistiques:")
    print(f"   Total de références photo: {len(all_photos)}")
    print(f"   Photos uniques: {len(set(all_photos))}")
    
    print(f"\n📁 Extensions:")
    for ext, count in photos_by_extension.most_common():
        print(f"   {ext or '(sans extension)'}: {count}")
    
    # Vérifier si un dossier images existe
    print(f"\n📂 Dossiers:")
    if os.path.exists("images"):
        print(f"   ✅ Dossier 'images/' existe")
        image_files = [f for f in os.listdir("images") if os.path.isfile(os.path.join("images", f))]
        print(f"   📸 Fichiers dans 'images/': {len(image_files)}")
        
        # Vérifier combien de photos de la BDD existent
        existing = 0
        missing = []
        for photo in set(all_photos[:100]):  # Échantillon
            if photo in image_files:
                existing += 1
            else:
                missing.append(photo)
        
        print(f"   ✅ Photos trouvées (échantillon): {existing}/{min(100, len(set(all_photos)))}")
        if missing:
            print(f"   ⚠️  Photos manquantes (échantillon, 5 premières):")
            for photo in missing[:5]:
                print(f"      - {photo}")
    else:
        print(f"   ❌ Dossier 'images/' n'existe pas")
        print(f"   💡 Créez un dossier 'images/' et placez-y les images")
    
    # Afficher quelques exemples
    print(f"\n📋 Exemples de photos dans la BDD:")
    unique_photos = list(set(all_photos))[:10]
    for photo in unique_photos:
        print(f"   - {photo}")
    
    # Statistiques par article
    articles_with_photos = sum(1 for a in articles if a.get('photo1'))
    print(f"\n📦 Articles avec au moins une photo: {articles_with_photos}/{len(articles)} ({articles_with_photos/len(articles)*100:.1f}%)")

if __name__ == "__main__":
    check_images()

