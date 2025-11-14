"""
Module de chargement des données depuis le fichier SQL
"""

import re
import sqlite3
from typing import List, Dict, Any
from pathlib import Path

class DataLoader:
    """Charge et parse les données depuis le fichier SQL"""
    
    def __init__(self, sql_file: str = "newflat_sograma_produits_seuls.sql"):
        self.sql_file = sql_file
        self.articles: List[Dict[str, Any]] = []
    
    def load_articles(self) -> List[Dict[str, Any]]:
        """Charge les articles depuis le fichier SQL"""
        print(f"Lecture du fichier {self.sql_file}...")
        
        with open(self.sql_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Extraction des INSERT INTO
        insert_pattern = r"INSERT INTO `article`[^;]+;"
        inserts = re.findall(insert_pattern, content, re.DOTALL | re.IGNORECASE)
        
        print(f"Trouvé {len(inserts)} blocs INSERT")
        
        # Colonnes de la table article (d'après le CREATE TABLE)
        columns = [
            'id', 'categorieId', 'groupeId', 'titreId', 'soustitreId', 'ficheId',
            'codeFournisseur', 'codeFabricant', 'reference', 'reference_interne', 'reference2',
            'ean', 'designation', 'libelle', 'motsClefs', 'uniteAchat', 'uniteVente',
            'uniteStock', 'conditionnement', 'url', 'photo1', 'photo2', 'photo3', 'photo4',
            'pdf1', 'pdf2', 'prix_vente', 'prix_vente_particulier', 'prix_vente_us',
            'prix_vente_particulier_us', 'tva', 'ecopart', 'ecopart_us', 'cart1', 'cart2',
            'cart3', 'express', 'remise_quantite', 'promo', 'date_creation', 'nb_consultation',
            'nb_vente', 'mainKeyWords', 'sales_ratio', 'is_deleted', 'is_active', 'deleted_at',
            'needs_reindex'
        ]
        
        articles = []
        
        for insert_block in inserts:
            # Extraction des valeurs entre parenthèses
            values_pattern = r"\(([^)]+)\)"
            value_rows = re.findall(values_pattern, insert_block)
            
            for row_values in value_rows:
                try:
                    # Parse les valeurs (gère les chaînes avec virgules et apostrophes)
                    values = self._parse_sql_values(row_values)
                    
                    if len(values) == len(columns):
                        article = dict(zip(columns, values))
                        # Conversion des types
                        article = self._convert_types(article)
                        # Filtrer les articles actifs et non supprimés
                        if article.get('is_active', 1) and not article.get('is_deleted', 0):
                            articles.append(article)
                except Exception as e:
                    # Ignorer les lignes mal formées
                    continue
        
        self.articles = articles
        print(f"{len(articles)} articles valides chargés")
        return articles
    
    def _parse_sql_values(self, row_str: str) -> List[Any]:
        """Parse une ligne de valeurs SQL"""
        values = []
        current_value = ""
        in_string = False
        string_char = None
        i = 0
        
        while i < len(row_str):
            char = row_str[i]
            
            if not in_string:
                if char in ("'", '"'):
                    in_string = True
                    string_char = char
                    # Ne pas ajouter la quote d'ouverture
                elif char == ',':
                    values.append(current_value.strip())
                    current_value = ""
                else:
                    current_value += char
            else:
                if char == string_char:
                    # Vérifier si c'est une échappement (deux quotes consécutives)
                    if i + 1 < len(row_str) and row_str[i + 1] == string_char:
                        # C'est un échappement, ajouter une quote
                        current_value += char
                        i += 1  # Passer la deuxième quote
                    else:
                        # C'est la fin de la chaîne, ne pas ajouter la quote de fermeture
                        in_string = False
                        string_char = None
                else:
                    # Caractère normal dans la chaîne
                    current_value += char
            
            i += 1
        
        if current_value.strip():
            values.append(current_value.strip())
        
        return values
    
    def _convert_types(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Convertit les types de données"""
        # Conversion des valeurs
        for key, value in article.items():
            if value is None or value == 'NULL' or value == '':
                article[key] = None
                continue
            
            # Nettoyage des chaînes
            if isinstance(value, str):
                value = value.strip()
                
                # Pour les noms de fichiers, nettoyage spécial et agressif
                if key in ['photo1', 'photo2', 'photo3', 'photo4', 'pdf1', 'pdf2']:
                    # Enlever TOUTES les quotes (simples et doubles) de partout
                    value = value.replace("'", "").replace('"', '')
                    value = value.strip()
                    # Si après nettoyage c'est vide, garder None
                    if not value:
                        value = None
                else:
                    # Pour les autres champs, nettoyage standard
                    original_value = value
                    while len(value) > 0 and ((value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"'))):
                        if value.startswith("'") and value.endswith("'"):
                            value = value[1:-1].strip()
                        elif value.startswith('"') and value.endswith('"'):
                            value = value[1:-1].strip()
                        else:
                            break
                        # Protection contre boucle infinie
                        if value == original_value:
                            break
                        original_value = value
                    
                    # Gérer les échappements
                    value = value.replace("''", "'")
                    value = value.replace('""', '"')
                    # Nettoyer les espaces multiples
                    value = ' '.join(value.split())
            
            # Conversion des types numériques
            if key in ['id', 'categorieId', 'groupeId', 'titreId', 'soustitreId', 'ficheId',
                      'nb_consultation', 'nb_vente', 'express', 'remise_quantite', 'promo',
                      'is_deleted', 'is_active', 'needs_reindex']:
                try:
                    article[key] = int(value) if value else 0
                except:
                    article[key] = 0
            elif key in ['prix_vente', 'prix_vente_particulier', 'prix_vente_us',
                        'prix_vente_particulier_us', 'tva', 'ecopart', 'ecopart_us', 'sales_ratio']:
                try:
                    article[key] = float(value) if value else 0.0
                except:
                    article[key] = 0.0
        
        return article

