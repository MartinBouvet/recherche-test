"""
Dictionnaire de synonymes pour le domaine des matériaux de construction
"""

from typing import List

SYNONYMS_METIER = {
    # Matériaux de base
    "béton": ["béton", "ciment", "mortier", "béton armé", "béton prêt à l'emploi"],
    "sable": ["sable", "agrégat fin", "sable alluvionnaire", "sable roulé"],
    "gravier": ["gravier", "gravillon", "agrégat", "caillou"],
    "pavé": ["pavé", "dalle", "pavé autobloquant", "pavé drainant", "pavé béton"],
    
    # Isolation
    "isolation": ["isolation", "isolant", "isolation thermique", "isolation phonique"],
    "laine": ["laine", "fibre", "matériau isolant"],
    "laine de roche": ["laine de roche", "rockwool", "laine minérale", "isolation roche"],
    "laine de verre": ["laine de verre", "fibre de verre", "isolation verre"],
    
    # Carrelage
    "carrelage": ["carrelage", "carreau", "faïence", "grès cérame", "carreau de céramique"],
    "extérieur": ["extérieur", "plein air", "terrasse", "jardin", "extérieur"],
    "intérieur": ["intérieur", "salle de bain", "cuisine", "intérieur"],
    
    # Enduits et mortiers
    "enduit": ["enduit", "plâtre", "enduit de finition", "enduit de lissage"],
    "mortier": ["mortier", "colle", "mortier colle", "adhésif", "colle carrelage"],
    
    # Terrasse et extérieur
    "terrasse": ["terrasse", "plage de piscine", "espace extérieur", "jardin", "allée"],
    "allée": ["allée", "chemin", "sentier", "accès"],
    
    # Dimensions et formats
    "petit": ["petit", "fin", "mince"],
    "grand": ["grand", "large", "épais"],
    
    # Autres matériaux
    "tuile": ["tuile", "ardoise", "couverture"],
    "brique": ["brique", "parpaing", "bloc"],
    "bois": ["bois", "lambourde", "lame", "planche"],
}

def expand_with_synonyms(terms: List[str]) -> List[str]:
    """
    Étend une liste de termes avec leurs synonymes
    
    Args:
        terms: Liste de termes à étendre
    
    Returns:
        Liste étendue avec synonymes
    """
    expanded = set(terms)
    
    for term in terms:
        term_lower = term.lower()
        # Chercher dans le dictionnaire
        for key, synonyms in SYNONYMS_METIER.items():
            if term_lower == key or term_lower in synonyms:
                # Ajouter tous les synonymes
                expanded.update(synonyms)
                break
    
    return list(expanded)

def get_synonyms(term: str) -> List[str]:
    """
    Retourne les synonymes d'un terme
    
    Args:
        term: Terme à chercher
    
    Returns:
        Liste des synonymes (incluant le terme original)
    """
    term_lower = term.lower()
    
    # Chercher directement
    if term_lower in SYNONYMS_METIER:
        return SYNONYMS_METIER[term_lower]
    
    # Chercher dans les valeurs
    for key, synonyms in SYNONYMS_METIER.items():
        if term_lower in synonyms:
            return synonyms
    
    return [term]  # Pas de synonymes trouvés

