"""
Base de connaissances expertes sur les techniques de construction
Utilisée par l'IA pour décomposer les projets en matériaux nécessaires
"""

from typing import List, Dict, Any

# Connaissances sur les techniques de construction
CONSTRUCTION_KNOWLEDGE = {
    "mur_parpaings": {
        "description": "Construction d'un mur en parpaings",
        "materiaux_necessaires": [
            {
                "type": "parpaing",
                "description": "Parpaings creux ou pleins selon l'usage (extérieur, intérieur, porteur)",
                "critères": "parpaing, aggloméré, bloc béton",
                "quantité_indicative": "varie selon la surface"
            },
            {
                "type": "mortier",
                "description": "Mortier pour assembler les parpaings",
                "critères": "mortier, mortier colle, mortier bâtard",
                "quantité_indicative": "environ 25-30 kg par m² de mur"
            },
            {
                "type": "ferraillage",
                "description": "Armatures métalliques pour renforcer le mur si nécessaire",
                "critères": "fer à béton, armature, treillis soudé",
                "quantité_indicative": "selon les normes de construction"
            },
            {
                "type": "enduit",
                "description": "Enduit de finition pour protéger et lisser le mur",
                "critères": "enduit, enduit de façade, enduit extérieur",
                "quantité_indicative": "environ 15-20 kg par m²"
            }
        ]
    },
    "cloison_parpaings": {
        "description": "Création d'une cloison intérieure en parpaings",
        "materiaux_necessaires": [
            {
                "type": "parpaing",
                "description": "Parpaings ou blocs creux pour cloison (épaisseur 10 ou 15 cm). Dans notre catalogue, ils peuvent être appelés 'bloc', 'moellon creux', 'bloc à bancher' ou 'bloc chainage'.",
                "critères": "bloc moellon creux cloison aggloméré béton bancher chainage",
                "quantité_indicative": "varie selon la surface"
            },
            {
                "type": "mortier",
                "description": "Mortier colle ou mortier bâtard pour assembler les parpaings",
                "critères": "mortier colle bloc béton mortier bâtard",
                "quantité_indicative": "environ 20-25 kg par m²"
            },
            {
                "type": "enduit",
                "description": "Enduit de finition intérieur pour lisser et protéger",
                "critères": "enduit intérieur enduit lissage enduit finition",
                "quantité_indicative": "environ 10-15 kg par m²"
            }
        ]
    },
    "terrasse_paves": {
        "description": "Création d'une terrasse en pavés",
        "materiaux_necessaires": [
            {
                "type": "pavé",
                "description": "Pavés autobloquants ou pavés béton pour terrasse",
                "critères": "pavé terrasse, pavé autobloquant, pavé drainant",
                "quantité_indicative": "varie selon la surface"
            },
            {
                "type": "sable",
                "description": "Sable pour lit de pose et jointoiement",
                "critères": "sable, sable de pose, sable à maçonnerie",
                "quantité_indicative": "environ 50-80 kg par m²"
            },
            {
                "type": "gravillon",
                "description": "Gravier pour la couche de fondation",
                "critères": "gravier, gravillon, tout-venant",
                "quantité_indicative": "environ 100-150 kg par m²"
            },
            {
                "type": "géotextile",
                "description": "Géotextile pour séparer les couches et éviter les remontées",
                "critères": "géotextile, bidim",
                "quantité_indicative": "surface de la terrasse"
            }
        ]
    },
    "chape_beton": {
        "description": "Réalisation d'une chape en béton",
        "materiaux_necessaires": [
            {
                "type": "ciment",
                "description": "Ciment pour le béton",
                "critères": "ciment, ciment portland",
                "quantité_indicative": "environ 300-350 kg par m³ de béton"
            },
            {
                "type": "sable",
                "description": "Sable pour le béton",
                "critères": "sable, sable à béton, sable alluvionnaire",
                "quantité_indicative": "environ 800-900 kg par m³"
            },
            {
                "type": "gravier",
                "description": "Gravier pour le béton",
                "critères": "gravier, gravillon, agrégat",
                "quantité_indicative": "environ 1100-1200 kg par m³"
            },
            {
                "type": "treillis_soude",
                "description": "Treillis soudé pour renforcer la chape",
                "critères": "treillis soudé, treillis, ferraillage",
                "quantité_indicative": "selon la surface"
            }
        ]
    },
    "isolation_mur": {
        "description": "Isolation d'un mur",
        "materiaux_necessaires": [
            {
                "type": "isolant",
                "description": "Isolant thermique (laine de roche, laine de verre, ou autre)",
                "critères": "isolation, isolant, laine de roche, laine de verre",
                "quantité_indicative": "varie selon l'épaisseur et la surface"
            },
            {
                "type": "pare_vapeur",
                "description": "Pare-vapeur pour protéger l'isolant",
                "critères": "pare-vapeur, film pare-vapeur",
                "quantité_indicative": "surface du mur"
            },
            {
                "type": "fixation",
                "description": "Chevilles et vis pour fixer l'isolant",
                "critères": "cheville, vis, fixation isolation",
                "quantité_indicative": "selon le type d'isolant"
            }
        ]
    }
}

def detect_project_type(query: str) -> str:
    """
    Détecte le type de projet mentionné dans la requête
    
    Args:
        query: Requête utilisateur
    
    Returns:
        Type de projet détecté ou None
    """
    query_lower = query.lower()
    
    # Détection des projets
    # Cloison en parpaings
    if any(word in query_lower for word in ["cloison", "cloisons", "créer une cloison", "faire une cloison"]) and any(word in query_lower for word in ["parpaing", "parpaings", "bloc", "blocs"]):
        return "cloison_parpaings"
    # Aussi détecter "créer une cloison" seul (on assume parpaings si pas spécifié)
    if any(word in query_lower for word in ["créer une cloison", "faire une cloison", "construire une cloison"]) and "parpaing" not in query_lower and "bloc" not in query_lower:
        return "cloison_parpaings"
    elif any(word in query_lower for word in ["mur", "murs"]) and any(word in query_lower for word in ["parpaing", "parpaings", "bloc"]):
        return "mur_parpaings"
    elif any(word in query_lower for word in ["terrasse", "terrasses"]) and any(word in query_lower for word in ["pavé", "pavés", "dalle"]):
        return "terrasse_paves"
    elif any(word in query_lower for word in ["chape", "dalle"]) and any(word in query_lower for word in ["béton", "beton", "ciment"]):
        return "chape_beton"
    elif any(word in query_lower for word in ["isolation", "isoler", "isolant"]) and any(word in query_lower for word in ["mur", "murs", "paroi"]):
        return "isolation_mur"
    
    return None

def get_project_materials(project_type: str) -> List[Dict[str, Any]]:
    """
    Retourne la liste des matériaux nécessaires pour un type de projet
    
    Args:
        project_type: Type de projet
    
    Returns:
        Liste des matériaux nécessaires
    """
    if project_type in CONSTRUCTION_KNOWLEDGE:
        return CONSTRUCTION_KNOWLEDGE[project_type]["materiaux_necessaires"]
    return []

def get_expert_prompt() -> str:
    """
    Retourne le prompt système pour transformer l'IA en expert en construction
    """
    return """Tu es un expert en construction et en matériaux de bâtiment avec plus de 20 ans d'expérience.
Tu connais parfaitement :
- Les techniques de construction (maçonnerie, béton, isolation, etc.)
- Les matériaux nécessaires pour chaque type de projet
- Les normes et bonnes pratiques du bâtiment
- Les quantités approximatives de matériaux selon les projets

Quand un utilisateur décrit un projet de construction, tu dois :
1. Identifier le type de projet
2. Lister TOUS les matériaux nécessaires pour réaliser ce projet
3. Pour chaque matériau, indiquer des critères de recherche précis
4. Proposer des alternatives si plusieurs options sont possibles

Tu dois être précis et complet, en pensant à tous les matériaux nécessaires (principaux et accessoires)."""

