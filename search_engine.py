"""
Moteur de recherche intelligent utilisant un modèle LLM
pour comprendre les requêtes et effectuer une recherche hybride
"""

import time
import re
import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ollama
from rapidfuzz import fuzz, process
from synonyms import expand_with_synonyms, get_synonyms
from expert_knowledge import detect_project_type, get_project_materials, get_expert_prompt
from knowledge_base import KnowledgeBase

class SearchEngine:
    """Moteur de recherche intelligent avec LLM"""
    
    def __init__(self, articles: List[Dict[str, Any]], model_name: str = "nchapman/ministral-8b-instruct-2410:8b", use_llm: bool = True):
        self.articles = articles
        self.model_name = model_name
        self.use_llm = use_llm  # Option pour désactiver le LLM et accélérer
        
        # Modèle d'embeddings pour la recherche sémantique
        print("Chargement du modèle d'embeddings...")
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Préparation des données pour la recherche
        print("Préparation des embeddings...")
        self._prepare_search_data()
        
        # Base de connaissances Eurocodes (optionnelle)
        self.knowledge_base = None
        eurocodes_dir = os.getenv("EUROCODES_DIR") or os.path.join(os.getcwd(), "6_EUROCODES")
        knowledge_cache_dir = os.getenv("KNOWLEDGE_CACHE_DIR", "knowledge_cache")
        try:
            self.knowledge_base = KnowledgeBase(
                documents_dir=eurocodes_dir,
                embedding_model=self.embedding_model,
                cache_dir=knowledge_cache_dir,
            )
            if self.knowledge_base and not self.knowledge_base.enabled:
                self.knowledge_base = None
        except Exception as exc:
            print(f"⚠️  Impossible d'initialiser la base Eurocodes: {exc}")
            self.knowledge_base = None
        
        print("Moteur de recherche initialisé!")
    
    def _prepare_search_data(self):
        """Prépare les données pour la recherche (embeddings, index)"""
        # Vérifier si les embeddings sont en cache
        embeddings_cache_path = "embeddings_cache.npy"
        search_texts_cache_path = "search_texts_cache.pkl"
        
        if os.path.exists(embeddings_cache_path) and os.path.exists(search_texts_cache_path):
            print("Chargement des embeddings depuis le cache...")
            try:
                import pickle
                self.embeddings = np.load(embeddings_cache_path)
                with open(search_texts_cache_path, 'rb') as f:
                    self.search_texts = pickle.load(f)
                    self.article_index = pickle.load(f)
                print(f"✅ {len(self.article_index)} articles chargés depuis le cache!")
                return
            except Exception as e:
                print(f"⚠️  Erreur lors du chargement du cache: {e}")
                print("   Régénération des embeddings...")
        
        # Créer un texte de recherche pour chaque article
        self.search_texts = []
        self.article_index = []
        
        for article in self.articles:
            # Combiner les champs pertinents pour la recherche
            search_text = self._build_search_text(article)
            self.search_texts.append(search_text)
            self.article_index.append(article)
        
        # Générer les embeddings
        print(f"Génération des embeddings pour {len(self.search_texts)} articles...")
        self.embeddings = self.embedding_model.encode(
            self.search_texts,
            show_progress_bar=True,
            batch_size=32
        )
        print("Embeddings générés!")
        
        # Sauvegarder en cache
        try:
            import pickle
            np.save(embeddings_cache_path, self.embeddings)
            with open(search_texts_cache_path, 'wb') as f:
                pickle.dump(self.search_texts, f)
                pickle.dump(self.article_index, f)
            print("✅ Embeddings sauvegardés en cache!")
        except Exception as e:
            print(f"⚠️  Impossible de sauvegarder le cache: {e}")
    
    def _build_search_text(self, article: Dict[str, Any]) -> str:
        """Construit le texte de recherche pour un article"""
        parts = []
        
        if article.get('libelle'):
            parts.append(article['libelle'])
        if article.get('designation'):
            parts.append(article['designation'])
        if article.get('motsClefs'):
            parts.append(article['motsClefs'])
        if article.get('mainKeyWords'):
            parts.append(article['mainKeyWords'])
        if article.get('reference'):
            parts.append(f"Référence: {article['reference']}")
        if article.get('codeFournisseur'):
            parts.append(f"Code: {article['codeFournisseur']}")
        
        return " | ".join(parts)
    
    def _understand_query_with_llm(self, query: str) -> Dict[str, Any]:
        """
        Utilise le modèle LLM pour comprendre la requête utilisateur
        et extraire les intentions, synonymes, termes clés et contraintes de prix
        Détecte aussi si c'est un projet de construction à décomposer
        """
        # Détecter si c'est un projet de construction
        project_type = detect_project_type(query)
        is_project = project_type is not None
        knowledge_context = ""
        if self.knowledge_base:
            try:
                top_k = 6 if is_project else 3
                knowledge_context = self.knowledge_base.build_context_block(
                    query=query,
                    top_k=top_k,
                )
            except Exception as exc:
                print(f"⚠️  Impossible de récupérer le contexte Eurocodes: {exc}")
                knowledge_context = ""
        
        context_prefix = f"{knowledge_context}\n\n" if knowledge_context else ""
        
        if is_project:
            # Mode expert : décomposer le projet en matériaux
            prompt = f"""{context_prefix}Tu es un expert en construction. Un utilisateur veut réaliser ce projet : "{query}"

Analyse ce projet et liste TOUS les matériaux nécessaires pour le réaliser.

Pour chaque matériau, indique :
- Le type de matériau (ex: "parpaing", "mortier", "enduit")
- Les critères de recherche précis pour trouver ce matériau dans un catalogue
- Une description de l'usage de ce matériau dans le projet

Réponds UNIQUEMENT au format JSON suivant (sans autre texte) :
{{
  "is_project": true,
  "project_type": "{project_type}",
  "reformulation": "description du projet",
  "prix_max": nombre ou null,
  "prix_min": nombre ou null,
  "materiaux": [
    {{
      "type": "nom du matériau",
      "critères_recherche": "mots-clés pour chercher ce matériau",
      "description": "description de l'usage"
    }}
  ]
}}

Exemple pour "je veux créer une cloison en parpaings" :
{{
  "is_project": true,
  "project_type": "cloison_parpaings",
  "reformulation": "cloison parpaings",
  "prix_max": null,
  "prix_min": null,
  "materiaux": [
    {{"type": "parpaing", "critères_recherche": "parpaing cloison creux", "description": "Parpaings pour cloison intérieure"}},
    {{"type": "mortier", "critères_recherche": "mortier colle", "description": "Mortier pour assembler les parpaings"}},
    {{"type": "enduit", "critères_recherche": "enduit intérieur lissage", "description": "Enduit de finition"}}
  ]
}}"""
        else:
            # Mode recherche classique
            prompt = f"""{context_prefix}Analyse cette requête de recherche de matériaux de construction et extrais les informations suivantes :
1. Une reformulation optimisée pour la recherche
2. Les contraintes de prix (prix maximum, prix minimum) si mentionnées

Requête: "{query}"

Réponds UNIQUEMENT au format JSON suivant (sans autre texte) :
{{
  "is_project": false,
  "reformulation": "reformulation optimisée",
  "prix_max": nombre ou null,
  "prix_min": nombre ou null
}}

Exemples :
- "carrelage à moins de 40 euros" → {{"is_project": false, "reformulation": "carrelage", "prix_max": 40, "prix_min": null}}
- "sable entre 20 et 50 euros" → {{"is_project": false, "reformulation": "sable", "prix_max": 50, "prix_min": 20}}
- "pavé terrasse" → {{"is_project": false, "reformulation": "pavé terrasse", "prix_max": null, "prix_min": null}}"""

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'system',
                        'content': get_expert_prompt() if is_project else 'Tu es un expert en matériaux de construction. Réponds UNIQUEMENT avec un JSON valide, sans autre texte ni explication.'
                    },
                    {
                        'role': 'user',
                        'content': prompt
                    }
                ],
                options={
                    'temperature': 0.2,  # Plus déterministe pour le JSON
                    'num_predict': 500 if is_project else 100  # Plus long pour les projets
                }
            )
            
            result_text = response['message']['content'].strip()
            
            # Extraire le JSON de la réponse (peut être entouré de markdown ou autre)
            # Utiliser une regex plus robuste pour capturer les JSON imbriqués
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    is_project = parsed.get('is_project', False)
                    reformulation = parsed.get('reformulation', query)
                    prix_max = parsed.get('prix_max')
                    prix_min = parsed.get('prix_min')
                    materiaux = parsed.get('materiaux', [])
                    
                    # Si c'est un projet mais que l'IA n'a pas fourni de matériaux, utiliser la base de connaissances
                    if is_project and not materiaux:
                        project_type = parsed.get('project_type') or detect_project_type(query)
                        if project_type:
                            materiaux_base = get_project_materials(project_type)
                            materiaux = [
                                {
                                    "type": m.get("type", ""),
                                    "critères_recherche": m.get("critères", ""),
                                    "description": m.get("description", "")
                                }
                                for m in materiaux_base
                            ]
                except Exception as e:
                    # Si le JSON est invalide, fallback
                    is_project = False
                    reformulation = query
                    prix_max = None
                    prix_min = None
                    materiaux = []
            else:
                # Pas de JSON trouvé, essayer d'extraire le prix manuellement
                reformulation = result_text.strip('"').strip("'").strip()
                prix_max = None
                prix_min = None
                
                # Extraction manuelle des prix avec regex
                prix_patterns = [
                    r'(?:à|moins de|sous|inférieur à|max|maximum)\s*(\d+)\s*(?:euros?|€|eur)',
                    r'(\d+)\s*(?:euros?|€|eur)\s*(?:ou moins|maximum|max)',
                    r'entre\s*(\d+)\s*(?:et|à)\s*(\d+)\s*(?:euros?|€|eur)',
                    r'(\d+)\s*(?:à|jusqu\'?à)\s*(\d+)\s*(?:euros?|€|eur)',
                    r'plus de\s*(\d+)\s*(?:euros?|€|eur)',
                    r'(\d+)\s*(?:euros?|€|eur)\s*(?:ou plus|minimum|min)'
                ]
                
                for pattern in prix_patterns:
                    match = re.search(pattern, query.lower())
                    if match:
                        if 'entre' in pattern or 'à' in pattern:
                            prix_min = float(match.group(1))
                            if len(match.groups()) > 1:
                                prix_max = float(match.group(2))
                        elif 'moins' in pattern or 'sous' in pattern or 'inférieur' in pattern or 'max' in pattern:
                            prix_max = float(match.group(1))
                        elif 'plus' in pattern or 'minimum' in pattern or 'min' in pattern:
                            prix_min = float(match.group(1))
                        break
            
            # Extraire les termes principaux de la reformulation
            terms = [t.lower() for t in reformulation.split() if len(t) > 2]
            
            return {
                "termes_principaux": terms,
                "synonymes": [],
                "caracteristiques": [],
                "reformulation": reformulation if reformulation else query,
                "prix_max": prix_max,
                "prix_min": prix_min,
                "is_project": is_project if 'is_project' in locals() else False,
                "materiaux": materiaux if 'materiaux' in locals() else [],
                "knowledge_context": knowledge_context,
            }
        except Exception as e:
            # Fallback silencieux en cas d'erreur
            terms = [t.lower() for t in query.split() if len(t) > 2]
            
            # Détecter si c'est un projet même en fallback
            project_type = detect_project_type(query)
            is_project = project_type is not None
            materiaux = []
            
            # Si c'est un projet, utiliser la base de connaissances
            if is_project:
                materiaux_base = get_project_materials(project_type)
                materiaux = [
                    {
                        "type": m.get("type", ""),
                        "critères_recherche": m.get("critères", ""),
                        "description": m.get("description", "")
                    }
                    for m in materiaux_base
                ]
            
            # Extraction manuelle des prix en fallback
            prix_max = None
            prix_min = None
            prix_patterns = [
                r'(?:à|moins de|sous|inférieur à|max|maximum)\s*(\d+)\s*(?:euros?|€|eur)',
                r'(\d+)\s*(?:euros?|€|eur)\s*(?:ou moins|maximum|max)',
                r'entre\s*(\d+)\s*(?:et|à)\s*(\d+)\s*(?:euros?|€|eur)',
                r'(\d+)\s*(?:à|jusqu\'?à)\s*(\d+)\s*(?:euros?|€|eur)',
                r'plus de\s*(\d+)\s*(?:euros?|€|eur)',
                r'(\d+)\s*(?:euros?|€|eur)\s*(?:ou plus|minimum|min)'
            ]
            
            for pattern in prix_patterns:
                match = re.search(pattern, query.lower())
                if match:
                    if 'entre' in pattern or 'à' in pattern:
                        prix_min = float(match.group(1))
                        if len(match.groups()) > 1:
                            prix_max = float(match.group(2))
                    elif 'moins' in pattern or 'sous' in pattern or 'inférieur' in pattern or 'max' in pattern:
                        prix_max = float(match.group(1))
                    elif 'plus' in pattern or 'minimum' in pattern or 'min' in pattern:
                        prix_min = float(match.group(1))
                    break
            
            return {
                "termes_principaux": terms,
                "synonymes": [],
                "caracteristiques": [],
                "reformulation": query,
                "prix_max": prix_max,
                "prix_min": prix_min,
                "is_project": is_project,
                "materiaux": materiaux,
                "knowledge_context": knowledge_context,
            }
    def _semantic_search(self, query: str, limit: int = 10) -> List[tuple]:
        """Recherche sémantique basée sur les embeddings"""
        # Embedding de la requête
        query_embedding = self.embedding_model.encode([query])
        
        # Calcul de similarité cosinus
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Tri par similarité
        top_indices = np.argsort(similarities)[::-1][:limit * 2]  # Prendre 2x pour le filtrage
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            results.append((idx, score))
        
        return results
    
    def _correct_typos(self, terms: List[str], all_libelles: List[str]) -> List[str]:
        """
        Corrige les fautes de frappe dans les termes de recherche
        
        Args:
            terms: Liste de termes à corriger
            all_libelles: Liste de tous les libellés pour référence
        
        Returns:
            Liste de termes corrigés
        """
        corrected = []
        
        # Extraire tous les mots uniques des libellés pour référence
        all_words = set()
        for libelle in all_libelles:
            words = [w.lower() for w in libelle.split() if len(w) > 2]
            all_words.update(words)
        
        for term in terms:
            if len(term) <= 2:
                corrected.append(term)
                continue
            
            # Chercher le mot le plus proche dans les libellés
            best_match = process.extractOne(
                term.lower(),
                list(all_words),
                scorer=fuzz.ratio,
                score_cutoff=70  # Minimum 70% de similarité
            )
            
            if best_match and best_match[1] >= 70:
                # Si très proche (>90%), utiliser la correction
                if best_match[1] >= 90:
                    corrected.append(best_match[0])
                else:
                    # Sinon, garder l'original mais ajouter le terme proche
                    corrected.append(term.lower())
                    if best_match[0] not in corrected:
                        corrected.append(best_match[0])
            else:
                corrected.append(term.lower())
        
        return corrected
    
    def _textual_search(self, query: str, understood_query: Dict[str, Any], limit: int = 10) -> List[tuple]:
        """Recherche textuelle classique avec scoring amélioré"""
        query_lower = query.lower()
        terms = understood_query.get("termes_principaux", [t for t in query.split() if len(t) > 2])
        
        # Filtrer les termes trop courts
        search_terms = [t.lower() for t in terms if len(t) > 2]
        
        if not search_terms:
            search_terms = [t.lower() for t in query.split() if len(t) > 2]
        
        # Correction des fautes de frappe
        all_libelles = [article.get('libelle', '') for article in self.article_index]
        search_terms = self._correct_typos(search_terms, all_libelles)
        
        # Expansion avec synonymes
        search_terms = expand_with_synonyms(search_terms)
        
        # Expansion de requête pour certains cas spécifiques
        # "sable pour béton" -> chercher aussi "mélange à béton"
        expanded_terms = set(search_terms)
        if "sable" in search_terms and "béton" in search_terms:
            expanded_terms.add("mélange")
            expanded_terms.add("alluvionnaire")
        if "pavé" in search_terms:
            # Pour "pavé", on veut que ce soit au début du libellé
            pass  # Géré par le scoring
        
        results = []
        
        for idx, article in enumerate(self.article_index):
            score = 0.0
            libelle = (article.get('libelle', '') or '').lower()
            designation = (article.get('designation', '') or '').lower()
            mots_clefs = (article.get('motsClefs', '') or '').lower()
            reference = (article.get('reference', '') or '').lower()
            
            # Score pour chaque terme
            terms_in_libelle = []
            terms_in_designation = []
            terms_in_mots_clefs = []
            
            # Vérifier aussi les termes étendus
            all_search_terms = list(expanded_terms)
            
            for term in all_search_terms:
                # Score très élevé si le terme est dans le libellé (titre principal)
                if term in libelle:
                    if term in search_terms:  # Terme original
                        terms_in_libelle.append(term)
                        # Bonus si le terme est au début du libellé
                        if libelle.startswith(term):
                            score += 5.0
                        else:
                            score += 4.0
                    else:  # Terme étendu
                        score += 2.0  # Moins de poids pour les termes étendus
                # Score élevé si dans la désignation
                elif term in designation:
                    if term in search_terms:
                        terms_in_designation.append(term)
                        score += 2.5
                    else:
                        score += 1.0
                # Score moyen si dans les mots-clés
                elif term in mots_clefs:
                    if term in search_terms:
                        terms_in_mots_clefs.append(term)
                        score += 2.0
                    else:
                        score += 0.5
                # Score faible si ailleurs dans le texte
                elif term in self.search_texts[idx].lower():
                    if term in search_terms:
                        score += 0.5
                    else:
                        score += 0.2
            
            # Bonus pour correspondance exacte de référence
            if query_lower.replace(' ', '') in reference:
                score += 3.0
            
            # Bonus MAJEUR si TOUS les termes ORIGINAUX sont présents (même dans des champs différents)
            all_terms_found = all(
                term in libelle or term in designation or term in mots_clefs or term in self.search_texts[idx].lower()
                for term in search_terms
            )
            if all_terms_found:
                # Bonus plus élevé si au moins un terme est dans le libellé
                if terms_in_libelle:
                    score += 4.0  # Augmenté
                else:
                    score += 2.0
            
            # Bonus spécial pour "sable pour béton" : si on a "sable" dans libellé ET "mélange" ou "béton" quelque part
            if "sable" in search_terms and "béton" in search_terms:
                # PRIORITÉ 1: Sable alluvionnaire (le plus pertinent pour béton)
                if "sable" in libelle and "alluvionnaire" in libelle:
                    score += 8.0  # Bonus TRÈS élevé pour sable alluvionnaire
                # PRIORITÉ 2: Mélange à béton
                elif "mélange" in libelle and "béton" in libelle:
                    score += 7.0  # Bonus très élevé pour mélange à béton
                # PRIORITÉ 3: Sable avec mélange/béton dans libellé
                elif "sable" in libelle and ("mélange" in libelle or "béton" in libelle):
                    score += 6.0
                # PRIORITÉ 4: Sable avec mélange/béton dans designation
                elif "sable" in libelle and ("mélange" in designation or "béton" in designation):
                    score += 5.0
                # PRIORITÉ 5: Sable avec mélange/béton ailleurs
                elif "sable" in libelle and ("mélange" in self.search_texts[idx].lower() or "béton" in self.search_texts[idx].lower()):
                    score += 4.0
                # PRIORITÉ 6: Sable simple (moins pertinent)
                elif "sable" in libelle and libelle.startswith("sable"):
                    score += 2.0
                # PÉNALITÉ: Sable polymère, équestre, etc. (pas pour béton)
                elif "sable" in libelle and ("polymère" in libelle or "équestre" in libelle or "pétanque" in libelle):
                    score *= 0.3  # Pénalité sévère
            
            # Bonus spécial pour "pavé" : doit être au début du libellé pour être vraiment un pavé
            if "pavé" in search_terms:
                if libelle.startswith("pavé") or libelle.startswith("pavés"):
                    score += 6.0  # Bonus MAJEUR pour vrais pavés
                elif "pavé" in libelle or "pavés" in libelle:
                    score += 2.0  # Bonus modéré si dans libellé
                else:
                    # Si "pavé" n'est pas dans le libellé, pénalité TRÈS sévère (accessoires)
                    score *= 0.1  # Pénalité encore plus sévère
            
            # Bonus si plusieurs termes sont dans le libellé
            if len(terms_in_libelle) >= 2:
                score += 2.0
            
            # Pénalité réduite si aucun terme principal n'est dans le libellé
            # Mais seulement si on a peu de termes trouvés
            if not terms_in_libelle and len(terms_in_designation) + len(terms_in_mots_clefs) < len(search_terms):
                score *= 0.5  # Pénalité moins sévère
            
            if score > 0:
                # Normaliser le score (en tenant compte des bonus spéciaux)
                # Calculer un max possible réaliste mais pas trop restrictif
                base_max = len(search_terms) * 5.0  # Termes dans libellé
                bonus_max = 8.0  # Bonus spéciaux (réduit pour ne pas trop normaliser)
                max_possible_score = base_max + bonus_max
                # Normalisation qui préserve les différences de score
                normalized_score = min(score / max_possible_score, 1.0)
                # Appliquer un facteur de boost pour les bons résultats
                if normalized_score > 0.3:
                    normalized_score = min(normalized_score * 1.2, 1.0)
                results.append((idx, normalized_score))
        
        # Trier par score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit * 3]  # Prendre plus de résultats pour le filtrage
    
    def _hybrid_search(self, query: str, understood_query: Dict[str, Any], limit: int = 10) -> List[tuple]:
        """Recherche hybride combinant sémantique et textuelle"""
        # Recherche textuelle d'abord (plus rapide et souvent plus pertinente)
        textual_results = self._textual_search(query, understood_query, limit * 2)
        
        # Détecter si c'est une requête qui nécessite une précision textuelle
        # (ex: "sable pour béton", "pavé terrasse")
        needs_textual_precision = any(
            (term in query.lower() and other_term in query.lower())
            for term, other_term in [("sable", "béton"), ("pavé", "terrasse"), ("pavé", "extérieur")]
        )
        
        # Si on a de bons résultats textuels, on les privilégie
        if textual_results and textual_results[0][1] > 0.3:
            # Recherche sémantique seulement pour enrichir
            semantic_results = self._semantic_search(query, limit)
            
            # Combiner les résultats avec un scoring hybride
            combined_scores = {}
            
            # Poids ajustés selon le type de requête
            if needs_textual_precision:
                # Pour les requêtes précises, privilégier encore plus le textuel
                textual_weight = 0.85
                semantic_weight = 0.15
            else:
                # Poids standard: 70% textuel, 30% sémantique
                textual_weight = 0.7
                semantic_weight = 0.3
            
            # Les résultats textuels sont prioritaires
            for idx, score in textual_results:
                combined_scores[idx] = score * textual_weight
            
            # Enrichir avec les résultats sémantiques
            for idx, score in semantic_results:
                if idx in combined_scores:
                    # Si déjà dans textuel, ajouter sémantique avec moins de poids
                    combined_scores[idx] += score * semantic_weight * 0.7
                else:
                    # Si pas dans textuel, moins de poids
                    combined_scores[idx] = score * semantic_weight * 0.3
        else:
            # Si pas de bons résultats textuels, utiliser surtout sémantique
            semantic_results = self._semantic_search(query, limit * 2)
            combined_scores = {idx: score * 0.8 for idx, score in semantic_results}
            
            # Ajouter quand même les résultats textuels avec moins de poids
            for idx, score in textual_results:
                if idx in combined_scores:
                    combined_scores[idx] += score * 0.2
                else:
                    combined_scores[idx] = score * 0.2
        
        # Trier et retourner
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [(idx, score) for idx, score in sorted_results[:limit * 2]]
    
    def _generate_confirmation_message(self, original_query: str, reformulation: str, num_results: int) -> str:
        """
        Génère une phrase de confirmation personnalisée pour la recherche vocale
        """
        if num_results == 0:
            return f"Je n'ai malheureusement trouvé aucun produit correspondant à votre recherche."
        
        # Extraire les termes principaux de la reformulation
        terms = [t.lower() for t in reformulation.split() if len(t) > 2]
        
        # Phrases de confirmation variées
        if num_results == 1:
            messages = [
                f"Voici le produit que nous pouvons vous proposer concernant {reformulation}.",
                f"J'ai trouvé un produit correspondant à votre recherche de {reformulation}.",
                f"Voici une solution pour {reformulation}."
            ]
        elif num_results <= 5:
            messages = [
                f"Voici les {num_results} produits que nous pouvons vous proposer concernant {reformulation}.",
                f"J'ai trouvé {num_results} produits correspondant à votre recherche de {reformulation}.",
                f"Voici {num_results} solutions pour {reformulation}."
            ]
        else:
            messages = [
                f"Voici les {num_results} produits que nous pouvons vous proposer concernant {reformulation}.",
                f"J'ai trouvé {num_results} produits correspondant à votre recherche de {reformulation}.",
                f"Voici une sélection de {num_results} produits pour {reformulation}."
            ]
        
        # Choisir une phrase aléatoirement (basé sur le hash de la requête pour la cohérence)
        import random
        random.seed(hash(reformulation) % 1000)
        return random.choice(messages)
    
    def search(self, query: str, limit: int = 10, min_score: float = 0.0) -> Dict[str, Any]:
        """
        Effectue une recherche intelligente avec cache
        
        Args:
            query: Requête de recherche en langage naturel
            limit: Nombre maximum de résultats
            min_score: Score minimum pour inclure un résultat
        
        Returns:
            Dictionnaire avec les résultats et métadonnées
        """
        start_time = time.time()
        
        # Hash de la requête pour le cache
        query_hash = hashlib.md5(f"{query.lower().strip()}_{limit}_{min_score}".encode()).hexdigest()
        
        # Vérifier le cache (simple dict en mémoire)
        if not hasattr(self, '_search_cache'):
            self._search_cache = {}
        
        if query_hash in self._search_cache:
            cached_result = self._search_cache[query_hash]
            # Ajouter le temps de recherche (très rapide depuis cache)
            cached_result['search_time'] = 0.001
            return cached_result
        
        # Comprendre la requête avec le LLM (peut être désactivé pour plus de rapidité)
        if self.use_llm:
            try:
                understood_query = self._understand_query_with_llm(query)
                reformulation = understood_query.get("reformulation", query)
                is_project = understood_query.get("is_project", False)
                materiaux = understood_query.get("materiaux", [])
            except:
                # Fallback rapide si LLM échoue
                terms = [t.lower() for t in query.split() if len(t) > 2]
                project_type = detect_project_type(query)
                is_project = project_type is not None
                materiaux = []
                if is_project:
                    materiaux_base = get_project_materials(project_type)
                    materiaux = [
                        {
                            "type": m.get("type", ""),
                            "critères_recherche": m.get("critères", ""),
                            "description": m.get("description", "")
                        }
                        for m in materiaux_base
                    ]
                understood_query = {
                    "termes_principaux": terms,
                    "synonymes": [],
                    "caracteristiques": [],
                    "reformulation": query,
                    "prix_max": None,
                    "prix_min": None,
                    "is_project": is_project,
                    "materiaux": materiaux
                }
                reformulation = query
        else:
            # Mode rapide sans LLM - extraction manuelle des prix
            terms = [t.lower() for t in query.split() if len(t) > 2]
            prix_max = None
            prix_min = None
        
            # Détecter si c'est un projet
            project_type = detect_project_type(query)
            is_project = project_type is not None
            materiaux = []
            if is_project:
                materiaux_base = get_project_materials(project_type)
                materiaux = [
                    {
                        "type": m.get("type", ""),
                        "critères_recherche": m.get("critères", ""),
                        "description": m.get("description", "")
                    }
                    for m in materiaux_base
                ]
        
            # Extraction manuelle des prix avec regex
            prix_patterns = [
                r'(?:à|moins de|sous|inférieur à|max|maximum)\s*(\d+)\s*(?:euros?|€|eur)',
                r'(\d+)\s*(?:euros?|€|eur)\s*(?:ou moins|maximum|max)',
                r'entre\s*(\d+)\s*(?:et|à)\s*(\d+)\s*(?:euros?|€|eur)',
                r'(\d+)\s*(?:à|jusqu\'?à)\s*(\d+)\s*(?:euros?|€|eur)',
                r'plus de\s*(\d+)\s*(?:euros?|€|eur)',
                r'(\d+)\s*(?:euros?|€|eur)\s*(?:ou plus|minimum|min)'
            ]
        
            for pattern in prix_patterns:
                match = re.search(pattern, query.lower())
                if match:
                    if 'entre' in pattern or 'à' in pattern:
                        prix_min = float(match.group(1))
                        if len(match.groups()) > 1:
                            prix_max = float(match.group(2))
                    elif 'moins' in pattern or 'sous' in pattern or 'inférieur' in pattern or 'max' in pattern:
                        prix_max = float(match.group(1))
                    elif 'plus' in pattern or 'minimum' in pattern or 'min' in pattern:
                        prix_min = float(match.group(1))
                    break
        
            understood_query = {
                "termes_principaux": terms,
                "synonymes": [],
                "caracteristiques": [],
                "reformulation": query,
                "prix_max": prix_max,
                "prix_min": prix_min,
                "is_project": is_project,
                "materiaux": materiaux
            }
            reformulation = query
        
        # Extraire les contraintes de prix
        prix_max = understood_query.get("prix_max")
        prix_min = understood_query.get("prix_min")
        is_project = understood_query.get("is_project", False)
        materiaux = understood_query.get("materiaux", [])
        results_by_material = None
        
        # Si c'est un projet avec des matériaux, effectuer plusieurs recherches
        if is_project and materiaux:
            # Mode projet : recherche par matériau
            # Pour les projets, augmenter significativement la limite
            project_limit = max(limit * 2, 30)  # Au moins 30 résultats pour un projet
            results_by_material = {}
            total_results = 0
            
            # Utiliser la base de connaissances plutôt que les matériaux inventés par l'IA
            project_type = detect_project_type(query)
            if project_type:
                materiaux_base = get_project_materials(project_type)
                # Convertir au format attendu
                materiaux_to_use = [
                    {
                        "type": m.get("type", ""),
                        "critères_recherche": m.get("critères", ""),
                        "description": m.get("description", "")
                    }
                    for m in materiaux_base
                ]
            else:
                # Fallback sur les matériaux de l'IA si pas de project_type
                materiaux_to_use = materiaux[:4]  # Limiter aux 4 premiers
            
            for materiau in materiaux_to_use:
                materiau_type = materiau.get("type", "")
                criteres = materiau.get("critères_recherche", materiau_type)
                description = materiau.get("description", "")
                
                # Créer une requête de recherche pour ce matériau
                search_query = criteres if criteres else materiau_type
                
                # Créer un understood_query simplifié pour ce matériau
                materiau_query = {
                    "termes_principaux": [t.lower() for t in search_query.split() if len(t) > 2],
                    "synonymes": [],
                    "caracteristiques": [],
                    "reformulation": search_query,
                    "prix_max": prix_max,
                    "prix_min": prix_min
                }
                
                # Recherche pour ce matériau (au moins 8-10 résultats par matériau)
                materiau_limit = max(project_limit // len(materiaux_to_use), 8)  # Au moins 8 résultats par matériau
                search_results = self._hybrid_search(search_query, materiau_query, materiau_limit * 3 if (prix_max or prix_min) else materiau_limit)
                
                # Formater les résultats pour ce matériau avec filtrage de pertinence
                materiau_results = []
                
                # Mots-clés à exclure selon le type de matériau
                exclude_keywords = {
                    "parpaing": [
                        "clou", "vis", "fixation", "cheville", "rivet", "bétonnière", "bétonniere",
                        "bloc porte", "bloc de soutènement", "bloc soutènement", "porte", "huisserie",
                        "poussant", "matricé", "isogyl", "arkose", "thomur", "mégalyr", "décoratif"
                    ],
                    "mortier": [
                        "clou", "vis", "fixation", "cheville", "rivet", "plâtre", "platre", "placo",
                        "bloc porte", "porte", "huisserie", "poussant"
                    ],
                    "enduit": [
                        "clou", "vis", "fixation", "cheville", "rivet", "plâtre", "platre", "placo",
                        "colle plâtre", "bloc porte", "porte", "huisserie", "poussant"
                    ],
                }
                
                # Mots-clés requis pour valider la pertinence (au moins un doit être présent)
                required_keywords = {
                    "parpaing": ["bloc", "aggloméré", "cloison", "creux", "béton", "moellon", "bancher", "chainage"],  # Inclut les termes utilisés dans la BDD
                    "mortier": ["mortier", "colle", "bâtard"],
                    "enduit": ["enduit", "lissage", "finition", "garnissage"],
                }
                
                exclude_list = exclude_keywords.get(materiau_type.lower(), [])
                required_list = required_keywords.get(materiau_type.lower(), [])
                
                for idx, score in search_results:
                    if score >= min_score:
                        article = self.article_index[idx]
                        
                        # Vérifier la pertinence : exclure les produits avec des mots-clés non pertinents
                        libelle_lower = article.get('libelle', '').lower()
                        designation_lower = (article.get('designation') or '').lower()
                        text_article = f"{libelle_lower} {designation_lower}"
                        
                        # Exclure si le produit contient des mots-clés non pertinents (sans exception de score)
                        if exclude_list and any(excl_word in text_article for excl_word in exclude_list):
                            continue  # Exclure systématiquement
                        
                        # Vérifier qu'au moins un mot-clé requis est présent
                        if required_list:
                            has_required = any(req_word in text_article for req_word in required_list)
                            if not has_required:
                                # Si aucun mot-clé requis n'est présent, exclure (sauf score très élevé)
                                if score < 0.5:  # Seuil plus élevé pour être sûr
                                    continue
                        
                        # Déterminer le prix à utiliser
                        prix_vente = float(article.get('prix_vente', 0.0))
                        prix_vente_particulier = float(article.get('prix_vente_particulier', 0.0)) if article.get('prix_vente_particulier') else None
                        prix_a_utiliser = prix_vente_particulier if prix_vente_particulier else prix_vente
                        
                        # Filtrer par prix
                        if prix_max is not None and prix_a_utiliser > prix_max:
                            continue
                        if prix_min is not None and prix_a_utiliser < prix_min:
                            continue
                        
                        result = {
                            "id": article.get('id'),
                            "libelle": article.get('libelle', ''),
                            "designation": article.get('designation'),
                            "reference": article.get('reference', ''),
                            "prix_vente": prix_vente,
                            "prix_vente_particulier": prix_vente_particulier,
                            "photo1": article.get('photo1'),
                            "score": round(score, 4),
                            "categorieId": article.get('categorieId', 0),
                            "motsClefs": article.get('motsClefs')
                        }
                        materiau_results.append(result)
                        total_results += 1
                        
                        if len(materiau_results) >= materiau_limit:
                            break
                
                # Stocker les résultats par matériau (même si vide, pour afficher la section)
                results_by_material[materiau_type] = {
                    "description": description,
                    "results": materiau_results
                }
            
            # Organiser les résultats pour le retour (ne pas limiter, garder tous les résultats par matériau)
            results = []
            for materiau_type, data in results_by_material.items():
                results.extend(data["results"])
            
            # Ne pas limiter pour les projets, on veut voir tous les matériaux
            
        else:
            # Mode recherche classique
            search_results = self._hybrid_search(query, understood_query, limit * 3 if (prix_max or prix_min) else limit)
            
            # Formater les résultats et filtrer par prix
            results = []
            for idx, score in search_results:
                if score >= min_score:
                    article = self.article_index[idx]
                    
                    # Déterminer le prix à utiliser (priorité au prix particulier)
                    prix_vente = float(article.get('prix_vente', 0.0))
                    prix_vente_particulier = float(article.get('prix_vente_particulier', 0.0)) if article.get('prix_vente_particulier') else None
                    prix_a_utiliser = prix_vente_particulier if prix_vente_particulier else prix_vente
                    
                    # Filtrer par prix si des contraintes sont définies
                    if prix_max is not None and prix_a_utiliser > prix_max:
                        continue
                    if prix_min is not None and prix_a_utiliser < prix_min:
                        continue
                    
                    result = {
                        "id": article.get('id'),
                        "libelle": article.get('libelle', ''),
                        "designation": article.get('designation'),
                        "reference": article.get('reference', ''),
                        "prix_vente": prix_vente,
                        "prix_vente_particulier": prix_vente_particulier,
                        "photo1": article.get('photo1'),
                        "score": round(score, 4),
                        "categorieId": article.get('categorieId', 0),
                        "motsClefs": article.get('motsClefs')
                    }
                    results.append(result)
                    
                    if len(results) >= limit:
                        break
        
        search_time = time.time() - start_time
        
        # Générer une phrase de confirmation personnalisée
        if is_project and materiaux:
            # Utiliser uniquement les matériaux de la base de connaissances (pas ceux inventés par l'IA)
            # Si on a un project_type, utiliser la base de connaissances
            project_type = detect_project_type(query)
            if project_type:
                materiaux_base = get_project_materials(project_type)
                # Utiliser les types de matériaux de la base de connaissances
                materiaux_noms = [m.get("type", "").capitalize() for m in materiaux_base]
            else:
                # Sinon utiliser ceux fournis par l'IA (mais limiter aux 3-4 premiers)
                materiaux_noms = [m.get("type", "").capitalize() for m in materiaux[:4]]
            
            if len(materiaux_noms) == 1:
                materiaux_text = materiaux_noms[0]
            elif len(materiaux_noms) == 2:
                materiaux_text = f"{materiaux_noms[0]} et {materiaux_noms[1]}"
            else:
                materiaux_text = ", ".join(materiaux_noms[:-1]) + f" et {materiaux_noms[-1]}"
            
            # Message dynamique selon le type de projet
            project_description = reformulation.lower()
            if "cloison" in project_description:
                confirmation_message = f"Pour créer une cloison en parpaings, vous aurez besoin de : {materiaux_text.lower()}. Voici les produits que nous pouvons vous proposer pour chaque matériau nécessaire."
            elif "mur" in project_description:
                confirmation_message = f"Pour construire un mur en parpaings, vous aurez besoin de : {materiaux_text.lower()}. Voici les produits que nous pouvons vous proposer pour chaque matériau nécessaire."
            elif "terrasse" in project_description:
                confirmation_message = f"Pour réaliser une terrasse en pavés, vous aurez besoin de : {materiaux_text.lower()}. Voici les produits que nous pouvons vous proposer pour chaque matériau nécessaire."
            elif "chape" in project_description or "dalle" in project_description:
                confirmation_message = f"Pour réaliser une chape en béton, vous aurez besoin de : {materiaux_text.lower()}. Voici les produits que nous pouvons vous proposer pour chaque matériau nécessaire."
            elif "isolation" in project_description:
                confirmation_message = f"Pour isoler un mur, vous aurez besoin de : {materiaux_text.lower()}. Voici les produits que nous pouvons vous proposer pour chaque matériau nécessaire."
            else:
                confirmation_message = f"Pour réaliser ce projet ({reformulation}), vous aurez besoin de : {materiaux_text.lower()}. Voici les produits que nous pouvons vous proposer pour chaque matériau nécessaire."
        else:
            confirmation_message = self._generate_confirmation_message(query, reformulation, len(results))
        
        # Préparer la liste des matériaux pour le frontend
        materiaux_list_for_frontend = None
        if is_project:
            project_type = detect_project_type(query)
            if project_type:
                materiaux_base = get_project_materials(project_type)
                materiaux_list_for_frontend = [
                    {"type": m.get("type", ""), "description": m.get("description", "")}
                    for m in materiaux_base
                ]
            elif materiaux:
                materiaux_list_for_frontend = [
                    {"type": m.get("type", ""), "description": m.get("description", "")}
                    for m in materiaux[:4]
                ]
        
        result = {
            "results": results,
            "query_understood": reformulation,
            "total_results": len(results),
            "search_time": round(search_time, 3),
            "confirmation_message": confirmation_message,
            "is_project": is_project,
            "materiaux": results_by_material if (is_project and results_by_material) else None,
            "materiaux_list": materiaux_list_for_frontend,
            "knowledge_context": understood_query.get("knowledge_context"),
        }
        
        # Mettre en cache (limiter à 500 entrées)
        if len(self._search_cache) >= 500:
            # Supprimer les plus anciennes (FIFO simple)
            oldest_key = next(iter(self._search_cache))
            del self._search_cache[oldest_key]
        
        self._search_cache[query_hash] = result
        
        return result
