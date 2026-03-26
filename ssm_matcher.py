"""
SSM Matching System using Machine Learning
Uses similarity-based recommendation to match patient SSM scores to appropriate resources
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


class SSMatcher:
    """
    ML-based Self-Sufficiency Matrix matching system
    
    Uses collaborative filtering and similarity scoring to recommend resources
    based on patient SSM scores and resource intensity levels.
    
    Algorithm:
    1. Vector representation of patient needs (18 SSM categories)
    2. Vector representation of resource capabilities (service-type intensities)
    3. Cosine similarity between patient vector and resource vectors
    4. Score boosting for resources slightly more intensive than patient need
    """
    
    # SSM Category to Service Type mapping
    CATEGORY_TO_SERVICE = {
        'Income': ['Money', 'Work'],
        'Employment': ['Work', 'Education'],
        'Housing': ['Housing'],
        'Food': ['Food'],
        'Childcare': ['Childrens'],
        "Children's Education": ['Childrens', 'Education'],
        'Adult Education': ['Education'],
        'Legal': ['Legal'],
        'Health Care': ['Healthcare', 'Health'],
        'Life Skills': ['Education'],
        'Mental Health': ['Health', 'Healthcare'],
        'Substance Abuse': ['Health', 'Healthcare'],
        'Mobility': ['Transit'],
        'Family Relations': ['Education', 'Childrens'],
        'Community Involvement': ['Education'],
        'Safety': ['Housing', 'Health'],
        'Parenting Skills': ['Childrens', 'Education'],
        'Credit History': ['Money', 'Education']
    }
    
    # Service types in database
    SERVICE_TYPES = [
        'Food', 'Housing', 'Healthcare', 'Health', 'Transit', 
        'Money', 'Work', 'Education', 'Childrens', 'Clothes', 'Language'
    ]
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize SSM matcher with resource database
        
        Args:
            df: DataFrame with columns: Name, Service Type, SSM Rating, etc.
        """
        self.df = df.copy()
        self.scaler = StandardScaler()
        
        # Parse SSM ratings for each resource
        self._parse_ssm_ratings()
        
        # Build resource feature vectors
        self._build_resource_vectors()
    
    def _parse_ssm_ratings(self):
        """Parse SSM Rating column into structured format"""
        self.df['SSM_Parsed'] = None
        
        for idx, row in self.df.iterrows():
            service_types = str(row.get('Service Type', '')).split(',')
            service_types = [s.strip() for s in service_types if s.strip()]
            
            ssm_ratings = str(row.get('SSM Rating', ''))
            if ssm_ratings and ssm_ratings != 'nan':
                ratings = ssm_ratings.split(',')
                ratings = [int(r.strip()) for r in ratings if r.strip().isdigit()]
                
                # Map ratings to service types
                ssm_dict = {}
                for service, rating in zip(service_types, ratings):
                    ssm_dict[service] = rating
                
                self.df.at[idx, 'SSM_Parsed'] = ssm_dict
            else:
                self.df.at[idx, 'SSM_Parsed'] = {}
    
    def _build_resource_vectors(self):
        """
        Build feature vectors for each resource
        Each vector represents intensity levels for each service type
        """
        self.df['Feature_Vector'] = None
        
        for idx, row in self.df.iterrows():
            ssm_dict = row['SSM_Parsed']
            
            # Create vector: [Food_intensity, Housing_intensity, ...]
            vector = []
            for service_type in self.SERVICE_TYPES:
                intensity = ssm_dict.get(service_type, 0)  # 0 if not offered
                vector.append(intensity)
            
            self.df.at[idx, 'Feature_Vector'] = np.array(vector)
    
    def build_patient_vector(self, ssm_scores: Dict[str, int]) -> np.ndarray:
        """
        Build patient need vector from SSM scores
        
        Args:
            ssm_scores: Dict of category -> score (1-6)
                       e.g., {'Food': 2, 'Housing': 1, 'Income': 3}
        
        Returns:
            Vector representing patient needs across all service types
        """
        # Initialize vector
        patient_vector = np.zeros(len(self.SERVICE_TYPES))
        
        # For each SSM category patient scored
        for category, score in ssm_scores.items():
            # Get corresponding service types
            service_types = self.CATEGORY_TO_SERVICE.get(category, [])
            
            # Map score to need intensity
            # Score 1-3 = high need (higher is better)
            # Score 4-6 = low/no need (lower score)
            if score <= 3:
                need_intensity = 4 - score  # 1->3, 2->2, 3->1
            else:
                need_intensity = 0  # No need
            
            # Add to vector for each relevant service type
            for service_type in service_types:
                if service_type in self.SERVICE_TYPES:
                    idx = self.SERVICE_TYPES.index(service_type)
                    patient_vector[idx] = max(patient_vector[idx], need_intensity)
        
        return patient_vector
    
    def calculate_ssm_match_score(
        self, 
        patient_vector: np.ndarray, 
        resource_vector: np.ndarray
    ) -> float:
        """
        Calculate how well a resource matches patient needs using ML similarity
        
        Uses:
        1. Cosine similarity (direction match)
        2. Intensity matching (prefer resources slightly more intensive than need)
        3. Penalty for under-serving high-need areas
        
        Args:
            patient_vector: Patient need vector
            resource_vector: Resource capability vector
        
        Returns:
            Match score (0-100)
        """
        # Handle zero vectors
        if np.all(patient_vector == 0) or np.all(resource_vector == 0):
            return 0.0
        
        # 1. Cosine similarity (direction/category match)
        cosine_sim = cosine_similarity(
            patient_vector.reshape(1, -1), 
            resource_vector.reshape(1, -1)
        )[0][0]
        cosine_score = (cosine_sim + 1) / 2 * 40  # 0-40 points
        
        # 2. Intensity matching
        # Prefer resources at same or slightly higher intensity
        intensity_scores = []
        for patient_need, resource_intensity in zip(patient_vector, resource_vector):
            if patient_need > 0:  # Patient has need in this area
                if resource_intensity == 0:
                    # Resource doesn't serve this need at all
                    intensity_scores.append(0)
                elif resource_intensity < patient_need:
                    # Resource under-serves (penalty)
                    score = 20 * (resource_intensity / patient_need)
                    intensity_scores.append(score)
                elif resource_intensity == patient_need:
                    # Perfect match
                    intensity_scores.append(30)
                elif resource_intensity <= patient_need + 1:
                    # Slightly more intensive (good!)
                    intensity_scores.append(25)
                else:
                    # Much more intensive than needed
                    score = 15 - (resource_intensity - patient_need) * 2
                    intensity_scores.append(max(score, 5))
        
        # Average intensity score
        if intensity_scores:
            intensity_score = np.mean(intensity_scores)
        else:
            intensity_score = 0
        
        # 3. Coverage bonus (how many patient needs are addressed)
        coverage = np.sum((patient_vector > 0) & (resource_vector > 0))
        total_needs = np.sum(patient_vector > 0)
        if total_needs > 0:
            coverage_score = (coverage / total_needs) * 30  # 0-30 points
        else:
            coverage_score = 0
        
        # Total score
        total_score = cosine_score + intensity_score + coverage_score
        return min(total_score, 100.0)  # Cap at 100
    
    def match_resources(
        self,
        ssm_scores: Dict[str, int],
        location: Optional[str] = None,
        demographics: Optional[List[str]] = None,
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Match resources to patient using ML-based SSM scoring
        
        Args:
            ssm_scores: Dict of SSM category -> score
            location: Optional location for filtering
            demographics: Optional demographics for filtering
            top_k: Number of results to return
        
        Returns:
            DataFrame with matched resources and scores
        """
        # Build patient need vector
        patient_vector = self.build_patient_vector(ssm_scores)
        
        # Calculate match scores for all resources
        results = self.df.copy()
        results['SSM_Match_Score'] = 0.0
        
        for idx, row in results.iterrows():
            resource_vector = row['Feature_Vector']
            match_score = self.calculate_ssm_match_score(
                patient_vector, 
                resource_vector
            )
            results.at[idx, 'SSM_Match_Score'] = match_score
        
        # Filter by location if provided
        if location:
            location_mask = results['Full Address (Formatted)'].str.contains(
                location, case=False, na=False
            )
            location_results = results[location_mask]
            if len(location_results) > 0:
                results = location_results
        
        # Filter by demographics if provided
        if demographics:
            demo_scores = []
            for idx, row in results.iterrows():
                req_text = str(row.get('Patient Requirements', '')).lower()
                score = 0
                for demo in demographics:
                    if demo.lower() in req_text:
                        score += 1
                if 'all ages' in req_text or 'all' in req_text:
                    score += 0.5
                demo_scores.append(score)
            
            results['Demo_Score'] = demo_scores
        else:
            results['Demo_Score'] = 0
        
        # Combine scores
        # SSM match is primary, demographics is secondary
        results['Total_Score'] = (
            results['SSM_Match_Score'] * 2.0 + 
            results['Demo_Score'] * 10.0
        )
        
        # Sort and return top results
        results = results.sort_values('Total_Score', ascending=False)
        return results.head(top_k)
    
    def get_critical_needs(self, ssm_scores: Dict[str, int]) -> List[str]:
        """
        Identify critical needs (SSM score = 1)
        
        Args:
            ssm_scores: Dict of category -> score
        
        Returns:
            List of categories at crisis level
        """
        return [cat for cat, score in ssm_scores.items() if score == 1]
    
    def explain_match(
        self, 
        ssm_scores: Dict[str, int], 
        resource_row: pd.Series
    ) -> str:
        """
        Explain why a resource was matched to a patient
        
        Args:
            ssm_scores: Patient SSM scores
            resource_row: Resource DataFrame row
        
        Returns:
            Human-readable explanation
        """
        patient_vector = self.build_patient_vector(ssm_scores)
        resource_vector = resource_row['Feature_Vector']
        ssm_dict = resource_row['SSM_Parsed']
        
        explanation = []
        
        # Identify matched services
        matched_services = []
        for i, service_type in enumerate(self.SERVICE_TYPES):
            if patient_vector[i] > 0 and resource_vector[i] > 0:
                patient_need = patient_vector[i]
                resource_level = resource_vector[i]
                
                if resource_level >= patient_need:
                    matched_services.append(
                        f"{service_type} (need level {int(patient_need)}, "
                        f"resource level {int(resource_level)})"
                    )
        
        if matched_services:
            explanation.append("Matches your needs in:")
            explanation.extend(f"  - {s}" for s in matched_services)
        
        # Overall match quality
        score = self.calculate_ssm_match_score(patient_vector, resource_vector)
        if score >= 80:
            explanation.append("\nExcellent match - strongly recommended")
        elif score >= 60:
            explanation.append("\nGood match - recommended")
        elif score >= 40:
            explanation.append("\nModerate match - may be helpful")
        else:
            explanation.append("\nPartial match - consider as option")
        
        return "\n".join(explanation)


class SSMPredictor:
    """
    Predict missing SSM scores using collaborative filtering
    (Advanced feature - predict patient needs from limited data)
    """
    
    def __init__(self):
        # Correlation matrix between SSM categories
        # Based on typical co-occurrence patterns
        self.correlations = {
            'Income': ['Employment', 'Housing', 'Credit History'],
            'Employment': ['Income', 'Adult Education', 'Credit History'],
            'Housing': ['Income', 'Safety'],
            'Food': ['Income', 'Housing'],
            'Mental Health': ['Substance Abuse', 'Life Skills'],
            'Substance Abuse': ['Mental Health', 'Employment'],
            'Parenting Skills': ["Children's Education", 'Childcare'],
        }
    
    def predict_related_needs(
        self, 
        known_scores: Dict[str, int]
    ) -> Dict[str, float]:
        """
        Predict likely scores for related categories
        
        Args:
            known_scores: Known SSM scores
        
        Returns:
            Dict of predicted scores (0-1 probability of need)
        """
        predictions = {}
        
        for category, score in known_scores.items():
            if score <= 2:  # High need
                # Check correlated categories
                related = self.correlations.get(category, [])
                for related_cat in related:
                    if related_cat not in known_scores:
                        # Predict moderate need
                        predictions[related_cat] = 0.6 + (3 - score) * 0.15
        
        return predictions


def demo_ssm_matcher():
    """Demo of SSM matching system"""
    import sys
    
    # Load database
    try:
        df = pd.read_excel('CAID Resources Database.xlsx', sheet_name='Resources')
    except:
        print("Error: Could not load database")
        print("Make sure CAID Resources Database.xlsx is in the current directory")
        return
    
    # Initialize matcher
    matcher = SSMatcher(df)
    
    print("SSM-Based ML Resource Matching Demo")
    print("=" * 80)
    
    # Example patient scores
    patient_scores = {
        'Food': 2,        # Vulnerable
        'Housing': 1,     # Crisis
        'Health Care': 3, # Safe
    }
    
    print("\nPatient SSM Scores:")
    for category, score in patient_scores.items():
        level = ['', 'CRISIS', 'VULNERABLE', 'SAFE', 'STABLE', 
                'BUILDING', 'EMPOWERED'][score]
        print(f"  {category}: {score} ({level})")
    
    # Identify critical needs
    critical = matcher.get_critical_needs(patient_scores)
    if critical:
        print(f"\nCritical Needs: {', '.join(critical)}")
    
    # Match resources
    print("\n" + "=" * 80)
    print("Top Matched Resources:")
    print("=" * 80)
    
    results = matcher.match_resources(
        ssm_scores=patient_scores,
        location='Hyannis',
        top_k=5
    )
    
    for i, (_, row) in enumerate(results.iterrows(), 1):
        print(f"\n{i}. {row['Name']}")
        print(f"   Organization: {row['Organization']}")
        print(f"   Services: {row['Service Type']}")
        print(f"   Match Score: {row['SSM_Match_Score']:.1f}/100")
        
        # Show explanation
        explanation = matcher.explain_match(patient_scores, row)
        print(f"   {explanation}")


if __name__ == "__main__":
    demo_ssm_matcher()
