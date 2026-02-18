import re
from typing import List, Optional
import pandas as pd

# Trying to ensure anything such as a phone number or email of a patient is discarded for confidentiality
potential_info_patterns = [
    r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
]

def looks_like_potential_info(text: str) -> bool:
    if not text:
        return False
    return any(re.search(p, text, flags=re.IGNORECASE) for p in potential_info_patterns)

class CAIDresource:
    def __init__(self, database_path:str, sheet_name: str = "Resources"):
        self.db = pd.read_excel(database_path, sheet_name = sheet_name)
        
        for columns in ["Service Type", "Address", "Patient Requirements", "Description"]:
            if columns in self.db.columns:
                self.db[columns] = self.db[columns].fillna("").astype(str)
            else:
                raise KeyError(f"Missing expected column within Database Excel Sheet: '{columns}'")
                
        self.db["_Service_Lc"] = self.db["Service Type"].str.lower()
        self.db["_Address_Lc"] = self.db["Address"].str.lower()
        self.db["_Req_Lc"] = self.db["Patient Requirements"].str.lower()
        self.db["_Description_Lc"] = self.db["Description"].str.lower()
                
    def search_resources(
        self,
        service_types: List[str],
        location: Optional[str] = None,
        demographics: Optional[List[str]] = None,
        top_k: int = 25,
    ) -> pd.DataFrame:
        
        if location and looks_like_potential_info(location):
            raise ValueError(" Location should be a town name or a ZIP Code only. Please do not use any phone numbers and or emails to stay confidential. ")
        
        results = self.db.copy()
        
        if service_types:
            service_types_lowercase = [s.lower() for s in service_types]
            mask = results["_Service_Lc"].apply(lambda svc: any(st in svc for st in service_types_lowercase))
            results = results[mask]
            
            
        results["_Location_Score"] = 0.0
        
        if location:
            loc = location.strip().lower()
            results["_Location_Score"] = results["_Address_Lc"].apply(lambda a: 2.0 if loc and loc in a else 0.0)
            
        results["_Demographic_Score"] = 0.0
        
        if demographics:
            demo = [d.strip().lower() for d in demographics if d.strip()]
            
            def Score_Req(req_text: str) -> float:
                score = 0.0
                for d in demo:
                    if d in req_text:
                        score += 1.0
                if "all ages" in req_text or "all" in req_text:
                    score += 0.5
                return score
            results["_Demographic_Score"] = results["_Req_Lc"].apply(Score_Req)
        
            
        results["_Description_Score"] = 0.0
        
        if service_types:
            service_types_lowercase = [s.lower() for s in service_types]
            results["_Description_Score"] = results["_Description_Lc"].apply(lambda d: 0.5 * sum(st in d for st in service_types_lowercase))
            
        results["_Score"] = results["_Location_Score"] + results["_Demographic_Score"] + results["_Description_Score"]
        
        if location:
            local = results[results["_Location_Score"] > 0]
            if len(local) > 0:
                results = local
                
        results = results.sort_values("_Score", ascending=False)
        
        return results.head(top_k)