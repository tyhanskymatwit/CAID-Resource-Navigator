from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os

from caid_resources import CAIDresource, looks_like_potential_info


DEFAULT_DB_PATH = os.getenv("CAID_DB_PATH", "CAID Resources Database.xlsx")
DEFAULT_SHEET = os.getenv("CAID_SHEET", "Resources")

app = FastAPI(
    title = "CAID Resource Navigator API",
    version = "1.0.0",
    description = "Community Resource Recommendations powered by up-to-date telehealth resources"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

agent: Optional[CAIDresource] = None

@app.on_event("startup")
def startup():
    global agent
    try:
        agent = CAIDresource(DEFAULT_DB_PATH, sheet_name=DEFAULT_SHEET)
    except Exception as e:
        raise RuntimeError(f"Failed to load CAID Database: {e}")
    
    
class ResourceRequest(BaseModel):
    service_types: List[str] = Field(..., description = "Requested needs/categories (e.g., ['Food', 'Housing']).")
    location: Optional[str] = Field(None, description = "Town nme or ZIP code only (No personal addresses please!).")
    demographics: Optional[List[str]] = Field(default_factory = list, description = "Eligibility keywords (e.g., ['Senior', 'Veteran']).")
    top_k: int = Field(10, ge=1, le=50, description = "Number of results to return (1-50).")
    
class ResourceInfo(BaseModel):
    Name: str = ""
    Organization: str = ""
    Address: str = ""
    Monday: str = ""
    Tuesday: str = ""
    Wednesday: str = ""
    Thursday: str = ""
    Friday: str = ""
    Saturday: str = ""
    Sunday: str = ""
    Service_Type: str = ""
    Patient_Requirements: str = ""
    Description: str = ""
    Score: float = 0.0
    
class RecommendedResource(BaseModel):
    count: int
    results: List[ResourceInfo]
    
def safety_check(req: ResourceRequest):
    if req.location and looks_like_potential_info(req.location):
        raise HTTPException(
            status_code = 400,
            detail = "Location must be a town name or ZIP code only. Do NOT enter phone numbers or emails to preserve confidentiality."
        )
        
    joined = " ".join(req.service_types + (req.demographics or []))
    if looks_like_potential_info(joined):
        raise HTTPException(
            status_code = 400,
            detail = "Please remove information that can be used to identify you. Use general categories ONLY!"
        )
        
def row_to_item(row) -> ResourceInfo:
    def get(col:str) -> str:
        return "" if col not in row or row[col] is None else str(row[col])
    
    return ResourceInfo(
        Name = get("Name"),
        Organization = get("Organization"),
        Address = get("Address"),
        Monday = get("Monday"),
        Tuesday = get("Tuesday"),
        Wednesday = get("Wednesday"),
        Thursday = get("Thursday"),
        Friday = get("Friday"),
        Saturday = get("Saturday"),
        Sunday = get("Sunday"),
        Service_Type = get("Service Type"),
        Patient_Requirements = get("Patient Requirements"),
        Description  = get("Description"),
        Score = float(row.get("_Score", 0.0)) if hasattr(row, "get") else 0.0
    )
    
@app.post("/recommend", response_model = RecommendedResource)
def recommend(req: ResourceRequest):
    if agent is None:
        raise HTTPException(
            status_code = 503,
            detail = "Resource Engine has not intialized"
        )
        
    safety_check(req)
    
    try:
        df = agent.search_resources(
            service_types = req.service_types,
            location = req.location,
            demographics = req.demographics,
            top_k = req.top_k,
        )
        
        results = []
        for _, r in df.iterrows():
            results.append(row_to_item(r))
            
        return RecommendedResource(count = len(results), results = results)
    
    except ValueError as VE:
        raise HTTPException (
           status_code = 400,
           detail = str(VE) 
        )
    except Exception as e:
        raise HTTPException (
            status_code = 500,
            detail = f"Server error: {e}"
        )