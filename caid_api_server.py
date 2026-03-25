from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os

from caid_resources import CAIDresource, looks_like_potential_info


DEFAULT_DB_PATH = os.getenv("CAID_DB_PATH", "CAID Resources Database.xlsx")
DEFAULT_SHEET = os.getenv("CAID_SHEET", "Resources")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

app = FastAPI(
    title="CAID Resource Navigator API",
    version="2.0.0",
    description="Community Resource Recommendations with proximity-based ranking powered by Google Maps"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent: Optional[CAIDresource] = None


@app.on_event("startup")
def startup():
    global agent
    try:
        agent = CAIDresource(
            DEFAULT_DB_PATH, 
            sheet_name=DEFAULT_SHEET,
            google_maps_api_key=GOOGLE_MAPS_API_KEY
        )
        print(f" CAID Database loaded: {DEFAULT_DB_PATH}")
        
        if agent.location_service.gmaps:
            print(" Google Maps proximity features ENABLED")
        else:
            print(" Google Maps API key not configured - proximity features DISABLED")
            print("  Set GOOGLE_MAPS_API_KEY environment variable to enable")
            
    except Exception as e:
        raise RuntimeError(f"Failed to load CAID Database: {e}")


class ResourceRequest(BaseModel):
    service_types: List[str] = Field(
        ..., 
        description="Requested needs/categories (e.g., ['Food', 'Housing'])."
    )
    location: Optional[str] = Field(
        None, 
        description="Town name, ZIP code, or address (No personal phone/email!)."
    )
    demographics: Optional[List[str]] = Field(
        default_factory=list, 
        description="Eligibility keywords (e.g., ['Senior', 'Veteran'])."
    )
    top_k: int = Field(
        10, 
        ge=1, 
        le=50, 
        description="Number of results to return (1-50)."
    )
    use_proximity: bool = Field(
        True,
        description="Use distance-based ranking (requires location and Google Maps API)"
    )


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
    Distance_Miles: Optional[float] = None
    Proximity_Score: Optional[float] = None


class RecommendedResource(BaseModel):
    count: int
    results: List[ResourceInfo]
    distance_summary: Optional[str] = None
    proximity_enabled: bool = False


class DistanceSummary(BaseModel):
    closest_miles: Optional[float] = None
    farthest_miles: Optional[float] = None
    average_miles: Optional[float] = None
    resources_with_distance: int = 0


def safety_check(req: ResourceRequest):
    """Check for PII in request"""
    if req.location and looks_like_potential_info(req.location):
        raise HTTPException(
            status_code=400,
            detail="Location must be a town name or ZIP code only. Do NOT enter phone numbers or emails to preserve confidentiality."
        )
    
    joined = " ".join(req.service_types + (req.demographics or []))
    if looks_like_potential_info(joined):
        raise HTTPException(
            status_code=400,
            detail="Please remove information that can be used to identify you. Use general categories ONLY!"
        )


def row_to_item(row) -> ResourceInfo:
    """Convert DataFrame row to ResourceInfo"""
    def get(col: str) -> str:
        return "" if col not in row or row[col] is None else str(row[col])
    
    # Get distance if available
    distance_miles = None
    proximity_score = None
    
    if "_Distance_Miles" in row and row["_Distance_Miles"] is not None:
        try:
            import pandas as pd
            if not pd.isna(row["_Distance_Miles"]):
                distance_miles = float(row["_Distance_Miles"])
        except:
            pass
    
    if "_Proximity_Score" in row and row["_Proximity_Score"] is not None:
        try:
            import pandas as pd
            if not pd.isna(row["_Proximity_Score"]):
                proximity_score = float(row["_Proximity_Score"])
        except:
            pass
    
    return ResourceInfo(
        Name=get("Name"),
        Organization=get("Organization"),
        Address=get("Address"),
        Monday=get("Monday"),
        Tuesday=get("Tuesday"),
        Wednesday=get("Wednesday"),
        Thursday=get("Thursday"),
        Friday=get("Friday"),
        Saturday=get("Saturday"),
        Sunday=get("Sunday"),
        Service_Type=get("Service Type"),
        Patient_Requirements=get("Patient Requirements"),
        Description=get("Description"),
        Score=float(row.get("_Score", 0.0)) if hasattr(row, "get") else 0.0,
        Distance_Miles=distance_miles,
        Proximity_Score=proximity_score
    )


@app.get("/")
def root():
    """API status and information"""
    return {
        "service": "CAID Resource Navigator API",
        "version": "2.0.0",
        "status": "online",
        "features": {
            "proximity_ranking": agent.location_service.gmaps is not None if agent else False,
            "google_maps_enabled": GOOGLE_MAPS_API_KEY is not None
        },
        "endpoints": {
            "POST /recommend": "Get resource recommendations",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
def health():
    """Health check endpoint"""
    if agent is None:
        raise HTTPException(status_code=503, detail="Resource engine not initialized")
    
    return {
        "status": "healthy",
        "database_loaded": True,
        "proximity_enabled": agent.location_service.gmaps is not None
    }


@app.post("/recommend", response_model=RecommendedResource)
def recommend(req: ResourceRequest):
    """
    Get resource recommendations based on patient needs
    
    The API will rank resources by:
    1. Distance from patient location (if proximity enabled)
    2. Demographic match
    3. Service type relevance
    4. Description match
    """
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="Resource Engine has not initialized"
        )
    
    safety_check(req)
    
    try:
        # Search with proximity ranking
        df = agent.search_resources(
            service_types=req.service_types,
            location=req.location,
            demographics=req.demographics,
            top_k=req.top_k,
            use_proximity=req.use_proximity
        )
        
        # Convert to response format
        results = []
        for _, r in df.iterrows():
            results.append(row_to_item(r))
        
        # Generate distance summary
        distance_summary = None
        if req.use_proximity and req.location and len(results) > 0:
            distances = [r.Distance_Miles for r in results if r.Distance_Miles is not None]
            if distances:
                distance_summary = f"Distance range: {min(distances):.1f} - {max(distances):.1f} miles. Average: {sum(distances)/len(distances):.1f} miles."
        
        return RecommendedResource(
            count=len(results),
            results=results,
            distance_summary=distance_summary,
            proximity_enabled=req.use_proximity and agent.location_service.gmaps is not None
        )
    
    except ValueError as ve:
        raise HTTPException(
            status_code=400,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Server error: {e}"
        )


@app.get("/distance_summary")
def get_distance_summary(
    service_types: List[str],
    location: str,
    demographics: Optional[List[str]] = None,
    top_k: int = 25
) -> DistanceSummary:
    """
    Get just the distance summary without full results
    Useful for UI to show "X resources within Y miles"
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Resource engine not initialized")
    
    if not agent.location_service.gmaps:
        raise HTTPException(
            status_code=400,
            detail="Proximity features not enabled. Set GOOGLE_MAPS_API_KEY."
        )
    
    try:
        df = agent.search_resources(
            service_types=service_types,
            location=location,
            demographics=demographics or [],
            top_k=top_k,
            use_proximity=True
        )
        
        distances = df["_Distance_Miles"].dropna()
        
        if len(distances) == 0:
            return DistanceSummary(resources_with_distance=0)
        
        return DistanceSummary(
            closest_miles=float(distances.min()),
            farthest_miles=float(distances.max()),
            average_miles=float(distances.mean()),
            resources_with_distance=len(distances)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
