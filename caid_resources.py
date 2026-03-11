import re
from typing import List, Optional, Tuple
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
import os

# Trying to ensure anything such as a phone number or email of a patient is discarded for confidentiality
potential_info_patterns = [
    r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
]

def looks_like_potential_info(text: str) -> bool:
    if not text:
        return False
    return any(re.search(p, text, flags=re.IGNORECASE) for p in potential_info_patterns)


class LocationService:
    """Handles geocoding and distance calculations"""
    
    def __init__(self, google_maps_api_key: Optional[str] = None):
        """
        Initialize location service with optional Google Maps API key
        
        Args:
            google_maps_api_key: Google Maps API key for geocoding
                                If None, will try to get from GOOGLE_MAPS_API_KEY env var
        """
        self.api_key = google_maps_api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        self.gmaps = None
        
        if self.api_key:
            try:
                import googlemaps
                self.gmaps = googlemaps.Client(key=self.api_key)
                print(" Google Maps geocoding enabled")
            except ImportError:
                print(" googlemaps package not installed. Install with: pip install googlemaps")
                print("  Proximity features will be disabled.")
            except Exception as e:
                print(f" Google Maps initialization failed: {e}")
        else:
            print(" No Google Maps API key provided. Proximity features disabled.")
            print("  Set GOOGLE_MAPS_API_KEY environment variable or pass api_key parameter.")
    
    def geocode(self, address: str) -> Optional[Tuple[float, float]]:
        """
        Geocode an address to latitude/longitude coordinates
        
        Args:
            address: Address string or location name
            
        Returns:
            Tuple of (latitude, longitude) or None if geocoding fails
        """
        if not self.gmaps:
            return None
        
        try:
            result = self.gmaps.geocode(address)
            if result:
                location = result[0]['geometry']['location']
                return (location['lat'], location['lng'])
        except Exception as e:
            print(f"Geocoding error for '{address}': {e}")
        
        return None
    
    @staticmethod
    def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate distance in miles between two geographic coordinates using Haversine formula
        
        This is the "as-the-crow-flies" distance, not driving distance.
        
        Args:
            lat1, lon1: Coordinates of first point
            lat2, lon2: Coordinates of second point
            
        Returns:
            Distance in miles
        """
        # Earth's radius in miles
        R = 3959.0
        
        # Convert to radians
        lat1_rad = radians(lat1)
        lon1_rad = radians(lon1)
        lat2_rad = radians(lat2)
        lon2_rad = radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = sin(dlat / 2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        
        distance = R * c
        return distance


class CAIDresource:
    def __init__(
        self, 
        database_path: str, 
        sheet_name: str = "Resources",
        google_maps_api_key: Optional[str] = None
    ):
        """
        Initialize CAID Resource finder
        
        Args:
            database_path: Path to Excel database
            sheet_name: Name of sheet in Excel file
            google_maps_api_key: Optional Google Maps API key for proximity features
        """
        self.db = pd.read_excel(database_path, sheet_name=sheet_name)
        
        # Check for expected columns
        for column in ["Service Type", "Patient Requirements", "Description"]:
            if column in self.db.columns:
                self.db[column] = self.db[column].fillna("").astype(str)
            else:
                raise KeyError(f"Missing expected column within Database Excel Sheet: '{column}'")
        
        # Handle address columns - support both old and new format
        if "Full Address (Formatted)" in self.db.columns:
            # New format with split addresses
            self.db["Address"] = self.db["Full Address (Formatted)"].fillna("").astype(str)
        elif "Address" in self.db.columns:
            # Old format
            self.db["Address"] = self.db["Address"].fillna("").astype(str)
        else:
            raise KeyError("Missing address column in database")
        
        # Create lowercase versions for searching
        self.db["_Service_Lc"] = self.db["Service Type"].str.lower()
        self.db["_Address_Lc"] = self.db["Address"].str.lower()
        self.db["_Req_Lc"] = self.db["Patient Requirements"].str.lower()
        self.db["_Description_Lc"] = self.db["Description"].str.lower()
        
        # Initialize location service
        self.location_service = LocationService(google_maps_api_key)
        
        # Cache for geocoded addresses
        self._geocode_cache = {}
    
    def _get_resource_coordinates(self, address: str) -> Optional[Tuple[float, float]]:
        """
        Get coordinates for a resource address with caching
        
        Args:
            address: Resource address
            
        Returns:
            Tuple of (lat, lng) or None
        """
        if address in self._geocode_cache:
            return self._geocode_cache[address]
        
        coords = self.location_service.geocode(address)
        self._geocode_cache[address] = coords
        return coords
    
    def search_resources(
        self,
        service_types: List[str],
        location: Optional[str] = None,
        demographics: Optional[List[str]] = None,
        top_k: int = 25,
        use_proximity: bool = True,
    ) -> pd.DataFrame:
        """
        Search for community resources with proximity-based ranking
        
        Args:
            service_types: List of service types to search for (e.g., ['Food', 'Housing'])
            location: Patient location (town name, zip code, or full address)
            demographics: List of demographic keywords (e.g., ['senior', 'veteran'])
            top_k: Maximum number of results to return
            use_proximity: Whether to use distance-based ranking (requires Google Maps API)
            
        Returns:
            DataFrame of matching resources sorted by relevance score
        """
        if location and looks_like_potential_info(location):
            raise ValueError(
                "Location should be a town name or a ZIP Code only. "
                "Please do not use any phone numbers and or emails to stay confidential."
            )
        
        results = self.db.copy()
        
        # Filter by service type
        if service_types:
            service_types_lowercase = [s.lower() for s in service_types]
            mask = results["_Service_Lc"].apply(
                lambda svc: any(st in svc for st in service_types_lowercase)
            )
            results = results[mask]
        
        # Initialize scoring columns
        results["_Location_Score"] = 0.0
        results["_Demographic_Score"] = 0.0
        results["_Description_Score"] = 0.0
        results["_Proximity_Score"] = 0.0
        results["_Distance_Miles"] = None
        
        # Proximity-based scoring
        patient_coords = None
        if location and use_proximity and self.location_service.gmaps:
            # Geocode patient location
            patient_coords = self.location_service.geocode(location)
            
            if patient_coords:
                print(f" Patient location geocoded: {location}")
                
                # Calculate distance to each resource
                distances = []
                for idx, row in results.iterrows():
                    resource_address = row["Address"]
                    
                    # Skip online/virtual resources
                    if "online" in resource_address.lower():
                        distances.append((idx, None))
                        continue
                    
                    resource_coords = self._get_resource_coordinates(resource_address)
                    
                    if resource_coords:
                        distance = self.location_service.haversine_distance(
                            patient_coords[0], patient_coords[1],
                            resource_coords[0], resource_coords[1]
                        )
                        distances.append((idx, distance))
                    else:
                        distances.append((idx, None))
                
                # Add distances to dataframe
                for idx, distance in distances:
                    if idx in results.index:
                        results.loc[idx, "_Distance_Miles"] = distance
                
                # Calculate proximity score (inverse of distance)
                # Closer resources get higher scores
                # Use logarithmic scale to avoid huge differences
                def proximity_score(distance):
                    if pd.isna(distance):
                        return 0.0
                    # Resources within 5 miles get max score of 10
                    # Score decreases logarithmically with distance
                    if distance < 0.5:  # Very close (< 0.5 miles)
                        return 10.0
                    elif distance < 5:  # Close (< 5 miles)
                        return 10.0 - (distance / 5) * 3  # Score 7-10
                    elif distance < 15:  # Moderate (< 15 miles)
                        return 7.0 - ((distance - 5) / 10) * 4  # Score 3-7
                    else:  # Far (> 15 miles)
                        return max(0.0, 3.0 - ((distance - 15) / 20) * 3)  # Score 0-3
                
                results["_Proximity_Score"] = results["_Distance_Miles"].apply(proximity_score)
                print(f" Calculated distances for {len([d for _, d in distances if d is not None])} resources")
            else:
                print(f" Could not geocode patient location: {location}")
        
        # Text-based location matching (fallback or supplement)
        if location:
            loc = location.strip().lower()
            results["_Location_Score"] = results["_Address_Lc"].apply(
                lambda a: 2.0 if loc and loc in a else 0.0
            )
        
        # Demographic matching
        if demographics:
            demo = [d.strip().lower() for d in demographics if d.strip()]
            
            def score_req(req_text: str) -> float:
                score = 0.0
                for d in demo:
                    if d in req_text:
                        score += 1.0
                if "all ages" in req_text or "all" in req_text:
                    score += 0.5
                return score
            
            results["_Demographic_Score"] = results["_Req_Lc"].apply(score_req)
        
        # Description matching
        if service_types:
            service_types_lowercase = [s.lower() for s in service_types]
            results["_Description_Score"] = results["_Description_Lc"].apply(
                lambda d: 0.5 * sum(st in d for st in service_types_lowercase)
            )
        
        # Calculate total score
        # Proximity score weighted heavily if available
        if patient_coords:
            results["_Score"] = (
                results["_Proximity_Score"] * 3.0 +  # Proximity is most important
                results["_Location_Score"] * 1.0 +   # Text matching bonus
                results["_Demographic_Score"] * 2.0 + # Demographics important
                results["_Description_Score"] * 0.5   # Description slight bonus
            )
        else:
            # Fallback to text-based scoring
            results["_Score"] = (
                results["_Location_Score"] * 2.0 +
                results["_Demographic_Score"] * 2.0 +
                results["_Description_Score"] * 0.5
            )
        
        # Filter to local results if we have location matches
        if location and not patient_coords:
            # Old behavior - only show exact text matches
            local = results[results["_Location_Score"] > 0]
            if len(local) > 0:
                results = local
        
        # Sort by total score
        results = results.sort_values("_Score", ascending=False)
        
        return results.head(top_k)
    
    def get_distance_summary(self, results: pd.DataFrame) -> str:
        """
        Get a summary of distances for the search results
        
        Args:
            results: Search results DataFrame
            
        Returns:
            Summary string
        """
        distances = results["_Distance_Miles"].dropna()
        
        if len(distances) == 0:
            return "Distance information not available"
        
        summary = f"Distance range: {distances.min():.1f} - {distances.max():.1f} miles\n"
        summary += f"Average distance: {distances.mean():.1f} miles\n"
        summary += f"Closest resource: {distances.min():.1f} miles away"
        
        return summary
