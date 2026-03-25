import pandas as pd
import json
from caid_resources import CAIDresource
from datetime import datetime
import os


class CAIDChatbot:
    def __init__(self, database_path, google_maps_api_key=None):
        """
        Initialize the chatbot with the resources database
        
        Args:
            database_path: Path to Excel database
            google_maps_api_key: Optional Google Maps API key for proximity features
        """
        # Get API key from parameter or environment variable
        api_key = google_maps_api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        
        self.agent = CAIDresource(database_path, google_maps_api_key=api_key)
        self.conversation_log = []
        self.patient_data = {}
        
        # Check if proximity features are available
        self.proximity_enabled = self.agent.location_service.gmaps is not None
        
    def log_interaction(self, user_input, bot_response):
        """Log all interactions for record keeping"""
        self.conversation_log.append({
            'timestamp': datetime.now().isoformat(),
            'user': user_input,
            'bot': bot_response
        })
    
    def start_conversation(self):
        """Begin the conversation flow"""
        print("\n" + "="*80)
        print("CAID COMMUNITY RESOURCES NAVIGATOR")
        print("="*80)
        
        if self.proximity_enabled:
            print(" Proximity-based search ENABLED")
        else:
            print(" Proximity features disabled (no Google Maps API key)")
            print("  Set GOOGLE_MAPS_API_KEY environment variable to enable")
        
        print("="*80 + "\n")
        
        greeting = """Greetings! Please provide a number from 1-6 describing which of the following 
services you struggle with (1 being crisis, 6 being fully functional):

1. Income
2. Employment
3. Housing
4. Food
5. Childcare
6. Children's Education
7. Adult Education
8. Legal
9. Health Care
10. Life Skills
11. Mental Health
12. Substance Abuse
13. Mobility
14. Family Relations
15. Community Involvement
16. Safety
17. Parenting Skills
18. Credit History

Please enter the category NUMBER followed by your rating (1-6).
Example: "4 2" means Food is at crisis level 2
You can enter multiple lines. Type 'done' when finished.
"""
        print(greeting)
        return greeting
    
    def collect_ssm_scores(self):
        """Self-Sufficiency Matrix scores"""
        categories = {
            1: "Income", 2: "Employment", 3: "Housing", 4: "Food",
            5: "Childcare", 6: "Children's Education", 7: "Adult Education",
            8: "Legal", 9: "Health Care", 10: "Life Skills", 11: "Mental Health",
            12: "Substance Abuse", 13: "Mobility", 14: "Family Relations",
            15: "Community Involvement", 16: "Safety", 17: "Parenting Skills",
            18: "Credit History"
        }
        
        scores = {}
        
        while True:
            user_input = input("\n> ").strip()
            
            if user_input.lower() == 'done':
                break
            
            try:
                parts = user_input.split()
                category_num = int(parts[0])
                score = int(parts[1])
                
                if category_num not in categories:
                    print(" Invalid category number. Please use 1-18.")
                    continue
                
                if score < 1 or score > 6:
                    print(" Score must be between 1 (crisis) and 6 (fully functional).")
                    continue
                
                category_name = categories[category_num]
                scores[category_name] = score
                print(f" Recorded: {category_name} = {score}")
                
            except (ValueError, IndexError):
                print(" Invalid format. Use: [category number] [score]")
                print("   Example: 4 2")
        
        self.patient_data['ssm_scores'] = scores
        return scores
    
    def collect_location_and_demographics(self):
        """Collect location and demographic information"""
        print("\n" + "-"*80)
        location_prompt = """Thanks! Please provide your town location and let us know if you apply 
to any of the following (type the numbers that apply, separated by commas):

1. Senior
2. Veteran
3. Active service
4. Homeless
5. Near homeless
6. Low-income
7. Single mother of child less than 6 months
8. Single father of child less than 6 months
9. Cancer patient
10. Less than 18 years old

Example: "Hyannis, 1, 6" for Senior and Low-income in Hyannis
"""
        
        if self.proximity_enabled:
            print(" Tip: Be specific with your location for best proximity results!")
            print("   You can enter: city name, zip code, or even 'current location'")
        
        print(location_prompt)
        
        user_input = input("\n> ").strip()
        
        # Parse location and demographics
        parts = user_input.split(',')
        location = parts[0].strip() if parts else ""
        
        demographics_map = {
            '1': 'senior', '2': 'veteran', '3': 'active service',
            '4': 'homeless', '5': 'near homeless', '6': 'low-income',
            '7': 'single mother of child <= 6 months',
            '8': 'single father of child <= 6 months',
            '9': 'cancer patient', '10': 'less than 18 years old'
        }
        
        demographics = []
        if len(parts) > 1:
            demo_nums = [p.strip() for p in parts[1:]]
            demographics = [demographics_map.get(num, '') for num in demo_nums if num in demographics_map]
        
        self.patient_data['location'] = location
        self.patient_data['demographics'] = demographics
        
        print(f"\n Location: {location}")
        print(f" Demographics: {', '.join(demographics) if demographics else 'None specified'}")
        
        return location, demographics
    
    def map_ssm_to_services(self, scores):
        """Map SSM categories to service types in database"""
        service_mapping = {
            'Income': ['Money', 'Work'],
            'Employment': ['Work', 'Education'],
            'Housing': ['Housing'],
            'Food': ['Food'],
            'Childcare': ['Childrens'],
            "Children's Education": ['Childrens', 'Education'],
            'Adult Education': ['Education'],
            'Health Care': ['Healthcare', 'Health'],
            'Mobility': ['Transit'],
            'Parenting Skills': ['Childrens', 'Education']
        }
        
        # Identify needs (scores 1-3 are crisis/vulnerable/safe)
        needed_services = set()
        critical_needs = []
        
        for category, score in scores.items():
            if score <= 3:  # Crisis, Vulnerable, or Safe (needs support)
                if score == 1:
                    critical_needs.append(category)
                
                if category in service_mapping:
                    for service in service_mapping[category]:
                        needed_services.add(service)
        
        return list(needed_services), critical_needs
    
    def format_resource(self, row):
        """Format a resource for display with distance information"""
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        hours_list = []
        for day in days:
            hours = row.get(day, '')
            if pd.notna(hours) and str(hours).strip():
                hours_list.append(f"    {day}: {hours}")
            else:
                hours_list.append(f"    {day}: Closed")
        
        hours_text = "\n".join(hours_list)
        
        # Add distance if available
        distance_info = ""
        if pd.notna(row.get('_Distance_Miles')):
            distance = row['_Distance_Miles']
            if distance < 1:
                distance_info = f"\n Distance: {distance:.2f} miles ({int(distance * 5280)} feet)"
            else:
                distance_info = f"\n Distance: {distance:.1f} miles"
        
        output = f"""
{'='*80}
{row['Name']}
{'='*80}
Organization: {row['Organization']}
Address: {row['Address']}{distance_info}

Hours:
{hours_text}

Services: {row['Service Type']}
Eligibility: {row.get('Patient Requirements', 'Not specified')}

Description:
{row['Description']}
"""
        return output
    
    def present_results(self, results, critical_needs):
        """Present search results to the patient."""
        print("\n" + "="*80)
        print("RECOMMENDED RESOURCES FOR YOU")
        print("="*80 + "\n")
        
        if critical_needs:
            print(f"  CRITICAL NEEDS IDENTIFIED: {', '.join(critical_needs)}")
            print("    We recommend contacting these resources as soon as possible.\n")
        
        if len(results) == 0:
            print(" Unfortunately, we couldn't find any exact matches in our database.")
            print("   Please call 211 for additional community resource referrals.")
            print("   Or visit: https://www.findhelp.org\n")
            return
        
        # Show distance summary if available
        if '_Distance_Miles' in results.columns:
            distances = results['_Distance_Miles'].dropna()
            if len(distances) > 0:
                print(f" DISTANCE SUMMARY:")
                print(f"   Closest resource: {distances.min():.1f} miles")
                print(f"   Farthest shown: {distances.max():.1f} miles")
                print(f"   Average distance: {distances.mean():.1f} miles")
                print()
        
        print(f"Found {len(results)} resource(s), showing top results sorted by relevance:\n")
        
        # Show top 5 results (or all if less than 5)
        top_results = results.head(5)
        
        for idx, (_, row) in enumerate(top_results.iterrows(), 1):
            print(f"\n{'*'*80}")
            print(f"RESOURCE #{idx}")
            print(self.format_resource(row))
        
        if len(results) > 5:
            print(f"\n{'='*80}")
            print(f"... and {len(results) - 5} more resources available.")
            print("\nAll results have been sorted by:")
            if self.proximity_enabled and '_Distance_Miles' in results.columns:
                print("  1. Distance from your location (closest first)")
            print("  2. Demographic match")
            print("  3. Service relevance")
    
    def run(self):
        """Run the complete chatbot conversation"""
        # Step 1: Greeting and SSM scores
        greeting = self.start_conversation()
        scores = self.collect_ssm_scores()
        
        if not scores:
            print("\n No scores provided. Exiting.")
            return
        
        # Step 2: Location and demographics
        location, demographics = self.collect_location_and_demographics()
        
        # Step 3: Map scores to services
        needed_services, critical_needs = self.map_ssm_to_services(scores)
        
        print(f"\n Searching for services based on SSM scores")
        if self.proximity_enabled and location:
            print(f" Calculating distances from: {location}")
        print()
        
        # Step 4: Search database with SSM-based matching
        try:
            results = self.agent.search_resources_with_ssm(
                ssm_scores=scores,
                location=location,
                demographics=demographics,
                use_proximity=True,
                top_k=10
            )
        except Exception as e:
            print(f"SSM matching failed, using standard search: {e}")
            # Fallback to standard search
            needed_services, _ = self.map_ssm_to_services(scores)
            results = self.agent.search_resources(
                service_types=needed_services,
                location=location,
                demographics=demographics,
                use_proximity=True
            )
        
        # Step 5: Present results
        self.present_results(results, critical_needs)
        
        print("\n" + "="*80)
        print("Thank you for using CAID Community Resources Navigator!")
        print("For additional help, call 211 or visit www.findhelp.org")
        print("="*80 + "\n")


# Main execution
if __name__ == "__main__":
    # Initialize chatbot with database
    # API key can be set via environment variable: GOOGLE_MAPS_API_KEY
    # Or pass directly: chatbot = CAIDChatbot('database.xlsx', google_maps_api_key='YOUR_KEY')
    
    chatbot = CAIDChatbot('CAID Resources Database.xlsx')
    
    # Run the conversation
    chatbot.run()
