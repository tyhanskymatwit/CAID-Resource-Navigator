import pandas as pd
import json
from datetime import datetime

class CAIDChatbot:
    def __init__(self, database_path):
        """Initialize the chatbot with the resources database"""
        self.db = pd.read_excel(database_path, sheet_name='Resources')
        self.conversation_log = []
        self.patient_data = {}
        
    def log_interaction(self, user_input, bot_response):
        """Log all interactions for record keeping"""
        self.conversation_log.append({
            'timestamp': datetime.now().isoformat(),
            'user': user_input,
            'bot': bot_response
        })
    
    def start_conversation(self):
        """Begin the conversation flow"""
        print("CAID COMMUNITY RESOURCES NAVIGATOR")
        
        greeting = """1. Income
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

Please enter the category number followed by your rating (1-6).
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
                    print(f" Invalid category number. Please use 1-18.")
                    continue
                
                if score < 1 or score > 6:
                    print(f" Score must be between 1 (crisis) and 6 (fully functional).")
                    continue
                
                category_name = categories[category_num]
                scores[category_name] = score
                print(f"Recorded: {category_name} = {score}")
                
            except (ValueError, IndexError):
                print(" Invalid format. Use: [category number] [score]")
                print("   Example: 4 2")
        
        self.patient_data['ssm_scores'] = scores
        return scores
    
    def collect_location_and_demographics(self):
        """Collect location and demographic information"""
        location_prompt = """Please provide your town location and let us know if you apply 
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

Example: "Provincetown, 1,6" for Senior and Low-income in Provincetown
"""
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
    
    def search_resources(self, service_types, location=None, demographics=None):
        """Search the database for matching resources"""
        # Start with all resources
        results = self.db.copy()
        
        # Filter by service type
        if service_types:
            mask = results['Service Type'].apply(
                lambda x: any(st in str(x) for st in service_types) if pd.notna(x) else False
            )
            results = results[mask]
        
        # Filter by location if specified
        if location:
            location_mask = results['Address'].str.contains(location, case=False, na=False)
            location_results = results[location_mask]
            
            # If we found local results, use them; otherwise show all
            if len(location_results) > 0:
                results = location_results
            else:
                print(f"\n  No resources found in {location}. Showing resources from nearby areas.\n")
        
        # Filter by demographics if specified
        if demographics:
            # Create a scoring system for demographic matches
            def score_demographics(req_text):
                if pd.isna(req_text):
                    return 0
                req_lower = str(req_text).lower()
                score = 0
                for demo in demographics:
                    if demo.lower() in req_lower:
                        score += 1
                # Also give points for "all ages" or general eligibility
                if 'all ages' in req_lower or 'all' in req_lower:
                    score += 0.5
                return score
            
            results['demo_score'] = results['Patient Requirements'].apply(score_demographics)
            # Sort by demographic match score
            results = results.sort_values('demo_score', ascending=False)
        
        return results
    
    def format_resource(self, row):
        """Format a resource for display"""
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        hours_list = []
        for day in days:
            hours = row.get(day, '')
            if pd.notna(hours) and str(hours).strip():
                hours_list.append(f"    {day}: {hours}")
            else:
                hours_list.append(f"    {day}: Closed")
        
        hours_text = "\n".join(hours_list)
        
        output = f"""
{row['Name']}
Organization: {row['Organization']}
Address: {row['Address']}

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
        print("\n Next steps: \n-Add proximity location functionality \n-Properly incorporate SSM \n-Add more detailed resource information \n-Add decent user interface \n-Save patient contact data and remind them of upcoming appoitnents/group meetings \nResources for you:")
        
        if critical_needs:
            print(f"  Critical Needs: {', '.join(critical_needs)}")
            print("    We recommend contacting these resources as soon as possible.\n")
        
        if len(results) == 0:
            print(" Unfortunately, we couldn't find any exact matches in our database.")
            print("   Please call 211 for additional community resource referrals.")
            print("   Or visit: https://www.findhelp.org\n")
            return
        
        print(f"Found {len(results)} resource(s) for you:\n")
        
        # Show top 5 results (or all if less than 5)
        top_results = results.head(5)
        
        for idx, (_, row) in enumerate(top_results.iterrows(), 1):
            print(f"RESOURCE #{idx}")
            print(self.format_resource(row))
        
        if len(results) > 5:
            print(f"\n... and {len(results) - 5} more resources available.")
    
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
        
        print(f"\n Searching for services: {', '.join(needed_services)}")
        
        # Step 4: Search database
        results = self.search_resources(needed_services, location, demographics)
        
        # Step 5: Present results
        self.present_results(results, critical_needs)

# Main execution
if __name__ == "__main__":
    # Initialize chatbot with database
    chatbot = CAIDChatbot('CAID Resources Database.xlsx')
    
    # Run the conversation
    chatbot.run()