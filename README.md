# CAID-Resource-Navigator
The culmination of 7 combined years of education, creating a database of community resources and services that users can find based on their SSM score, all prompted and sorted using AI and ML algorithms.



# COULD run in one single deployment given a container
# Workflow -->
#                   Browser (5173)
#                   fetch ("/recommend")
#                   FastAPI server (8000)
#                   Database search + score
#                   JSON results

# Deployment in medical setting --> One address "resources.outercapehealth.org"
#                                   Reverse proxy (Nginx) --> Serves asfrontend at "/" and routes API at "/api" --> Real Database instead of excel file
#                                   Postgres for DB?
# Deployment flow? --> User - Nginx - FastAPI - DB - security measures

REQUIRES 2 SEPARATE TERMINALS: both bash

Terminal 1:

# To get the Back-End application process up and running
# Boots fastAPI engine (/recommend endpoint)
# 8000 - set as backend port
uvicorn caid_api_server:app --reload



Terminal 2:

# To get the static Front-End UI runnning
# Differentiating ports so the OS can distinguish services (Could be 5000 port if transitioning to Flask / 3000 if React/Node)
# 5173 - set as frontend port
python -m http.server 5173


# 127.0.0.1:5173/caid_portal.html