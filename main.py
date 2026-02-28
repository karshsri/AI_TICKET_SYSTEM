# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI(title="AI Ticket Classifier")

# Load trained model
model = joblib.load("ticket_model.pkl")

class TicketRequest(BaseModel):
    text: str

def generate_response(intent):
    if intent == "authentication":
        return {
            "category": "Authentication Issue",
            "response": "Please click on 'Forgot Password' on the login page and follow the reset instructions."
        }
    elif intent == "leave_query":
        return {
            "category": "Leave Management",
            "response": "You can check your leave balance under the HR Portal → Leave Section."
        }
    else:
        return {
            "category": "Unknown",
            "response": "Your ticket has been forwarded to the support team."
        }

@app.post("/predict")
def classify_ticket(ticket: TicketRequest):
    prediction = model.predict([ticket.text])[0]
    return generate_response(prediction)