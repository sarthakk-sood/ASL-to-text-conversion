from django.http import JsonResponse
from rest_framework.decorators import api_view
import pickle
import os
from textblob import TextBlob

# Load Model during startup
try:
    with open("mlp_asl_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None
    print("Warning: mlp_asl_model.pkl not found")

@api_view(['POST'])
def predict_asl(request):
    if not model:
        return JsonResponse({"error": "Model not loaded"}, status=500)
    
    try:
        # Expected input: {"landmarks": [x1, y1, z1, x2, y2, z2, ...]} (63 values)
        landmarks = request.data.get('landmarks')
        if not landmarks or len(landmarks) != 63:
            return JsonResponse({"error": "Invalid landmarks. Expected 63 floats."}, status=400)
            
        # Get standard prediction
        pred = model.predict([landmarks])[0]
        
        # Get probability (confidence accuracy)
        proba = model.predict_proba([landmarks])[0]
        confidence = float(max(proba)) * 100 # Highest percentage
        
        return JsonResponse({"prediction": pred, "confidence": confidence})
        
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

@api_view(['POST'])
def autocorrect(request):
    text = request.data.get("text", "")
    if not text:
        return JsonResponse({"corrected": ""})
    
    # Use TextBlob correctly to fix spelling mistakes
    # Convert to lowercase since TextBlob ignores uppercase words (treats them as acronyms)
    is_upper = text.isupper()
    is_title = text.istitle()
    
    corrected_text = str(TextBlob(text.lower()).correct())
    
    if is_upper:
        corrected_text = corrected_text.upper()
    elif is_title:
        corrected_text = corrected_text.title()
        
    return JsonResponse({"corrected": corrected_text})