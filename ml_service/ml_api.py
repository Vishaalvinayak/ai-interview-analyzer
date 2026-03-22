# ml_service/ml_api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import re
import os

# ============================================
# FASTAPI APP INITIALIZATION
# ============================================

app = FastAPI(
    title="Interview Answer Analyzer API",
    description="AI-powered interview answer evaluation system",
    version="1.0.0"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# MODEL LOADING (AT STARTUP)
# ============================================

print("=" * 60)
print("  INTERVIEW ANSWER ANALYZER API")
print("=" * 60)
print("\n🔄 Loading AI model...")

class ModelManager:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = 'model_saved'
        
        if not os.path.exists(model_path):
            raise Exception(f"❌ Model not found at {model_path}")
        
        print(f"💻 Device: {self.device}")
        print(f"📂 Loading from: {model_path}")
        
        # Load tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        self.label_names = ['weak', 'average', 'strong']
        
        # Weak language patterns
        self.weak_patterns = [
            r'\bi think\b', r'\bmaybe\b', r'\bprobably\b',
            r'\bkinda\b', r'\bsorta\b', r'\bnot sure\b',
            r'\bi guess\b', r'\bperhaps\b', r'\bmight be\b',
            r'\bunsure\b', r'\bdon\'t know\b'
        ]
        
        # Keywords database for different topics
        self.keywords_db = {
            "REST API": ["HTTP", "GET", "POST", "PUT", "DELETE", "endpoint", 
                        "stateless", "resource", "JSON", "RESTful", "API"],
            "SQL": ["database", "table", "query", "SELECT", "JOIN", "WHERE",
                   "relational", "ACID", "transaction", "index"],
            "NoSQL": ["MongoDB", "document", "collection", "schema-less",
                     "scalability", "distributed", "CAP", "eventual consistency"],
            "OOP": ["class", "object", "inheritance", "polymorphism",
                   "encapsulation", "abstraction", "interface", "method"],
            "Git": ["commit", "branch", "merge", "repository", "clone",
                   "push", "pull", "version control", "conflict"],
            "MVC": ["Model", "View", "Controller", "separation of concerns",
                   "architecture", "presentation", "business logic"],
            "deadline": ["prioritize", "schedule", "planning", "time management",
                        "communication", "milestone", "deliverable"],
            "project": ["challenge", "team", "solution", "result", "implemented",
                       "developed", "outcome", "success"],
            "optimize": ["performance", "index", "query", "bottleneck",
                        "cache", "profiling", "execution plan", "latency"]
        }
        
        print("✅ Model loaded successfully!")
        print(f"✅ Ready to analyze answers\n")

# Initialize model manager
model_manager = ModelManager()

# ============================================
# REQUEST/RESPONSE MODELS
# ============================================

class AnalyzeRequest(BaseModel):
    question: str = Field(..., min_length=10, max_length=500, 
                         description="Interview question")
    answer: str = Field(..., min_length=20, max_length=2000,
                       description="Candidate's answer")

class KeywordAnalysis(BaseModel):
    found: List[str]
    missing: List[str]
    coverage: float

class WeakPhrase(BaseModel):
    phrase: str
    position: int
    context: str

class AnalyzeResponse(BaseModel):
    classification: str
    score: int
    confidence: Dict[str, float]
    keyword_analysis: KeywordAnalysis
    weak_language: List[WeakPhrase]
    suggestions: List[str]
    metadata: Dict

# ============================================
# CORE ANALYSIS FUNCTIONS
# ============================================

def classify_answer(answer: str) -> tuple:
    """Get model classification and confidence scores"""
    
    # Tokenize
    inputs = model_manager.tokenizer(
        answer,
        add_special_tokens=True,
        max_length=256,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(model_manager.device)
    
    # Get prediction
    with torch.no_grad():
        outputs = model_manager.model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
    
    # Convert to dict
    confidences = {
        label: float(prob) 
        for label, prob in zip(model_manager.label_names, probabilities)
    }
    
    classification = model_manager.label_names[predicted_class]
    
    return classification, confidences


def calculate_score(classification: str, confidences: Dict[str, float], 
                    answer: str, keyword_coverage: float) -> int:
    """Calculate 0-100 score based on multiple factors"""
    
    # Base score from classification
    base_scores = {
        'weak': 25,
        'average': 55,
        'strong': 80
    }
    base = base_scores[classification]
    
    # Confidence adjustment (-10 to +15)
    confidence_adjustment = (confidences[classification] - 0.5) * 30
    
    # Word count factor
    word_count = len(answer.split())
    if word_count < 50:
        length_adjustment = -15
    elif word_count > 150:
        length_adjustment = +10
    elif word_count > 100:
        length_adjustment = +5
    else:
        length_adjustment = 0
    
    # Keyword coverage bonus (0 to +10)
    keyword_bonus = keyword_coverage * 10
    
    # Calculate final score
    final_score = base + confidence_adjustment + length_adjustment + keyword_bonus
    
    # Clamp between 0-100
    return max(0, min(100, int(final_score)))


def extract_keywords(question: str, answer: str) -> Dict:
    """Extract and analyze keywords based on question topic"""
    
    # Find relevant keyword set
    relevant_keywords = []
    question_lower = question.lower()
    
    for topic, keywords in model_manager.keywords_db.items():
        if topic.lower() in question_lower or any(
            word.lower() in question_lower for word in topic.split()
        ):
            relevant_keywords.extend(keywords)
    
    # If no specific match, use general keywords
    if not relevant_keywords:
        relevant_keywords = ["example", "experience", "implemented", 
                            "developed", "solution", "result"]
    
    # Remove duplicates
    relevant_keywords = list(set(relevant_keywords))
    
    # Check which keywords are present
    answer_lower = answer.lower()
    found_keywords = []
    for kw in relevant_keywords:
        if kw.lower() in answer_lower:
            found_keywords.append(kw)
    
    missing_keywords = [kw for kw in relevant_keywords if kw not in found_keywords]
    
    # Calculate coverage
    coverage = len(found_keywords) / max(len(relevant_keywords), 1)
    
    return {
        'found': found_keywords[:10],  # Top 10
        'missing': missing_keywords[:5],  # Top 5 missing
        'coverage': round(coverage, 2)
    }


def detect_weak_language(answer: str) -> List[Dict]:
    """Detect confidence-reducing phrases"""
    
    weak_phrases = []
    answer_lower = answer.lower()
    
    for pattern in model_manager.weak_patterns:
        matches = re.finditer(pattern, answer_lower, re.IGNORECASE)
        for match in matches:
            start = match.start()
            end = match.end()
            
            # Get context (20 chars before and after)
            context_start = max(0, start - 20)
            context_end = min(len(answer), end + 20)
            context = answer[context_start:context_end]
            
            weak_phrases.append({
                'phrase': match.group(),
                'position': start,
                'context': context.strip()
            })
    
    return weak_phrases


def generate_suggestions(classification: str, weak_phrases: List, 
                        keyword_analysis: Dict, word_count: int,
                        confidences: Dict) -> List[str]:
    """Generate actionable improvement suggestions"""
    
    suggestions = []
    
    # Classification-based suggestions
    if classification == 'weak':
        suggestions.append("⚠️ Provide specific examples from your experience to strengthen your answer")
        suggestions.append("📝 Structure your answer with clear introduction, body, and conclusion")
        suggestions.append("🎯 Focus on demonstrating concrete skills and outcomes")
    
    elif classification == 'average':
        suggestions.append("📈 Add more technical depth and specific details to your explanation")
        suggestions.append("💡 Include quantifiable results or metrics where possible")
        suggestions.append("🔍 Elaborate on the 'why' and 'how' of your decisions")
    
    else:  # strong
        suggestions.append("✅ Excellent answer! Maintain this level of detail and specificity")
        suggestions.append("🎯 Consider adding one more concrete example to make it even stronger")
    
    # Weak language suggestions
    if len(weak_phrases) >= 3:
        suggestions.append(f"💬 Remove {len(weak_phrases)} uncertain phrases to sound more confident")
    elif len(weak_phrases) > 0:
        examples = [p['phrase'] for p in weak_phrases[:2]]
        suggestions.append(f"💬 Replace uncertain phrases like '{examples[0]}' with confident statements")
    
    # Keyword suggestions
    if keyword_analysis['coverage'] < 0.3:
        missing = keyword_analysis['missing'][:3]
        if missing:
            suggestions.append(f"🔑 Include key terms: {', '.join(missing)}")
    
    # Length suggestions
    if word_count < 60:
        suggestions.append(f"📏 Expand your answer (current: {word_count} words, aim for 100-150)")
    elif word_count > 300:
        suggestions.append(f"✂️ Consider making your answer more concise (current: {word_count} words)")
    
    # Confidence-based suggestion
    if confidences[classification] < 0.7:
        suggestions.append("⚡ Your answer is on the borderline - adding more details will solidify it")
    
    return suggestions[:5]  # Return top 5 suggestions


# ============================================
# API ENDPOINTS
# ============================================

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_answer(request: AnalyzeRequest):
    """
    Main endpoint to analyze interview answer
    
    Returns classification, score, confidence, keywords, and suggestions
    """
    try:
        # Step 1: Get AI classification
        classification, confidences = classify_answer(request.answer)
        
        # Step 2: Analyze keywords
        keyword_analysis = extract_keywords(request.question, request.answer)
        
        # Step 3: Calculate score
        score = calculate_score(
            classification, 
            confidences, 
            request.answer,
            keyword_analysis['coverage']
        )
        
        # Step 4: Detect weak language
        weak_phrases = detect_weak_language(request.answer)
        
        # Step 5: Generate suggestions
        word_count = len(request.answer.split())
        suggestions = generate_suggestions(
            classification,
            weak_phrases,
            keyword_analysis,
            word_count,
            confidences
        )
        
        # Step 6: Metadata
        metadata = {
            'word_count': word_count,
            'sentence_count': len(re.split(r'[.!?]+', request.answer.strip())),
            'avg_word_length': round(
                sum(len(word) for word in request.answer.split()) / max(word_count, 1), 
                2
            ),
            'has_examples': bool(re.search(r'\b(for example|for instance|such as|like)\b', 
                                          request.answer.lower())),
            'question_length': len(request.question.split())
        }
        
        # Return response
        return AnalyzeResponse(
            classification=classification,
            score=score,
            confidence=confidences,
            keyword_analysis=KeywordAnalysis(**keyword_analysis),
            weak_language=weak_phrases,
            suggestions=suggestions,
            metadata=metadata
        )
        
    except Exception as e:
        print(f"❌ Error in analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_manager.model is not None,
        "device": str(model_manager.device),
        "version": "1.0.0"
    }


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Interview Answer Analyzer API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "analyze": "POST /analyze - Analyze interview answer",
            "health": "GET /health - Health check",
            "docs": "GET /docs - Interactive API documentation"
        },
        "model_info": {
            "architecture": "DistilBERT",
            "accuracy": "96.15%",
            "classes": ["weak", "average", "strong"]
        }
    }


# ============================================
# STARTUP MESSAGE
# ============================================

@app.on_event("startup")
async def startup_event():
    print("\n" + "=" * 60)
    print("  ✅ API SERVER READY!")
    print("=" * 60)
    print(f"  📍 Access API at: http://localhost:8000")
    print(f"  📚 View docs at: http://localhost:8000/docs")
    print(f"  🎯 Model Accuracy: 96.15%")
    print("=" * 60 + "\n")


# Run with: uvicorn ml_api:app --reload --port 8000