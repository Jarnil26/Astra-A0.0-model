import json

class PatternEngine:
    def __init__(self):
        # 1. Defined symptom clusters (Patterns)
        self.PATTERNS = [
            {
                "name": "Infectious Fever",
                "symptoms": ["fever", "headache", "nausea"],
                "diseases": ["Dengue", "Malaria", "Viral Fever", "Typhoid"],
                "boost": 1.6,
                "category": "infectious"
            },
            {
                "name": "Respiratory Distress",
                "symptoms": ["cough", "shortness of breath", "fatigue"],
                "diseases": ["COVID-19", "Asthma", "Pneumonia", "Bronchitis"],
                "boost": 1.8,
                "category": "respiratory"
            },
            {
                "name": "Diabetes Classic",
                "symptoms": ["frequent urination", "increased thirst", "weight loss"],
                "diseases": ["Diabetes"],
                "boost": 2.0,
                "category": "metabolic"
            },
            {
                "name": "Digestive Distress",
                "symptoms": ["abdominal pain", "diarrhea", "vomiting"],
                "diseases": ["Gastroenteritis", "Food poisoning", "Gastritis", "IBS"],
                "boost": 1.6,
                "category": "digestive"
            },
            {
                "name": "Cardiac Emergency",
                "symptoms": ["chest pain", "shortness of breath", "sweating"],
                "diseases": ["Myocardial infarction", "Angina", "Coronary artery disease"],
                "boost": 2.0,
                "category": "cardiac"
            },
            {
                "name": "Joint Inflammation",
                "symptoms": ["joint pain", "stiffness", "swelling"],
                "diseases": ["Rheumatoid arthritis", "Osteoarthritis", "Gout"],
                "boost": 1.7,
                "category": "musculoskeletal"
            },
            {
                "name": "Urinary Infection",
                "symptoms": ["burning urination", "frequent urination", "lower abdominal pain"],
                "diseases": ["Urinary tract infection", "Cystitis", "Prostatitis"],
                "boost": 1.7,
                "category": "urinary"
            }
        ]

        # 2. Critical Rules (Force top logic)
        self.CRITICAL_RULES = [
            {
                "symptoms": ["frequent urination", "increased thirst"],
                "force_top": "Diabetes",
                "min_confidence": 0.6
            },
            {
                "symptoms": ["cough", "shortness of breath"],
                "force_top": "COVID-19",
                "min_confidence": 0.6
            }
        ]

        # 3. Category Mapping for Diseases
        self.CATEGORY_MAP = {
            "Dengue": "infectious", "Malaria": "infectious", "Viral Fever": "infectious", "Typhoid": "infectious",
            "COVID-19": "respiratory", "Asthma": "respiratory", "Pneumonia": "respiratory", "Bronchitis": "respiratory",
            "Diabetes": "metabolic", "Diabetes Mellitus": "metabolic", "Type 1 diabetes": "metabolic",
            "Gastritis": "digestive", "Food Poisoning": "digestive", "GERD": "digestive", "Indigestion": "digestive",
            "Psoriasis": "dermatology", "Atopic dermatitis": "dermatology", "Eczema": "dermatology", "Skin allergy": "dermatology"
        }

    def detect_patterns(self, input_symptoms):
        """Detect matched patterns based on keyword matching (>70% or >=2 keywords)."""
        s_input = [s.lower().strip() for s in input_symptoms]
        active_patterns = []
        
        for p in self.PATTERNS:
            match_count = 0
            for p_symp in p["symptoms"]:
                p_symp_lower = p_symp.lower().strip()
                # Check if this keyword exists in any of the user's symptoms
                if any(p_symp_lower in s for s in s_input):
                    match_count += 1
            
            # Threshold: >= 2 or 70% of pattern symptoms
            if match_count >= 2 or (match_count / len(p["symptoms"])) >= 0.7:
                active_patterns.append(p)
        
        return active_patterns

    def get_critical_enforcements(self, input_symptoms):
        """Check for must-detect critical rules."""
        s_input = set(s.lower().strip() for s in input_symptoms)
        enforcements = []
        
        for rule in self.CRITICAL_RULES:
            r_symptoms = set(s.lower().strip() for s in rule["symptoms"])
            if r_symptoms.issubset(s_input):
                enforcements.append(rule)
                
        return enforcements

    def get_disease_category(self, disease):
        """Utility to get category for negative filtering."""
        d_lower = disease.lower()
        for d_name, cat in self.CATEGORY_MAP.items():
            if d_name.lower() in d_lower:
                return cat
        return "other"
