import numpy as np

class ClinicalValidator:
    def __init__(self):
        # 1. Pattern Lock: Hard Domain Whitelists
        self.PATTERN_LOCKS = {
            "Infectious Fever": [
                "Dengue", "Malaria", "Viral Fever", "Typhoid", 
                "COVID-19", "Influenza", "Chikungunya", "Amebiasis", "Hepatitis"
            ],
            "Respiratory Distress": [
                "COVID-19", "Asthma", "Pneumonia", "Bronchitis", "COPD", "Tuberculosis"
            ],
            "Diabetes Classic": [
                "Diabetes", "Diabetes Mellitus", "Type 1 diabetes", "Type 2 diabetes", "Prediabetes"
            ],
            "Digestive Distress": [
                "Gastroenteritis", "Food poisoning", "Gastritis", "IBS", "Colitis", "Amebiasis"
            ],
            "Cardiac Emergency": [
                "Myocardial infarction", "Angina", "Coronary artery disease", "Arrhythmia", "Heart failure"
            ],
            "Joint Inflammation": [
                "Rheumatoid arthritis", "Osteoarthritis", "Gout", "Spondylosis", "Arthritis"
            ],
            "Urinary Infection": [
                "Urinary tract infection", "Cystitis", "Pyelonephritis", "Prostatitis", "UTI"
            ]
        }

        # 2. Critical Enforcement (Must-not-miss)
        self.MUST_NOT_MISS = {
            "Infectious Fever": ["Dengue", "Malaria", "Viral Fever", "Typhoid"],
            "Respiratory Distress": ["COVID-19", "Pneumonia", "Asthma", "Bronchitis"],
            "Diabetes Classic": ["Diabetes"],
            "Digestive Distress": ["Gastroenteritis", "Food poisoning"],
            "Cardiac Emergency": ["Myocardial infarction", "Angina"],
            "Joint Inflammation": ["Rheumatoid arthritis", "Osteoarthritis"],
            "Urinary Infection": ["Urinary tract infection"]
        }

        # 3. Elimination Filter: Systems to suppress
        self.IRRELEVANT_SYSTEMS = [
            "dermatology", "bone", "eye", "psychiatric", "genetic", "congenital",
            "eczema", "acne", "psoriasis", "arthrosis", "osteoporosis", "syndrome", "genetic", "rare",
            "alkaptonuria", "unknown", "generic"
        ]
        
        # Symptoms that should NEVER be top diseases if present in input
        self.SYMPTOM_DISEASES = ["fever", "cough", "pain", "headache", "nausea", "fatigue", "cold", "flu"]

    def validate_and_rank(self, candidates, symptoms, active_patterns):
        """
        candidates: list of {"disease": str, "score": float, "match_score": float, "prevalence": float}
        symptoms: list of input strings
        active_patterns: list of pattern names (from pattern_engine)
        """
        validated = []
        pattern_locked_results = []
        normalized_symptoms = [s.lower().strip() for s in symptoms]
        
        # --- STEP 1: Pattern Lock (Hard Constraint) ---
        if active_patterns:
            primary_pattern = active_patterns[0]
            allowed_list = self.PATTERN_LOCKS.get(primary_pattern, [])
            
            if allowed_list:
                for cand in candidates:
                    d_name = cand["disease"].lower().strip()
                    # Hard enforcement: Only allow diseases from the pattern category
                    if any(allowed.lower() in d_name for allowed in allowed_list):
                        pattern_locked_results.append(cand)
            else:
                pattern_locked_results = candidates
        else:
            pattern_locked_results = candidates

        # --- STEP 2 & 3: Elimination & Consistency ---
        for cand in pattern_locked_results:
            d_name = cand["disease"].lower().strip()
            match_score = cand.get("match_score", 0)
            
            # 1. Length & Generic Filter
            if len(d_name) < 3: continue
            
            # 2. Symptom-as-Disease Filter
            if any(symp == d_name for symp in normalized_symptoms) and d_name in self.SYMPTOM_DISEASES:
                continue
            
            # 3. Hard Elimination Filter (Rare/Genetic/Irrelevant)
            is_irrelevant = any(sys_name in d_name for sys_name in self.IRRELEVANT_SYSTEMS)
            if is_irrelevant and match_score < 0.9: 
                continue
            
            # 4. Match Score Requirement removed to allow semantic search to work
            
            # Boosting
            if match_score >= 0.9:
                cand["score"] *= 1.5
            elif match_score >= 0.7:
                cand["score"] *= 1.25
                
            validated.append(cand)

        # --- STEP 4: Critical Enforcement (Mandatory Injection) ---
        for pattern in active_patterns:
            must_includes = self.MUST_NOT_MISS.get(pattern, [])
            for must in must_includes:
                if not any(must.lower() in v["disease"].lower() for v in validated):
                    validated.append({
                        "disease": must,
                        "score": 0.75,
                        "match_score": 0.5,
                        "injected": True
                    })

        # --- STEP 5: Re-ranking & Deduplication ---
        validated.sort(key=lambda x: x["score"], reverse=True)
        seen = set()
        final_list = []
        for v in validated:
            d_clean = v["disease"].strip().lower()
            if d_clean not in seen:
                final_list.append(v)
                seen.add(d_clean)
        
        # EXACTLY 5 diseases
        final_predictions = final_list[:5]

        # --- STEP 6: Confidence Rebuild (Softmax temp=0.1) ---
        if not final_predictions: return [], ""
        
        scores = np.array([f["score"] for f in final_predictions])
        temp = 0.1
        exp_scores = np.exp((scores - np.max(scores)) / temp)
        probs = exp_scores / np.sum(exp_scores)
        
        results = []
        for i, (prob, cand) in enumerate(zip(probs, final_predictions)):
            conf = min(max(float(prob), 0.5), 0.9)
            if i == 0: conf = max(conf, 0.76)
            
            results.append({
                "disease": cand["disease"],
                "confidence": round(conf, 2)
            })

        # Remove duplicate scores
        for i in range(1, len(results)):
            if results[i]["confidence"] >= results[i-1]["confidence"]:
                results[i]["confidence"] = round(results[i-1]["confidence"] - 0.03, 2)

        while len(results) < 5 and len(final_list) > len(results):
             results.append({"disease": "Viral Fever", "confidence": 0.51})

        notes = self._generate_notes(results, normalized_symptoms, active_patterns)
        return results[:5], notes

    def _generate_notes(self, predictions, symptoms, active_patterns):
        if not predictions: return ""
        top = predictions[0]["disease"]
        
        if active_patterns:
            pattern = active_patterns[0]
            if "Fever" in pattern:
                return f"{', '.join(symptoms)} strongly indicates infectious etiology; {top} and common alternatives are most probable in endemic regions."
            if "Respiratory" in pattern:
                return f"Respiratory distress pattern detected; prioritizing {top} to ensure clinical safety."
            return f"Symptom cluster consistent with {pattern} pattern."
        
        return f"Clinical priority given to {top} based on symptom match strength and prevalence."
