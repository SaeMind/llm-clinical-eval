"""
synthetic_cases.py
Synthetic clinical case generator for LLM evaluation harness.
Generates realistic clinical documents with NO PHI.
Uses templated clinical scenarios across 10 disease categories.
"""

import hashlib
import random
import uuid
from dataclasses import dataclass, field
from typing import Literal

DISEASE_CATEGORIES = [
    "congestive_heart_failure",
    "copd_exacerbation",
    "type2_diabetes",
    "sepsis",
    "acute_mi",
    "stroke",
    "ckd",
    "oncology",
    "psychiatric",
    "trauma",
]

TaskType = Literal[
    "discharge_summary_compression",
    "radiology_report_impression",
    "progress_note_soap",
    "multi_visit_synthesis",
]

# ---------------------------------------------------------------------------
# Synthetic discharge summary templates by disease category
# ---------------------------------------------------------------------------
DISCHARGE_TEMPLATES = {
    "congestive_heart_failure": """
ADMISSION DATE: [DATE]  DISCHARGE DATE: [DATE+5]
ATTENDING: [PHYSICIAN_NAME], MD

CHIEF COMPLAINT: Progressive dyspnea, bilateral lower extremity edema x 3 days.

HISTORY OF PRESENT ILLNESS:
Patient is a [AGE]-year-old [SEX] with a history of ischemic cardiomyopathy (EF 30%), hypertension, type 2 diabetes mellitus, and chronic kidney disease stage 3 who presented to the emergency department with worsening dyspnea on exertion, orthopnea, and bilateral ankle swelling over the past three days. Patient reports weight gain of 8 lbs over one week. Denies chest pain, fever, or chills. BNP on presentation was 2,840 pg/mL. Chest X-ray demonstrated bilateral pleural effusions and vascular congestion consistent with decompensated heart failure.

HOSPITAL COURSE:
Patient was admitted for acute decompensated heart failure. IV furosemide was initiated at 80 mg BID with goal of 1-1.5L net negative fluid balance per day. Strict I&Os and daily weights were maintained. Cardiology was consulted and recommended continuation of carvedilol and lisinopril with uptitration at outpatient follow-up. Echocardiogram performed on hospital day 2 demonstrated LVEF 28%, moderate mitral regurgitation, and elevated LVEDP. Patient diuresed 4.2L over hospital course with improvement in dyspnea and resolution of bilateral edema. BNP at discharge: 1,120 pg/mL. Creatinine remained stable at 1.6 mg/dL throughout admission.

DISCHARGE MEDICATIONS (CHANGES FROM PRIOR):
1. Furosemide 80 mg PO daily (increased from 40 mg)
2. Carvedilol 12.5 mg PO BID (unchanged)
3. Lisinopril 10 mg PO daily (unchanged; dose increase deferred pending renal function monitoring)
4. Spironolactone 25 mg PO daily (new addition per cardiology)
5. Metformin 500 mg PO BID — HELD at discharge; restart only after nephrology clearance
6. Atorvastatin 40 mg PO QHS (unchanged)

FOLLOW-UP:
1. Cardiology clinic in 1 week — weight monitoring, repeat BMP
2. Primary care in 2 weeks
3. Daily weight monitoring; call if weight increases >3 lbs in 24 hours or >5 lbs in 1 week
4. Sodium restriction: <2g daily; fluid restriction: 1.5L daily

DISCHARGE CONDITION: Stable. Ambulating independently.
""",

    "sepsis": """
ADMISSION DATE: [DATE]  DISCHARGE DATE: [DATE+7]
ATTENDING: [PHYSICIAN_NAME], MD

CHIEF COMPLAINT: Fever, altered mental status, hypotension.

HISTORY OF PRESENT ILLNESS:
[AGE]-year-old [SEX] with history of diabetes mellitus type 2, hypertension, and recent urinary tract infection (completed nitrofurantoin course 10 days prior) who was brought by EMS with fever to 39.8C, altered mental status, and systolic blood pressure of 82 mmHg. Initial lactate 3.2 mmol/L. Urinalysis demonstrated pyuria with >50 WBCs/hpf, positive nitrites. Blood cultures x2 drawn. Urine culture subsequently grew E. coli >100,000 CFU/mL, sensitive to ciprofloxacin and ceftriaxone.

HOSPITAL COURSE:
Patient met sepsis criteria and was admitted to the MICU. Sepsis bundle initiated: 30mL/kg IV crystalloid bolus administered, vasopressors not required after fluid resuscitation. Empiric piperacillin-tazobactam initiated. Blood cultures finalized as no growth at 48 hours. Urine culture sensitivities allowed de-escalation to ceftriaxone IV on hospital day 2, then oral ciprofloxacin on hospital day 4. Patient was transferred to the floor on hospital day 3. Mental status returned to baseline on hospital day 2. Completed 7-day course of antibiotics.

DISCHARGE MEDICATIONS:
1. Ciprofloxacin 500 mg PO BID — complete 3-day course after discharge (total 7-day course)
2. Metformin 1000 mg PO BID — RESTARTED (held during admission due to acute illness)
3. Lisinopril 20 mg PO daily (unchanged)
4. Home medications otherwise unchanged

FOLLOW-UP:
1. Primary care within 1 week for repeat urinalysis and urine culture
2. Urology referral placed for recurrent UTI evaluation
DISCHARGE CONDITION: Good. Afebrile x 48 hours prior to discharge.
""",
}

RADIOLOGY_TEMPLATES = {
    "congestive_heart_failure": """
EXAMINATION: CT Chest with IV Contrast
DATE: [DATE]
INDICATION: Dyspnea, rule out pulmonary embolism; known CHF

FINDINGS:
PULMONARY VASCULATURE: CT angiography of the pulmonary vasculature demonstrates no evidence of pulmonary embolism to the subsegmental level. Mild pulmonary arterial hypertension pattern with pulmonary artery diameter measuring 3.2 cm (upper limits of normal).

LUNGS AND AIRWAYS: Bilateral dependent atelectasis in the lower lobes. Moderate bilateral pleural effusions, right greater than left, with associated passive compressive atelectasis. Peribronchial cuffing and septal thickening consistent with pulmonary edema. No focal consolidation to suggest pneumonia. No pneumothorax. No suspicious pulmonary nodules.

MEDIASTINUM: Cardiomegaly with cardiothoracic ratio of 0.62. Moderate pericardial effusion. No mediastinal or hilar lymphadenopathy.

SOFT TISSUES/BONES: Mild subcutaneous edema of the chest wall, bilateral. No aggressive bone lesion identified.

IMPRESSION:
1. No pulmonary embolism.
2. Findings consistent with decompensated congestive heart failure: bilateral pleural effusions (moderate), pulmonary edema pattern, cardiomegaly.
3. Pericardial effusion, moderate — recommend echocardiogram for further characterization.
4. No acute pneumonia.
""",
}


@dataclass
class SyntheticCase:
    case_id:              str
    task_type:            str
    disease_category:     str
    source_document:      str
    reference_summary:    str
    metadata:             dict = field(default_factory=dict)


class SyntheticCaseGenerator:
    """
    Generates synthetic clinical evaluation cases from templates.
    All output is synthetic — no PHI.
    """

    AGE_RANGE     = (45, 85)
    SEX_OPTIONS   = ["male", "female"]
    PHYSICIAN_NAMES = ["J. Smith", "M. Johnson", "A. Patel", "R. Chen", "L. Garcia"]

    def _fill_template(self, template: str) -> str:
        """Replace placeholder tokens with synthetic values."""
        import datetime
        base_date = datetime.date(2023, random.randint(1, 12), random.randint(1, 28))
        return (
            template
            .replace("[DATE]",        base_date.strftime("%m/%d/%Y"))
            .replace("[DATE+5]",      (base_date + datetime.timedelta(days=5)).strftime("%m/%d/%Y"))
            .replace("[DATE+7]",      (base_date + datetime.timedelta(days=7)).strftime("%m/%d/%Y"))
            .replace("[AGE]",         str(random.randint(*self.AGE_RANGE)))
            .replace("[SEX]",         random.choice(self.SEX_OPTIONS))
            .replace("[PHYSICIAN_NAME]", random.choice(self.PHYSICIAN_NAMES))
        )

    def _get_reference_summary(self, source: str, task_type: str, disease: str) -> str:
        """Return a pre-validated reference summary for the case."""
        # In production: reference summaries are human-validated and stored in
        # data/reference_summaries/{task_type}/{disease}.json
        # For scaffold: return a structured placeholder reference
        if task_type == "discharge_summary_compression":
            if disease == "congestive_heart_failure":
                return ("Patient admitted for acute decompensated heart failure with EF 28%, treated with IV diuresis achieving 4.2L net negative balance. "
                        "Furosemide increased to 80 mg daily and spironolactone added at discharge. "
                        "Follow up with cardiology in 1 week; metformin held pending nephrology clearance.")
            elif disease == "sepsis":
                return ("Patient admitted with urosepsis; E. coli bacteriuria, de-escalated to ciprofloxacin after culture sensitivities. "
                        "Completed 7-day antibiotic course; metformin restarted at discharge. "
                        "Follow up primary care in 1 week for repeat urine culture; urology referral placed.")
        return "Reference summary not available for this case type."

    def generate(self, task_type: str, n: int = 20) -> list[SyntheticCase]:
        """Generate n synthetic evaluation cases for a given task type."""
        cases = []
        # Distribute cases across disease categories
        categories = DISEASE_CATEGORIES * (n // len(DISEASE_CATEGORIES) + 1)
        random.shuffle(categories)
        categories = categories[:n]

        for i, disease in enumerate(categories):
            # Select appropriate template
            if task_type == "discharge_summary_compression":
                template = DISCHARGE_TEMPLATES.get(disease, DISCHARGE_TEMPLATES["congestive_heart_failure"])
            elif task_type == "radiology_report_impression":
                template = RADIOLOGY_TEMPLATES.get(disease, RADIOLOGY_TEMPLATES["congestive_heart_failure"])
            else:
                template = DISCHARGE_TEMPLATES.get(disease, DISCHARGE_TEMPLATES["congestive_heart_failure"])

            source = self._fill_template(template)
            reference = self._get_reference_summary(source, task_type, disease)
            case_id = hashlib.md5(f"{task_type}_{disease}_{i}".encode()).hexdigest()[:12]

            cases.append(SyntheticCase(
                case_id=case_id,
                task_type=task_type,
                disease_category=disease,
                source_document=source.strip(),
                reference_summary=reference,
                metadata={"template_index": i, "synthetic": True},
            ))

        return cases
