"""Railway knowledge base + multilingual response templates.

A compact, self-contained corpus so the support assistant can give *grounded*
answers (and the governance layer can measure groundedness / hallucination)
without any external API.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Knowledge base: policy documents the assistant retrieves from
# ---------------------------------------------------------------------------
KB_DOCS = [
    {"id": "cancel-1", "topic": "cancellation",
     "text": "Confirmed tickets cancelled more than 48 hours before the scheduled departure "
             "attract a cancellation charge of 5 percent of the fare, subject to a minimum of "
             "60 rupees for AC classes and 30 rupees for sleeper."},
    {"id": "cancel-2", "topic": "cancellation",
     "text": "Cancelling between 48 hours and 12 hours before departure incurs a charge of 25 "
             "percent of the fare. Cancellation after the 12-hour window and up to 4 hours before "
             "departure incurs 50 percent. No refund is given after that."},
    {"id": "refund-1", "topic": "refund",
     "text": "Refunds for cancelled e-tickets are credited to the original payment method within "
             "3 to 7 working days. For UPI and card payments the credit usually appears in 3 working days."},
    {"id": "refund-2", "topic": "refund",
     "text": "If a train is cancelled by Indian Railways, a full automatic refund is processed and "
             "no cancellation charge is deducted. The passenger need not file a request."},
    {"id": "tatkal-1", "topic": "tatkal",
     "text": "Tatkal booking opens at 10:00 AM one day before the journey date for AC classes and at "
             "11:00 AM for non-AC classes. Tatkal fares carry a premium over the normal fare."},
    {"id": "rac-1", "topic": "rac",
     "text": "RAC stands for Reservation Against Cancellation. An RAC passenger is guaranteed travel "
             "but may share a side-lower berth. RAC tickets usually get confirmed if cancellations occur."},
    {"id": "wl-1", "topic": "waitlist",
     "text": "A waiting-list ticket is not confirmed at the time of booking. Confirmation depends on "
             "cancellations. If a waiting-list ticket does not confirm, the fare is refunded automatically."},
    {"id": "baggage-1", "topic": "baggage",
     "text": "Sleeper class allows up to 40 kilograms of luggage free. AC two-tier and three-tier allow "
             "up to 50 kilograms. Excess luggage is charged at the luggage rate per kilogram."},
    {"id": "id-1", "topic": "travel_documents",
     "text": "Carry a valid photo identity card such as Aadhaar, PAN, voter ID or passport. An e-ticket "
             "is valid without a printout as long as a valid identity proof is produced."},
    {"id": "board-1", "topic": "boarding",
     "text": "Reach the boarding station at least 20 minutes before the scheduled departure. The train "
             "does not wait beyond its scheduled halt time at intermediate stations."},
    {"id": "pnr-1", "topic": "pnr",
     "text": "A PNR number is a 10-digit number printed on your ticket that holds the current booking "
             "status, coach and berth details for all passengers in the ticket."},
    {"id": "meal-1", "topic": "onboard",
     "text": "Meals can be pre-ordered on many long-distance trains through the catering service at "
             "least 4 hours before departure. Rajdhani and Duronto trains include meals in the fare."},
    {"id": "senior-1", "topic": "concession",
     "text": "Senior citizens aged 58 and above can avail of a concession in sleeper and AC classes, "
             "subject to the prevailing railway policy and valid age proof."},
]

INTENTS = ["pnr_status", "refund", "cancellation_policy", "tatkal_rules", "baggage",
           "travel_documents", "boarding", "concession", "travel_recommendation",
           "journey_planning"]

# Keyword hints per intent (used by the lightweight classifier)
INTENT_KEYWORDS = {
    "pnr_status": ["pnr", "status", "confirm", "waitlist", "wl", "rac", "berth", "seat number",
                   "पीएनआर", "स्थिति", "நிலை", "స్థితి", "অবস্থা"],
    "refund": ["refund", "money back", "credited", "reimburse", "cashback",
               "धनवापसी", "परतावा", "பணத்திரும்ப", "వాపసు", "ফেরত"],
    "cancellation_policy": ["cancel", "cancellation", "charges", "deduct",
                            "रद्द", "रद्दीकरण", "रத்து", "ரத்து", "రద్దు", "বাতিল"],
    "tatkal_rules": ["tatkal", "premium", "last minute booking",
                     "तत्काल", "तत्काळ", "தத்கால்", "తత్కాల్", "তৎকাল"],
    "baggage": ["luggage", "baggage", "bags", "kg", "kilogram",
                "सामान", "பொருள்", "సామాను", "মালপত্র"],
    "travel_documents": ["id", "identity", "aadhaar", "pan", "passport", "document",
                         "दस्तावेज़", "ஆவண", "పత్రాలు", "নথি"],
    "boarding": ["board", "boarding", "reach station", "platform", "how early",
                 "बोर्डिंग", "बोर्ड", "बोर्डिंग"],
    "concession": ["senior", "concession", "discount", "age",
                   "रियायत", "सवलत", "சலுகை", "రాయితీ", "ছাড়"],
    "travel_recommendation": ["recommend", "suggest", "best train", "which train", "options",
                              "सुझाव", "शिफारस", "பரிந்துரை", "సూచన", "সুপারিশ"],
    "journey_planning": ["plan", "itinerary", "route", "reach by", "budget",
                         "योजना", "नियोजन", "திட்டமிடல்", "ప్రణాళిక", "পরিকল্পনা"],
}

# Multilingual templates. {fact} is filled with a retrieved, grounded fact.
_I18N_DIR = Path(__file__).resolve().parent / "kb_i18n"


def _parse_i18n(text: str) -> dict:
    """Parse the '<id>: <translated text>' blocks separated by ===DOC=== markers."""
    docs = {}
    for chunk in text.split("===DOC==="):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        doc_id, body = chunk.split(":", 1)
        docs[doc_id.strip()] = body.strip()
    return docs


def load_i18n_docs() -> dict:
    """Build {lang: [ {id, topic, text} ]} from the translated KB files.

    Keeps the English topic mapping so retrieval-by-intent works in any language.
    """
    out = {}
    if not _I18N_DIR.is_dir():
        return out
    for f in sorted(_I18N_DIR.glob("*.txt")):
        parsed = _parse_i18n(f.read_text(encoding="utf-8"))
        docs = [{"id": d["id"], "topic": d["topic"], "text": parsed[d["id"]]}
                for d in KB_DOCS if d["id"] in parsed]
        if docs:
            out[f.stem] = docs
    return out


# {lang: [docs]} for the translated languages (en stays in KB_DOCS).
KB_DOCS_ML = load_i18n_docs()

# Multilingual templates. {fact} is filled with a retrieved, grounded fact.
TEMPLATES = {
    "en": {"pnr_status": "Your PNR status request has been noted. {fact}",
           "refund": "Here is the refund policy: {fact}",
           "cancellation_policy": "Regarding cancellation: {fact}",
           "tatkal_rules": "Tatkal booking rules: {fact}",
           "baggage": "Baggage allowance: {fact}",
           "travel_documents": "Travel documents: {fact}",
           "boarding": "Boarding guidance: {fact}",
           "concession": "Concession rules: {fact}",
           "travel_recommendation": "Here is what I can suggest: {fact}",
           "journey_planning": "For journey planning: {fact}",
           "fallback": "I can help with PNR status, refunds, cancellations, Tatkal, baggage and journey planning."},
    "hi": {"pnr_status": "आपकी PNR स्थिति के लिए: {fact}",
           "refund": "धनवापसी नियम: {fact}",
           "cancellation_policy": "रद्दीकरण के बारे में: {fact}",
           "tatkal_rules": "तत्काल नियम: {fact}",
           "baggage": "सामान नियम: {fact}",
           "travel_documents": "यात्रा दस्तावेज़: {fact}",
           "boarding": "बोर्डिंग मार्गदर्शन: {fact}",
           "concession": "रियायत नियम: {fact}",
           "travel_recommendation": "सुझाव: {fact}",
           "journey_planning": "यात्रा योजना: {fact}",
           "fallback": "मैं PNR, धनवापसी, रद्दीकरण, तत्काल, सामान और यात्रा योजना में मदद कर सकता हूँ।"},
    "ta": {"pnr_status": "உங்கள் PNR நிலை: {fact}",
           "refund": "பணத்திரும்ப நடைமுறை: {fact}",
           "cancellation_policy": "ரத்து விதிகள்: {fact}",
           "tatkal_rules": "தத்கால் விதிகள்: {fact}",
           "baggage": "பொருள் அளவு: {fact}",
           "travel_documents": "பயண ஆவணங்கள்: {fact}",
           "boarding": "பயண வழிகாட்டல்: {fact}",
           "concession": "சலுகை விதிகள்: {fact}",
           "travel_recommendation": "பரிந்துரை: {fact}",
           "journey_planning": "பயண திட்டமிடல்: {fact}",
           "fallback": "PNR, பணத்திரும்பம், ரத்து, தத்கால், பொருள் மற்றும் பயண திட்டமிடலில் உதவ முடியும்."},
    "te": {"pnr_status": "మీ PNR స్థితి: {fact}",
           "refund": "వాపసు విధానం: {fact}",
           "cancellation_policy": "రద్దు నియమాలు: {fact}",
           "tatkal_rules": "తత్కాల్ నియమాలు: {fact}",
           "baggage": "సామాను పరిమితి: {fact}",
           "travel_documents": "ప్రయాణ పత్రాలు: {fact}",
           "boarding": "బోర్డింగ్ మార్గదర్శకం: {fact}",
           "concession": "రాయితీ నియమాలు: {fact}",
           "travel_recommendation": "సూచన: {fact}",
           "journey_planning": "ప్రయాణ ప్రణాళిక: {fact}",
           "fallback": "PNR, వాపసు, రద్దు, తత్కాల్, సామాను మరియు ప్రయాణ ప్రణాళికలో సహాయం చేయగలను."},
    "bn": {"pnr_status": "আপনার PNR অবস্থা: {fact}",
           "refund": "ফেরত নীতি: {fact}",
           "cancellation_policy": "বাতিলের নিয়ম: {fact}",
           "tatkal_rules": "তৎকাল নিয়ম: {fact}",
           "baggage": "মালপত্রের সীমা: {fact}",
           "travel_documents": "ভ্রমণ নথি: {fact}",
           "boarding": "বোর্ডিং নির্দেশনা: {fact}",
           "concession": "ছাড়ের নিয়ম: {fact}",
           "travel_recommendation": "সুপারিশ: {fact}",
           "journey_planning": "ভ্রমণ পরিকল্পনা: {fact}",
           "fallback": "PNR, ফেরত, বাতিল, তৎকাল, মালপত্র ও ভ্রমণ পরিকল্পনায় সাহায্য করতে পারি।"},
    "mr": {"pnr_status": "तुमची PNR स्थिती: {fact}",
           "refund": "परतावा धोरण: {fact}",
           "cancellation_policy": "रद्द करण्याचे नियम: {fact}",
           "tatkal_rules": "तत्काळ नियम: {fact}",
           "baggage": "सामान मर्यादा: {fact}",
           "travel_documents": "प्रवास कागदपत्रे: {fact}",
           "boarding": "बोर्डिंग मार्गदर्शन: {fact}",
           "concession": "सवलत नियम: {fact}",
           "travel_recommendation": "शिफारस: {fact}",
           "journey_planning": "प्रवास नियोजन: {fact}",
           "fallback": "मी PNR, परतावा, रद्द, तत्काळ, सामान आणि प्रवास नियोजनात मदत करू शकतो."},
}
