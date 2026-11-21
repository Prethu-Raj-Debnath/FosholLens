"""Curated crop advisory knowledge base.

This is the only component allowed to state a fact: a disease name, a symptom, a
medicine, a dose. The LLM rewrites prose around these entries and is validated
against them; it never invents one.

Scope is rice and potato only. Those are the two crops the shipped EfficientNet
checkpoint can actually name, so they are the two crops the app offers. Tomato
and chili were carried here while the ViT checkpoint was the primary model; they
have no class in the main checkpoint, so every photo of them could only ever be
answered with "could not identify". Offering them was a dead end, not a feature.

Every `chemical` entry carries a Bangla transliteration of the active ingredient
(`bnName`) as well as the Latin one. That is not cosmetic: the TTS model silently
deletes Latin-script spans, so a Latin-only product name would vanish from the
spoken advice while leaving a fluent, complete-sounding instruction behind.

Sources are listed per entry so the UI can cite them.
"""

from __future__ import annotations

from typing import TypedDict


class Chemical(TypedDict):
    product: str
    bnName: str
    dose: str
    note: str


class DiseaseEntry(TypedDict):
    id: str
    cropId: str
    cropName: str
    diseaseLabel: str
    bnName: str
    symptoms: list[str]
    organic: list[str]
    chemical: list[Chemical]
    prevention: list[str]
    source: str


class CropEntry(TypedDict):
    id: str
    name: str
    diseases: list[DiseaseEntry]


#: The crops this app supports. Anything else is out of scope, and the API
#: rejects it rather than storing a row it can never give sourced advice for.
SUPPORTED_CROPS: tuple[str, ...] = ("rice", "potato")


KB: list[DiseaseEntry] = [
    # ------------------------------------------------------------------ rice --
    {
        "id": "rice_blast",
        "cropId": "rice",
        "cropName": "ধান",
        "diseaseLabel": "ধানের ব্লাস্ট রোগ",
        "bnName": "ব্লাস্ট",
        "symptoms": [
            "পাতায় ধূসর-সবুজ ডিম্বাকৃতি দাগ, কিনারা বাদামি",
            "দাগের মাঝখানে হালকা রঙ, পরে সেটি সাদাটে হয়ে যায়",
            "শীষের গোড়া পচে শীষ সাদা হয়ে যায়",
        ],
        "organic": [
            "আক্রান্ত পাতা ও শীষ আলাদা করে জমি থেকে সরিয়ে ফেলুন",
            "জমিতে পানি নিষ্কাশনের ব্যবস্থা রাখুন",
            "রোগমুক্ত বীজ ব্যবহার করুন",
        ],
        "chemical": [
            {
                "product": "Tricyclazole",
                "bnName": "ট্রাইসাইক্লাজল",
                "dose": "প্রতি লিটারে ০.৬ গ্রাম",
                "note": "৭ থেকে ১০ দিন পরপর সর্বোচ্চ দুইবার",
            },
            {
                "product": "Isoprothiolane",
                "bnName": "আইসোপ্রোথায়োলেন",
                "dose": "প্রতি লিটারে ১.৫ মিলি",
                "note": "শীষ বেরোনোর আগে প্রয়োগ করুন",
            },
        ],
        "prevention": [
            "অতিরিক্ত নাইট্রোজেন সার দেবেন না",
            "জাতভেদে প্রতিরোধী ধানের বীজ বপন করুন",
            "জমিতে গাছের ঘনত্ব কম রাখুন",
        ],
        "source": "BRRI - আধুনিক ধানের চাষ",
    },
    {
        "id": "rice_bacterial_blight",
        "cropId": "rice",
        "cropName": "ধান",
        "diseaseLabel": "ধানের ব্যাকটেরিয়াল ব্লাইট",
        "bnName": "ব্যাকটেরিয়াল ব্লাইট",
        "symptoms": [
            "পাতার কিনারা থেকে হলুদ হয়ে শুকিয়ে যেতে শুরু করে",
            "শুকনো অংশ ঢেউয়ের মতো দেখায়",
            "ভোরের শিশিরে পাতায় দুধের মতো ফোঁটা দেখা যায়",
        ],
        "organic": [
            "আক্রান্ত গাছের অংশ কেটে সরিয়ে ফেলুন",
            "জমিতে পানি জমতে দেবেন না",
            "বীজ বপনের আগে শুকনো করে নিন",
        ],
        "chemical": [
            {
                "product": "Copper hydroxide",
                "bnName": "কপার হাইড্রোক্সাইড",
                "dose": "প্রতি লিটারে ২ গ্রাম",
                "note": "আক্রান্ত জমিতে ১০ দিন পরপর",
            },
        ],
        "prevention": [
            "প্রতিরোধী জাত নির্বাচন করুন",
            "অতিরিক্ত ইউরিয়া প্রয়োগ কমান",
            "আগাছা পরিষ্কার রাখুন",
        ],
        "source": "BRRI - ধানের রোগ ও প্রতিকার",
    },
    {
        "id": "rice_brown_spot",
        "cropId": "rice",
        "cropName": "ধান",
        "diseaseLabel": "ধানের বাদামি দাগ রোগ",
        "bnName": "বাদামি দাগ",
        "symptoms": [
            "পাতায় ডিম্বাকৃতি বাদামি দাগ, চারপাশে হলদে আভা",
            "দাগের মাঝখানে ধূসর রঙ ও ছোট কালো বিন্দু দেখা যায়",
            "চারার গোড়া পচে চারা শুকিয়ে মরে যায়",
        ],
        "organic": [
            "আক্রান্ত পাতা সংগ্রহ করে জমি থেকে সরিয়ে ফেলুন",
            "সুষম সার প্রয়োগ করুন, পটাশ ও নাইট্রোজেনের ঘাটতি রাখবেন না",
            "রোগমুক্ত বীজ ব্যবহার করুন",
        ],
        "chemical": [
            {
                "product": "Mancozeb",
                "bnName": "ম্যানকোজেব",
                "dose": "প্রতি লিটারে ২ গ্রাম",
                "note": "৭ থেকে ১০ দিন পরপর",
            },
            {
                "product": "Carbendazim",
                "bnName": "কার্বেন্ডাজিম",
                "dose": "প্রতি লিটারে ১ গ্রাম",
                "note": "রোগ বেশি হলে ১০ দিন পরপর",
            },
        ],
        "prevention": [
            "প্রতিরোধী ধানের জাত বপন করুন",
            "বীজ বপনের আগে বীজ শোধন করুন",
            "জমিতে পানি নিষ্কাশনের ব্যবস্থা রাখুন",
        ],
        "source": "BRRI - ধানের রোগ ও প্রতিকার",
    },
    {
        "id": "rice_tungro",
        "cropId": "rice",
        "cropName": "ধান",
        "diseaseLabel": "ধানের টুংরো রোগ",
        "bnName": "টুংরো",
        "symptoms": [
            "পাতায় হলদে-কমলা রঙ ছড়িয়ে পড়ে",
            "গাছ খাটো হয়ে যায় ও কুশি কম হয়",
            "শীষ ছোট হয় ও চিটা বেশি থাকে",
        ],
        "organic": [
            "আক্রান্ত গাছ তুলে জমি থেকে নষ্ট করে ফেলুন",
            "সবুজ পাতা ফড়িং নিয়ন্ত্রণে হলুদ আঠালো ফাঁদ ব্যবহার করুন",
            "আশপাশের জমিতে একই সময়ে ধান রোপণ করুন",
        ],
        "chemical": [
            {
                "product": "Imidacloprid",
                "bnName": "ইমিডাক্লোপ্রিড",
                "dose": "প্রতি লিটারে ০.৩ মিলি",
                "note": "সবুজ পাতা ফড়িং দমনে ৭ দিন পরপর",
            },
        ],
        "prevention": [
            "টুংরো প্রতিরোধী ধানের জাত ব্যবহার করুন",
            "পাতা ফড়িং দেখা দিলে সঙ্গে সঙ্গে ব্যবস্থা নিন",
            "আগাছা পরিষ্কার রাখুন",
        ],
        "source": "BRRI - ধানের রোগ ও প্রতিকার",
    },
    # ---------------------------------------------------------------- potato --
    {
        "id": "potato_late_blight",
        "cropId": "potato",
        "cropName": "আলু",
        "diseaseLabel": "আলুর নাবি ধসা রোগ",
        "bnName": "নাবি ধসা",
        "symptoms": [
            "পাতার কিনারায় পানিতে ভেজা মতো বাদামি দাগ",
            "ভোরে পাতার নিচে সাদা ছত্রাকের আস্তরণ",
            "আক্রান্ত পাতা দ্রুত শুকিয়ে যাচ্ছে",
        ],
        "organic": [
            "আক্রান্ত গাছ তুলে জমি থেকে সরিয়ে ফেলুন",
            "জমিতে পানি জমতে দেবেন না",
            "ফসল আবর্তন করুন",
        ],
        "chemical": [
            {
                "product": "Mancozeb",
                "bnName": "ম্যানকোজেব",
                "dose": "প্রতি লিটারে ২ গ্রাম",
                "note": "৭ দিন পরপর",
            },
            {
                "product": "Metalaxyl + Mancozeb",
                "bnName": "মেটালাক্সিল ও ম্যানকোজেব",
                "dose": "প্রতি লিটারে ২ গ্রাম",
                "note": "রোগ বেশি হলে ৭ দিন পরপর",
            },
        ],
        "prevention": [
            "সুস্থ বীজ আলু ব্যবহার করুন",
            "গাছের মাঝে যথেষ্ট দূরত্ব রাখুন",
            "আগাম জাত হলে আগে রোপণ করুন",
        ],
        "source": "BARI - আলুর রোগ ব্যবস্থাপনা",
    },
    {
        "id": "potato_early_blight",
        "cropId": "potato",
        "cropName": "আলু",
        "diseaseLabel": "আলুর আগাম ধসা রোগ",
        "bnName": "আগাম ধসা",
        "symptoms": [
            "নিচের পুরনো পাতায় বাদামি কিনারাযুক্ত গোল দাগ",
            "দাগের ভেতরে একাধিক রিং দেখা যায়",
            "পাতা হলদে হয়ে শুকিয়ে ঝরে পড়ে",
        ],
        "organic": [
            "আক্রান্ত নিচের পাতা সরিয়ে ফেলুন",
            "ফসল আবর্তন করুন",
        ],
        "chemical": [
            {
                "product": "Chlorothalonil",
                "bnName": "ক্লোরোথ্যালোনিল",
                "dose": "প্রতি লিটারে ২ গ্রাম",
                "note": "১০ দিন পরপর",
            },
        ],
        "prevention": [
            "সুষম সার প্রয়োগ করুন",
            "জমিতে পানি নিষ্কাশন রাখুন",
        ],
        "source": "BARI - আলুর রোগ ব্যবস্থাপনা",
    },
]


#: The checkpoint's `*_healthy` classes. They are a real result — the model
#: looked and saw no disease — so they get their own entries. They are kept out
#: of `KB` on purpose: `crops()` lists diseases, and a healthy plant is not one.
HEALTHY: dict[str, DiseaseEntry] = {
    "rice": {
        "id": "rice_healthy",
        "cropId": "rice",
        "cropName": "ধান",
        "diseaseLabel": "সুস্থ ধানের গাছ",
        "bnName": "সুস্থ",
        "symptoms": [],
        "organic": [
            "নিয়মিত পরিচর্যা চালিয়ে যান",
            "সুষম সার ও সঠিক পানি ব্যবস্থাপনা বজায় রাখুন",
        ],
        # No chemical for a healthy plant: naming one would invite a spray
        # nobody needs, which costs money and puts the farmer at risk.
        "chemical": [],
        "prevention": [
            "নিয়মিত মাঠ পরিদর্শন করুন",
            "রোগের প্রাথমিক লক্ষণ দেখা দিলে সঙ্গে সঙ্গে ব্যবস্থা নিন",
        ],
        "source": "BRRI - আধুনিক ধানের চাষ",
    },
    "potato": {
        "id": "potato_healthy",
        "cropId": "potato",
        "cropName": "আলু",
        "diseaseLabel": "সুস্থ আলুর গাছ",
        "bnName": "সুস্থ",
        "symptoms": [],
        "organic": [
            "নিয়মিত পরিচর্যা চালিয়ে যান",
            "সুষম সার ও সঠিক পানি ব্যবস্থাপনা বজায় রাখুন",
        ],
        "chemical": [],
        "prevention": [
            "নিয়মিত মাঠ পরিদর্শন করুন",
            "রোগের প্রাথমিক লক্ষণ দেখা দিলে সঙ্গে সঙ্গে ব্যবস্থা নিন",
        ],
        "source": "BARI - আলুর রোগ ব্যবস্থাপনা",
    },
}

_DISEASES = {entry["id"]: entry for entry in KB}
_HEALTHY_BY_ID = {entry["id"]: entry for entry in HEALTHY.values()}


def get_disease(disease_id: str) -> DiseaseEntry | None:
    """Look up an entry. Healthy verdicts are addressable here too."""
    return _DISEASES.get(disease_id) or _HEALTHY_BY_ID.get(disease_id)


def get_healthy(crop_id: str) -> DiseaseEntry | None:
    return HEALTHY.get(crop_id)


def diseases_for_crop(crop_id: str) -> list[DiseaseEntry]:
    return [entry for entry in KB if entry["cropId"] == crop_id]


def crops() -> list[CropEntry]:
    seen: list[CropEntry] = []
    for entry in KB:
        if not any(crop["id"] == entry["cropId"] for crop in seen):
            seen.append(
                {
                    "id": entry["cropId"],
                    "name": entry["cropName"],
                    "diseases": diseases_for_crop(entry["cropId"]),
                }
            )
    return seen
