"""Curated crop advisory knowledge base.

This is the only component allowed to state a fact: a disease name, a symptom, a
medicine, a dose. The LLM rewrites prose around these entries and is validated
against them; it never invents one.

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
    # ---------------------------------------------------------------- tomato --
    {
        "id": "tomato_early_blight",
        "cropId": "tomato",
        "cropName": "টমেটো",
        "diseaseLabel": "টমেটোর আগাম ধসা রোগ",
        "bnName": "আগাম ধসা",
        "symptoms": [
            "নিচের পাতায় ছোট বাদামি দাগ",
            "দাগ ধীরে ধীরে বড় হয়ে পাতাকে হলদে করে দেয়",
            "পাতা ঝরে পড়ে গাছ খালি হয়ে যায়",
        ],
        "organic": [
            "আক্রান্ত নিচের পাতা সরিয়ে ফেলুন",
            "গাছের গোড়ায় খড় বিছিয়ে দিন",
            "পাতায় পানি জমতে দেবেন না",
        ],
        "chemical": [
            {
                "product": "Mancozeb",
                "bnName": "ম্যানকোজেব",
                "dose": "প্রতি লিটারে ২ গ্রাম",
                "note": "১০ দিন পরপর",
            },
        ],
        "prevention": [
            "গাছের মাঝে যথেষ্ট দূরত্ব রাখুন",
            "ফসল আবর্তন করুন",
        ],
        "source": "BARI - টমেটোর রোগ ব্যবস্থাপনা",
    },
    {
        "id": "tomato_late_blight",
        "cropId": "tomato",
        "cropName": "টমেটো",
        "diseaseLabel": "টমেটোর নাবি ধসা রোগ",
        "bnName": "নাবি ধসা",
        "symptoms": [
            "পাতায় বড় বাদামি দাগ, কিনারা পানিতে ভেজা",
            "ফলে বাদামি শক্ত দাগ",
            "সকালে পাতার নিচে সাদা আস্তরণ",
        ],
        "organic": [
            "আক্রান্ত পাতা ও ফল সরিয়ে ফেলুন",
            "জমিতে বাতাস চলাচল বাড়ান",
        ],
        "chemical": [
            {
                "product": "Metalaxyl + Mancozeb",
                "bnName": "মেটালাক্সিল ও ম্যানকোজেব",
                "dose": "প্রতি লিটারে ২ গ্রাম",
                "note": "৭ দিন পরপর",
            },
        ],
        "prevention": [
            "বীজ শোধন করে বপন করুন",
            "জমিতে পানি জমতে দেবেন না",
        ],
        "source": "BARI - টমেটোর রোগ ব্যবস্থাপনা",
    },
    # ----------------------------------------------------------------- chili --
    {
        "id": "chili_leaf_curl",
        "cropId": "chili",
        "cropName": "মরিচ",
        "diseaseLabel": "মরিচের পাতা কোঁকড়ানো রোগ",
        "bnName": "পাতা কোঁকড়ানো",
        "symptoms": [
            "নতুন পাতা কুঁকড়ে যায় ও ছোট হয়ে যায়",
            "গাছের বৃদ্ধি থেমে যায়",
            "ফল কম ধরে ও ফলের আকার ছোট হয়",
        ],
        "organic": [
            "আক্রান্ত গাছ তুলে ফেলুন",
            "সাদা মাছি নিয়ন্ত্রণে হলুদ আঠালো ফাঁদ ব্যবহার করুন",
            "আগাছা পরিষ্কার রাখুন",
        ],
        "chemical": [
            {
                "product": "Imidacloprid",
                "bnName": "ইমিডাক্লোপ্রিড",
                "dose": "প্রতি লিটারে ০.৩ মিলি",
                "note": "সাদা মাছি দমনে ৭ দিন পরপর",
            },
        ],
        "prevention": [
            "রোগমুক্ত চারা রোপণ করুন",
            "পাশে অন্য ফসল চাষ করুন",
            "সাদা মাছি দেখা দিলে সঙ্গে সঙ্গে ব্যবস্থা নিন",
        ],
        "source": "BARI - মরিচের রোগ ও পোকা ব্যবস্থাপনা",
    },
    {
        "id": "chili_anthracnose",
        "cropId": "chili",
        "cropName": "মরিচ",
        "diseaseLabel": "মরিচের অ্যানথ্রাকনোজ রোগ",
        "bnName": "অ্যানথ্রাকনোজ",
        "symptoms": [
            "ফলে গোল বাদামি দাগ, মাঝখানে গাঢ় রিং",
            "দাগ বড় হয়ে ফল পচে যায়",
            "আক্রান্ত ফল গাছে ঝুলে থাকে",
        ],
        "organic": [
            "আক্রান্ত ফল তুলে ফেলুন",
            "ফসল আবর্তন করুন",
        ],
        "chemical": [
            {
                "product": "Carbendazim",
                "bnName": "কার্বেন্ডাজিম",
                "dose": "প্রতি লিটারে ১ গ্রাম",
                "note": "১০ দিন পরপর",
            },
        ],
        "prevention": [
            "রোগমুক্ত বীজ ব্যবহার করুন",
            "ফল সংগ্রহের পর জমি পরিষ্কার করুন",
        ],
        "source": "BARI - মরিচের রোগ ও পোকা ব্যবস্থাপনা",
    },
]

_DISEASES = {entry["id"]: entry for entry in KB}


def get_disease(disease_id: str) -> DiseaseEntry | None:
    return _DISEASES.get(disease_id)


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
