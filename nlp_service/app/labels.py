"""Label sets for the three classification tasks.

These mirror what the existing GovPrep-AI app already uses, so predictions
slot straight into the current UI and database fields.

* SUBJECTS     -> copied from ``frontend/src/pages/UploadPDF.js`` (the subject a
                  user picks when uploading a PDF).
* DIFFICULTIES -> the values the Groq question-generation prompt in
                  ``backend/server.js`` asks for.
* TOPICS       -> NOT defined yet. The app currently stores free-text topics
                  written by the LLM (mixed English/Hindi), which cannot be used
                  as class labels. A controlled topic taxonomy will be defined
                  in the data phase, once the public dataset is chosen.

Which subjects are actually trainable (e.g. "Other" and "Current Affairs" have
no clean public-dataset equivalent) is decided in the data phase, based on the
class counts that really exist.
"""

SUBJECTS = [
    "History",
    "Geography",
    "Mathematics",
    "Science",
    "Reasoning",
    "General Awareness",
    "English",
    "Economics",
    "Polity",
    "Current Affairs",
    "Other",
]

DIFFICULTIES = ["easy", "medium", "hard"]

# Optional provenance column. When present it is carried through every split and is the
# grouping key for source-grouped splitting (see data.stratified_split(group_col=...)).
OPTIONAL_COLUMNS = ["source"]

# Names that public datasets and exam syllabi use for the subjects above. Used ONLY by
# training.build_dataset to translate a raw label to the app's label. A raw label that is not
# here (and is not already a SUBJECTS value) is reported and dropped; it is never guessed.
# "Computer Knowledge" is deliberately absent: the app has no such subject, so adding it is a
# project decision (it would also have to be added to SUBJECTS and the frontend dropdown).
SUBJECT_ALIASES = {
    "indian polity": "Polity",
    "polity": "Polity",
    "political science": "Polity",
    "civics": "Polity",
    "indian constitution": "Polity",
    "general science": "Science",
    "science": "Science",
    "physics": "Science",
    "chemistry": "Science",
    "biology": "Science",
    "maths": "Mathematics",
    "math": "Mathematics",
    "mathematics": "Mathematics",
    "quantitative aptitude": "Mathematics",
    "logical reasoning": "Reasoning",
    "reasoning": "Reasoning",
    "indian history": "History",
    "history": "History",
    "indian geography": "Geography",
    "geography": "Geography",
    "economy": "Economics",
    "indian economy": "Economics",
    "economics": "Economics",
    "general knowledge": "General Awareness",
    "gk": "General Awareness",
    "general awareness": "General Awareness",
    "english language": "English",
    "english": "English",
    "current affairs": "Current Affairs",
}

# Column names expected in every CSV the service trains or evaluates on.
DATA_COLUMNS = ["text", "subject", "topic", "difficulty"]
