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

# Column names expected in every CSV the service trains or evaluates on.
DATA_COLUMNS = ["text", "subject", "topic", "difficulty"]
