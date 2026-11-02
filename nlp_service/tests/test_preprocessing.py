import pytest

from app.preprocessing import NEGATIONS, PreprocessConfig, TextPreprocessor


def test_cleaning_removes_noise():
    pre = TextPreprocessor(PreprocessConfig(normalizer="none"))
    out = pre("Visit https://x.com/a or mail me@site.org - Page 12 of 70 Mauryan Empire")
    assert out == ["visit", "mail", "mauryan", "empire"]


def test_lowercase_and_digits_kept():
    out = TextPreprocessor(PreprocessConfig(normalizer="none"))("Battle of PANIPAT 1526")
    assert "panipat" in out and "1526" in out and "battle" in out


def test_stopwords_removed_but_negations_kept():
    pre = TextPreprocessor(PreprocessConfig(normalizer="none"))
    out = pre("Which of the following is NOT a fundamental right")
    assert "not" in out
    assert "the" not in out and "of" not in out and "which" not in out
    assert "not" in NEGATIONS


def test_negations_can_be_dropped():
    pre = TextPreprocessor(PreprocessConfig(normalizer="none", keep_negations=False))
    assert "not" not in pre("this is not correct")


def test_stopword_removal_can_be_disabled():
    pre = TextPreprocessor(PreprocessConfig(normalizer="none", remove_stopwords=False))
    assert "the" in pre("the empire")


def test_contractions_and_possessives():
    pre = TextPreprocessor(PreprocessConfig(normalizer="none"))
    out = pre("Ashoka's edicts didn't mention it")
    assert "ashoka" in out and "edicts" in out and "not" in out
    assert "s" not in out and "t" not in out


def test_stemming_default():
    out = TextPreprocessor()("empires battles fought fundamental")
    assert "empir" in out and "battl" in out and "fundament" in out


def test_devanagari_preserved_not_stemmed_or_split():
    out = TextPreprocessor()("भारत की राजधानी क्या है? Babur won.")
    assert "भारत" in out and "राजधानी" in out  # matras intact
    assert "babur" in out


def test_empty_and_non_string_inputs():
    pre = TextPreprocessor()
    assert pre(None) == [] and pre("") == [] and pre("   ") == [] and pre(float("nan")) == []
    assert pre("the of and") == []


def test_deterministic_and_config_round_trip():
    cfg = PreprocessConfig(normalizer="none", min_token_len=3)
    again = PreprocessConfig.from_dict(cfg.to_dict())
    assert again == cfg
    text = "The Indus Valley Civilisation flourished around 2500 BCE"
    assert TextPreprocessor(cfg)(text) == TextPreprocessor(again)(text)
    assert TextPreprocessor(cfg).to_string(text) == " ".join(TextPreprocessor(cfg)(text))


def test_min_token_len():
    out = TextPreprocessor(PreprocessConfig(normalizer="none", min_token_len=4))("ice age 1526 war")
    assert out == ["1526"]


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        PreprocessConfig(normalizer="bogus")
    with pytest.raises(ValueError):
        PreprocessConfig(min_token_len=0)


def test_explain_has_all_stages():
    ex = TextPreprocessor().explain("Which is NOT correct? Empires fell.")
    assert set(ex) >= {"config", "cleaned", "tokens", "after_stopword_removal",
                       "final_tokens", "removed_stopwords"}
    assert "not" in ex["final_tokens"] and "which" in ex["removed_stopwords"]


def test_lemma_mode_or_clear_error():
    pre = TextPreprocessor(PreprocessConfig(normalizer="lemma"))
    try:
        out = pre("empires battles")
    except LookupError as exc:
        assert "nltk.downloader wordnet" in str(exc)  # clear message when data missing
    else:
        assert "empire" in out and "battle" in out
