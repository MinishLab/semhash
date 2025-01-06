from dataclasses import dataclass


@dataclass
class DatasetRecord:
    """Dataset record."""

    name: str
    text_name: str | None = None
    label_name: str | None = None
    sub_directory: str = ""
    columns: list[str] | None = None
    split_one: str = "train"
    split_two: str = "test"


DATASET_DICT: dict[str, DatasetRecord] = {
    "bbc": DatasetRecord(name="SetFit/bbc-news", text_name="text", label_name="label_text"),
    "senteval_cr": DatasetRecord(name="SetFit/SentEval-CR", text_name="text", label_name="label_text"),
    "tweet_sentiment_extraction": DatasetRecord(
        name="SetFit/tweet_sentiment_extraction", text_name="text", label_name="label_text"
    ),
    "emotion": DatasetRecord(name="SetFit/emotion", text_name="text", label_name="label_text"),
    "amazon_counterfactual": DatasetRecord(
        name="SetFit/amazon_counterfactual_en", text_name="text", label_name="label_text"
    ),
    "ag_news": DatasetRecord(name="SetFit/ag_news", text_name="text", label_name="label_text"),
    "enron_spam": DatasetRecord(name="SetFit/enron_spam", text_name="text", label_name="label_text"),
    "subj": DatasetRecord(name="SetFit/subj", text_name="text", label_name="label_text"),
    "sst5": DatasetRecord(name="SetFit/sst5", text_name="text", label_name="label_text"),
    "20_newgroups": DatasetRecord(name="SetFit/20_newsgroups", text_name="text", label_name="label_text"),
    "hatespeech_offensive": DatasetRecord(name="SetFit/hate_speech_offensive", text_name="text", label_name="label"),
    "ade": DatasetRecord(name="SetFit/ade_corpus_v2_classification", text_name="text", label_name="label"),
    "imdb": DatasetRecord(name="SetFit/imdb", text_name="text", label_name="label"),
    "massive_scenario": DatasetRecord(
        name="SetFit/amazon_massive_scenario_en-US", text_name="text", label_name="label"
    ),
    "student": DatasetRecord(name="SetFit/student-question-categories", text_name="text", label_name="label"),
    "squad_v2": DatasetRecord(name="squad_v2", columns=["question", "context", "answers"], split_two="validation"),
    "wikitext": DatasetRecord(
        name="Salesforce/wikitext", text_name="text", label_name="text", sub_directory="wikitext-103-raw-v1"
    ),
}
