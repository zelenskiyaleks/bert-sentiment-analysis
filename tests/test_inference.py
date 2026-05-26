from src.inference.predict import predict_sentiment


def test_predict_sentiment_returns_valid_output():

    prediction, confidence = predict_sentiment(
        "This movie was normal"
    )

    assert prediction in [
        "Positive",
        "Negative"
    ]

    assert 0 <= confidence <= 1


def test_negative_review_prediction():

    prediction, _ = predict_sentiment(
        "Terrible movie."
    )

    assert prediction == "Negative"


def test_positive_review_prediction():

    prediction, _ = predict_sentiment(
        "Amazing movie!"
    )

    assert prediction == "Positive"