from src.data.preprocessing import clean_text


def test_clean_text_removes_html():

    text = "Great movie! <br /><br /> Must watch."

    cleaned = clean_text(text)

    assert cleaned == "Great movie! Must watch."


def test_clean_text_removes_extra_spaces():

    text = "Great     movie!"

    cleaned = clean_text(text)

    assert cleaned == "Great movie!"


def test_clean_text_empty_string():

    text = ""

    cleaned = clean_text(text)

    assert cleaned == ""