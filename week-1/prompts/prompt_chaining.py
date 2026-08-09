SOURCE_TEXT = (
    "Acme Robotics today announced the general availability of its new warehouse picking robot, Falcon-2. "
    "Falcon-2 can identify and pick over 1,200 SKUs per hour, a 40% improvement over the previous generation, "
    "while using 25% less power thanks to a redesigned battery management system. The company says "
    "early customers have cut picking errors by more than half. Falcon-2 ships starting next quarter, "
    "with pricing available on request."
)


def summary_prompt(text):
    return f"Summarize this in 2 sentences:\n\n{text}"


def tweet_prompt(summary):
    return f"Turn this summary into a punchy tweet (under 280 characters) with 2 relevant hashtags:\n\n{summary}"
