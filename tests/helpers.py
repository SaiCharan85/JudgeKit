from judgekit import Answer, JudgeResponse


def respond(**answers: str) -> JudgeResponse:
    return JudgeResponse(answers=[Answer(id=k, answer=v) for k, v in answers.items()])  # type: ignore[arg-type]


ALL_YES = respond(crit="yes", maj="yes", mino="yes")
