class TalkyError(Exception):
    "Talky Package에서 사용되는 global exception"
    pass

class PromptMissingPlaceholderError(TalkyError):
    """지정된 placeholder가 prompt에 존재하지 않는 경우 발생되는 에러"""

class PromptMissingInputError(TalkyError):
    """전달되어야 하는 input parameter가 전달되지 않은 경우"""