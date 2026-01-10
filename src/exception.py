import sys

def error_message_detail(error, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info() # exc_tb -> exception traceback
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_num = exc_tb.tb_lineno
    return (
        f"Error in [{file_name}] at line [{line_num}] "
        f"with message [{error}]"
    )

class CustomException(Exception):
    def __init__(self, error_message: str, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail
        )

    def __str__(self):
        return self.error_message
