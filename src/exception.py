import sys

## function to generate a detailed error message
def error_message_detail(error, error_detail: sys) -> str:
    
    '''returns a formatted error message with file name, line number, and error description.'''
    
    _, _, exc_tb = error_detail.exc_info()    ##get traceback object
    file_name = exc_tb.tb_frame.f_code.co_filename  ##get file name
    error_message = (
        f"Error occurred in python script name [{file_name}]"
        f"line number [{exc_tb.tb_lineno}]"
        f"error message [{str(error)}]"

    )
    return error_message


## custom exception class
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
    


'''Normally, if Python throws an exception, you just see a traceback in the console.
  This code’s purpose is to catch the exception and print a custom, formatted message that includes:

   ->The file name where the error occurred
   -> The exact line number
   ->The error message text
This is very useful for debugging ML projects where you want clean logs instead of messy tracebacks.'''
