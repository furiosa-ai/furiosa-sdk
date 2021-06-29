class CliError(Exception):
    def __init__(self, message, exit_code=2):
        self.message = message
        self.exit_code = exit_code


class NoCommandException(CliError):
    def __init__(self):
        super().__init__('Need command', 2)


class ApiError(CliError):
    def __init__(self, message, response):
        http_status = response.status_code
        try:
            body = response.json()
            error_code = body['error_code']
            error_message = body['message']
            super().__init__('{} (http_status: {}, error_code: {}, message: {})'
                             .format(message, http_status, error_code, error_message), 4)
        except:
            error_message = response.text
            super().__init__('{} (http_status: {}, message: {})'
                             .format(message, http_status, error_message), 4)

    def __str__(self):
        return "(exit code: {}, message: {})".format(self.exit_code, self.message)

    def __repr__(self):
        return "ApiError(exit code: {}, message: {})".format(self.exit_code, self.message)
