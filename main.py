import os

# replacement strings
WINDOWS_LINE_ENDING = b'\r\n'
UNIX_LINE_ENDING = b'\n'


for dirpath, dnames, fnames in os.walk("./"):
    for f in fnames:
        if f.endswith(".py") and '.venv' not in dirpath:
            file_path = os.path.join(dirpath, f)

            with open(file_path, 'rb') as open_file:
                content = open_file.read()

            content = content.replace(WINDOWS_LINE_ENDING, UNIX_LINE_ENDING)

            with open(file_path, 'wb') as open_file:
                open_file.write(content)
