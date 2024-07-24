import os

def get_next_available_filename(base_filename):
    base_name, extension = os.path.splitext(base_filename)
    counter = 1
    new_filename = base_filename

    while os.path.exists(new_filename):
        new_filename = f"{base_name}({counter}){extension}"
        counter += 1

    return new_filename