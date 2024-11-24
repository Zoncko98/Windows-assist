import os
import subprocess

def open_application(app_name):
    """
    Open applications in Ubuntu.
    
    :param app_name: The name of the application to open (e.g., "calculator").
    :return: A message indicating whether the application was successfully opened or not.
    """
    app_mapping = {
        "calculator": "gnome-calculator",
        "firefox": "firefox",
        "terminal": "gnome-terminal",
        "text editor": "gedit",
        "file manager": "nautilus",
    }
    command = app_mapping.get(app_name.lower())
    if command:
        try:
            subprocess.Popen([command])
            return f"{app_name.capitalize()} is now open."
        except Exception as e:
            return f"Failed to open {app_name}: {str(e)}"
    return f"Application '{app_name}' is not recognized."

def close_application(app_name):
    """
    Close an application in Ubuntu.
    
    :param app_name: The name of the application to close (e.g., "firefox").
    :return: A message indicating whether the application was successfully closed or not.
    """
    try:
        subprocess.run(["pkill", app_name], check=True)
        return f"{app_name.capitalize()} has been closed."
    except subprocess.CalledProcessError:
        return f"No process named '{app_name}' was found to close."
    except Exception as e:
        return f"Failed to close {app_name}: {str(e)}"

def list_directory(path="."):
    """
    List files in the specified directory.
    
    :param path: The directory path to list files from.
    :return: A string with the list of files or an error message.
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Directory '{path}' does not exist.")
        if not os.access(path, os.R_OK):
            raise PermissionError(f"No permission to access directory: {path}")
        files = os.listdir(path)
        return "\n".join(files) if files else "The directory is empty."
    except FileNotFoundError as e:
        return str(e)
    except PermissionError as e:
        return str(e)
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"