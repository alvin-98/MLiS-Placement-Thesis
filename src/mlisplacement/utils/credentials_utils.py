import os

def set_mlflow_credentials(credentials_file: str = None):
    """Function to read the mlflow credentials from a file and set the environment variables.
    [mlflow]
    mlflow_tracking_username = username
    mlflow_tracking_password = password
    mlflow_tracking_uri = uri

    Default file location is ~/.mlflow/credentials, as set by the mlis package.

    """
    DEFAULT_CRED_FILE = os.path.abspath(os.path.join(os.path.expanduser("~"), ".mlflow", "credentials"))

    credentials_file = credentials_file or DEFAULT_CRED_FILE

    # read the credentials and set the environment variables
    with open(credentials_file, "r") as f:
        credentials = f.read().split("\n")
        os.environ["MLFLOW_TRACKING_USERNAME"] = credentials[1].split("=")[1].strip()
        os.environ["MLFLOW_TRACKING_PASSWORD"] = credentials[2].split("=")[1].strip()
        os.environ["MLFLOW_TRACKING_URI"] = credentials[3].split("=")[1].strip()

def set_hf_token(credentials_file: str = None):
    """Function to read the mlflow token from a file and set the environment variable.
    [mlflow]
    mlflow_tracking_username = username
    mlflow_tracking_password = password
    mlflow_tracking_uri = uri
    hf_token = token

    Default file location is ~/.mlflow/credentials, as set by the mlis package.

    """
    DEFAULT_CRED_FILE = os.path.abspath(os.path.join(os.path.expanduser("~"), ".mlflow", "credentials"))

    credentials_file = credentials_file or DEFAULT_CRED_FILE

    # read the credentials and set the environment variables
    # if line not found, do not set
    with open(credentials_file, "r") as f:
        credentials = f.read().split("\n")
        try:
            os.environ["HF_TOKEN"] = credentials[4].split("=")[1].strip()
        except IndexError:
            pass