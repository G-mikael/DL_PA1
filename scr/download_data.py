import os
import zipfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
ROOT_DIR = CURRENT_FILE.parent.parent
os.chdir(ROOT_DIR)

os.environ['KAGGLE_USERNAME'] = os.getenv("KAGGLE_USERNAME")
os.environ['KAGGLE_API_TOKEN'] = os.getenv("KAGGLE_API_TOKEN")

DATA_DIR = Path("data/raw")
STAGE1_TRAIN_DIR = DATA_DIR / "stage1_train"
ZIP_PATH = DATA_DIR / "stage1_train.zip"

def download_and_extract_dsb2018():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    if STAGE1_TRAIN_DIR.exists() and any(STAGE1_TRAIN_DIR.iterdir()):
        print("Dataset stage1_train já existe em:", STAGE1_TRAIN_DIR)
        return

    print("Iniciando download do Data Science Bowl 2018...")
    
    from kaggle.api.kaggle_api_extended import KaggleApi
    
    api = KaggleApi()
    api.authenticate()

    api.competition_download_file(
        competition="data-science-bowl-2018",
        file_name="stage1_train.zip",
        path=DATA_DIR
    )
    print("Extraindo arquivos em:", STAGE1_TRAIN_DIR)
    STAGE1_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(STAGE1_TRAIN_DIR)

    if ZIP_PATH.exists():
        os.remove(ZIP_PATH)
        
    print("Download e extração concluídos com sucesso!")

if __name__ == "__main__":
    download_and_extract_dsb2018()