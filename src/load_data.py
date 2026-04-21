import kagglehub
from kagglehub import KaggleDatasetAdapter
import os

def download_data():
    print("Descargando dataset de fraude bancario desde Kaggle...")
    
    df = kagglehub.load_dataset(
        KaggleDatasetAdapter.PANDAS,
        "mlg-ulb/creditcardfraud",
        "creditcard.csv",
    )
    
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/creditcard.csv", index=False)
    
    print(f"Dataset guardado en data/raw/creditcard.csv")
    print(f"Shape: {df.shape}")
    print(f"Primeros 5 registros:\n{df.head()}")

if __name__ == "__main__":
    download_data()