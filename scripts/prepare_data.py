import pandas as pd
import requests
from pathlib import Path

URLS = [
    'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv',
    'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv',
]
BASE_DIR = Path('data')


def download_data(url, filename):
    """
    Download data from url and save it to filename
    """
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)


def main():
    """
    Main function
    """
    dataset_path = BASE_DIR / 'dataset.csv'
    for url in URLS:
        filename = BASE_DIR / url.split('/')[-1]
        if not filename.exists():
            download_data(url, filename)
        else:
            print(f'{filename} already exists')

    train_df = pd.read_csv(BASE_DIR / 'train.csv',
                           names=['category', 'title', 'description'])
    test_df = pd.read_csv(BASE_DIR / 'test.csv',
                          names=['category', 'title', 'description'])
    dataset = pd.concat([train_df, test_df])
    dataset.to_csv(dataset_path, index=False)
    dataset['description'].values.tofile(BASE_DIR / 'content.txt', sep='\n')


if __name__ == '__main__':
    main()
