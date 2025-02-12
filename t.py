from numerapi import NumerAPI
import dotenv, os

import ssl
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ssl._create_default_https_context = ssl._create_unverified_context


dotenv.load_dotenv('./.env')

P_ID = os.getenv('NUMERAI_PUBLIC_ID')
S_ID = os.getenv('NUMERAI_SECRET_ID')

# Pass the custom session to NumerAPI
napi = NumerAPI(public_id = P_ID, secret_key = S_ID)
napi.download_dataset("v5.0/train.parquet")
