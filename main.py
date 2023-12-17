
from fauna import fql
from fauna.client import Client
from fauna.encoding import QuerySuccess
from fauna.errors import FaunaException
import os
import csv
from openai import OpenAI
import pinecone
from langchain.embeddings.llamacpp import LlamaCppEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma


fauna_api_tok = 'fnacapi_omd2ZXJzaW9uAWdwYXlsb2FkWFiiYmlkcjM4NDE4MjI3MjcwMTQ5NzkzN2ZzZWNyZXR4OGY2cU4xdTRRb256YkxBOXhaSXVzbzdSOFdUN21WYUxTTHgwWXlJMDA1dFR2dEtHSHRPaW5iZz09'


os.environ['FAUNA_SECRET'] = "fnAFVOeu_9AAzGI8mQP0eqt5I8mkfUHJCLBsyxhS"
client = Client()
q1 = fql("""
  User.create({ 
    name: "Shadid",
    age: 29
  })
""")

csv_file_path = "places.csv"

with open(csv_file_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      query = fql("Place.create(${x})", x=row)
      res: QuerySuccess = client.query(query)



os.environ['OPENAI_API_KEY'] = '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'

client = OpenAI(organization= "org-yHdbKL8wyLocilOUoSgLvyut")


pinecone.init(api_key="13d845cb-6abf-4a47-9d62-d01f54b7a886", environment="gcp-starter")

index = pinecone.Index("places")

output = []

with open('places.csv', newline='') as csvfile:
  reader = csv.DictReader(csvfile)
  for i, row in enumerate(reader):
    activities = row['activities']
    generate_embeddings = client.embeddings.create(input=activities, model="text-embedding-ada-002")
    emb = generate_embeddings['data'][0]['embedding']
    output.append((str(i) + "", emb, { 'state': row['state'], 'name': row['name'] }))

# output
index.upsert(output)