import os
from openai import OpenAI

client = OpenAI(
    organization = os.getenv("OPENAI_ORGANIZATION_KEY"),
    api_key = os.getenv("OPENAI_API_KEY"),                                                                                     
)


client.batches.cancel("batch_name")
print('done')