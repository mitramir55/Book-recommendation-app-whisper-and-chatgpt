from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
import semantic_kernel as sk
from pathlib import Path
import os
import openai
from dotenv import load_dotenv

# Reading the transcript
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
with open(Path.cwd() / "src" / "03_06.mp3", "rb") as audio_file:
    transcript = openai.Audio.transcribe("whisper-1", audio_file).text

kernel = sk.Kernel()

# Prepare OpenAI service using credentials stored in the `.env` file
api_key, org_id = sk.openai_settings_from_dot_env()
kernel.add_text_completion_service("dv", OpenAIChatCompletion("gpt-4", api_key, org_id))



prompt_prefix = """
You are a librarian. Read the user description and give a book recommendation. say why you chose that book.
"""
example_prompt = """
here is an example input:
I like fantasy novels and like to get enthusiastic when reading.
response: 
I recommend Harry Potter as it has all the elements you're looking for.
It's full of character development and has a lot of magic and fantasy. I hope you enjoy!
"""
prompt = f"{prompt_prefix} " + example_prompt + "\n The input = {{$input}} " + "output:\n"
kernel.add_text_completion_service("dv", OpenAIChatCompletion("gpt-4", api_key, org_id))
creator = kernel.create_semantic_function(prompt)

print(creator(transcript))
