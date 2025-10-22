import sys,os
from agents import Agent, Runner, ModelSettings
import asyncio

os.environ["OPENAI_API_KEY"]=os.environ.get("OPENAI_PROJ_API_KEY")
model_settings = ModelSettings(
    temperature=0.0,   
    top_p=1.0,       
)

code_agent = Agent(
    name="Code Generator",
    instructions="Generate complete standalone code based on user instructions. Respond only with code without markdown (```).",
    model="gpt-3.5-turbo-0125",
    model_settings=model_settings
)

async def main():
    prompt = sys.argv[1]
    result = await Runner.run(code_agent, prompt)
    print(result.final_output)

if __name__ == "__main__":
    asyncio.run(main())

