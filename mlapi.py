from fastapi import FastAPI
import pickle
import gradio as gr

CUSTOM_PATH = "/gradio"

with open('sentalyzer.joblib' , 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

@app.get('/')
async def root():
    return {'example':'sample'}


@app.get("/predict")
async def scoring_endpoint(INPUT_TEXT: str):
    yhat = model.polarity_scores(INPUT_TEXT)['compound']
    
    if yhat > 0:
        value = 'Positive'
    if yhat < 0:
        value = 'Negative'
    if yhat == 0:
        value = 'Neutral'

    return f"This is {value} sentiment"

io = gr.Interface(scoring_endpoint,
                  inputs=gr.inputs.Textbox(lines=5, placeholder="Enter your comment here.."),
                  outputs="text",
                  description="Enter a comment to be scored as a Positive, Neutral or Negative statement",
                  title="Sentiment Analyzer",
                  theme="dark")
# io = gr.Interface(fn=score_text,
#                     inputs=gr.inputs.Textbox(lines=5, placeholder='Type comment'),
#                     outputs='text')

gradio_app = gr.mount_gradio_app(app, io, path=CUSTOM_PATH) 