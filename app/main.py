from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from .model import train_load
import numpy as np
import pandas as pd
import logging
from aiogram import types
from .bot import WEBHOOK_URL, WEBHOOK_PATH, bot,dispatcher

app = FastAPI()
model = train_load()
# Fastapi Routes


@app.on_event("startup")
async def startup_event():
    if WEBHOOK_URL:
        full_webhook_url = WEBHOOK_URL.rstrip("/") + WEBHOOK_PATH
        logging.info(f"Setting Telegram webhook to: {full_webhook_url}")
        await bot.delete_webhook(drop_pending_updates=True)
        await bot.set_webhook(full_webhook_url)
    else:
        logging.warning(
            "WEBHOOK_URL not set. Telegram webhook will not be configured automatically. "
            "Set WEBHOOK_URL for deployment environments."
        )


@app.on_event("shutdown")
async def shutdown_event():
    await bot.delete_webhook()
    await bot.session.close()


@app.get("/")
async def root():
    return {"status": "ok", "webhook_path": WEBHOOK_PATH}


@app.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request):
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc

    update = types.Update(**payload)
    await dispatcher.feed_update(bot, update)
    return {"ok": True}

class PredictionRequest(BaseModel):
    payment_inc_ratio: float
    dti: float

@app.post('/predict')
async def predict(data: PredictionRequest):
    prob = model.predict_proba(pd.DataFrame(
        [dict(payment_inc_ratio=data.payment_inc_ratio,
              dti=data.dti)]
    ))[0,1]
    return {
        'probability': prob,
        'actual': str(np.where(prob>.5,"default","paid off"))
    }
