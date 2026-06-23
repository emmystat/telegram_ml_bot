import logging
import os

from aiogram import Bot, Dispatcher, Router, types
from aiogram.filters import Command
import numpy as np
from aiogram.types import Message
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
import pandas as pd
from .model import train_load

model = train_load()
logging.basicConfig(level=logging.INFO)

TELEGRAM_TOKEN = os.getenv("TOKEN")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
CODE = os.getenv("CODE")
WEBHOOK_PATH = f"/webhook/{CODE}" if CODE else "/webhook/undefined"

if not TELEGRAM_TOKEN:
    raise RuntimeError("Environment variable TOKEN is required for Telegram bot integration.")


bot = Bot(token=TELEGRAM_TOKEN)
dispatcher = Dispatcher()
router = Router()


def format_prediction(data: dict) -> str:
    prob = model.predict_proba(pd.DataFrame([data]))[0,1]

    return f"Prediction: {np.where(prob>.5,"default","paid off")} \nprobability: {prob:.2f%}"

class PredictionState(StatesGroup):
    payment_inc_ratio = State()
    dti = State()


@router.message(Command("predict"))
async def start_prediction(message: Message, state: FSMContext):
    await state.set_state(PredictionState.payment_inc_ratio)
    await message.answer("Enter payment_inc_ratio: ")

@router.message(PredictionState.payment_inc_ratio)
async def start_prediction(message: Message, state: FSMContext):
    try:
        payment_inc_ratio = float(message.text)
        await state.update_data(payment_inc_ratio=payment_inc_ratio)
        await state.set_state(PredictionState.dti)
        await message.answer("Enter dti: ")
    except ValueError:
        await message.answer("Please enter a valid number: ")

@router.message(PredictionState.dti)
async def start_prediction(message: Message, state: FSMContext):
    try:
        dti = float(message.text)
        data = await state.get_data()
        payload = {
            'payment_inc_ratio': data['payment_inc_ratio'],
            'dti': dti,
        }
        
        await message.answer(
            format_prediction(payload)
        )
        await state.clear()
    except ValueError:
        await message.answer("Please enter a valid number: ")
    except Exception:
        await message.answer("Unable to generate prediction: ")
        await state.clear()

@router.message(Command("predict2"))
async def cmd_predict(message: types.Message):
    parts = message.text.split()
    if len(parts) != 3:
        await message.reply("Usage: /predict2 <payment_inc_ratio> <dti>")
        return

    try:
        payment_inc_ratio = float(parts[1])
        dti = float(parts[2])
    except ValueError:
        await message.reply("Both payment_inc_ratio and dti must be numbers.")
        return

    await message.answer(format_prediction({
        'payment_inc_ratio':payment_inc_ratio, 
        'dti': dti}))


dispatcher.include_router(router)
