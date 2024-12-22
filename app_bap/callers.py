import multiprocessing
import requests
import json
import aiohttp
import asyncio
import pandas as pd

from sklearn.linear_model import LinearRegression


SEASONS = {"Зима": "winter", "Весна": "spring", "Осень": "autumn", "Лето": "summer"}


def get_weather_data(city, df):
    df = df[df["city"] == city].copy()

    min_temperature = df["temperature"].min()
    max_temperature = df["temperature"].max()
    mean_temperature = df["temperature"].mean()

    anomalies = df.copy()
    anomalies["moving_average"] = anomalies["temperature"].rolling(window=30, min_periods=1).mean()
    anomalies["moving_std"] = anomalies["temperature"].rolling(window=30, min_periods=1).std()
    anomalies["is_anomaly"] = anomalies.apply(
        lambda column: 
            (column["temperature"] >= column["moving_average"] + 2 * column["moving_std"]) |\
            (column["temperature"] <= column["moving_average"] - 2 * column["moving_std"]),
        axis=1
    )
    anomalies = anomalies[["timestamp", "temperature", "is_anomaly"]]

    season_profile = df.copy()
    season_profile = df.groupby("season")["temperature"].agg(average="mean", std="std")
    
    trend = df.copy()
    
    trend["timestamp_ordinal"] = pd.to_datetime(trend["timestamp"])
    trend["timestamp_ordinal"] = trend["timestamp_ordinal"].map(pd.Timestamp.toordinal)

    x = trend[["timestamp_ordinal"]]
    y = trend[["temperature"]]
    
    linreg = LinearRegression()
    linreg.fit(x, y)
    trend["trend"] = linreg.predict(x)
    
    trend = trend[["timestamp", "trend"]]

    return {
        city: [
            mean_temperature, 
            min_temperature, 
            max_temperature,
            season_profile,
            trend,
            anomalies
        ]
    }

def collect_weather_data_multiprocess(cities, df):
    temp_weather_data = []
    weather_data = {}

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        args = [(city, df) for city in cities]
        temp_weather_data = pool.starmap(get_weather_data, args)
    
    for temp_weather_object in temp_weather_data:
        weather_data.update(temp_weather_object)

    return weather_data

async def async_get_temperatures(city, api_key):
    base_url = "https://api.openweathermap.org/data/2.5/weather?"
    params = {
        'q': city,
        "appid": api_key,
        "units": "metric"
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url=base_url, params=params) as response:
            temperature = await response.text()
            temperature = json.loads(temperature)["main"]["temp"]

    return {city: temperature}

async def collect_temperatures(cities, api_key):
    temperature = {}
    tasks = [async_get_temperatures(city=city, api_key=api_key) for city in cities]
    ts = await asyncio.gather(*tasks)
    
    for t in ts:
        temperature.update(t)

    return temperature

async def async_validate_temperature(cities, df, api_key, season):
    validations = {}
    
    weather_data = collect_weather_data_multiprocess(cities, df) 
    temperatures = await collect_temperatures(cities, api_key=api_key)
    
    for city in cities:
        season_profile = weather_data[city][3]

        season_top = season_profile.loc[season, "average"] + season_profile.loc[season, "std"]
        season_bottom = season_profile.loc[season, "average"] - season_profile.loc[season, "std"]
        
        if temperatures[city] is None:
            validations.update({city: "Нет данных"})
        elif temperatures[city] > season_top: 
            validations.update({city: "рекорд жары"})
        elif temperatures[city] < season_bottom:
             validations.update({city: "Рекорд холода"})
        else:
             validations.update({city: "Ок"})

    return validations
    