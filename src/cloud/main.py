import os

from fastapi import (
    FastAPI,
    Header,
    HTTPException
)

from pydantic import BaseModel

from src.cloud.predictor import (
    predict_thermal
)


app = FastAPI(

    title=(
        "PC Health ML API"
    ),

    description=(
        "Cloud-assisted machine learning "
        "API for personal computer "
        "hardware anomaly detection."
    ),

    version="0.1.0"
)


API_KEY = os.getenv(
    "PC_HEALTH_API_KEY",
    "development-key"
)


class ThermalTelemetry(
    BaseModel
):

    machine_id: str

    cpu_load_mean: float

    cpu_temp_mean: float

    cpu_power_mean: float

    cpu_p_clock_mean: float

    cpu_e_clock_mean: float

    cpu_fan_mean: float

    pump_fan_mean: float

    gpu_load_mean: float

    gpu_temp_mean: float

    gpu_power_mean: float

    ssd_temp_mean: float

    motherboard_system_temp_mean: float

    motherboard_vrm_temp_mean: float

    motherboard_pch_temp_mean: float

    cpu_socket_temp_mean: float

    cpu_temp_delta: float


@app.get("/")
def root():

    return {

        "service":
            "PC Health ML API",

        "version":
            "0.1.0",

        "status":
            "running"
    }


@app.get("/health")
def health():

    return {

        "status":
            "healthy",

        "model_loaded":
            True
    }


@app.post(
    "/predict/thermal"
)
def thermal_prediction(

    telemetry:
        ThermalTelemetry,

    x_api_key:
        str = Header(default="")
):

    if x_api_key != API_KEY:

        raise HTTPException(

            status_code=401,

            detail="Invalid API key"
        )


    data = telemetry.model_dump()


    machine_id = data.pop(
        "machine_id"
    )


    prediction = predict_thermal(
        data
    )


    return {

        "machine_id":
            machine_id,

        "model":
            "thermal_expert_v0.1",

        **prediction
    }