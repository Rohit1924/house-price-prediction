import io
import os
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="India House Price Prediction API",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================
# LOAD MODELS
# =========================================================

def load_model(filename):
    path = os.path.join(BASE_DIR, filename)

    if not os.path.exists(path):
        raise RuntimeError(f"Model file not found: {filename}")

    return joblib.load(path)


models = {
    "mumbai": load_model("mumbai_model.pkl"),
    "bengaluru": load_model("bengaluru_model.pkl"),
    "hyderabad": load_model("hyderabad_model.pkl"),
    "delhi": load_model("delhi_model.pkl")
}


# =========================================================
# PYDANTIC INPUT MODELS
# =========================================================

class House(BaseModel):
    area: float = Field(gt=0, le=10000)
    bedrooms: int = Field(ge=1, le=10)
    location: str = Field(min_length=1)


class DelhiHouse(House):
    bathrooms: float = Field(ge=0, le=20)
    balcony: float = Field(ge=0, le=20)

    status: str = Field(min_length=1)
    neworold: str = Field(min_length=1)

    parking: float = Field(ge=0)

    furnished: str = Field(min_length=1)

    lift: float = Field(ge=0)

    type_of_building: str = Field(min_length=1)


# =========================================================
# HELPERS
# =========================================================

def clean(value):
    return " ".join(
        str(value).strip().lower().split()
    )


def normalize_city(city):
    city = clean(city)

    if city == "bangalore":
        city = "bengaluru"

    return city


def format_price(city, price):

    price = max(float(price), 0)

    return {
        "success": True,
        "city": city.title(),
        "predicted_price": round(price, 2),
        "price_lakh": round(price / 100000, 2),
        "price_crore": round(price / 10000000, 2)
    }


# =========================================================
# LOCATION EXTRACTION
# =========================================================

def extract_locations(model):
    """
    Model structure:

    Pipeline
        -> preprocessor
            -> ColumnTransformer
                -> cat
                    -> OneHotEncoder
                        -> categories_[0]
    """

    try:

        preprocessor = model.named_steps["preprocessor"]

        encoder = preprocessor.named_transformers_["cat"]

        categories = encoder.categories_[0]

        locations = []

        for value in categories:

            location = str(value).strip()

            if (
                location
                and location.lower() != "nan"
            ):
                locations.append(location)

        # Remove duplicate locations
        locations = list(dict.fromkeys(locations))

        # Sort locations
        locations.sort(
            key=lambda x: x.lower()
        )

        return locations

    except Exception as e:

        print(
            "Location extraction failed:",
            e
        )

        return []


# =========================================================
# LOAD LOCATIONS ONCE
# =========================================================

city_locations = {}

for city_name, model in models.items():

    city_locations[city_name] = (
        extract_locations(model)
    )

    print(
        f"{city_name.upper()} -> "
        f"{len(city_locations[city_name])} locations loaded"
    )


# =========================================================
# API INFORMATION
# =========================================================

@app.get("/api")
def api_home():

    return {
        "success": True,
        "message": "House Price Prediction API Running",
        "cities": list(models.keys())
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "models_loaded": len(models),

        "locations_loaded": {
            city: len(locations)
            for city, locations
            in city_locations.items()
        }
    }


# =========================================================
# CITY LOCATIONS
# IMPORTANT FOR FRONTEND DROPDOWN
# =========================================================

@app.get("/locations/{city}")
def get_locations(city: str):

    city = normalize_city(city)

    if city not in models:

        raise HTTPException(
            status_code=404,
            detail="City not supported."
        )

    locations = city_locations.get(
        city,
        []
    )

    if not locations:

        raise HTTPException(
            status_code=500,
            detail=(
                f"No trained locations could be "
                f"loaded for {city.title()}."
            )
        )

    return {
        "success": True,
        "city": city.title(),
        "count": len(locations),
        "locations": locations
    }


# =========================================================
# BASIC PREDICTION
# MUMBAI / BENGALURU / HYDERABAD
# =========================================================

def predict_basic(city, data):

    input_df = pd.DataFrame([
        {
            "area": data.area,
            "bedrooms": data.bedrooms,
            "location": clean(data.location)
        }
    ])

    try:

        prediction = models[city].predict(
            input_df
        )[0]

        return format_price(
            city,
            prediction
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# =========================================================
# MUMBAI
# =========================================================

@app.post("/predict/mumbai")
def predict_mumbai(data: House):

    return predict_basic(
        "mumbai",
        data
    )


# =========================================================
# BENGALURU
# =========================================================

@app.post("/predict/bengaluru")
def predict_bengaluru(data: House):

    return predict_basic(
        "bengaluru",
        data
    )


# =========================================================
# HYDERABAD
# =========================================================

@app.post("/predict/hyderabad")
def predict_hyderabad(data: House):

    return predict_basic(
        "hyderabad",
        data
    )


# =========================================================
# DELHI
# =========================================================

@app.post("/predict/delhi")
def predict_delhi(data: DelhiHouse):

    input_df = pd.DataFrame([
        {
            "area": data.area,

            "bedrooms": data.bedrooms,

            "location":
                clean(data.location),

            "bathrooms":
                data.bathrooms,

            "balcony":
                data.balcony,

            "status":
                clean(data.status),

            "neworold":
                clean(data.neworold),

            "parking":
                data.parking,

            "furnished":
                clean(data.furnished),

            "lift":
                data.lift,

            "type_of_building":
                clean(data.type_of_building)
        }
    ])

    try:

        prediction = models[
            "delhi"
        ].predict(
            input_df
        )[0]

        return format_price(
            "delhi",
            prediction
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# =========================================================
# CSV BULK PREDICTION
# =========================================================

@app.post("/predict-file")
async def predict_file(
    city: str = Form(...),
    file: UploadFile = File(...)
):

    city = normalize_city(city)


    # =====================================================
    # CHECK CITY
    # =====================================================

    if city not in models:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid city. Use Mumbai, Bengaluru, "
                "Hyderabad or Delhi."
            )
        )


    # =====================================================
    # CHECK FILE
    # =====================================================

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Please select a CSV file."
        )


    if not file.filename.lower().endswith(".csv"):

        raise HTTPException(
            status_code=400,
            detail="Only CSV files are allowed."
        )


    # =====================================================
    # READ CSV
    # =====================================================

    try:

        contents = await file.read()

        df = pd.read_csv(
            io.BytesIO(contents)
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Could not read CSV: {str(e)}"
        )


    if df.empty:

        raise HTTPException(
            status_code=400,
            detail="CSV file is empty."
        )


    # =====================================================
    # CLEAN COLUMN NAMES
    # =====================================================

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(
            " ",
            "_",
            regex=False
        )
        .str.replace(
            ".",
            "",
            regex=False
        )
    )


    # =====================================================
    # COMMON COLUMN NAME SUPPORT
    # =====================================================

    aliases = {

        # Bedrooms
        "bhk": "bedrooms",
        "bedroom": "bedrooms",
        "bed_rooms": "bedrooms",
        "no_of_bedrooms": "bedrooms",
        "no_of_bedroom": "bedrooms",

        # Area
        "total_sqft": "area",
        "total_sq_feet": "area",
        "sqft": "area",
        "sq_ft": "area",
        "square_feet": "area",

        # Location
        "region": "location",
        "locality": "location",
        "address": "location",

        # Bathroom
        "bath": "bathrooms",
        "bathroom": "bathrooms",

        # Furnishing
        "furnished_status": "furnished",
        "furnishing": "furnished",
        "furnishing_status": "furnished",

        # Building
        "type": "type_of_building",
        "building_type": "type_of_building",

        # New / Old
        "new_or_old": "neworold",
        "new_or_resale": "neworold"
    }


    for old_name, new_name in aliases.items():

        if (
            old_name in df.columns
            and new_name not in df.columns
        ):

            df.rename(
                columns={
                    old_name: new_name
                },
                inplace=True
            )


    # =====================================================
    # REQUIRED FEATURES
    # =====================================================

    if city == "delhi":

        required_columns = [
            "area",
            "bedrooms",
            "location",
            "bathrooms",
            "balcony",
            "status",
            "neworold",
            "parking",
            "furnished",
            "lift",
            "type_of_building"
        ]


        text_columns = [
            "location",
            "status",
            "neworold",
            "furnished",
            "type_of_building"
        ]

    else:

        required_columns = [
            "area",
            "bedrooms",
            "location"
        ]


        text_columns = [
            "location"
        ]


    # =====================================================
    # CHECK MISSING COLUMNS
    # =====================================================

    missing_columns = [

        column

        for column in required_columns

        if column not in df.columns

    ]


    if missing_columns:

        raise HTTPException(
            status_code=400,

            detail={
                "message":
                    "Required columns are missing.",

                "missing_columns":
                    missing_columns,

                "available_columns":
                    list(df.columns)
            }
        )


    # =====================================================
    # CLEAN TEXT COLUMNS
    # =====================================================

    for column in text_columns:

        df[column] = (
            df[column]
            .astype(str)
            .apply(clean)
        )


    # =====================================================
    # NUMERIC COLUMNS
    # =====================================================

    numeric_columns = [

        column

        for column in required_columns

        if column not in text_columns

    ]


    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    # =====================================================
    # VALID ROWS
    # =====================================================

    valid_rows = (
        df[required_columns]
        .notna()
        .all(axis=1)
    )


    valid_rows &= (
        df["area"] > 0
    )


    valid_rows &= (
        df["area"] <= 10000
    )


    valid_rows &= (
        df["bedrooms"] >= 1
    )


    valid_rows &= (
        df["bedrooms"] <= 10
    )


    prediction_df = (
        df.loc[valid_rows]
        .copy()
    )


    if prediction_df.empty:

        raise HTTPException(
            status_code=400,
            detail="No valid rows found in CSV."
        )


    # =====================================================
    # PREDICTION
    # =====================================================

    try:

        predictions = models[
            city
        ].predict(
            prediction_df[
                required_columns
            ]
        )


        predictions = [
            max(float(price), 0)
            for price in predictions
        ]


        prediction_df[
            "predicted_price"
        ] = predictions


        prediction_df[
            "price_lakh"
        ] = (
            prediction_df[
                "predicted_price"
            ] / 100000
        ).round(2)


        prediction_df[
            "price_crore"
        ] = (
            prediction_df[
                "predicted_price"
            ] / 10000000
        ).round(2)


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


    # =====================================================
    # CREATE DOWNLOAD CSV
    # =====================================================

    output = io.StringIO()


    prediction_df.to_csv(
        output,
        index=False
    )


    output.seek(0)


    return StreamingResponse(

        iter([
            output.getvalue()
        ]),

        media_type="text/csv",

        headers={
            "Content-Disposition":
                f'attachment; filename="{city}_predictions.csv"'
        }
    )


# =========================================================
# FRONTEND
#
# VERY IMPORTANT:
# THIS MUST REMAIN AFTER ALL API ROUTES.
# =========================================================

STATIC_DIR = os.path.join(
    BASE_DIR,
    "static"
)


if os.path.isdir(STATIC_DIR):

    app.mount(
        "/",
        StaticFiles(
            directory=STATIC_DIR,
            html=True
        ),
        name="frontend"
    )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )