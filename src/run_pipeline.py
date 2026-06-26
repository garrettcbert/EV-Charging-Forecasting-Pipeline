import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import extract_api
import transform_load
import simulate_target
import train_model

def run_full_pipeline():
    print('=' * 60)
    print("STEP 1/4: Extracting EV stations and weather data")
    print('=' * 60)
    extract_api.run()

    print("\n" + '=' * 60)
    print("STEP 2/4: Transforming and loading data into SQLite")
    transform_load.run()

    print('\n' + '=' * 60)
    print("STEP 3/4: Simulating utilization target variable")
    print('=' * 60)
    simulate_target.run()

    print("\n" + "=" * 60)
    print("STEP 4/4: Training and saving model")
    print("=" * 60)
    train_model.run()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_full_pipeline()