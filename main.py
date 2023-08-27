import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import heapq
from datetime import datetime


# Define the calculate_moves_required function
def calculate_moves_required(
    container_incoming_time, container_predicted_departure, all_other_containers
):
    container_duration = (
        container_predicted_departure - container_incoming_time
    ).total_seconds()

    moves_required = 0
    for other_container in all_other_containers:
        if other_container["CON_NUM"] == container["CON_NUM"]:
            continue  # Skip the current container
        other_duration = (
            container_predicted_departure - other_container["IN_TIME"]
        ).total_seconds()
        if other_duration > container_duration:
            moves_required += 1

    return moves_required


# Define the find_optimal_location function
def find_optimal_location(container, available_space):
    con_size = container["CON_SIZE"]
    departure_time = container["IN_TIME"]

    optimal_location = None
    min_moves = float("inf")

    for block, block_data in available_space.items():
        for row, row_data in block_data.items():
            for bay, bay_data in row_data.items():
                for tier, tier_data in bay_data.items():
                    if con_size == tier_data["Container Size"]:
                        other_containers = [
                            c
                            for c in incoming_containers
                            if c["CON_NUM"] != container["CON_NUM"]
                        ]
                        moves_required = calculate_moves_required(
                            container["IN_TIME"], departure_time, other_containers
                        )

                        if moves_required < min_moves:
                            min_moves = moves_required
                            bay_formatted = f"{int(bay):02}"
                            optimal_location = f"{block}{bay_formatted}{row}{tier}"

    return optimal_location


# Load data from CSV files
yard_locations = pd.read_csv("Yard Locations.csv")
past_in_out_data = pd.read_csv("Past In and Out Container Data.csv")
incoming_containers = pd.read_csv("Incoming Conatiners.csv")

# Preprocess data
yard_locations["Location"] = (
    yard_locations["Area"].astype(str)
    + yard_locations["Row"].astype(str)
    + yard_locations["Bay"].astype(str)
    + yard_locations["Level"].astype(str)
)

past_in_out_data["IN_TIME"] = pd.to_datetime(
    past_in_out_data["IN_TIME"], format="%d-%m-%Y %H:%M", errors="coerce"
)
past_in_out_data["OUT_TIME"] = pd.to_datetime(
    past_in_out_data["OUT_TIME"], format="%d-%m-%Y %H:%M", errors="coerce"
)
incoming_containers["IN_TIME"] = pd.to_datetime(
    incoming_containers["IN_TIME"], format="%d-%m-%Y %H:%M", dayfirst=True
)

past_in_out_data.dropna(subset=["IN_TIME", "OUT_TIME"], inplace=True)
incoming_containers.dropna(subset=["IN_TIME"], inplace=True)

# Train Linear Regression model with data preprocessing
imputer = SimpleImputer(strategy="mean")
X = past_in_out_data[["CON_SIZE"]]
X_imputed = imputer.fit_transform(X)
model = LinearRegression()
y = (past_in_out_data["OUT_TIME"] - datetime(1970, 1, 1)).dt.total_seconds()
model.fit(X_imputed, y)

# Initialize data structures
priority_queue = []
available_space = {loc: {} for loc in yard_locations["Location"]}
assigned_locations = {}

# Populate container list and sort by predicted departure time
container_list = []
for _, container in incoming_containers.iterrows():
    predicted_departure_time = model.predict([[container["CON_SIZE"]]])[0]
    predicted_departure_unix = float(predicted_departure_time)
    container_dict = container.to_dict()
    container_list.append((predicted_departure_unix, container_dict))

# Sort the container list by predicted departure time
container_list.sort(key=lambda x: x[0])

generated_strings = []

for letter1 in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    for num1 in range(1, 100):
        num_str = str(num1).zfill(2)
        for letter2 in "ABCDEF":
            for digit in range(1, 10):
                generated_string = f"{letter1}{num_str}{letter2}{digit}"
                generated_strings.append(generated_string)

# Placement algorithm
assigned_locations = {}
for _, container_dict in container_list:
    container_num = container_dict["CON_NUM"]  # Use CON_NUM as the unique identifier
    con_size = container_dict["CON_SIZE"]
    in_time = container_dict["IN_TIME"]

    if generated_strings:
        assigned_location = generated_strings.pop(0)
    else:
        assigned_location = None  # Handle the case when all generated strings are used

    assigned_locations[container_num] = {
        "CON_NUM": container_dict["CON_NUM"],
        "CON_SIZE": container_dict["CON_SIZE"],
        "STATUS": container_dict["STATUS"],
        "Assigned Location": assigned_location,
    }

# Save assigned locations to CSV
resultant_data = list(assigned_locations.values())
resultant = pd.DataFrame(resultant_data)
resultant.to_csv("ResultTab.csv", index=False)
