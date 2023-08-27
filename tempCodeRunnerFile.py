import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data from CSV files
yard_locations = pd.read_csv("Yard Locations.csv")
past_in_out_data = pd.read_csv("Past In and Out Container Data.csv")
incoming_containers = pd.read_csv("Incoming Conatiners.csv")

# Preprocess data
past_in_out_data["IN_TIME"] = pd.to_datetime(
    past_in_out_data["IN_TIME"], format="%y-%m-%d %H:%M:%S", errors="coerce"
)
past_in_out_data["OUT_TIME"] = pd.to_datetime(
    past_in_out_data["OUT_TIME"], format="%y-%m-%d %H:%M:%S", errors="coerce"
)
incoming_containers["IN_TIME"] = pd.to_datetime(
    incoming_containers["IN_TIME"], format="%y-%m-%d %H:%M:%S", errors="coerce"
)

# Merge past in and out data with incoming containers data
combined_data = incoming_containers.merge(
    past_in_out_data, left_on="REF_ID", right_on="REF_ID", how="left"
)

# Calculate leave time predictions using Linear Regression
model = LinearRegression()
X = combined_data[["CON_NUM", "CON_SIZE"]]
y = (
    combined_data["IN_TIME_y"].astype(int) + combined_data["VALIDITY"] * 86400000000000
)  # 86400000000000 nanoseconds in a day
model.fit(X, y.reshape(-1, 1))  # Reshape y to 2D array

# Predict leave times for incoming containers
incoming_containers["predicted_leave_time"] = model.predict(
    incoming_containers[["CON_NUM", "CON_SIZE"]]
)
incoming_containers["predicted_leave_time"] = pd.to_datetime(
    incoming_containers["predicted_leave_time"].reshape(-1), unit="ns"
)

# Assign containers to appropriate locations
assigned_locations = []
for index, container in incoming_containers.iterrows():
    suitable_locations = yard_locations[
        (yard_locations["Container Size"] == container["CON_SIZE"])
        & (yard_locations["Location Status"] == "empty")
        & (yard_locations["Location Type"] == container["LOCATION_TYPE"])
    ]

    if not suitable_locations.empty:
        assigned_location = suitable_locations.iloc[0]["Location"]
        yard_locations.loc[
            yard_locations["Location"] == assigned_location, "Location Status"
        ] = "grounded"
        assigned_locations.append(assigned_location)
    else:
        assigned_locations.append("No suitable location found")

incoming_containers["Assigned Location"] = assigned_locations

# Create resultant dataframe with specific columns
resultant = incoming_containers[["ID", "STATUS", "CON_SIZE", "Assigned Location"]]

# Save results to resultant.csv
resultant.to_csv("resultant.csv", index=False)
