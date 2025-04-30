import pandas as pd
from deepecho import PARModel

df = pd.read_json("data/real_data_train_pre.json")

# add sequence index by adding the date back together. This will later on not be generated
df['validity_start'] = pd.to_datetime(
    df.rename(columns={
        'validity_start_year': 'year',
        'validity_start_month': 'month',
        'validity_start_day': 'day'
    })[['year', 'month', 'day']]
)

categorical = "categorical"
count = "count"
continuous = "continuous"
entity_columns = ["player_id"]
sequence_index = "validity_start"
context_columns = ["first_name", "last_name", "pseudonym", "position", "foot", "citizenship", "height",
                   "date_of_birth_year", "date_of_birth_month", "date_of_birth_day"]

data_types = {
    "injury_category": categorical,
    "market_value_category": categorical,
    "age": count,
    "coach_id": categorical,
    "club_id": categorical,
    "league_id": categorical,
    "club": categorical,
    "league": categorical,
    "season_id": categorical,
    "injury": categorical,
    "last_transfer_fee": count,
    "coach": categorical,
    "market_value": continuous,
    "league_played_matches": count,
    "league_minutes_played": count,
    "league_goals": count,
    "international_goals": count,
    "international_minutes_played": count,
    "international_played_matches": count,
    "international_competition": categorical,
    "missed_matches": count,
    "validity_start_year": categorical,
    "validity_start_month": categorical,
    "validity_start_day": categorical,
    "validity_end_year": categorical,
    "validity_end_month": categorical,
    "validity_end_day": categorical,
    "date_of_birth_year": categorical,
    "date_of_birth_month": categorical,
    "date_of_birth_day": categorical,
    "reason_regular_interval": categorical,
    "reason_new_coach": categorical,
    "reason_transfer": categorical,
    "reason_market_value_update": categorical,
    "reason_injury": categorical,
    "reason_injury_end": categorical
}

entities_in_real_data = df.player_id.nunique()
print(f"entities in real data: {entities_in_real_data}")

model = PARModel(epochs=1024, cuda=True)
model.fit(
    data=df,
    entity_columns=entity_columns,
    context_columns=context_columns,
    data_types=data_types,
    sequence_index=sequence_index,
)
samples = model.sample(entities_in_real_data*2)
samples.to_json("par_samples.json", orient="records")
