import pandas as pd
df = pd.read_json("data/real_data_test_pre.json")

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
context_columns = ["first_name", "last_name", "pseudonym", "position", "foot", "citizenship", "height", ]

#todo maybe use datetime type for the dates
# can use suffix _idx for validity_start for the idx and validity_start fpr the real one
data_types = {
    "injury_category": categorical,
    "market_value_category": categorical,
    "age": count,
    "coach_id": categorical,
    "club_id": categorical,
    "league_id": categorical,
    "club": categorical,
    "league": categorical,
    "season_id": count,
    "injury": categorical,
    "last_transfer_fee": count,
    "coach": categorical,
    "market_value": continuous,
    "league_played_matches": count,
     "league_minutes_played": count,
    "league_goals": count,
    "international_goals": count,
    "international_minutes_played": count,
    "international_playd_matches": count,
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
from deepecho import PARModel

model = PARModel(epochs=1024, cuda=True)
model.fit(
    data=df,
    entity_columns=entity_columns,
    context_columns=context_columns,
    data_types=data_types,
    sequence_index=sequence_index,
)
samples = model.sample(entities_in_real_data)
samples.to_json("par_samples.json", orient="records")
