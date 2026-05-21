import pickle

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TEAM_RENAMES = {
    "Deccan Chargers": "Sunrisers Hyderabad",
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
}

ACTIVE_TEAMS = sorted(
    [
        "Chennai Super Kings",
        "Delhi Capitals",
        "Gujarat Titans",
        "Kolkata Knight Riders",
        "Lucknow Super Giants",
        "Mumbai Indians",
        "Punjab Kings",
        "Rajasthan Royals",
        "Royal Challengers Bangalore",
        "Sunrisers Hyderabad",
    ]
)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


def normalize_teams(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        df[col] = df[col].replace(TEAM_RENAMES)
    return df


def build_training_frame(matches_path: str, deliveries_path: str) -> pd.DataFrame:
    matches = normalize_columns(pd.read_csv(matches_path))
    deliveries = normalize_columns(pd.read_csv(deliveries_path))

    matches = normalize_teams(matches, ["team1", "team2", "toss_winner", "winner"])
    deliveries = normalize_teams(deliveries, ["batting_team", "bowling_team"])

    numeric_delivery_cols = [
        "match_id",
        "inning",
        "over",
        "ball",
        "total_runs",
        "is_wicket",
    ]
    for col in numeric_delivery_cols:
        deliveries[col] = pd.to_numeric(deliveries[col], errors="coerce")

    deliveries = deliveries.dropna(subset=numeric_delivery_cols)
    deliveries[numeric_delivery_cols] = deliveries[numeric_delivery_cols].astype(int)

    matches["id"] = pd.to_numeric(matches["id"], errors="coerce").astype(int)
    matches["city"] = matches["city"].replace({"": pd.NA})

    matches = matches[
        matches["team1"].isin(ACTIVE_TEAMS)
        & matches["team2"].isin(ACTIVE_TEAMS)
        & matches["winner"].isin(ACTIVE_TEAMS)
        & matches["city"].notna()
        & (matches["result"] != "no result")
        & (matches["super_over"] == "N")
        & matches["method"].isna()
    ].copy()

    first_innings = (
        deliveries[deliveries["inning"] == 1]
        .groupby("match_id", as_index=False)["total_runs"]
        .sum()
        .rename(columns={"total_runs": "total_runs_x"})
    )

    match_df = matches.merge(first_innings, left_on="id", right_on="match_id")

    chase_df = match_df.merge(deliveries, left_on="id", right_on="match_id")
    chase_df = chase_df[chase_df["inning"] == 2].copy()
    chase_df = chase_df[
        chase_df["batting_team"].isin(ACTIVE_TEAMS)
        & chase_df["bowling_team"].isin(ACTIVE_TEAMS)
    ].copy()
    chase_df["match_key"] = chase_df["id"]

    chase_df["current_score"] = chase_df.groupby("match_key")["total_runs"].cumsum()
    chase_df["wickets_fallen"] = chase_df.groupby("match_key")["is_wicket"].cumsum()
    chase_df["runs_left"] = chase_df["total_runs_x"] - chase_df["current_score"] + 1
    chase_df["balls_left"] = 120 - (chase_df["over"] * 6 + chase_df["ball"])
    chase_df["wickets"] = 10 - chase_df["wickets_fallen"]
    chase_df["balls_bowled"] = chase_df["over"] * 6 + chase_df["ball"]
    chase_df["crr"] = chase_df["current_score"] * 6 / chase_df["balls_bowled"]
    chase_df["rrr"] = chase_df["runs_left"] * 6 / chase_df["balls_left"]
    chase_df["winner"] = (chase_df["batting_team"] == chase_df["winner"]).astype(int)

    final_df = chase_df[
        [
            "batting_team",
            "bowling_team",
            "city",
            "runs_left",
            "balls_left",
            "wickets",
            "total_runs_x",
            "crr",
            "rrr",
            "winner",
        ]
    ].copy()

    final_df = final_df.replace([pd.NA, float("inf"), float("-inf")], pd.NA).dropna()
    final_df = final_df[
        (final_df["balls_left"] > 0)
        & (final_df["runs_left"] >= 0)
        & (final_df["wickets"] > 0)
    ].copy()

    return final_df


def train_and_save_model(matches_path: str, deliveries_path: str, output_path: str) -> None:
    final_df = build_training_frame(matches_path, deliveries_path)

    X = final_df.drop(columns="winner")
    y = final_df["winner"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = Pipeline(
        steps=[
            (
                "step1",
                ColumnTransformer(
                    transformers=[
                        (
                            "trf",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                drop="first",
                            ),
                            ["batting_team", "bowling_team", "city"],
                        )
                    ],
                    remainder="passthrough",
                ),
            ),
            ("step2", LogisticRegression(solver="liblinear", max_iter=1000)),
        ]
    )

    pipe.fit(X_train, y_train)

    with open(output_path, "wb") as model_file:
        pickle.dump(pipe, model_file)

    print(f"Training rows: {len(final_df)}")
    print(f"Train score: {pipe.score(X_train, y_train):.4f}")
    print(f"Test score: {pipe.score(X_test, y_test):.4f}")

    encoder = pipe.named_steps["step1"].named_transformers_["trf"]
    print("Batting teams:", list(encoder.categories_[0]))
    print("Bowling teams:", list(encoder.categories_[1]))
    print("Cities:", list(encoder.categories_[2]))


if __name__ == "__main__":
    train_and_save_model(
        "matches_2008-2024.csv",
        "deliveries_2008-2024.csv",
        "pipe.pkl",
    )
