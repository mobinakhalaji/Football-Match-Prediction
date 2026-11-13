import os
import re
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


def load_dataset() -> pd.DataFrame:
    primary_file = "football_matches.csv"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_file = os.path.join(base_dir, primary_file)

    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

    print(f"Loading dataset: {dataset_file}")
    return pd.read_csv(dataset_file)


def extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    date_cols = [
        c for c in df.columns if re.search(r"_recent_fixture_date_\d+$", c)
    ]

    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    if "fixture_date" in df.columns:
        df["fixture_date"] = pd.to_datetime(df["fixture_date"], errors="coerce")
        df["fixture_year"] = df["fixture_date"].dt.year
        df["fixture_month"] = df["fixture_date"].dt.month
        df["fixture_day"] = df["fixture_date"].dt.day
        df["fixture_weekday"] = df["fixture_date"].dt.weekday
    else:
        df["fixture_year"] = -1
        df["fixture_month"] = -1
        df["fixture_day"] = -1
        df["fixture_weekday"] = -1

    def add_recent_date_features(prefix: str, suffixes: range):
        date_columns = [f"{prefix}_recent_fixture_date_{i}" for i in suffixes]
        valid_columns = [c for c in date_columns if c in df.columns]

        if not valid_columns:
            return

        df[f"{prefix}_recent_matches_count"] = df[valid_columns].notna().sum(axis=1)
        df[f"{prefix}_days_since_last_match"] = (
            df["fixture_date"] - df[valid_columns[0]]
        ).dt.days
        df[f"{prefix}_days_since_last_match"] = df[f"{prefix}_days_since_last_match"].fillna(999)

    add_recent_date_features("home_club", range(1, 11))
    add_recent_date_features("away_club", range(1, 11))

    return df


def aggregate_recent_stats(df: pd.DataFrame, prefix: str, stats: list) -> pd.DataFrame:
    for stat_name in stats:
        pattern = f"{prefix}_recent_{stat_name}_"
        columns = [c for c in df.columns if c.startswith(pattern)]
        columns = sorted(columns, key=lambda x: int(re.search(r"_(\d+)$", x).group(1)))

        if not columns:
            continue

        df[f"{prefix}_recent_{stat_name}_mean"] = df[columns].mean(axis=1)
        df[f"{prefix}_recent_{stat_name}_sum"] = df[columns].sum(axis=1)
        df[f"{prefix}_recent_{stat_name}_last"] = df[columns[0]]
        df[f"{prefix}_recent_{stat_name}_last3_mean"] = df[columns[:3]].mean(axis=1)

    return df


def report_cleaning(df: pd.DataFrame):
    initial_rows = len(df)
    duplicate_rows = df.duplicated().sum()
    critical_cols = [
        "match_result",
        "home_club",
        "away_club",
        "fixture_date",
        "competition",
    ]
    df_no_duplicates = df.drop_duplicates()
    df_cleaned = df_no_duplicates.dropna(subset=critical_cols)
    removed_duplicates = initial_rows - len(df_no_duplicates)
    removed_missing_critical = len(df_no_duplicates) - len(df_cleaned)
    total_removed = initial_rows - len(df_cleaned)

    print("\nData cleaning report:")
    print(f"  Initial rows: {initial_rows}")
    print(f"  Duplicate rows: {duplicate_rows}")
    print(f"  Rows with missing critical values ({', '.join(critical_cols)}): {removed_missing_critical}")
    print(f"  Rows that would be removed by duplicate/critical cleaning: {total_removed}")
    print(f"  Rows with any missing value: {df.isna().any(axis=1).sum()}")
    print("  Note: current pipeline uses imputation and does not drop rows except for dropped duplicates or missing critical data if you choose to enable it.")


def preprocess(df: pd.DataFrame):
    if "match_id" in df.columns:
        df = df.drop("match_id", axis=1)

    df = extract_date_features(df)

    recent_stats = [
        "goals_scored",
        "goals_conceded",
        "team_rating",
        "opponent_rating",
        "played_at_home",
        "cup_game",
    ]

    df = aggregate_recent_stats(df, "home_club", recent_stats)
    df = aggregate_recent_stats(df, "away_club", recent_stats)

    # Drop raw date columns after extracting features
    raw_date_cols = [
        c
        for c in df.columns
        if re.search(r"_recent_fixture_date_\d+$", c) or c == "fixture_date"
    ]
    df = df.drop(columns=raw_date_cols, errors="ignore")

    # Separate target
    y = df["match_result"].astype(str)
    X = df.drop("match_result", axis=1)

    # Encode categorical object columns
    label_encoders = {}
    for col in X.select_dtypes(include=["object", "string"]).columns:
        X[col] = X[col].fillna("missing").astype(str)
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Impute missing numeric values
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)

    return X, y, target_encoder


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    return model, X_test, y_test, y_pred


if __name__ == "__main__":
    df = load_dataset()
    report_cleaning(df)
    X, y, target_encoder = preprocess(df)
    model, X_test, y_test, y_pred = train_and_evaluate(X, y)

    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))
