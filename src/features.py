import pandas as pd


def add_feats(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    X["HasCabin"] = X["Cabin"].notna().astype(int)
    X["Deck"] = X["Cabin"].apply(lambda x: x[0] if pd.notna(x) and x != "" else "Unknown")
    X["Company"] = X["SibSp"] + X["Parch"]
    X["Alone"] = (X["Company"] == 0).astype(int)
    X["Title"] = X["Name"].apply(lambda x: x.split(",")[1].split()[0])
    incl_title = ["Mr.", "Miss.", "Mrs.", "Master."]
    X["Title"] = X["Title"].apply(lambda x: x if x in incl_title else "others")
    X["Family"] = X["Name"].apply(lambda x: x.split(",")[0])
    X["Married"] = (X["Title"] == "Mrs.").astype(int)

    def deck_class(deck: str) -> str:
        if deck == "Unknown":
            return "Unknown"
        if deck in ["C", "B", "D", "E"]:
            return "Common"
        return "Rare"

    X["Deck"] = X["Deck"].apply(deck_class)
    return X
