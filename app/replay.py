import pandas as pd


class ReplayFeed:
    def __init__(self, df: pd.DataFrame):
        self.records = {server: group.reset_index(drop=True).to_dict("records")
                        for server, group in df.groupby("server_id")}

    def has(self, server_id: str) -> bool:
        return bool(self.records.get(server_id))

    def next_reading(self, server_id: str, cursors: dict):
        records = self.records[server_id]
        position = cursors.get(server_id, 0)
        wrapped = position >= len(records)
        if wrapped:
            position = 0
        cursors[server_id] = position + 1
        return dict(records[position]), wrapped
