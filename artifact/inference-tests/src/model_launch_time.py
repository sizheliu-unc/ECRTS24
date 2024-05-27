import pandas as pd
import sqlite3
import argparse

parser = argparse.ArgumentParser(
                    prog='CUDA launch time parser',
                    description='parse sqlite from nsys into pure statistics of cuda program launch time')

class Processor:
    def __init__(self, filename) -> None:
        self.con = sqlite3.connect(filename)

    def process(self) -> None:
        pass

class ModelLaunchTimeProcessor(Processor):
    def __init__(self, filename, graph=True) -> None:
        super().__init__(filename)
        stringIds = pd.read_sql_query("SELECT id,value FROM StringIds WHERE value IN ('cudaMemcpyAsync','cudaGraphLaunch')", self.con)
        if graph:
            nameId = stringIds[stringIds['value']=='cudaGraphLaunch'].iloc[0]['id']
        else:
            nameId = stringIds[stringIds['value']=='cudaMemcpyAsync'].iloc[0]['id']
        self.memcpy_df = pd.read_sql_query( f"SELECT start,end,correlationId,nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE nameId={nameId} order by start ASC", self.con).iloc[30:]
        self.gpumemcpy_df = pd.read_sql_query ("SELECT start,end,correlationId,copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE copyKind=1 order by start ASC", self.con).iloc[30:]

    def process(self) -> None:
        df = pd.merge(self.memcpy_df, self.gpumemcpy_df, 'inner', ["correlationId"])
        df['launchTime'] = df['start_y'] - df['start_x']
        print(f"Launch time: {round(df['launchTime'].mean() / 1000, 2)} us")

parser.add_argument('filename', type=str, help="file name, must use sqlite file.")
parser.add_argument('--graph', action='store_true', default=False, help="used cuda graph?")
def main():
    args = parser.parse_args()
    processor = ModelLaunchTimeProcessor(args.filename, args.graph)
    processor.process()

if __name__ == "__main__":
    main()
