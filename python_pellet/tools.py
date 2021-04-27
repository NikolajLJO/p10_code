import datetime
import os
import csv
from pathlib import Path

def get_writer():
    _, writer = os.pipe()
    return os.fdopen(writer, 'w')

# MontezumaRevengeDeterministic-v4 10000000 y 2021-04-27-1441-actor[2]-log <- this works
def process_score_over_steps(in_file_name: str):
    now = datetime.datetime.now()
    now_but_text = str(now.date()) + '-' + str(now.hour) + str(now.minute)
    # TODO if file name not end txt add it
    path = Path(__file__).parent
    csv_path = path / "logs" / (now_but_text + ".csv")
    in_file_path = path / "logs" / (in_file_name + ".txt")
    with open(csv_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile, )
        with open(in_file_path, 'r') as infile:
            for line in infile:
                items = line.split('|')
                printable_items = [items[1].strip(),
                                   items[3].strip(),
                                   items[5].strip(),
                                   items[7].strip()]
                writer.writerow(printable_items)
