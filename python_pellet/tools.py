import datetime
import os
import csv
from pathlib import Path

def get_writer():
    _, writer = os.pipe()
    return os.fdopen(writer, 'w')


# game name                        steps    should i process file instead of running?   log file name
# MontezumaRevengeDeterministic-v4 10000000 y                                           2021-04-27-1441-actor[2]-log
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
                if line.__contains__("|"):
                    items = line.split('|')
                    printable_items = [items[1].strip(),
                                       items[3].strip(),
                                       items[5].strip(),
                                       items[7].strip()]
                    writer.writerow(printable_items)


def process_dis(in_file_name: str):
    now = datetime.datetime.now()
    now_but_text = str(now.date()) + '-' + str(now.hour) + str(now.minute)
    if in_file_name.split('.')[-1] is not ".txt":
        in_file_name += ".text"
    path = Path(__file__).parent
    csv_path = path / "logs" / (now_but_text + ".csv")
    in_file_path = path / "logs" / (in_file_name + ".txt")
    with open(csv_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile, )
        with open(in_file_path, 'r') as infile:
            start = infile.readline().split('|')[1]
            end = ''
            for line in infile:
                end = line.split('|')[1]
            start = datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S.%f')
            end = datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S.%f')
            running_time = (end-start).total_seconds()
            running_time = str(int(running_time))

        in_file_path = in_file_path.__str__().replace('manager', 'learner')
        with open(in_file_path, 'r') as infile:
            j = 0
            for line in infile:
                if line.endswith("que\n"):
                    j += 1

        back_props_done = j * 1000
        writer.writerow([running_time, back_props_done])
