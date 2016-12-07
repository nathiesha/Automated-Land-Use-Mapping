import csv

with open("original.csv", "rb") as inp, open("reliable.csv", "wb") as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if int(row[6]) < 2:
            writer.writerow(row)

