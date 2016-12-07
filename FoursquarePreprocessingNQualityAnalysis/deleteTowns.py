import csv


with open("original.csv", "rb") as inp, open("towns_deleted.csv", "wb") as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[3] != "Village":
            writer.writerow(row)

