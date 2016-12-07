import csv

counter = 0
with open("original.csv", "rb") as inp, open("neighbourhoods_deleted.csv", "wb") as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[3] != "Neighborhood":
            writer.writerow(row)
        else:
            counter += 1
    print counter

