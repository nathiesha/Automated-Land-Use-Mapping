import csv

counter = 0
with open("original.csv", "rb") as inp, open("city_deleted.csv", "wb") as out:
    writer = csv.writer(out)
    for row in csv.reader(inp):
        if row[3] != "City":
            writer.writerow(row)
        else:
            counter += 1
    print counter